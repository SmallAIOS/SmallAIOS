// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-user denial rate limiter.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-audit-log/spec.md`
//! Requirement "Denial audit with rate limit".
//!
//! Each user is allowed [`DENIAL_RATE_PER_SECOND`] = 10 individual
//! `DENY:<syscall>` records per rolling 1-second window. Excess
//! denials are coalesced into one `DENY_BURST { count: N }` record
//! emitted on the next denial that falls outside the window.
//!
//! ## State
//!
//! For each user the limiter tracks:
//!
//! - The window-start timestamp (ns since epoch).
//! - The number of records emitted in this window (`emitted`).
//! - The number of records swallowed in this window (`swallowed`).
//!
//! On every `observe(user, ts)`:
//!
//! 1. If `ts - window_start >= 1s`: window rolled. If `swallowed > 0`
//!    we return `FlushBurst { count: swallowed }` and reset the
//!    window with `emitted = 1`. Otherwise reset and return `Allow`.
//! 2. Else (still inside the window): if `emitted < limit`, increment
//!    `emitted` and return `Allow`. Else increment `swallowed` and
//!    return `Drop`.
//!
//! ## Bookkeeping cap
//!
//! The limiter keeps state for at most [`MAX_TRACKED_USERS`] active
//! users. Beyond that, the oldest entry is evicted — its swallowed
//! count is forfeited. This is a deliberate trade-off: the only way
//! to hit this cap is for thousands of distinct users to denial-burst
//! in the same second, which is itself a security event the kernel
//! should escalate via other means.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

/// Spec rate limit — 10 individual denials per user per second.
pub const DENIAL_RATE_PER_SECOND: u32 = 10;

/// Maximum number of distinct users the limiter holds state for.
pub const MAX_TRACKED_USERS: usize = 256;

/// Window length in nanoseconds — 1 second per spec.
const WINDOW_NS: u64 = 1_000_000_000;

/// Decision returned by [`DenialRateLimiter::observe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenialDecision {
    /// Append the denial as an individual `DENY:<syscall>` record.
    Allow,
    /// Swallow the denial — it's part of an active burst.
    Drop,
    /// Window closed with `count > 0` swallowed denials. Caller SHALL
    /// emit a `DENY_BURST { count }` record before recording the
    /// current denial individually.
    FlushBurst {
        /// Number of denials coalesced.
        count: u32,
    },
}

/// Per-user denial rate-limit state.
#[derive(Debug, Clone, PartialEq, Eq)]
struct UserState {
    window_start_ns: u64,
    emitted: u32,
    swallowed: u32,
    last_seen_ns: u64,
}

/// Token-bucket-flavoured rate limiter — see module docs.
#[derive(Debug, Default)]
pub struct DenialRateLimiter {
    users: BTreeMap<String, UserState>,
}

impl DenialRateLimiter {
    /// Construct a fresh limiter.
    pub fn new() -> Self {
        Self {
            users: BTreeMap::new(),
        }
    }

    /// Observe a denial from `user` at `ts_ns`. See module docs for
    /// the decision matrix.
    pub fn observe(&mut self, user: &str, ts_ns: u64) -> DenialDecision {
        // Bound the bookkeeping. If we'd insert a new entry past the
        // cap, evict the oldest.
        if !self.users.contains_key(user) && self.users.len() >= MAX_TRACKED_USERS {
            self.evict_oldest();
        }
        let state = self
            .users
            .entry(user.to_string())
            .or_insert_with(|| UserState {
                window_start_ns: ts_ns,
                emitted: 0,
                swallowed: 0,
                last_seen_ns: ts_ns,
            });

        // Did the rolling 1-second window close?
        let elapsed = ts_ns.saturating_sub(state.window_start_ns);
        if elapsed >= WINDOW_NS {
            // Window closed. Maybe flush a burst from the previous
            // window before starting a new one.
            let pending = state.swallowed;
            state.window_start_ns = ts_ns;
            state.emitted = 1;
            state.swallowed = 0;
            state.last_seen_ns = ts_ns;
            if pending > 0 {
                return DenialDecision::FlushBurst { count: pending };
            } else {
                return DenialDecision::Allow;
            }
        }

        state.last_seen_ns = ts_ns;
        if state.emitted < DENIAL_RATE_PER_SECOND {
            state.emitted += 1;
            DenialDecision::Allow
        } else {
            state.swallowed = state.swallowed.saturating_add(1);
            DenialDecision::Drop
        }
    }

    /// Number of users currently tracked.
    pub fn tracked_users(&self) -> usize {
        self.users.len()
    }

    /// Pull pending swallowed counts and clear them — used by a hard
    /// reboot / shutdown path so swallowed denials don't disappear
    /// silently on a clean shutdown.
    pub fn drain_pending(&mut self) -> alloc::vec::Vec<(String, u32)> {
        let mut out = alloc::vec::Vec::new();
        for (k, v) in self.users.iter_mut() {
            if v.swallowed > 0 {
                out.push((k.clone(), v.swallowed));
                v.swallowed = 0;
            }
        }
        out
    }

    fn evict_oldest(&mut self) {
        // Find the entry with the smallest `last_seen_ns`.
        let mut oldest_key: Option<String> = None;
        let mut oldest_ts = u64::MAX;
        for (k, v) in self.users.iter() {
            if v.last_seen_ns < oldest_ts {
                oldest_ts = v.last_seen_ns;
                oldest_key = Some(k.clone());
            }
        }
        if let Some(k) = oldest_key {
            self.users.remove(&k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_ten_allow() {
        let mut l = DenialRateLimiter::new();
        for i in 0..10 {
            assert_eq!(l.observe("alice", 1_000 + i), DenialDecision::Allow);
        }
    }

    #[test]
    fn eleventh_drops_within_window() {
        let mut l = DenialRateLimiter::new();
        for i in 0..10 {
            l.observe("alice", 1_000 + i);
        }
        let d = l.observe("alice", 1_010);
        assert_eq!(d, DenialDecision::Drop);
    }

    #[test]
    fn burst_flushed_on_window_roll() {
        let mut l = DenialRateLimiter::new();
        for i in 0..200 {
            l.observe("alice", 1_000 + i);
        }
        // Cross the 1-second window (1_000_000_000 ns).
        let d = l.observe("alice", 1_000 + 1_500_000_000);
        assert_eq!(d, DenialDecision::FlushBurst { count: 190 });
        // 200 calls: 10 emitted, 190 swallowed, then the window-roll
        // observe counts as 1 emitted in the new window. The next
        // call within the new window stays in `Allow` until limit.
        let d = l.observe("alice", 1_001 + 1_500_000_000);
        assert_eq!(d, DenialDecision::Allow);
    }

    #[test]
    fn per_user_isolated() {
        let mut l = DenialRateLimiter::new();
        for i in 0..10 {
            l.observe("alice", 1_000 + i);
        }
        // Bob in the same window starts fresh.
        for i in 0..10 {
            assert_eq!(
                l.observe("bob", 1_000 + i),
                DenialDecision::Allow,
                "bob iter {}",
                i
            );
        }
        // Bob's 11th drops independently.
        assert_eq!(l.observe("bob", 1_010), DenialDecision::Drop);
    }

    #[test]
    fn window_roll_with_zero_swallowed_does_not_emit_burst() {
        let mut l = DenialRateLimiter::new();
        for i in 0..5 {
            l.observe("alice", 1_000 + i);
        }
        // Cross window with no excess.
        let d = l.observe("alice", 1_000 + 2_000_000_000);
        assert_eq!(d, DenialDecision::Allow);
    }

    #[test]
    fn evict_oldest_when_capacity_hit() {
        let mut l = DenialRateLimiter::new();
        // Fill to the cap with distinct users.
        for i in 0..MAX_TRACKED_USERS {
            l.observe(&alloc::format!("u{}", i), i as u64);
        }
        assert_eq!(l.tracked_users(), MAX_TRACKED_USERS);
        // One more user causes oldest to be evicted.
        l.observe("new_user", (MAX_TRACKED_USERS as u64) + 1);
        assert_eq!(l.tracked_users(), MAX_TRACKED_USERS);
    }

    #[test]
    fn drain_pending_returns_swallowed_counts() {
        let mut l = DenialRateLimiter::new();
        for i in 0..200 {
            l.observe("alice", 1_000 + i);
        }
        let pending = l.drain_pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, "alice".to_string());
        assert_eq!(pending[0].1, 190);
        // Second drain returns nothing.
        let again = l.drain_pending();
        assert!(again.is_empty());
    }

    #[test]
    fn multiple_users_burst_independently() {
        let mut l = DenialRateLimiter::new();
        for u in ["alice", "bob"] {
            for i in 0..200 {
                l.observe(u, 1_000 + i);
            }
        }
        let pending = l.drain_pending();
        assert_eq!(pending.len(), 2);
        for (_, count) in pending {
            assert_eq!(count, 190);
        }
    }

    #[test]
    fn rate_limit_const_matches_spec() {
        assert_eq!(DENIAL_RATE_PER_SECOND, 10);
    }

    #[test]
    fn tracked_users_starts_zero() {
        let l = DenialRateLimiter::new();
        assert_eq!(l.tracked_users(), 0);
    }

    #[test]
    fn window_starts_at_first_observe() {
        let mut l = DenialRateLimiter::new();
        l.observe("alice", 1_000_000);
        assert_eq!(l.tracked_users(), 1);
    }
}
