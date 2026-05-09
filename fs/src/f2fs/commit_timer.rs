// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock-driven background checkpoint commit timer.
//!
//! Per `fs-f2fs-readwrite/spec.md` (fsync section): "A 5-second
//! background timer SHALL trigger a checkpoint commit if any dirty
//! data is pending and `fsync` has not been called."
//!
//! Phase 7 implements [`super::write::F2fs::tick_timer`] which takes
//! synthetic 1 Hz tick increments. Phase 8 adds this thinner wrapper
//! that the kernel boot path can call from the cooperative scheduler
//! with a wall-clock Unix-second timestamp; the wrapper records
//! `last_commit_unix` and trips when `now - last_commit >= 5`.
//!
//! The split keeps the two ticking mechanisms decoupled:
//!
//! - `tick_timer` is an internal, increment-style API that integrates
//!   trivially with the existing [`super::write::WriteState`].
//! - [`CommitTimer`] is the kernel-facing API: feed it a wall-clock
//!   tick on every scheduler idle pass, get back a `Decision::Commit`
//!   when it is time to invoke the commit closure.
//!
//! Owned by `embedded-filesystem-v1` Phase 8.

extern crate alloc;

use core::cell::Cell;

/// Default commit window in seconds, matching the spec.
pub const DEFAULT_COMMIT_WINDOW_SECONDS: u64 = 5;

/// Outcome of a single [`CommitTimer::tick`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// No commit required this tick — either the filesystem is clean
    /// or the window has not yet elapsed.
    Idle,
    /// The commit window elapsed and dirty data is pending. The caller
    /// SHALL invoke its `fsync` / checkpoint-commit closure.
    Commit,
}

/// Minimal wall-clock-driven commit timer.
///
/// `CommitTimer` carries:
///
/// 1. A `dirty` flag the writer flips on every mutation.
/// 2. A `last_commit_unix` snapshot, the most recent wall-clock second
///    at which the writer was clean.
/// 3. A `window_seconds` knob (default [`DEFAULT_COMMIT_WINDOW_SECONDS`]).
///
/// On every [`tick`](Self::tick) call:
///
/// - If `!dirty`, returns [`Decision::Idle`] and refreshes
///   `last_commit_unix` so the window starts fresh on the next dirty
///   transition.
/// - If `dirty` and `now - last_commit >= window_seconds`, returns
///   [`Decision::Commit`].
/// - Otherwise returns [`Decision::Idle`].
///
/// Callers are expected to:
///
/// 1. Call [`mark_dirty`](Self::mark_dirty) after every successful write.
/// 2. Call [`tick`](Self::tick) at ~1 Hz from the cooperative
///    scheduler's idle path, with the current Unix-seconds clock.
/// 3. On `Decision::Commit`, invoke their commit closure (typically
///    `F2fs::fsync`), then call [`mark_clean`](Self::mark_clean).
///
/// [`mark_clean`](Self::mark_clean) MUST be called from the same
/// `fsync` path so a long write→fsync cycle does not double-fire the
/// timer.
///
/// The struct is `#![no_std]`-friendly and uses [`Cell`] for interior
/// mutability so callers can pass it by `&self` from inside scheduler
/// idle hooks.
#[derive(Debug)]
pub struct CommitTimer {
    dirty: Cell<bool>,
    last_commit_unix: Cell<u64>,
    window_seconds: u64,
}

impl Default for CommitTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl CommitTimer {
    /// Construct a timer with the spec's default 5-second window,
    /// `dirty=false`, `last_commit_unix=0`.
    pub fn new() -> Self {
        Self {
            dirty: Cell::new(false),
            last_commit_unix: Cell::new(0),
            window_seconds: DEFAULT_COMMIT_WINDOW_SECONDS,
        }
    }

    /// Construct a timer with a custom window. `seconds == 0` is
    /// treated as "fire on every dirty tick" — useful for tests.
    pub fn with_window(seconds: u64) -> Self {
        Self {
            dirty: Cell::new(false),
            last_commit_unix: Cell::new(0),
            window_seconds: seconds,
        }
    }

    /// Configured window in seconds.
    pub fn window_seconds(&self) -> u64 {
        self.window_seconds
    }

    /// Returns `true` iff dirty data is pending.
    pub fn is_dirty(&self) -> bool {
        self.dirty.get()
    }

    /// Last Unix-seconds value at which the timer was clean.
    pub fn last_commit_unix(&self) -> u64 {
        self.last_commit_unix.get()
    }

    /// Mark the filesystem as dirty. Idempotent.
    pub fn mark_dirty(&self) {
        self.dirty.set(true);
    }

    /// Mark the filesystem as clean and update `last_commit_unix` to
    /// `now_unix`. Called from the writer's `fsync` path.
    pub fn mark_clean(&self, now_unix: u64) {
        self.dirty.set(false);
        self.last_commit_unix.set(now_unix);
    }

    /// Advance the timer by one wall-clock tick.
    ///
    /// Returns [`Decision::Commit`] iff dirty data is pending and the
    /// configured window has elapsed since the last clean transition.
    /// On `Commit`, the caller is responsible for invoking its
    /// `fsync`-equivalent and then [`mark_clean`](Self::mark_clean).
    ///
    /// On `Idle` while clean, the `last_commit_unix` watermark is
    /// refreshed so the window starts fresh on the next dirty
    /// transition.
    pub fn tick(&self, now_unix: u64) -> Decision {
        if !self.dirty.get() {
            // While clean, slide the watermark forward so the window
            // does not pre-charge before the next mutation.
            self.last_commit_unix.set(now_unix);
            return Decision::Idle;
        }
        let last = self.last_commit_unix.get();
        let elapsed = now_unix.saturating_sub(last);
        if elapsed >= self.window_seconds {
            Decision::Commit
        } else {
            Decision::Idle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_timer_returns_idle() {
        let t = CommitTimer::new();
        assert_eq!(t.tick(0), Decision::Idle);
        assert_eq!(t.tick(100), Decision::Idle);
    }

    #[test]
    fn dirty_within_window_returns_idle() {
        let t = CommitTimer::new();
        t.mark_clean(10);
        t.mark_dirty();
        // 4 < 5: still idle
        assert_eq!(t.tick(14), Decision::Idle);
    }

    #[test]
    fn dirty_at_window_returns_commit() {
        let t = CommitTimer::new();
        t.mark_clean(10);
        t.mark_dirty();
        assert_eq!(t.tick(15), Decision::Commit);
    }

    #[test]
    fn dirty_past_window_returns_commit() {
        let t = CommitTimer::new();
        t.mark_clean(10);
        t.mark_dirty();
        assert_eq!(t.tick(99), Decision::Commit);
    }

    #[test]
    fn commit_resets_window() {
        let t = CommitTimer::new();
        t.mark_clean(10);
        t.mark_dirty();
        assert_eq!(t.tick(15), Decision::Commit);
        // Caller honours the contract: invokes fsync, calls mark_clean.
        t.mark_clean(15);
        assert!(!t.is_dirty());
        // Subsequent dirty transition starts a fresh window.
        t.mark_dirty();
        assert_eq!(t.tick(19), Decision::Idle);
        assert_eq!(t.tick(20), Decision::Commit);
    }

    #[test]
    fn idle_tick_slides_watermark() {
        let t = CommitTimer::new();
        // Clean ticks at t=100 — watermark moves forward.
        assert_eq!(t.tick(100), Decision::Idle);
        // Dirty transition.
        t.mark_dirty();
        // tick at 104: elapsed=4 since watermark=100 — still idle.
        assert_eq!(t.tick(104), Decision::Idle);
        // tick at 105: elapsed=5 — commit.
        assert_eq!(t.tick(105), Decision::Commit);
    }

    #[test]
    fn fsync_resets_timer_window() {
        // Spec: "fsync resets the timer."
        let t = CommitTimer::new();
        t.mark_clean(10);
        t.mark_dirty();
        // Some background activity advances the clock to t=13.
        assert_eq!(t.tick(13), Decision::Idle);
        // Application calls fsync — caller invokes mark_clean(13).
        t.mark_clean(13);
        // Now mark dirty again at t=14.
        t.mark_dirty();
        // At t=17 we are only 3 sec past the most recent fsync — idle.
        assert_eq!(t.tick(17), Decision::Idle);
        // At t=18 we hit the 5s window.
        assert_eq!(t.tick(18), Decision::Commit);
    }

    #[test]
    fn no_dirty_data_no_commit() {
        // Spec: "no dirty data → no commit."
        let t = CommitTimer::new();
        for sec in 0..600u64 {
            assert_eq!(t.tick(sec), Decision::Idle);
        }
    }

    #[test]
    fn custom_window() {
        let t = CommitTimer::with_window(10);
        assert_eq!(t.window_seconds(), 10);
        t.mark_clean(0);
        t.mark_dirty();
        assert_eq!(t.tick(9), Decision::Idle);
        assert_eq!(t.tick(10), Decision::Commit);
    }

    #[test]
    fn zero_window_fires_immediately() {
        let t = CommitTimer::with_window(0);
        t.mark_dirty();
        assert_eq!(t.tick(0), Decision::Commit);
    }

    #[test]
    fn last_commit_unix_round_trip() {
        let t = CommitTimer::new();
        assert_eq!(t.last_commit_unix(), 0);
        t.mark_clean(42);
        assert_eq!(t.last_commit_unix(), 42);
    }

    #[test]
    fn mark_dirty_idempotent() {
        let t = CommitTimer::new();
        t.mark_dirty();
        t.mark_dirty();
        t.mark_dirty();
        assert!(t.is_dirty());
    }

    #[test]
    fn saturating_sub_protects_against_clock_rewind() {
        // If wall-clock NTP rewinds, saturating_sub keeps elapsed at 0
        // rather than wrapping to a huge value. The invariant is that
        // we never spuriously fire a commit on a clock rewind.
        let t = CommitTimer::new();
        t.mark_clean(100);
        t.mark_dirty();
        // Now clock rewinds to 50.
        assert_eq!(t.tick(50), Decision::Idle);
    }

    #[test]
    fn default_window_is_five_seconds() {
        assert_eq!(DEFAULT_COMMIT_WINDOW_SECONDS, 5);
        let t = CommitTimer::default();
        assert_eq!(t.window_seconds(), 5);
    }
}
