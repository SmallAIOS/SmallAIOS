// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cooperative idle-timeout sweeper.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-roles/spec.md`
//! Q5 ("Per-role idle timeout").
//!
//! Idle thresholds (defaults; will be overridden by `mgmt/policy.toml`
//! in `management-login-v1` Phase 6):
//!
//! | Role     | Default idle timeout |
//! |----------|----------------------|
//! | Root     | 15 minutes           |
//! | Operator | 60 minutes           |
//! | Viewer   | 60 minutes           |
//!
//! ## Cooperative scheduling
//!
//! [`Sweeper::tick`] is called from the kernel's main loop on a low
//! cadence (every ~1 s). It walks the [`SessionTable`] in O(N=32) and
//! invalidates any session whose `last_activity_unix_time` is older
//! than the role's threshold. Each invalidation is reported through an
//! [`AuditSink`] for the audit log Phase 10 will consume.

use super::session_table::{Session, SessionId, SessionTable};
use super::Role;

/// Default idle timeout for `Root`, in seconds (15 minutes).
pub const DEFAULT_ROOT_IDLE_SECS: u64 = 15 * 60;

/// Default idle timeout for `Operator`, in seconds (60 minutes).
pub const DEFAULT_OPERATOR_IDLE_SECS: u64 = 60 * 60;

/// Default idle timeout for `Viewer`, in seconds (60 minutes).
pub const DEFAULT_VIEWER_IDLE_SECS: u64 = 60 * 60;

/// Per-role idle policy. Each value is a count of seconds of inactivity
/// after which a session is force-logged-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdlePolicy {
    /// Idle threshold for `Root` sessions.
    pub root_secs: u64,
    /// Idle threshold for `Operator` sessions.
    pub operator_secs: u64,
    /// Idle threshold for `Viewer` sessions.
    pub viewer_secs: u64,
}

impl IdlePolicy {
    /// Construct the spec-default policy (15/60/60 min).
    pub const fn defaults() -> Self {
        Self {
            root_secs: DEFAULT_ROOT_IDLE_SECS,
            operator_secs: DEFAULT_OPERATOR_IDLE_SECS,
            viewer_secs: DEFAULT_VIEWER_IDLE_SECS,
        }
    }

    /// Look up the threshold for a role.
    pub const fn for_role(&self, role: Role) -> u64 {
        match role {
            Role::Root => self.root_secs,
            Role::Operator => self.operator_secs,
            Role::Viewer => self.viewer_secs,
        }
    }
}

impl Default for IdlePolicy {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Reasons the sweeper invalidated a session. Carried into the audit
/// log so post-mortem analysis can distinguish operator-initiated
/// logout from idle-eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepReason {
    /// `now - last_activity > threshold(role)`.
    IdleTimeout,
}

/// Audit-sink trait — the sweeper reports every invalidation here. The
/// kernel installs a [`NoopAuditSink`] in production builds until
/// Phase 10 lands the audit chain; tests use [`CapturingAuditSink`].
pub trait AuditSink {
    /// Record that `id` (now invalid) was swept.
    fn on_session_invalidated(&mut self, id: SessionId, session: &Session, reason: SweepReason);
}

/// Production no-op audit sink. Phase 10 of `management-login-v1` will
/// add a real `AuditChainSink` that hash-chains each event.
#[derive(Debug, Default)]
pub struct NoopAuditSink;

impl AuditSink for NoopAuditSink {
    fn on_session_invalidated(&mut self, _id: SessionId, _session: &Session, _reason: SweepReason) {
        // Phase 10 wire-up.
    }
}

/// Capturing audit sink for unit tests.
///
/// Records every invalidation event in an in-memory `Vec`, exposed via
/// [`Self::events`].
#[derive(Debug, Default)]
pub struct CapturingAuditSink {
    events: alloc::vec::Vec<(SessionId, u32, SweepReason)>,
}

impl CapturingAuditSink {
    /// Construct an empty capturing sink.
    pub fn new() -> Self {
        Self {
            events: alloc::vec::Vec::new(),
        }
    }

    /// Borrow the recorded events. Each tuple is
    /// `(SessionId, user_id, SweepReason)`.
    pub fn events(&self) -> &[(SessionId, u32, SweepReason)] {
        &self.events
    }

    /// Drain the captured events.
    pub fn drain(&mut self) -> alloc::vec::Vec<(SessionId, u32, SweepReason)> {
        core::mem::take(&mut self.events)
    }
}

impl AuditSink for CapturingAuditSink {
    fn on_session_invalidated(&mut self, id: SessionId, session: &Session, reason: SweepReason) {
        self.events.push((id, session.user_id, reason));
    }
}

/// The sweeper itself. Thin glue between an [`IdlePolicy`] and a
/// [`SessionTable`] / [`AuditSink`] pair.
#[derive(Debug, Default)]
pub struct Sweeper {
    /// Active idle policy. Phase 6 (`mgmt/policy.toml` loader) will
    /// hot-swap this in via [`Self::set_policy`].
    pub policy: IdlePolicy,
}

impl Sweeper {
    /// Construct a sweeper with default thresholds.
    pub const fn new() -> Self {
        Self {
            policy: IdlePolicy::defaults(),
        }
    }

    /// Construct with an explicit policy (tests + mgmt loader).
    pub const fn with_policy(policy: IdlePolicy) -> Self {
        Self { policy }
    }

    /// Replace the active idle policy. Existing sessions are evaluated
    /// against the *new* threshold on the next [`Self::tick`].
    pub fn set_policy(&mut self, policy: IdlePolicy) {
        self.policy = policy;
    }

    /// Walk the table, invalidate idle sessions, and audit each one.
    ///
    /// Returns the number of sessions invalidated this tick.
    pub fn tick<S: AuditSink>(
        &self,
        now_unix_time: u64,
        table: &mut SessionTable,
        audit: &mut S,
    ) -> usize {
        // Two-pass: collect ids first (we cannot mutate while iterating
        // since `iter()` borrows immutably), then release each in a
        // dedicated mutable pass. The table is at most 32 slots so the
        // collection cost is bounded.
        let mut to_kill: alloc::vec::Vec<SessionId> = alloc::vec::Vec::new();
        for (id, session) in table.iter() {
            let threshold = self.policy.for_role(session.role);
            // Saturating math protects against clock-skew / cold-boot.
            let elapsed = now_unix_time.saturating_sub(session.last_activity_unix_time);
            if elapsed > threshold {
                to_kill.push(id);
            }
        }

        let mut killed = 0;
        for id in to_kill {
            // Lookup, audit, then release. If lookup fails (concurrent
            // logout via syscall), skip — the session is already gone.
            let snapshot = match table.lookup(id) {
                Ok(s) => *s,
                Err(_) => continue,
            };
            audit.on_session_invalidated(id, &snapshot, SweepReason::IdleTimeout);
            if table.release(id).is_ok() {
                killed += 1;
            }
        }
        killed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::session_table::Session as KSession;

    fn mk_session(user_id: u32, role: Role, login_at: u64) -> KSession {
        KSession::new(user_id, b"alice", role, login_at, false).unwrap()
    }

    #[test]
    fn defaults_match_spec_table() {
        let p = IdlePolicy::defaults();
        assert_eq!(p.root_secs, 15 * 60);
        assert_eq!(p.operator_secs, 60 * 60);
        assert_eq!(p.viewer_secs, 60 * 60);
    }

    #[test]
    fn for_role_dispatches_correctly() {
        let p = IdlePolicy::defaults();
        assert_eq!(p.for_role(Role::Root), 15 * 60);
        assert_eq!(p.for_role(Role::Operator), 60 * 60);
        assert_eq!(p.for_role(Role::Viewer), 60 * 60);
    }

    #[test]
    fn idle_root_session_swept_after_15_minutes() {
        let mut tbl = SessionTable::new();
        let id = tbl.acquire(mk_session(1, Role::Root, 0)).unwrap();

        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();

        // 14 min — still alive.
        let n = sweeper.tick(14 * 60, &mut tbl, &mut audit);
        assert_eq!(n, 0);
        assert!(tbl.lookup(id).is_ok());

        // 15 min + 1s — swept.
        let n = sweeper.tick(15 * 60 + 1, &mut tbl, &mut audit);
        assert_eq!(n, 1);
        assert!(tbl.lookup(id).is_err());
        assert_eq!(audit.events().len(), 1);
        assert_eq!(audit.events()[0].1, 1);
        assert_eq!(audit.events()[0].2, SweepReason::IdleTimeout);
    }

    #[test]
    fn operator_threshold_is_higher_than_root() {
        let mut tbl = SessionTable::new();
        let _id_root = tbl.acquire(mk_session(1, Role::Root, 0)).unwrap();
        let _id_op = tbl.acquire(mk_session(2, Role::Operator, 0)).unwrap();

        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();

        // 16 min — root swept, operator alive.
        let n = sweeper.tick(16 * 60, &mut tbl, &mut audit);
        assert_eq!(n, 1);
        assert_eq!(audit.events()[0].1, 1);
    }

    #[test]
    fn reset_idle_renews_the_threshold() {
        let mut tbl = SessionTable::new();
        let id = tbl.acquire(mk_session(1, Role::Root, 0)).unwrap();

        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();

        // 14 min — refresh.
        tbl.reset_idle(id, 14 * 60).unwrap();
        // 28 min — only 14 min since last activity, still alive.
        let n = sweeper.tick(28 * 60, &mut tbl, &mut audit);
        assert_eq!(n, 0);
        assert!(tbl.lookup(id).is_ok());

        // 30 min — 16 min since last activity, swept.
        let n = sweeper.tick(30 * 60, &mut tbl, &mut audit);
        assert_eq!(n, 1);
    }

    #[test]
    fn empty_table_tick_is_a_noop() {
        let mut tbl = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let n = sweeper.tick(1_000_000, &mut tbl, &mut audit);
        assert_eq!(n, 0);
        assert!(audit.events().is_empty());
    }

    #[test]
    fn noop_sink_is_zero_sized_and_compiles() {
        let mut sink = NoopAuditSink;
        let mut tbl = SessionTable::new();
        let _id = tbl.acquire(mk_session(1, Role::Root, 0)).unwrap();
        let sweeper = Sweeper::new();
        // Should not panic.
        let n = sweeper.tick(15 * 60 + 1, &mut tbl, &mut sink);
        assert_eq!(n, 1);
    }

    #[test]
    fn set_policy_takes_effect_on_next_tick() {
        let mut tbl = SessionTable::new();
        let _id = tbl.acquire(mk_session(1, Role::Viewer, 0)).unwrap();

        let mut sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();

        // Default viewer: 60 min — at 30 min, alive.
        let n = sweeper.tick(30 * 60, &mut tbl, &mut audit);
        assert_eq!(n, 0);

        // Tighten viewer policy to 10 min.
        sweeper.set_policy(IdlePolicy {
            root_secs: 15 * 60,
            operator_secs: 60 * 60,
            viewer_secs: 10 * 60,
        });
        let n = sweeper.tick(31 * 60, &mut tbl, &mut audit);
        assert_eq!(n, 1);
    }

    #[test]
    fn saturating_clock_skew_does_not_misfire() {
        // last_activity in the future → elapsed saturates to 0.
        let mut tbl = SessionTable::new();
        let _id = tbl.acquire(mk_session(1, Role::Root, 1_000_000)).unwrap();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        // now < login → elapsed = 0, do not sweep.
        let n = sweeper.tick(0, &mut tbl, &mut audit);
        assert_eq!(n, 0);
    }
}
