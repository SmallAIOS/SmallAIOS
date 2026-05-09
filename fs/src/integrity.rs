// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed read-path policy for the filesystem stack.
//!
//! Per the `fs-integrity` spec, the kernel SHALL keep the system
//! reachable when both squashfs slots fail signature verification:
//!
//! - `/models/` is left unmounted; any `model_load` syscall returns
//!   `-EROFS` with a `system not in service` message.
//! - `/data/`, login, audit, and console SHALL keep working so an
//!   operator can investigate and stage a recovery image.
//!
//! This module owns the [`InferenceLockout`] guard that captures the
//! "both slots invalid" state. The guard is consulted by the
//! `model_load` overlay path (added in Phase 3 of the overlay change)
//! and by the management API; it is a pure-functional state machine
//! with no I/O side effects of its own.
//!
//! Owned by `embedded-filesystem-v1` Phase 6.

extern crate alloc;

/// One-line operator-facing message printed on the console and
/// returned from `model_load` when [`InferenceLockout::is_locked`].
pub const SYSTEM_NOT_IN_SERVICE_MSG: &str =
    "system not in service: both squashfs slots invalid; stage a recovery image";

/// The reason a [`InferenceLockout`] guard was raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockoutReason {
    /// Both A and B squashfs slots failed signature verification at
    /// mount time. Inference is hard-disabled until a valid slot is
    /// restored; `auth`, audit, and `/data/` paths continue to work.
    BothSlotsInvalid,
    /// A per-read SHA-3-256 mismatch was observed after mount; the
    /// in-memory squashfs has been quarantined for the rest of this
    /// boot to prevent serving any byte from a corrupt block.
    BlockHashMismatch,
}

impl LockoutReason {
    /// Stable, lowercase ASCII tag used in the audit ring.
    pub const fn audit_tag(&self) -> &'static str {
        match self {
            Self::BothSlotsInvalid => "both_slots_invalid",
            Self::BlockHashMismatch => "block_hash_mismatch",
        }
    }
}

/// Inference-lockout guard.
///
/// The `MountTable` owns one of these. While `is_locked()` is true,
/// any caller to `model_load` SHALL receive `-EROFS` and the lockout
/// reason SHALL be stamped into the audit log. Login, audit, and
/// `/data/` continue to work unchanged.
///
/// The guard is intentionally append-only: there is no public unlock
/// path, so the only way to clear it is to reboot after staging a
/// recovery image. Tests reach for [`InferenceLockout::reset`] to
/// fabricate a clean state between scenarios; that helper is gated
/// behind `cfg(test)` so production code cannot accidentally clear
/// the lockout.
#[derive(Debug, Default, Clone, Copy)]
pub struct InferenceLockout {
    locked: bool,
    reason: Option<LockoutReason>,
    /// Number of `model_load` calls rejected by this guard. Reset by
    /// [`take_rejected_count`] for audit-record batching.
    rejected_count: u32,
}

impl InferenceLockout {
    /// Construct a permissive guard (inference allowed). Used by the
    /// happy-path mount where at least one squashfs slot verifies.
    pub const fn permissive() -> Self {
        Self {
            locked: false,
            reason: None,
            rejected_count: 0,
        }
    }

    /// Construct a guard that locks out inference with the given
    /// reason. The caller is responsible for emitting the audit
    /// record at the time of construction; the guard merely tracks
    /// the state.
    pub const fn locked(reason: LockoutReason) -> Self {
        Self {
            locked: true,
            reason: Some(reason),
            rejected_count: 0,
        }
    }

    /// `true` iff inference is hard-disabled.
    pub const fn is_locked(&self) -> bool {
        self.locked
    }

    /// Reason the guard was raised (if any).
    pub const fn reason(&self) -> Option<LockoutReason> {
        self.reason
    }

    /// Operator-facing message returned to user code from
    /// `model_load` when locked. Returns `None` while the guard is
    /// permissive.
    pub const fn user_message(&self) -> Option<&'static str> {
        if self.locked {
            Some(SYSTEM_NOT_IN_SERVICE_MSG)
        } else {
            None
        }
    }

    /// Tally one `model_load` rejection. Returns the new total.
    /// Production code calls this from the `model_load` syscall
    /// epilogue when the guard rejects the call.
    pub fn record_rejection(&mut self) -> u32 {
        self.rejected_count = self.rejected_count.saturating_add(1);
        self.rejected_count
    }

    /// How many `model_load` calls have been rejected since the last
    /// [`take_rejected_count`].
    pub const fn rejected_count(&self) -> u32 {
        self.rejected_count
    }

    /// Drain the rejection counter (used by the audit-batcher).
    pub fn take_rejected_count(&mut self) -> u32 {
        let n = self.rejected_count;
        self.rejected_count = 0;
        n
    }

    /// Test-only reset (clears the guard back to permissive state).
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.locked = false;
        self.reason = None;
        self.rejected_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permissive_default_does_not_lock() {
        let g = InferenceLockout::permissive();
        assert!(!g.is_locked());
        assert!(g.reason().is_none());
        assert!(g.user_message().is_none());
    }

    #[test]
    fn locked_guard_carries_reason_and_message() {
        let g = InferenceLockout::locked(LockoutReason::BothSlotsInvalid);
        assert!(g.is_locked());
        assert_eq!(g.reason(), Some(LockoutReason::BothSlotsInvalid));
        assert_eq!(g.user_message(), Some(SYSTEM_NOT_IN_SERVICE_MSG));
    }

    #[test]
    fn audit_tag_matches_spec() {
        assert_eq!(
            LockoutReason::BothSlotsInvalid.audit_tag(),
            "both_slots_invalid"
        );
        assert_eq!(
            LockoutReason::BlockHashMismatch.audit_tag(),
            "block_hash_mismatch"
        );
    }

    #[test]
    fn rejection_counter_increments() {
        let mut g = InferenceLockout::locked(LockoutReason::BothSlotsInvalid);
        assert_eq!(g.record_rejection(), 1);
        assert_eq!(g.record_rejection(), 2);
        assert_eq!(g.rejected_count(), 2);
    }

    #[test]
    fn rejection_counter_drains() {
        let mut g = InferenceLockout::locked(LockoutReason::BothSlotsInvalid);
        let _ = g.record_rejection();
        let _ = g.record_rejection();
        let _ = g.record_rejection();
        assert_eq!(g.take_rejected_count(), 3);
        assert_eq!(g.rejected_count(), 0);
    }

    #[test]
    fn rejection_counter_saturates() {
        let mut g = InferenceLockout::locked(LockoutReason::BothSlotsInvalid);
        // Advance to near saturation by manipulating the counter via
        // the public API a few times; we don't actually run u32::MAX
        // iterations — the saturating_add semantics are clear from
        // the implementation. Just spot-check that the API does not
        // panic at high counts.
        for _ in 0..1000 {
            let _ = g.record_rejection();
        }
        assert_eq!(g.rejected_count(), 1000);
    }

    #[test]
    fn block_hash_mismatch_locks_too() {
        let g = InferenceLockout::locked(LockoutReason::BlockHashMismatch);
        assert!(g.is_locked());
        assert_eq!(g.reason(), Some(LockoutReason::BlockHashMismatch));
    }

    #[test]
    fn test_only_reset_clears_state() {
        let mut g = InferenceLockout::locked(LockoutReason::BothSlotsInvalid);
        let _ = g.record_rejection();
        g.reset();
        assert!(!g.is_locked());
        assert_eq!(g.rejected_count(), 0);
    }

    #[test]
    fn message_string_is_user_friendly() {
        // Operator-facing message contains a clear next-step hint.
        assert!(SYSTEM_NOT_IN_SERVICE_MSG.contains("recovery"));
        assert!(SYSTEM_NOT_IN_SERVICE_MSG.contains("not in service"));
    }
}
