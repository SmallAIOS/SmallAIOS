// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX signal handling.
//!
//! Implements a minimal signal subsystem covering the signals commonly
//! required by AI runtime workloads: `SIGINT`, `SIGTERM` for graceful
//! shutdown, `SIGUSR1`/`SIGUSR2` for model reload triggers, and the
//! non-catchable `SIGKILL`.
//!
//! Signals are represented as a 64-bit bitmask.  Signal numbers correspond
//! to bit positions (1-based, matching POSIX conventions).
//!
//! `SIGKILL` is protected: its action cannot be changed from `Default` and
//! it cannot be masked.

use crate::errno::Errno;

// ─── Signal Numbers ─────────────────────────────────────────────────────────

/// Standard POSIX signals relevant to AI workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Signal {
    /// Hangup detected on controlling terminal.
    SIGHUP = 1,
    /// Interrupt from keyboard (Ctrl-C).
    SIGINT = 2,
    /// Quit from keyboard (Ctrl-\).
    SIGQUIT = 3,
    /// Kill signal — cannot be caught or ignored.
    SIGKILL = 9,
    /// User-defined signal 1 (model reload, etc.).
    SIGUSR1 = 10,
    /// User-defined signal 2.
    SIGUSR2 = 12,
    /// Termination signal — graceful shutdown.
    SIGTERM = 15,
}

impl Signal {
    /// Maximum signal number we support (bits 1..=31).
    pub const MAX: usize = 31;

    /// Try to convert a raw signal number to a `Signal` variant.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            1 => Some(Self::SIGHUP),
            2 => Some(Self::SIGINT),
            3 => Some(Self::SIGQUIT),
            9 => Some(Self::SIGKILL),
            10 => Some(Self::SIGUSR1),
            12 => Some(Self::SIGUSR2),
            15 => Some(Self::SIGTERM),
            _ => None,
        }
    }

    /// Return the signal number as a `usize` for bitmask indexing.
    pub const fn number(self) -> usize {
        self as i32 as usize
    }
}

// ─── Signal Action ──────────────────────────────────────────────────────────

/// What to do when a signal is delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigAction {
    /// Perform the default action (terminate, core dump, ignore, etc.).
    Default,
    /// Explicitly ignore the signal.
    Ignore,
    /// Invoke a user-supplied handler at the given function address.
    Handler(usize),
}

// ─── Signal Mask ────────────────────────────────────────────────────────────

/// A 64-bit bitmask tracking which signals are set.
///
/// Bit *n* corresponds to signal number *n* (1-based).  Bit 0 is unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalMask(u64);

impl SignalMask {
    /// Create an empty signal mask (no signals set).
    pub const fn new() -> Self {
        Self(0)
    }

    /// Set the bit for `signal`.
    pub fn set(&mut self, signal: Signal) {
        self.0 |= 1u64 << signal.number();
    }

    /// Clear the bit for `signal`.
    pub fn clear(&mut self, signal: Signal) {
        self.0 &= !(1u64 << signal.number());
    }

    /// Check whether the bit for `signal` is set.
    pub fn is_set(&self, signal: Signal) -> bool {
        self.0 & (1u64 << signal.number()) != 0
    }

    /// Set all signal bits (1..=31).
    pub fn block_all(&mut self) {
        // Set bits 1 through 31.
        self.0 = 0xFFFF_FFFE; // bits 1..=31
    }

    /// Clear all signal bits.
    pub fn unblock_all(&mut self) {
        self.0 = 0;
    }

    /// Return the raw bitmask value.
    pub const fn bits(&self) -> u64 {
        self.0
    }

    /// Check whether *any* bit is set.
    pub const fn any(&self) -> bool {
        self.0 != 0
    }

    /// Return the bitwise AND of two masks.
    pub const fn and(&self, other: &SignalMask) -> SignalMask {
        SignalMask(self.0 & other.0)
    }

    /// Return the bitwise complement (NOT) of the mask.
    pub const fn not(&self) -> SignalMask {
        SignalMask(!self.0)
    }
}

// ─── Signal State ───────────────────────────────────────────────────────────

/// Per-task signal state.
pub struct SignalState {
    /// Signals that have been delivered but not yet handled.
    pub pending: SignalMask,
    /// Signals that are currently blocked (masked).
    pub mask: SignalMask,
    /// Per-signal disposition table (indexed 0..31; index 0 unused).
    pub actions: [SigAction; 32],
}

impl SignalState {
    /// Create a new signal state with all defaults.
    pub fn new() -> Self {
        Self {
            pending: SignalMask::new(),
            mask: SignalMask::new(),
            actions: [SigAction::Default; 32],
        }
    }
}

// ─── Signal Operations ──────────────────────────────────────────────────────

/// Check whether there are any pending signals that are not masked.
pub fn signal_pending(state: &SignalState) -> bool {
    // pending & ~mask != 0  <==>  there is at least one deliverable signal.
    state.pending.and(&state.mask.not()).any()
}

/// Deliver (raise) a signal: set its pending bit.
///
/// Returns `EINVAL` if the signal is `SIGKILL` or `SIGTERM` and the
/// disposition is `Ignore` (those signals must never be silently dropped).
pub fn signal_deliver(state: &mut SignalState, signal: Signal) -> Result<(), Errno> {
    let action = state.actions[signal.number()];
    // SIGKILL and SIGTERM cannot be ignored.
    if matches!(signal, Signal::SIGKILL | Signal::SIGTERM) && action == SigAction::Ignore {
        return Err(Errno::EINVAL);
    }
    state.pending.set(signal);
    Ok(())
}

/// Handle (dequeue) a pending signal: clear its pending bit and return
/// the registered action.
pub fn signal_handle(state: &mut SignalState, signal: Signal) -> SigAction {
    state.pending.clear(signal);
    state.actions[signal.number()]
}

/// Set the disposition for a signal.
///
/// Returns `EINVAL` if attempting to change the action for `SIGKILL`
/// (whose disposition is always `Default`).
pub fn sigaction_set(
    state: &mut SignalState,
    signal: Signal,
    action: SigAction,
) -> Result<(), Errno> {
    if signal == Signal::SIGKILL {
        return Err(Errno::EINVAL);
    }
    state.actions[signal.number()] = action;
    Ok(())
}

/// Replace the signal mask, returning the old mask.
///
/// `SIGKILL` is forcibly unmasked — it can never be blocked.
pub fn sigprocmask_set(
    state: &mut SignalState,
    mut new_mask: SignalMask,
) -> Result<SignalMask, Errno> {
    // SIGKILL must never be masked.
    new_mask.clear(Signal::SIGKILL);

    let old = state.mask;
    state.mask = new_mask;
    Ok(old)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Signal enum ──────────────────────────────────────────────────────

    #[test]
    fn signal_from_raw() {
        assert_eq!(Signal::from_raw(1), Some(Signal::SIGHUP));
        assert_eq!(Signal::from_raw(2), Some(Signal::SIGINT));
        assert_eq!(Signal::from_raw(9), Some(Signal::SIGKILL));
        assert_eq!(Signal::from_raw(15), Some(Signal::SIGTERM));
        assert_eq!(Signal::from_raw(0), None);
        assert_eq!(Signal::from_raw(99), None);
    }

    #[test]
    fn signal_number_matches_repr() {
        assert_eq!(Signal::SIGHUP.number(), 1);
        assert_eq!(Signal::SIGINT.number(), 2);
        assert_eq!(Signal::SIGKILL.number(), 9);
        assert_eq!(Signal::SIGUSR1.number(), 10);
        assert_eq!(Signal::SIGUSR2.number(), 12);
        assert_eq!(Signal::SIGTERM.number(), 15);
    }

    // ── SignalMask ───────────────────────────────────────────────────────

    #[test]
    fn mask_set_clear_is_set() {
        let mut mask = SignalMask::new();
        assert!(!mask.is_set(Signal::SIGINT));

        mask.set(Signal::SIGINT);
        assert!(mask.is_set(Signal::SIGINT));
        assert!(!mask.is_set(Signal::SIGTERM));

        mask.clear(Signal::SIGINT);
        assert!(!mask.is_set(Signal::SIGINT));
    }

    #[test]
    fn mask_block_all_unblock_all() {
        let mut mask = SignalMask::new();
        mask.block_all();
        assert!(mask.is_set(Signal::SIGHUP));
        assert!(mask.is_set(Signal::SIGINT));
        assert!(mask.is_set(Signal::SIGKILL));
        assert!(mask.is_set(Signal::SIGTERM));

        mask.unblock_all();
        assert!(!mask.any());
    }

    #[test]
    fn mask_multiple_signals() {
        let mut mask = SignalMask::new();
        mask.set(Signal::SIGUSR1);
        mask.set(Signal::SIGUSR2);
        assert!(mask.is_set(Signal::SIGUSR1));
        assert!(mask.is_set(Signal::SIGUSR2));
        assert!(!mask.is_set(Signal::SIGINT));
    }

    // ── signal_pending ───────────────────────────────────────────────────

    #[test]
    fn no_pending_signals() {
        let state = SignalState::new();
        assert!(!signal_pending(&state));
    }

    #[test]
    fn pending_unmasked_signal() {
        let mut state = SignalState::new();
        state.pending.set(Signal::SIGINT);
        assert!(signal_pending(&state));
    }

    #[test]
    fn pending_masked_signal_not_visible() {
        let mut state = SignalState::new();
        state.pending.set(Signal::SIGINT);
        state.mask.set(Signal::SIGINT);
        assert!(!signal_pending(&state));
    }

    // ── signal_deliver / signal_handle ───────────────────────────────────

    #[test]
    fn deliver_and_handle() {
        let mut state = SignalState::new();
        signal_deliver(&mut state, Signal::SIGINT).unwrap();
        assert!(state.pending.is_set(Signal::SIGINT));

        let action = signal_handle(&mut state, Signal::SIGINT);
        assert_eq!(action, SigAction::Default);
        assert!(!state.pending.is_set(Signal::SIGINT));
    }

    #[test]
    fn deliver_sigkill_with_ignore_fails() {
        let mut state = SignalState::new();
        // Cannot set SIGKILL to Ignore via sigaction_set (tested below),
        // but force it manually to verify deliver rejects it.
        state.actions[Signal::SIGKILL.number()] = SigAction::Ignore;
        assert_eq!(
            signal_deliver(&mut state, Signal::SIGKILL),
            Err(Errno::EINVAL)
        );
    }

    #[test]
    fn deliver_sigterm_with_ignore_fails() {
        let mut state = SignalState::new();
        // SIGTERM set to Ignore should also be rejected on deliver.
        state.actions[Signal::SIGTERM.number()] = SigAction::Ignore;
        assert_eq!(
            signal_deliver(&mut state, Signal::SIGTERM),
            Err(Errno::EINVAL)
        );
    }

    // ── sigaction_set ────────────────────────────────────────────────────

    #[test]
    fn sigaction_set_handler() {
        let mut state = SignalState::new();
        let handler_addr = 0xDEAD_BEEF;
        sigaction_set(&mut state, Signal::SIGINT, SigAction::Handler(handler_addr)).unwrap();
        assert_eq!(
            state.actions[Signal::SIGINT.number()],
            SigAction::Handler(handler_addr)
        );
    }

    #[test]
    fn sigaction_set_sigkill_rejected() {
        let mut state = SignalState::new();
        assert_eq!(
            sigaction_set(&mut state, Signal::SIGKILL, SigAction::Ignore),
            Err(Errno::EINVAL)
        );
        assert_eq!(
            sigaction_set(
                &mut state,
                Signal::SIGKILL,
                SigAction::Handler(0x1234)
            ),
            Err(Errno::EINVAL)
        );
        // Disposition should remain Default.
        assert_eq!(
            state.actions[Signal::SIGKILL.number()],
            SigAction::Default
        );
    }

    // ── sigprocmask_set ──────────────────────────────────────────────────

    #[test]
    fn sigprocmask_returns_old_mask() {
        let mut state = SignalState::new();
        state.mask.set(Signal::SIGHUP);

        let mut new_mask = SignalMask::new();
        new_mask.set(Signal::SIGINT);
        let old = sigprocmask_set(&mut state, new_mask).unwrap();

        assert!(old.is_set(Signal::SIGHUP));
        assert!(!old.is_set(Signal::SIGINT));
        assert!(state.mask.is_set(Signal::SIGINT));
        assert!(!state.mask.is_set(Signal::SIGHUP));
    }

    #[test]
    fn sigprocmask_cannot_mask_sigkill() {
        let mut state = SignalState::new();
        let mut new_mask = SignalMask::new();
        new_mask.block_all();

        sigprocmask_set(&mut state, new_mask).unwrap();
        // SIGKILL must remain unmasked.
        assert!(!state.mask.is_set(Signal::SIGKILL));
        // Other signals should be masked.
        assert!(state.mask.is_set(Signal::SIGINT));
        assert!(state.mask.is_set(Signal::SIGTERM));
    }
}
