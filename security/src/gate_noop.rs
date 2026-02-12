// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! No-op SecurityGate when `formal-gate` feature is disabled.
//!
//! Provides the same public API surface as the real gate but as a zero-sized
//! type that unconditionally allows all crossings. This lets integration
//! points compile without feature-flag conditionals around every call site.

/// No-op verdict: always allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// Data is always allowed to cross.
    Allowed,
}

impl GateVerdict {
    /// Always true — the no-op gate never denies.
    pub fn is_allowed(&self) -> bool {
        true
    }

    /// Always false.
    pub fn is_denied(&self) -> bool {
        false
    }

    /// Always false.
    pub fn is_permissive_pass(&self) -> bool {
        false
    }
}

/// Zero-sized security gate that unconditionally allows all crossings.
///
/// When the `formal-gate` feature is disabled, this replaces the full
/// 5-layer verification pipeline. All checks return `GateVerdict::Allowed`
/// with zero runtime overhead.
pub struct SecurityGate;

impl SecurityGate {
    /// Create a no-op gate (ignores the mode parameter).
    pub fn new() -> Self {
        Self
    }

    /// No-op check: always returns `Allowed`.
    pub fn check_noop(&self) -> GateVerdict {
        GateVerdict::Allowed
    }

    /// Statistics: always 0 (no tracking in no-op mode).
    pub fn total_checks(&self) -> u64 {
        0
    }
    pub fn total_allowed(&self) -> u64 {
        0
    }
    pub fn total_denied(&self) -> u64 {
        0
    }
    pub fn total_permissive_passes(&self) -> u64 {
        0
    }
}

impl Default for SecurityGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_gate_is_zst() {
        assert_eq!(core::mem::size_of::<SecurityGate>(), 0);
    }

    #[test]
    fn noop_verdict_always_allowed() {
        let gate = SecurityGate::new();
        let verdict = gate.check_noop();
        assert!(verdict.is_allowed());
        assert!(!verdict.is_denied());
        assert!(!verdict.is_permissive_pass());
        assert_eq!(verdict, GateVerdict::Allowed);
    }

    #[test]
    fn noop_statistics_zero() {
        let gate = SecurityGate::new();
        assert_eq!(gate.total_checks(), 0);
        assert_eq!(gate.total_allowed(), 0);
        assert_eq!(gate.total_denied(), 0);
        assert_eq!(gate.total_permissive_passes(), 0);
    }

    #[test]
    fn noop_default() {
        let gate = SecurityGate::default();
        assert_eq!(gate.check_noop(), GateVerdict::Allowed);
    }
}
