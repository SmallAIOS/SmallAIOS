// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Physical-presence indicator trait used to gate destructive boot-time
//! actions on `/flash/`.
//!
//! ## Why a presence indicator?
//!
//! The first time a SmallAIOS unit boots against a fresh QSPI NOR or
//! NAND part, the medium is unformatted: the root metadata-pair is raw
//! `0xFF`. A naive boot path would silently format it. That gives
//! attackers a wipe-and-reboot weapon if they can force a reboot before
//! the rest of the platform comes up. The fix, mirrored on every secure
//! embedded platform, is a *physical-presence* gate: a hardware signal
//! (jumper, chassis-intrusion switch, recovery button, TPM PP bit) that
//! the operator must assert to authorize a one-time format.
//!
//! The same trait is reused by `management-login-v1` Phase 4 for its
//! `auth.skip-firstboot` recovery flow, so a single per-arch impl
//! covers both call sites.
//!
//! ## v1 scope
//!
//! Phase 3 of `embedded-flash-fs-v1` lands the trait + a kernel default
//! that returns `false` (real per-arch impls land when the first MCU/
//! FPGA target boards arrive). A `TestPresenceProvider` is provided
//! gated behind `cfg(test)` and `fs-flash-mock` for the conformance
//! suite. No attempt is made here to *bind* a trait object into the
//! kernel's mount sequencer; that wiring lands in `embedded-filesystem-v1`
//! Phase 6 alongside the F2FS first-boot path.

/// Trait every per-arch flash-presence indicator must implement.
///
/// ## Contract
///
/// `asserted()` SHALL return `true` if and only if the operator has
/// physically asserted the presence signal *at the time of the call*.
/// Implementations MUST NOT cache an "ever-asserted" bit across reboots
/// — the property the gate enforces is that an attacker cannot trigger
/// a destructive action without a fresh physical gesture per-boot.
///
/// `asserted()` SHALL be safe to call repeatedly. The kernel mount
/// sequencer queries it exactly once at the format/halt decision point;
/// subsequent calls (e.g. from telemetry) may observe the signal change
/// state freely.
pub trait PhysicalPresenceProvider {
    /// `true` iff the operator has asserted the physical-presence
    /// signal right now.
    fn asserted(&self) -> bool;
}

/// Default presence indicator used by the kernel when no per-arch
/// override has been registered. Returns `false` unconditionally so
/// the format-on-presence path is closed by default; an unprovisioned
/// kernel SHALL halt with a recovery message rather than silently wipe
/// the medium.
///
/// Real per-arch implementations (`arch::aarch64::flash::presence`,
/// `arch::riscv64::flash::presence`, etc.) land alongside the QSPI/ONFI
/// controller bring-ups in Phase 8 of `embedded-flash-fs-v1`.
#[derive(Debug, Default, Clone, Copy)]
pub struct KernelPresenceProvider;

impl PhysicalPresenceProvider for KernelPresenceProvider {
    fn asserted(&self) -> bool {
        false
    }
}

/// Test-only [`PhysicalPresenceProvider`] that returns a constant.
///
/// Lets the conformance suite drive the format-on-presence decision
/// without spinning up real hardware. Reused by `management-login-v1`
/// Phase 4 tests for the `auth.skip-firstboot` recovery flow.
#[derive(Debug, Clone, Copy)]
pub struct TestPresenceProvider {
    asserted: bool,
}

impl TestPresenceProvider {
    /// Construct a provider that reports the given assertion state.
    pub const fn asserted(asserted: bool) -> Self {
        Self { asserted }
    }
}

impl PhysicalPresenceProvider for TestPresenceProvider {
    fn asserted(&self) -> bool {
        self.asserted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_provider_defaults_to_not_asserted() {
        let p = KernelPresenceProvider;
        assert!(!p.asserted());
    }

    #[test]
    fn test_provider_returns_configured_state() {
        assert!(TestPresenceProvider::asserted(true).asserted());
        assert!(!TestPresenceProvider::asserted(false).asserted());
    }
}
