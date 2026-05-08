// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Physical-presence detection.
//!
//! Spec: `openspec/changes/management-login-v1/specs/console-login/spec.md`
//! Q7 — Physical-presence assertion sources.
//!
//! The recovery path (`auth.skip-firstboot`) SHALL only be honored when at
//! least one [`PhysicalPresenceProvider`] for the current architecture
//! reports presence. Each provider abstracts a per-arch source:
//!
//! - x86: GPIO line check ([`PresenceX86Gpio`]).
//! - AArch64 / Jetson: GPIO line check ([`PresenceJetsonGpio`]).
//! - QEMU: `fw_cfg` flag ([`PresenceQemuFwCfg`]).
//! - RISC-V: GPIO line check ([`PresenceRiscvGpio`]).
//! - Verified-boot trusted boot arg ([`PresenceTrustedBootArg`]) — only
//!   honored when the kernel is signature-verified, gated by the
//!   `verified-boot` Cargo feature.
//!
//! [`PhysicalPresenceRegistry`] aggregates an arbitrary set of providers
//! and asserts presence iff *any* provider asserts. The kernel boot path
//! constructs the registry once from compile-time-selected impls and
//! reuses it across the recovery path.
//!
//! ## Status
//!
//! The MMIO / fw_cfg reads are stubbed; the trait dispatch and
//! aggregation logic is the load-bearing surface. Real GPIO probes will
//! land alongside `embedded-filesystem-v1` Phase 7 / `unikernel-orin-bringup-v1`
//! Phase 3.

use alloc::boxed::Box;
use alloc::vec::Vec;

mod jetson;

pub use jetson::PresenceJetsonGpio;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Asks "is the operator physically present at this machine?".
///
/// Implementations SHALL be pure-functional with respect to their own
/// internal state; the registry calls [`Self::asserted`] once per recovery
/// attempt. Returning `true` enables a high-trust operation
/// (`auth.skip-firstboot`) so implementations SHALL be conservative and
/// only assert when the underlying signal is unambiguous.
pub trait PhysicalPresenceProvider {
    /// Stable, lowercase ASCII identifier — used in audit records.
    fn name(&self) -> &'static str;

    /// Read the underlying signal and return `true` iff presence is
    /// asserted.
    fn asserted(&self) -> bool;
}

// ─── Aggregation ─────────────────────────────────────────────────────────────

/// Aggregates one or more [`PhysicalPresenceProvider`] impls.
///
/// `asserted()` returns `true` iff *any* member provider asserts. An
/// empty registry SHALL return `false` (fail-closed).
pub struct PhysicalPresenceRegistry {
    providers: Vec<Box<dyn PhysicalPresenceProvider>>,
}

impl Default for PhysicalPresenceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalPresenceRegistry {
    /// Construct an empty registry. The kernel adds providers via
    /// [`Self::register`] during boot.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Add a provider.
    pub fn register(&mut self, p: Box<dyn PhysicalPresenceProvider>) {
        self.providers.push(p);
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether the registry has any providers.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Returns `true` iff at least one provider reports presence.
    pub fn asserted(&self) -> bool {
        self.providers.iter().any(|p| p.asserted())
    }

    /// Return the name of the first provider asserting presence, if any
    /// (used by the audit record).
    pub fn asserting_provider_name(&self) -> Option<&'static str> {
        self.providers
            .iter()
            .find_map(|p| if p.asserted() { Some(p.name()) } else { None })
    }
}

// ─── x86 GPIO line ───────────────────────────────────────────────────────────

/// x86 GPIO presence provider. The actual MMIO read is stubbed for v1;
/// the trait surface is the load-bearing piece. Set `assert` at
/// construction so tests / kernel boot flag the line.
pub struct PresenceX86Gpio {
    assert: bool,
    line_id: u32,
}

impl PresenceX86Gpio {
    /// Construct a provider for a specific GPIO line. `assert` is the
    /// initial / stub state — production wire-up reads MMIO.
    pub const fn new(line_id: u32, assert: bool) -> Self {
        Self { assert, line_id }
    }

    /// Borrow the line id (for audit records).
    pub fn line_id(&self) -> u32 {
        self.line_id
    }
}

impl PhysicalPresenceProvider for PresenceX86Gpio {
    fn name(&self) -> &'static str {
        "x86-gpio"
    }

    fn asserted(&self) -> bool {
        // Production: read MMIO at base + (line_id / 32) * 4, mask
        // bit (line_id % 32). Stubbed for v1.
        self.assert
    }
}

// ─── QEMU fw_cfg ─────────────────────────────────────────────────────────────

/// QEMU `fw_cfg` presence provider — looks at the
/// `opt/smallaios/phys-presence` flag passed via `qemu -fw_cfg
/// name=opt/smallaios/phys-presence,string=1`.
pub struct PresenceQemuFwCfg {
    assert: bool,
}

impl PresenceQemuFwCfg {
    /// Construct with the `fw_cfg` value (`true` if the flag is `1`).
    pub const fn new(assert: bool) -> Self {
        Self { assert }
    }
}

impl PhysicalPresenceProvider for PresenceQemuFwCfg {
    fn name(&self) -> &'static str {
        "qemu-fw-cfg"
    }

    fn asserted(&self) -> bool {
        self.assert
    }
}

// ─── RISC-V GPIO line ────────────────────────────────────────────────────────

/// RISC-V GPIO presence provider. Same shape as
/// [`PresenceX86Gpio`].
pub struct PresenceRiscvGpio {
    assert: bool,
    line_id: u32,
}

impl PresenceRiscvGpio {
    /// Construct a provider for a specific GPIO line.
    pub const fn new(line_id: u32, assert: bool) -> Self {
        Self { assert, line_id }
    }

    /// Borrow the line id.
    pub fn line_id(&self) -> u32 {
        self.line_id
    }
}

impl PhysicalPresenceProvider for PresenceRiscvGpio {
    fn name(&self) -> &'static str {
        "riscv-gpio"
    }

    fn asserted(&self) -> bool {
        self.assert
    }
}

// ─── Verified-boot trusted boot arg ──────────────────────────────────────────

/// Honors the `auth.skip-firstboot` boot arg only when the kernel is
/// signature-verified.
///
/// In v1 this provider trusts the caller (boot path) to flip
/// `kernel_verified` to `true` only when the verified-boot stack
/// confirmed the kernel image. Builds without the `verified-boot`
/// feature SHALL pass `false` so the provider is permanently disabled.
pub struct PresenceTrustedBootArg {
    kernel_verified: bool,
    boot_arg_present: bool,
}

impl PresenceTrustedBootArg {
    /// Construct with the verified-boot kernel signature state and
    /// whether the boot arg is present in the cmdline.
    pub const fn new(kernel_verified: bool, boot_arg_present: bool) -> Self {
        Self {
            kernel_verified,
            boot_arg_present,
        }
    }
}

impl PhysicalPresenceProvider for PresenceTrustedBootArg {
    fn name(&self) -> &'static str {
        "verified-boot-arg"
    }

    fn asserted(&self) -> bool {
        self.kernel_verified && self.boot_arg_present
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_returns_false() {
        let r = PhysicalPresenceRegistry::new();
        assert!(!r.asserted());
        assert!(r.is_empty());
        assert!(r.asserting_provider_name().is_none());
    }

    #[test]
    fn registry_returns_true_when_any_asserts() {
        let mut r = PhysicalPresenceRegistry::new();
        r.register(Box::new(PresenceX86Gpio::new(7, false)));
        r.register(Box::new(PresenceQemuFwCfg::new(true)));
        r.register(Box::new(PresenceRiscvGpio::new(9, false)));
        assert!(r.asserted());
        assert_eq!(r.asserting_provider_name(), Some("qemu-fw-cfg"));
    }

    #[test]
    fn registry_returns_false_when_all_negate() {
        let mut r = PhysicalPresenceRegistry::new();
        r.register(Box::new(PresenceX86Gpio::new(7, false)));
        r.register(Box::new(PresenceQemuFwCfg::new(false)));
        r.register(Box::new(PresenceRiscvGpio::new(9, false)));
        r.register(Box::new(PresenceTrustedBootArg::new(true, false)));
        assert!(!r.asserted());
    }

    #[test]
    fn x86_gpio_provider_reports_correctly() {
        let on = PresenceX86Gpio::new(11, true);
        assert!(on.asserted());
        assert_eq!(on.name(), "x86-gpio");
        assert_eq!(on.line_id(), 11);
        let off = PresenceX86Gpio::new(11, false);
        assert!(!off.asserted());
    }

    #[test]
    fn qemu_fwcfg_provider_reports_correctly() {
        let on = PresenceQemuFwCfg::new(true);
        assert!(on.asserted());
        assert_eq!(on.name(), "qemu-fw-cfg");
        let off = PresenceQemuFwCfg::new(false);
        assert!(!off.asserted());
    }

    #[test]
    fn riscv_gpio_provider_reports_correctly() {
        let on = PresenceRiscvGpio::new(3, true);
        assert!(on.asserted());
        assert_eq!(on.name(), "riscv-gpio");
        assert_eq!(on.line_id(), 3);
    }

    #[test]
    fn jetson_gpio_provider_reports_correctly() {
        let on = PresenceJetsonGpio::new(401, true);
        assert!(on.asserted());
        assert_eq!(on.name(), "jetson-gpio");
        let off = PresenceJetsonGpio::new(401, false);
        assert!(!off.asserted());
    }

    #[test]
    fn verified_boot_arg_requires_both_signals() {
        // Verified + arg → assert.
        assert!(PresenceTrustedBootArg::new(true, true).asserted());
        // Unverified kernel: ignored even with arg.
        assert!(!PresenceTrustedBootArg::new(false, true).asserted());
        // Verified but no boot arg: not asserted.
        assert!(!PresenceTrustedBootArg::new(true, false).asserted());
    }

    #[test]
    fn registry_register_grows_len() {
        let mut r = PhysicalPresenceRegistry::new();
        assert_eq!(r.len(), 0);
        r.register(Box::new(PresenceX86Gpio::new(0, false)));
        assert_eq!(r.len(), 1);
        r.register(Box::new(PresenceQemuFwCfg::new(false)));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn unverified_kernel_ignores_boot_arg() {
        // Spec scenario "Skip-firstboot ignored without presence" via
        // the verified-boot-arg path specifically: the boot arg can
        // never assert presence on a non-verified kernel.
        let p = PresenceTrustedBootArg::new(false, true);
        assert!(!p.asserted());
    }

    #[test]
    fn registry_asserting_name_is_first_match() {
        let mut r = PhysicalPresenceRegistry::new();
        r.register(Box::new(PresenceX86Gpio::new(0, true)));
        r.register(Box::new(PresenceQemuFwCfg::new(true)));
        // First match wins.
        assert_eq!(r.asserting_provider_name(), Some("x86-gpio"));
    }
}
