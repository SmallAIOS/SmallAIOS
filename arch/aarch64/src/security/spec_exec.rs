// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AArch64 speculative-execution silicon detection + profile selection.
//!
//! Part of OpenSpec change `spec-exec-mitigations-v1` (Phase 3 — aarch64).
//! `tasks.md` is tracked on the spine branch.
//!
//! This module owns the *silicon-level* half of the aarch64 Spectre
//! mitigations: it reads `ID_AA64PFR0_EL1`, decodes the `CSV2`/`CSV3`
//! fields, boot-logs the result, and selects the mitigation profile. The
//! *barrier-emit* half (`csdb` / `dsb sy` at the syscall trust boundary)
//! lives in `kernel/src/syscall/spec_exec.rs` — the kernel crate is Layer 0
//! of the strict acyclic layer model and cannot depend on this Layer-2 arch
//! crate, so `init()` here publishes the selected profile *down* into the
//! kernel via [`smallaios_kernel::syscall::spec_exec::set_software_profile`].
//!
//! # `ID_AA64PFR0_EL1` field decoding (ARM ARM, ARMv8.5-A)
//!
//! Per the *Arm Architecture Reference Manual for A-profile*, the
//! AArch64 Processor Feature Register 0 (`ID_AA64PFR0_EL1`) is a 64-bit
//! read-only system register whose top two nibbles are:
//!
//! | Field  | Bits      | Width | Meaning                                   |
//! |--------|-----------|-------|-------------------------------------------|
//! | `CSV3` | `[63:60]` | 4     | Speculative-fault-address side-channel    |
//! |        |           |       | (Meltdown-class) hardening level          |
//! | `CSV2` | `[59:56]` | 4     | Branch-prediction (Spectre-v2-class)      |
//! |        |           |       | context isolation level                   |
//!
//! `CSV2 >= 1` means the branch predictor cannot be trained across a
//! hardware context boundary using a separate address — i.e. the silicon
//! provides the Spectre-v2 hardening, so no software Retpoline-shape is
//! required. `CSV2 == 0` means the property is *not* advertised. On the
//! Cortex-A78AE (the SmallAIOS reference part, on the Jetson Orin) `CSV2`
//! and `CSV3` are both advertised; the `CSV2 == 0` branch is defensive and
//! must never panic — the software profile is functional.

/// Decoded `CSV2` level (`ID_AA64PFR0_EL1` bits `[59:56]`).
pub const CSV2_SHIFT: u64 = 56;
/// Decoded `CSV3` level (`ID_AA64PFR0_EL1` bits `[63:60]`).
pub const CSV3_SHIFT: u64 = 60;
/// 4-bit field mask shared by `CSV2` and `CSV3`.
pub const FIELD_MASK: u64 = 0xF;

/// Mitigation profile selected at boot from the silicon feature bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// `CSV2 >= 1`: silicon provides Spectre-v2 hardening; no software
    /// Retpoline-shape needed. `csdb`/`dsb sy` barriers are still emitted
    /// unconditionally at the trust boundary (they defend the
    /// Spectre-v1-class load past the capability gate, which CSV2 does not
    /// cover).
    HardwareMitigated,
    /// `CSV2 == 0`: silicon does *not* advertise Spectre-v2 hardening.
    /// Boot continues with a warning; the dispatch path additionally treats
    /// indirect-call sites as needing an explicit `csdb` software shape.
    Software,
}

impl Profile {
    /// Stable string used in the boot log line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Profile::HardwareMitigated => "hardware-mitigated",
            Profile::Software => "software",
        }
    }
}

/// Decode `(CSV2, CSV3)` from a raw `ID_AA64PFR0_EL1` value.
///
/// Split out from [`init`] so it is unit-testable on the host without an
/// `mrs` (which would `SIGILL` off-target). The shifts/mask are exactly the
/// ARM ARM field positions documented at the top of this module.
#[inline]
pub fn decode_csv(id_aa64pfr0_el1: u64) -> (u8, u8) {
    let csv2 = ((id_aa64pfr0_el1 >> CSV2_SHIFT) & FIELD_MASK) as u8;
    let csv3 = ((id_aa64pfr0_el1 >> CSV3_SHIFT) & FIELD_MASK) as u8;
    (csv2, csv3)
}

/// Select the mitigation profile from a decoded `CSV2` level.
///
/// `CSV2 >= 1` → [`Profile::HardwareMitigated`]; `CSV2 == 0` →
/// [`Profile::Software`]. Pure function for host unit tests.
#[inline]
pub fn select_profile(csv2: u8) -> Profile {
    if csv2 >= 1 {
        Profile::HardwareMitigated
    } else {
        Profile::Software
    }
}

/// Read `ID_AA64PFR0_EL1` (EL1, requires running at EL1+).
///
/// # Safety
///
/// `mrs <x>, ID_AA64PFR0_EL1` is a read-only system-register access that is
/// always permitted at EL1 and has no side effects. It is `unsafe` only
/// because it is inline assembly. Must not be called on a non-aarch64 target
/// (the function does not exist there).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn read_id_aa64pfr0_el1() -> u64 {
    let val: u64;
    // SAFETY: read-only ID register; legal at EL1; no memory/stack/flags.
    core::arch::asm!(
        "mrs {}, ID_AA64PFR0_EL1",
        out(reg) val,
        options(nomem, nostack, preserves_flags),
    );
    val
}

/// Probe `ID_AA64PFR0_EL1`, decode CSV2/CSV3, log the profile, and publish
/// it to the kernel dispatch path.
///
/// Boot-logs exactly:
/// `[spec-exec-aarch64] CSV2=<n> CSV3=<n> profile=<hardware-mitigated|software>`
///
/// When `CSV2 < 1` an additional warning line is logged and the software
/// profile is selected. **Boot always continues — this never panics.**
///
/// Call once, early in `kernel_main`, before the syscall path is reachable.
pub fn init() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: kernel runs at EL1; reading the ID register is always legal
    // and side-effect free.
    let raw = unsafe { read_id_aa64pfr0_el1() };

    // Host/test builds (non-aarch64): there is no `ID_AA64PFR0_EL1`. Model
    // the reference Cortex-A78AE (CSV2=1, CSV3=1) so the boot-log/profile
    // logic is still exercised and the log line is representative.
    #[cfg(not(target_arch = "aarch64"))]
    let raw: u64 = (1u64 << CSV2_SHIFT) | (1u64 << CSV3_SHIFT);

    let (csv2, csv3) = decode_csv(raw);
    let profile = select_profile(csv2);

    crate::uart::puts("[spec-exec-aarch64] CSV2=");
    crate::uart::put_dec(csv2 as u64);
    crate::uart::puts(" CSV3=");
    crate::uart::put_dec(csv3 as u64);
    crate::uart::puts(" profile=");
    crate::uart::puts(profile.as_str());
    crate::uart::puts("\n");

    if profile == Profile::Software {
        // CSV2 == 0: hardware Spectre-v2 mitigation absent. Warn, do NOT
        // panic — the software profile is functional. The extra `csdb`
        // software shape would be inserted at every indirect-call site
        // (today: the ONNX op-dispatch indirect call — see
        // `spec-exec-mitigations-v1` Phase 5; and the post-cap-check site
        // already always emits `csdb` via
        // `kernel::syscall::spec_exec::speculation_barrier`). The dispatch
        // path reads `software_profile_active()` to know it is in this mode.
        crate::uart::puts(
            "[spec-exec-aarch64] WARN: CSV2=0 — hardware Spectre-v2 \
             mitigation absent; selecting software profile (extra csdb at \
             indirect-call sites). Boot continues.\n",
        );
    }

    // Publish the selection DOWN into the kernel (Layer 0) so the syscall
    // dispatch path can read it without an upward kernel→arch dependency.
    smallaios_kernel::syscall::spec_exec::set_software_profile(profile == Profile::Software);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_positions_match_arm_arm() {
        // ARM ARM (ARMv8.5-A): CSV2 = [59:56], CSV3 = [63:60].
        assert_eq!(CSV2_SHIFT, 56);
        assert_eq!(CSV3_SHIFT, 60);
        assert_eq!(FIELD_MASK, 0xF);
    }

    #[test]
    fn test_decode_csv_reference_part() {
        // Cortex-A78AE reference: CSV2=1, CSV3=1.
        let raw = (1u64 << 56) | (1u64 << 60);
        assert_eq!(decode_csv(raw), (1, 1));
    }

    #[test]
    fn test_decode_csv_isolates_fields() {
        // CSV2=0b1010 (10), CSV3=0b0011 (3); ensure no bleed from low bits.
        let raw = (0xA_u64 << 56) | (0x3_u64 << 60) | 0xFFFF_FFFF_FFFF;
        assert_eq!(decode_csv(raw), (10, 3));
    }

    #[test]
    fn test_decode_csv_all_zero() {
        assert_eq!(decode_csv(0), (0, 0));
    }

    #[test]
    fn test_decode_csv_max_fields() {
        let raw = (0xF_u64 << 56) | (0xF_u64 << 60);
        assert_eq!(decode_csv(raw), (15, 15));
    }

    #[test]
    fn test_select_profile_hardware_when_csv2_ge_1() {
        assert_eq!(select_profile(1), Profile::HardwareMitigated);
        assert_eq!(select_profile(2), Profile::HardwareMitigated);
        assert_eq!(select_profile(15), Profile::HardwareMitigated);
    }

    #[test]
    fn test_select_profile_software_when_csv2_zero() {
        assert_eq!(select_profile(0), Profile::Software);
    }

    #[test]
    fn test_profile_log_strings() {
        assert_eq!(Profile::HardwareMitigated.as_str(), "hardware-mitigated");
        assert_eq!(Profile::Software.as_str(), "software");
    }
}
