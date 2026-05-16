// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 speculative-execution mitigations (Spectre v1 / v2).
//!
//! Phase 2 of OpenSpec change `spec-exec-mitigations-v1`. See
//! `docs/spec-exec-audit.md` for the threat model and design rationale; this
//! module implements the *runtime* (CPU-feature-gated) half. The *compile
//! time* half (Retpoline + Speculative Load Hardening) is emitted by
//! `arch/x86_64/build.rs`.
//!
//! ## What this module configures
//!
//! | Mitigation | Mechanism | Knob |
//! |------------|-----------|------|
//! | Spectre-v2 (BTI), enhanced HW path | `IA32_SPEC_CTRL.IBRS` set once, sticky | CPUID.07H:EDX[29] |
//! | Spectre-v2 (BTI), legacy SW path   | `IA32_SPEC_CTRL.IBRS` toggled per syscall entry | CPUID.07H:EDX[26..27] |
//! | Cross-SMT branch sharing           | `IA32_SPEC_CTRL.STIBP` | CPUID.01H:EBX[23] (HTT) |
//! | Cross-domain BTB poisoning         | `IA32_PRED_CMD.IBPB` on syscall entry | CPUID.07H:EDX[26] |
//!
//! ## Design decisions (authoritative — mirrors the audit doc)
//!
//! * **Enhanced IBRS preferred.** When CPUID leaf 7 EDX[29] reports Enhanced
//!   IBRS, IBRS is set *once* and left set ("sticky" — the CPU keeps the
//!   higher privilege domain protected without per-transition writes). On
//!   parts with only legacy IBRS we fall back to writing IBRS in the syscall
//!   entry trampoline and emit a boot warning (legacy IBRS is slow).
//! * **IBPB on every syscall entry**, *after* the capability check, gated by
//!   the `spec-exec-ibpb-off` opt-out feature (residual-risk trade-off
//!   documented on the feature in `Cargo.toml`).
//! * **STIBP** is set whenever SMT is detected (CPUID leaf 1 EBX[23]).
//!
//! Everything here is clean-room `#![no_std]` (no external crates).

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// `IA32_SPEC_CTRL` MSR — speculation control (IBRS/STIBP/SSBD).
const IA32_SPEC_CTRL: u32 = 0x0000_0048;
/// `IA32_PRED_CMD` MSR — write-only, `bit0 = IBPB`.
const IA32_PRED_CMD: u32 = 0x0000_0049;

/// `IA32_SPEC_CTRL.IBRS` (bit 0).
const SPEC_CTRL_IBRS: u64 = 1 << 0;
/// `IA32_SPEC_CTRL.STIBP` (bit 1).
const SPEC_CTRL_STIBP: u64 = 1 << 1;
/// `IA32_PRED_CMD.IBPB` (bit 0).
const PRED_CMD_IBPB: u64 = 1 << 0;

/// Detected IBRS support level. Stored as a plain `u8` in an atomic so the
/// syscall entry path can read it without locking.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum IbrsLevel {
    /// No IBRS support reported by CPUID.
    None = 0,
    /// Legacy IBRS — must be (re)written on every privilege transition.
    Legacy = 1,
    /// Enhanced IBRS — set once, sticky; no per-transition write needed.
    Enhanced = 2,
}

impl IbrsLevel {
    fn from_u8(v: u8) -> Self {
        match v {
            2 => IbrsLevel::Enhanced,
            1 => IbrsLevel::Legacy,
            _ => IbrsLevel::None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            IbrsLevel::None => "none",
            IbrsLevel::Legacy => "legacy",
            IbrsLevel::Enhanced => "enhanced",
        }
    }
}

/// Detected IBRS level (see [`IbrsLevel`]). `255` = "not yet initialized".
static IBRS_LEVEL: AtomicU8 = AtomicU8::new(255);
/// Whether SMT (Hyper-Threading) was detected, so STIBP is wanted.
static SMT_DETECTED: AtomicU8 = AtomicU8::new(0);
/// The `IA32_SPEC_CTRL` value to (re)write on the *legacy* per-entry path.
/// Composed once at [`init`] time from the detected IBRS/STIBP bits so the
/// hot path is a single relaxed load + `wrmsr`.
static LEGACY_SPEC_CTRL: AtomicU64 = AtomicU64::new(0);

/// Read a Model-Specific Register.
///
/// # Safety
/// `msr` must be a valid, readable MSR address on this CPU.
#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") low,
            out("edx") high,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((high as u64) << 32) | (low as u64)
}

/// Write a Model-Specific Register.
///
/// # Safety
/// `msr` must be a valid, writable MSR address and `value` a legal value for
/// it on this CPU.
#[inline]
unsafe fn wrmsr(msr: u32, value: u64) {
    let low = value as u32;
    let high = (value >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") low,
            in("edx") high,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// Raw CPUID with sub-leaf. Returns `(eax, ebx, ecx, edx)`.
///
/// `rbx` is the LLVM-reserved base pointer under some codegen models, so it is
/// shuffled through `rsi` per the standard inline-asm idiom.
///
/// # Safety
/// CPUID is unprivileged and always safe on x86-64; marked `unsafe` only
/// because it is raw `asm!`.
#[inline]
unsafe fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "mov {tmp:r}, rbx",
            "cpuid",
            "xchg {tmp:r}, rbx",
            tmp = out(reg) ebx,
            inout("eax") leaf => eax,
            inout("ecx") subleaf => ecx,
            out("edx") edx,
            options(nostack, preserves_flags),
        );
    }
    (eax, ebx, ecx, edx)
}

/// Initialize x86-64 speculative-execution mitigations.
///
/// Probes CPUID for IBRS / Enhanced-IBRS / STIBP / IBPB support, programs the
/// enhanced (sticky) path when available, composes the legacy per-entry MSR
/// value otherwise, and emits the Phase-2 boot log line:
///
/// ```text
/// [spec-exec-x86] IBRS=<none|legacy|enhanced> STIBP=<n|y> IBPB-on-entry=on|off
/// ```
///
/// Must be called once during early kernel bring-up (after the serial console
/// is up so the boot log lands). Idempotent: a second call simply re-probes
/// and re-programs with the same result.
///
/// # Safety
/// Reads/writes `IA32_SPEC_CTRL`. Caller must run in kernel (ring 0) context.
pub unsafe fn init(log: impl Fn(&str)) {
    // --- CPUID feature probe ---------------------------------------------
    // Leaf 7, sub-leaf 0, EDX:
    //   bit 26 = IBRS/IBPB supported (IA32_SPEC_CTRL + IA32_PRED_CMD)
    //   bit 27 = STIBP supported
    //   bit 29 = IA32_ARCH_CAPABILITIES present (gates Enhanced-IBRS query)
    let (_, _, _, leaf7_edx) = unsafe { cpuid(7, 0) };
    let ibrs_ibpb_supported = (leaf7_edx & (1 << 26)) != 0;
    let stibp_supported = (leaf7_edx & (1 << 27)) != 0;
    let arch_caps_present = (leaf7_edx & (1 << 29)) != 0;

    // Enhanced IBRS is advertised in IA32_ARCH_CAPABILITIES (MSR 0x10A) bit 1
    // (IBRS_ALL). The Phase-2 design references "CPUID leaf 7 EDX[29]" as the
    // gate; EDX[29] is exactly the IA32_ARCH_CAPABILITIES-present bit, and the
    // sticky/enhanced behaviour is then confirmed via that MSR.
    const IA32_ARCH_CAPABILITIES: u32 = 0x0000_010A;
    const ARCH_CAP_IBRS_ALL: u64 = 1 << 1;
    let enhanced_ibrs = arch_caps_present
        && ibrs_ibpb_supported
        && (unsafe { rdmsr(IA32_ARCH_CAPABILITIES) } & ARCH_CAP_IBRS_ALL) != 0;

    let ibrs_level = if !ibrs_ibpb_supported {
        IbrsLevel::None
    } else if enhanced_ibrs {
        IbrsLevel::Enhanced
    } else {
        IbrsLevel::Legacy
    };

    // Leaf 1 EBX[23] = HTT (logical-processor count > 1 → SMT possible).
    let (_, leaf1_ebx, _, _) = unsafe { cpuid(1, 0) };
    let smt = (leaf1_ebx & (1 << 23)) != 0;

    // --- Program the chosen path -----------------------------------------
    let want_stibp = smt && stibp_supported;

    match ibrs_level {
        IbrsLevel::Enhanced => {
            // Set IBRS once and leave it set ("sticky"). Also set STIBP here
            // if SMT — it is likewise sticky under Enhanced IBRS.
            let mut v = unsafe { rdmsr(IA32_SPEC_CTRL) };
            v |= SPEC_CTRL_IBRS;
            if want_stibp {
                v |= SPEC_CTRL_STIBP;
            }
            unsafe { wrmsr(IA32_SPEC_CTRL, v) };
            // Legacy path unused; keep it harmless if ever read.
            LEGACY_SPEC_CTRL.store(v, Ordering::Relaxed);
        }
        IbrsLevel::Legacy => {
            // Legacy IBRS must be re-asserted on every kernel entry. Compose
            // the value the trampoline will write each time.
            let mut v = SPEC_CTRL_IBRS;
            if want_stibp {
                v |= SPEC_CTRL_STIBP;
            }
            LEGACY_SPEC_CTRL.store(v, Ordering::Relaxed);
            // Assert it now for the current (boot) context too.
            unsafe { wrmsr(IA32_SPEC_CTRL, v) };
            log("[spec-exec-x86] WARNING: only legacy IBRS available — \
                 per-entry MSR writes (perf cost). Prefer Enhanced-IBRS HW.\n");
        }
        IbrsLevel::None => {
            // No IBRS at all. Retpoline (compiler) still mitigates BTI; STIBP
            // may still be independently settable on some parts.
            if want_stibp {
                let mut v = unsafe { rdmsr(IA32_SPEC_CTRL) };
                v |= SPEC_CTRL_STIBP;
                unsafe { wrmsr(IA32_SPEC_CTRL, v) };
                LEGACY_SPEC_CTRL.store(v, Ordering::Relaxed);
            }
            log("[spec-exec-x86] WARNING: no IBRS support — relying on \
                 Retpoline (compiler) for Spectre-v2.\n");
        }
    }

    IBRS_LEVEL.store(ibrs_level as u8, Ordering::Relaxed);
    SMT_DETECTED.store(u8::from(smt), Ordering::Relaxed);

    // --- Boot log line (task 1.9) ----------------------------------------
    let ibpb_on = ibpb_on_entry_enabled() && ibrs_ibpb_supported;
    log("[spec-exec-x86] IBRS=");
    log(ibrs_level.as_str());
    log(" STIBP=");
    log(if want_stibp { "y" } else { "n" });
    log(" IBPB-on-entry=");
    log(if ibpb_on { "on" } else { "off" });
    log("\n");
}

/// Whether per-syscall IBPB is compiled in (compile-time `cfg`).
///
/// True only when the spec-exec mitigations are active (`spec-exec-x86`) AND
/// the `spec-exec-ibpb-off` opt-out feature is *not* set (residual-risk
/// trade-off documented on the feature in `Cargo.toml`). The actual IBPB
/// `wrmsr` is issued kernel-side after the capability check
/// (`smallaios_kernel::spec_exec::predictor_barrier_ibpb`); this query gates
/// only the boot-log status string.
#[inline(always)]
pub const fn ibpb_on_entry_enabled() -> bool {
    cfg!(feature = "spec-exec-x86") && !cfg!(feature = "spec-exec-ibpb-off")
}

/// The detected IBRS support level (valid after [`init`]).
#[inline]
pub fn ibrs_level() -> IbrsLevel {
    IbrsLevel::from_u8(IBRS_LEVEL.load(Ordering::Relaxed))
}

/// Whether SMT was detected at [`init`] time.
#[inline]
pub fn smt_detected() -> bool {
    SMT_DETECTED.load(Ordering::Relaxed) != 0
}

/// Re-assert `IA32_SPEC_CTRL` on the **legacy** IBRS path.
///
/// Called from the syscall entry trampoline (task 1.6). A no-op fast-return on
/// Enhanced-IBRS / no-IBRS parts (the value is sticky / unused there), so the
/// trampoline can call it unconditionally.
///
/// # Safety
/// Writes `IA32_SPEC_CTRL`; ring-0 only. Designed to be called on every kernel
/// entry before any indirect branch into kernel code.
#[inline]
pub unsafe fn legacy_ibrs_on_entry() {
    if IBRS_LEVEL.load(Ordering::Relaxed) == IbrsLevel::Legacy as u8 {
        let v = LEGACY_SPEC_CTRL.load(Ordering::Relaxed);
        unsafe { wrmsr(IA32_SPEC_CTRL, v) };
    }
}

/// Emit an Indirect Branch Predictor Barrier (IBPB) at syscall entry.
///
/// Flushes the indirect-branch predictor so a prior (user/other-domain)
/// context cannot have trained a branch the kernel is about to take. Per the
/// Phase-2 design this is invoked **after** the capability check on the
/// syscall path (see the kernel-side wiring), and compiled out entirely when
/// the `spec-exec-ibpb-off` feature is active.
///
/// # Safety
/// Writes the write-only `IA32_PRED_CMD` MSR; ring-0 only.
#[inline]
pub unsafe fn issue_ibpb() {
    if ibpb_on_entry_enabled() {
        unsafe { wrmsr(IA32_PRED_CMD, PRED_CMD_IBPB) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibrs_level_roundtrips() {
        assert_eq!(IbrsLevel::from_u8(0), IbrsLevel::None);
        assert_eq!(IbrsLevel::from_u8(1), IbrsLevel::Legacy);
        assert_eq!(IbrsLevel::from_u8(2), IbrsLevel::Enhanced);
        assert_eq!(IbrsLevel::from_u8(99), IbrsLevel::None);
    }

    #[test]
    fn ibrs_level_strings() {
        assert_eq!(IbrsLevel::None.as_str(), "none");
        assert_eq!(IbrsLevel::Legacy.as_str(), "legacy");
        assert_eq!(IbrsLevel::Enhanced.as_str(), "enhanced");
    }

    #[test]
    fn ibpb_default_on_unless_opted_out() {
        assert_eq!(
            ibpb_on_entry_enabled(),
            cfg!(feature = "spec-exec-x86") && !cfg!(feature = "spec-exec-ibpb-off")
        );
    }

    #[test]
    fn msr_and_bit_constants_are_canonical() {
        // Guards against accidental edits to the architectural constants.
        assert_eq!(IA32_SPEC_CTRL, 0x48);
        assert_eq!(IA32_PRED_CMD, 0x49);
        assert_eq!(SPEC_CTRL_IBRS, 1);
        assert_eq!(SPEC_CTRL_STIBP, 2);
        assert_eq!(PRED_CMD_IBPB, 1);
    }
}
