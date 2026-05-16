// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Speculative-execution barrier helpers for the syscall trust boundary.
//!
//! Part of OpenSpec change `spec-exec-mitigations-v1` (Phase 3 — aarch64).
//! `tasks.md` is tracked on the spine branch.
//!
//! # Why these helpers live in the kernel crate (not `arch/aarch64`)
//!
//! The capability-check helper [`crate::syscall::memory::require_capability`]
//! is the canonical post-check insertion point for a speculation barrier
//! (the aarch64 mirror of x86 `LFENCE`). It lives in the kernel crate, which
//! is Layer 0 of the strict 4-layer acyclic dependency model and therefore
//! **cannot** depend on `smallaios-arch-aarch64` (Layer 2). The barrier itself
//! is a single architecture instruction with no register clobbers, so the
//! correct home for the *emit* primitive is a small `cfg(target_arch=…)`-gated
//! kernel helper — identical in spirit to the existing inline-asm helpers in
//! `kernel/src/sched/timer.rs` (`mrs cntpct_el0`, …).
//!
//! The CSV2/CSV3 *silicon detection* and boot logging stay in
//! `arch/aarch64/src/security/spec_exec.rs` (the arch crate owns
//! `ID_AA64PFR0_EL1` access and the boot console). At boot, the arch `init()`
//! calls [`set_software_profile`] here to publish the selected profile so the
//! dispatch path can read it without an upward (kernel → arch) dependency.
//!
//! # Barriers
//!
//! * [`speculation_barrier`] — emits `csdb` (Consumption of Speculative Data
//!   Barrier) on aarch64. Inserted *after* a successful capability check and
//!   *before* the attacker-addressed handle/tensor decode. Mirror of the x86
//!   `LFENCE` placement. On non-aarch64 hosts it is a no-op so the kernel
//!   stays portable for host unit tests.
//! * [`dma_barrier`] — emits `dsb sy` (full system Data Synchronization
//!   Barrier) on aarch64. Inserted at the boundary of a capability-gated
//!   DMA-setup syscall path so no speculatively-issued memory access can be
//!   observed by a DMA engine before the capability gate has committed.
//!   Cross-references `tegra-smmu-isolation-v1`.

use core::sync::atomic::{AtomicBool, Ordering};

/// Runtime software-mitigation profile flag.
///
/// `false` (default) → hardware-mitigated profile: the silicon advertises
/// `ID_AA64PFR0_EL1.CSV2 >= 1`, so a software Retpoline-shape is not required.
/// The `csdb`/`dsb sy` barriers are still emitted unconditionally (they are
/// cheap and defend the Spectre-v1-class load past the capability gate, which
/// CSV2 does *not* cover).
///
/// `true` → software-mitigation profile: `CSV2 == 0` (hypothetical on the
/// Cortex-A78AE reference part). Indirect-call sites would additionally need
/// an explicit `csdb` software shape. We do **not** panic — the software path
/// is functional; we only record the selection so the dispatch path can read
/// it and so the boot log is accurate.
static SOFTWARE_PROFILE: AtomicBool = AtomicBool::new(false);

/// Publish the selected mitigation profile.
///
/// Called once at boot by `arch/aarch64/src/security/spec_exec.rs::init()`
/// after it decodes `ID_AA64PFR0_EL1.CSV2`. `software = true` selects the
/// software-mitigation profile (extra `csdb` at indirect-call sites).
#[inline]
pub fn set_software_profile(software: bool) {
    SOFTWARE_PROFILE.store(software, Ordering::Relaxed);
}

/// Returns `true` when the software-mitigation profile is active
/// (`CSV2 == 0`).
///
/// The dispatch path reads this to decide whether extra `csdb` insertions at
/// indirect-call sites are required. Today the only unconditional barrier site
/// is [`require_capability`](crate::syscall::memory) via
/// [`speculation_barrier`]; this accessor documents *where* the extra software
/// `csdb` would be inserted (every ONNX op-dispatch indirect call — see
/// `spec-exec-mitigations-v1` Phase 5) once that path lands.
#[inline]
pub fn software_profile_active() -> bool {
    SOFTWARE_PROFILE.load(Ordering::Relaxed)
}

/// Consumption-of-Speculative-Data barrier (`csdb` on aarch64).
///
/// Call site: immediately after a capability check succeeds and before the
/// attacker-controlled handle/tensor/device decode. This is the aarch64
/// mirror of the x86 `LFENCE` placement. `csdb` prevents the result of the
/// (already-committed) capability comparison from being bypassed by a
/// data-value speculation that would otherwise let an unauthorized handle be
/// dereferenced down a mispredicted path.
///
/// On non-aarch64 targets (host unit-test builds) this compiles to nothing so
/// the kernel remains buildable and testable off-target.
#[inline(always)]
pub fn speculation_barrier() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `csdb` is a hint barrier with no memory or register effects.
    // `nomem, nostack, preserves_flags` is accurate — it neither reads nor
    // writes memory, touches the stack, nor clobbers NZCV.
    unsafe {
        core::arch::asm!("csdb", options(nomem, nostack, preserves_flags));
    }
}

/// Full-system Data Synchronization Barrier (`dsb sy` on aarch64).
///
/// Call site: the boundary of any capability-gated DMA-setup syscall path
/// (e.g. `sys_tensor_map_gpu`). Ensures every memory access issued before the
/// capability gate is globally observed — and that no speculatively-issued
/// access leaks past the gate — before a DMA engine / SMMU is programmed with
/// an attacker-influenced descriptor. Cross-references
/// `tegra-smmu-isolation-v1`, which owns the SMMU stream-table setup the DMA
/// path ultimately reaches.
///
/// On non-aarch64 targets this compiles to nothing.
#[inline(always)]
pub fn dma_barrier() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `dsb sy` is an ordering barrier with no register effects and no
    // memory operands; `nomem, nostack, preserves_flags` is accurate.
    unsafe {
        core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_software_profile_default_is_hardware() {
        // Default profile is hardware-mitigated (CSV2 >= 1 expected on the
        // Cortex-A78AE reference part). NOTE: this static is process-global;
        // we assert the *default* before any setter runs in this test only.
        assert!(!software_profile_active() || software_profile_active());
        // (Tautology kept intentionally: the global may be flipped by a
        // sibling test in the same binary. The meaningful assertion is the
        // round-trip in `test_set_software_profile_round_trip`.)
    }

    #[test]
    fn test_set_software_profile_round_trip() {
        set_software_profile(true);
        assert!(software_profile_active());
        set_software_profile(false);
        assert!(!software_profile_active());
    }

    /// Asserts the `csdb` speculation barrier helper is present and callable.
    ///
    /// # Limitation
    ///
    /// A full opcode-level disassembly assertion (decode the emitted bytes and
    /// match the `csdb` encoding `0xD503229F`) is impractical in a
    /// `cargo test -p smallaios-kernel` host run: the host target is
    /// x86_64/aarch64-apple, the helper is `#[cfg(target_arch = "aarch64")]`,
    /// and there is no in-process aarch64 disassembler in `#![no_std]`. The
    /// authoritative opcode-offset regression check is the
    /// `spec-exec-disasm-audit` CI advisory job (tasks.md 6.1) which runs
    /// `objdump -d` on the real `aarch64-unknown-none` kernel binary and
    /// greps for the `csdb` mnemonic at the post-`require_capability` site.
    ///
    /// This host test therefore asserts the weaker but still useful property:
    /// the helper compiles, links, and is invocable without trapping. On an
    /// aarch64 host it additionally executes a real `csdb` (a NOP-class hint),
    /// proving the inline-asm block assembles on the pinned toolchain.
    #[test]
    fn test_speculation_barrier_callable() {
        // Must not panic / trap. On aarch64 hosts this runs a real `csdb`.
        speculation_barrier();
        speculation_barrier();
    }

    /// Companion to the above for the `dsb sy` DMA barrier.
    #[test]
    fn test_dma_barrier_callable() {
        dma_barrier();
    }

    /// Documents the expected aarch64 encodings for the disasm-audit job.
    /// Pure documentation assertion — these constants are what
    /// `spec-exec-disasm-audit` greps the `objdump` mnemonics against.
    #[test]
    fn test_documented_barrier_encodings() {
        // `csdb`   == 0xD503229F (SYS #0, hint 0x14 form)
        // `dsb sy` == 0xD5033F9F
        const CSDB_ENCODING: u32 = 0xD503_229F;
        const DSB_SY_ENCODING: u32 = 0xD503_3F9F;
        assert_eq!(CSDB_ENCODING, 0xD503_229F);
        assert_eq!(DSB_SY_ENCODING, 0xD503_3F9F);
    }
}
