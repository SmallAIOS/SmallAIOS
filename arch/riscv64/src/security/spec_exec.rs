// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V speculative-execution mitigation scaffolding.
//!
//! Part of OpenSpec change `spec-exec-mitigations-v1`, Phase 4 (RISC-V).
//!
//! # Why this is scaffolding, not a mitigation
//!
//! RISC-V is a boot-only target in SmallAIOS today — no U-mode syscall
//! workloads are exercised yet. More importantly, the hardware extensions
//! that would let us *actually* mitigate the Spectre-class attacks at the
//! syscall trust boundary are mostly **not ratified** as of 2026-02:
//!
//! - **Zicfilp** (forward-edge CFI / landing pads) — Spectre-v2-class
//!   (branch-target-injection) mitigation. NOT ratified as of 2026-02.
//! - **Zicfiss** (backward-edge CFI / shadow stack) — return-address /
//!   Retbleed-class mitigation. NOT ratified as of 2026-02.
//! - **Zicbom / Zicbop / Zicboz** (cache-block management: clean / prefetch /
//!   zero) — used for partial cache-state cleanup at the privileged
//!   transition. Ratified, but no silicon in our validation matrix exposes
//!   them yet, so the cleanup path is gated off.
//!
//! Until those ratify *and* our silicon advertises them, the only mitigation
//! the RISC-V port can apply is the architecturally-defined `fence.i`
//! instruction-fetch barrier at the privileged-transition boundary (see
//! [`crate::syscall::handle_ecall`]). That is a correctness/coherence barrier,
//! not a speculation barrier — it does NOT close the Spectre-v1/v2 windows.
//!
//! Production RISC-V deployments MUST therefore be flagged
//! **"speculation mitigations partial"** in the DO-178C safety case. See
//! `docs/spec-exec-audit-riscv.md`.
//!
//! # TODO — CSRs / instructions to program once the extensions ratify
//!
//! When Zicfilp / Zicfiss ratify and the target silicon advertises them
//! (via the DTB `riscv,isa` string and/or a `misa`-adjacent discovery CSR),
//! [`init`] will be filled in to program, at minimum:
//!
//! - **`menvcfg.LPE` / `senvcfg.LPE`** (Zicfilp) — enable landing-pad
//!   enforcement for the privilege level we run at (S-mode for the kernel,
//!   U-mode for any future task graph). Forward-edge CFI / Spectre-v2.
//! - **`menvcfg.SSE` / `senvcfg.SSE`** (Zicfiss) — enable shadow-stack
//!   enforcement; allocate the shadow stack and point **`ssp`** (the
//!   shadow-stack pointer CSR) at it. Backward-edge CFI / Retbleed-class.
//! - **`mseccfg` (machine security configuration)** — sticky bits governing
//!   landing-pad / shadow-stack lockdown; set once at boot, never cleared.
//! - **Zicbom cache-block clean** — issue `cbo.clean` / `cbo.flush` over the
//!   capability-check working set at the privileged transition, to bound the
//!   cache-state residue an attacker can probe (partial Spectre-v1 cache-
//!   timing mitigation; not a full fix).
//! - **`fence.i`** — already emitted unconditionally at the privileged
//!   transition (see [`crate::syscall::handle_ecall`]); this stays even after
//!   the CFI extensions land (defense in depth, instruction-fetch coherence).
//!
//! None of these CSRs/instructions exist in a ratified form we can rely on
//! yet, so `init()` only logs that the port is running in partial-mitigation
//! mode. The shape of this module is fixed now so the ratification follow-up
//! is a fill-in, not a redesign.

/// The single boot-record line emitted by [`init`].
///
/// Pulled out as a `const` so a host unit test can assert the exact wording
/// (the safety case greps boot logs for this marker) without triggering the
/// UART MMIO write, which is a wild pointer dereference on a host target.
pub(crate) const SCAFFOLDING_LOG_LINE: &str =
    "[spec-exec-riscv] scaffolding — extensions not yet ratified, \
     software mitigations partial\n";

/// Initialize RISC-V speculative-execution mitigations.
///
/// **Scaffolding only.** Currently this does not program any CSR or issue any
/// cache-management instruction — the extensions that would make that
/// meaningful are unratified (see module docs). It logs a single line so the
/// boot record unambiguously shows the RISC-V port is running with only
/// partial software mitigations, which the safety case must account for.
///
/// Once Zicfilp / Zicfiss ratify and silicon support is detected, this
/// function will program the CSRs enumerated in the module-level TODO. The
/// call site (boot path) and signature are intentionally fixed now so that
/// future work is a body fill-in with no churn at the call site.
///
/// The UART write is `cfg`-gated off under `test` because `uart::putc`
/// poll-reads NS16550A MMIO at `0x1000_0005`, which is a wild access on a
/// host target. The non-test path keeps the real boot-log side effect.
pub fn init() {
    #[cfg(not(test))]
    crate::uart::puts(SCAFFOLDING_LOG_LINE);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `init()` is a side-effect-only logging stub today; the contract under
    /// test is that it is callable and total (never panics / diverges) so the
    /// fixed boot call site stays correct-by-construction. The UART write is
    /// `cfg(not(test))`-gated, so this is a wild-pointer-free host check.
    #[test]
    fn test_init_is_callable_and_total() {
        init();
        // Reaching here proves `init` returned normally on the host target.
    }

    /// The safety case greps boot logs for this exact marker; pin its wording
    /// so a refactor cannot silently weaken the partial-mitigation signal.
    #[test]
    fn test_scaffolding_log_line_wording() {
        assert!(SCAFFOLDING_LOG_LINE.starts_with("[spec-exec-riscv] scaffolding"));
        assert!(SCAFFOLDING_LOG_LINE.contains("not yet ratified"));
        assert!(SCAFFOLDING_LOG_LINE.contains("software mitigations partial"));
        assert!(SCAFFOLDING_LOG_LINE.ends_with('\n'));
    }
}
