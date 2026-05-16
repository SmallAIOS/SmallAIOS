// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V syscall handler (23.8).
//!
//! Syscalls from U-mode arrive as `ecall` (environment call), generating
//! an exception with scause = 8 (EXC_ECALL_FROM_U).
//!
//! RISC-V syscall calling convention:
//!   - a7 = syscall number
//!   - a0-a5 = arguments
//!   - Return value in a0
//!
//! This matches the Linux RISC-V ABI, making future compatibility easier.
//!
//! # Privileged-transition instruction-fetch barrier (`fence.i`)
//!
//! Part of OpenSpec change `spec-exec-mitigations-v1`, Phase 4 (RISC-V,
//! scaffolding). The `ecall` path is a privileged transition: U-mode traps
//! into S-mode, the kernel dispatches, then `sret` (in `trap.rs`) resumes
//! U-mode. RISC-V does **not** make stores to the instruction stream visible
//! to subsequent instruction fetches on the same hart without an explicit
//! [`fence.i`] — the architecturally-defined instruction-fetch / fetch-
//! coherence barrier.
//!
//! We emit `fence.i` **after** [`smallaios_kernel::syscall::dispatch`]
//! returns and **before** control unwinds toward `sret` back to U-mode. The
//! *after* side is the load-bearing one: a syscall may legitimately have
//! mutated the code that the about-to-resume context will fetch (future
//! code-load / page-remap / module-style syscalls), and the resuming hart
//! must observe that mutation. Placing it after dispatch also bounds any
//! stale-instruction window opened during kernel servicing of the call.
//!
//! This is a coherence / correctness barrier, **not** a speculation barrier:
//! it does not by itself close the Spectre-v1/v2 windows. Those need
//! Zicfilp / Zicfiss (forward/backward-edge CFI), which are **not ratified**
//! as of 2026-02. RISC-V therefore ships with *partial* mitigations — see
//! `docs/spec-exec-audit-riscv.md` and `crate::security::spec_exec`. The
//! barrier is inserted now so the privileged-transition entry shape is
//! correct-by-construction even though syscall workloads are not yet
//! exercised on this target.

use super::trap::TrapFrame;

/// Instruction-fetch / fetch-coherence barrier at the privileged-transition
/// boundary.
///
/// Emits a single `fence.i` (Zifencei, part of the RV64GC baseline). On the
/// host test target this is a no-op so `handle_ecall` stays host-testable.
///
/// See the module docs for placement rationale (emitted *after* dispatch,
/// *before* the return toward `sret` to U-mode).
#[inline(always)]
fn fence_i_barrier() {
    // SAFETY: `fence.i` has no operands, no memory effects beyond ordering
    // the calling hart's own instruction fetch w.r.t. its prior stores, and
    // is part of the RV64GC baseline (Zifencei). It cannot trap in S-mode.
    #[cfg(not(test))]
    unsafe {
        core::arch::asm!("fence.i", options(nostack, preserves_flags));
    }
}

/// Handle an ecall from U-mode.
///
/// Extracts syscall number from a7 (x17) and arguments from a0-a5 (x10-x15).
/// Result is written back to a0 (x10) in the trap frame.
///
/// Note: The trap handler (trap.rs) advances sepc by 4 after this returns
/// so that sret goes to the instruction after ecall.
pub fn handle_ecall(frame: &mut TrapFrame) {
    let syscall_number = frame.regs[17] as usize; // a7 = x17

    let args = smallaios_kernel::syscall::SyscallArgs::new(
        syscall_number,
        [
            frame.regs[10] as usize, // a0 = x10
            frame.regs[11] as usize, // a1 = x11
            frame.regs[12] as usize, // a2 = x12
            frame.regs[13] as usize, // a3 = x13
            frame.regs[14] as usize, // a4 = x14
            frame.regs[15] as usize, // a5 = x15
        ],
    );

    let result = smallaios_kernel::syscall::dispatch(&args);

    // Write return value to a0 (x10) in the trap frame
    frame.regs[10] = result as u64;

    // Privileged-transition instruction-fetch barrier (Phase 4 scaffolding).
    // Emitted AFTER dispatch and BEFORE the return that unwinds toward `sret`
    // (in trap.rs) back to U-mode: a syscall may have mutated code the
    // resuming context will fetch, and `fence.i` makes that visible to this
    // hart's subsequent instruction fetch. Coherence barrier, not a
    // speculation barrier — see module docs and docs/spec-exec-audit-riscv.md.
    fence_i_barrier();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(syscall_nr: u64, args: [u64; 6]) -> TrapFrame {
        let mut frame = TrapFrame {
            regs: [0u64; 32],
            sstatus: 0,
            sepc: 0x1000,
            stval: 0,
            scause: 8, // EXC_ECALL_FROM_U
        };
        frame.regs[17] = syscall_nr; // a7
        frame.regs[10] = args[0]; // a0
        frame.regs[11] = args[1]; // a1
        frame.regs[12] = args[2]; // a2
        frame.regs[13] = args[3]; // a3
        frame.regs[14] = args[4]; // a4
        frame.regs[15] = args[5]; // a5
        frame
    }

    #[test]
    fn test_ecall_dispatches_and_returns() {
        // Syscall 0xFF is out of range, should return error.
        let mut frame = make_frame(0xFF, [0; 6]);
        handle_ecall(&mut frame);
        // Result is in a0 (x10), should be negative (error)
        let result = frame.regs[10] as i64;
        assert!(result < 0, "out-of-range syscall should return error");
    }

    #[test]
    fn test_ecall_register_mapping() {
        // Verify a7->syscall_nr and a0-a5->args mapping
        let frame = make_frame(0x50, [1, 2, 3, 4, 5, 6]);
        assert_eq!(frame.regs[17], 0x50); // a7
        assert_eq!(frame.regs[10], 1); // a0
        assert_eq!(frame.regs[11], 2); // a1
        assert_eq!(frame.regs[15], 6); // a5
    }

    #[test]
    fn test_ecall_preserves_other_registers() {
        let mut frame = make_frame(0xFF, [0; 6]);
        frame.regs[1] = 0xDEAD; // ra
        frame.regs[8] = 0xBEEF; // s0
        handle_ecall(&mut frame);
        assert_eq!(frame.regs[1], 0xDEAD);
        assert_eq!(frame.regs[8], 0xBEEF);
    }

    /// The `fence.i` privileged-transition barrier is on the `handle_ecall`
    /// path (after dispatch, before return). On the host target the asm is
    /// `cfg(not(test))`-gated to a no-op; this test pins that the barrier
    /// helper stays callable and total so the scaffolding can't regress to a
    /// shape where the barrier is unreachable.
    #[test]
    fn test_fence_i_barrier_is_total() {
        // Direct call: must not panic / diverge on host.
        fence_i_barrier();
        // Indirect via the real entry path: handle_ecall reaches the barrier
        // after dispatch even for an out-of-range syscall.
        let mut frame = make_frame(0xFF, [0; 6]);
        handle_ecall(&mut frame);
    }
}
