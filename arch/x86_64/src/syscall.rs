// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 syscall/sysret entry point.
//!
//! In VM mode, SmallAIOS uses the `syscall` instruction for system calls.
//! The `syscall` instruction:
//!   - Saves RIP in RCX, RFLAGS in R11
//!   - Loads RIP from IA32_LSTAR MSR
//!   - Loads CS from IA32_STAR MSR bits [47:32]
//!   - Masks RFLAGS with IA32_FMASK MSR
//!
//! Calling convention (Linux-compatible):
//!   - Syscall number in RAX
//!   - Arguments: RDI, RSI, RDX, R10, R8, R9
//!   - Return value in RAX
//!   - RCX and R11 are clobbered by syscall/sysret

use core::arch::{asm, naked_asm};

/// MSR addresses for syscall configuration.
const IA32_STAR: u32 = 0xC000_0081;
const IA32_LSTAR: u32 = 0xC000_0082;
const IA32_FMASK: u32 = 0xC000_0084;
const IA32_EFER: u32 = 0xC000_0080;

/// EFER.SCE bit — enables syscall/sysret.
const EFER_SCE: u64 = 1 << 0;

/// RFLAGS bits to mask on syscall entry.
/// Mask IF (interrupts) and TF (trap flag) for safety.
const FMASK_VALUE: u64 = 0x0000_0000_0000_0200 | 0x0000_0000_0000_0100;

/// Read a Model-Specific Register.
///
/// # Safety
/// Must be called with a valid MSR address.
unsafe fn rdmsr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        asm!(
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
/// Must be called with a valid MSR address and appropriate value.
unsafe fn wrmsr(msr: u32, value: u64) {
    let low = value as u32;
    let high = (value >> 32) as u32;
    unsafe {
        asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") low,
            in("edx") high,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// Initialize the syscall/sysret mechanism.
///
/// Must be called after GDT is loaded. Sets up:
/// - IA32_EFER.SCE = 1 (enable syscall instruction)
/// - IA32_STAR = kernel CS/SS selectors
/// - IA32_LSTAR = address of syscall entry point
/// - IA32_FMASK = mask IF and TF on entry
///
/// # Safety
/// Must be called exactly once during kernel initialization, after GDT setup.
pub unsafe fn init() {
    // Enable SYSCALL/SYSRET in EFER
    let efer = unsafe { rdmsr(IA32_EFER) };
    unsafe {
        wrmsr(IA32_EFER, efer | EFER_SCE);
    }

    // STAR: bits [47:32] = kernel CS selector (0x08), bits [63:48] = user CS base
    // For unikernel mode, user CS/SS don't matter, but we set them for correctness.
    // Kernel CS = 0x08, Kernel SS = 0x10
    // User CS = 0x18 | 3, User SS = 0x20 | 3 (ring 3 RPL)
    let star_value: u64 = (0x0008u64 << 32) | (0x0018u64 << 48);
    unsafe {
        wrmsr(IA32_STAR, star_value);
    }

    // LSTAR: syscall entry point address
    unsafe {
        wrmsr(IA32_LSTAR, syscall_entry as *const () as usize as u64);
    }

    // FMASK: clear IF and TF on syscall entry
    unsafe {
        wrmsr(IA32_FMASK, FMASK_VALUE);
    }
}

/// Re-assert legacy IBRS on kernel entry (spec-exec-mitigations-v1 Phase 2,
/// task 1.6).
///
/// On Enhanced-IBRS / no-IBRS parts this is a single relaxed-atomic compare
/// and immediate return (IBRS is sticky / unused there), so the trampoline
/// calls it unconditionally. On legacy-IBRS parts it re-writes
/// `IA32_SPEC_CTRL` so the indirect-branch predictor is in the protected
/// state *before* the kernel takes any indirect branch in `dispatch`.
///
/// IBPB is intentionally NOT issued here: per the audit it must fire *after*
/// the capability check (so it cannot be used as an unconditional pre-auth
/// DoS amplifier and so it brackets exactly the privileged window). That call
/// site lives kernel-side in `smallaios_kernel::state::check_capability`.
///
/// `extern "C"` + no-mangle-free `sym` reference so the naked entry can `call`
/// it with the SysV ABI; it preserves no syscall-arg registers itself because
/// the trampoline calls it before marshalling `SyscallArgs`... see the call
/// placement below.
extern "C" fn spec_exec_on_entry() {
    // SAFETY: ring-0 syscall entry context.
    unsafe {
        crate::security::spec_exec::legacy_ibrs_on_entry();
    }
}

/// Low-level syscall entry point.
///
/// This is the target of IA32_LSTAR. It:
/// 1. Saves caller-saved registers
/// 2. Re-asserts legacy IBRS (no-op on Enhanced-IBRS hardware)
/// 3. Builds a SyscallArgs struct
/// 4. Calls the kernel dispatch function
/// 5. Restores registers
/// 6. Returns via `sysretq`
///
/// Register state on entry:
/// - RAX = syscall number
/// - RDI = arg0, RSI = arg1, RDX = arg2, R10 = arg3, R8 = arg4, R9 = arg5
/// - RCX = saved RIP (for sysret)
/// - R11 = saved RFLAGS (for sysret)
#[unsafe(naked)]
extern "C" fn syscall_entry() {
    // SAFETY: This is a naked function implementing the syscall entry/exit path.
    // All register saves/restores are handled explicitly in assembly.
    naked_asm!(
        // Save callee-saved registers that the Rust ABI requires us to preserve
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        // Save RCX (return RIP) and R11 (return RFLAGS)
        "push rcx",
        "push r11",

        // Build SyscallArgs on the stack:
        // SyscallArgs { number: usize, args: [usize; 6] }
        // Push args[5] through args[0], then number
        "push r9",       // args[5]
        "push r8",       // args[4]
        "push r10",      // args[3]
        "push rdx",      // args[2]
        "push rsi",      // args[1]
        "push rdi",      // args[0]
        "push rax",      // number

        // spec-exec-mitigations-v1 Phase 2 (task 1.6): re-assert legacy IBRS
        // BEFORE the indirect call into `dispatch`. All syscall-argument
        // registers are now safely marshalled onto the stack, so this
        // `call` clobbering SysV caller-saved registers is harmless. The
        // shim is a cheap no-op on Enhanced-IBRS / no-IBRS hardware.
        "call {spec_exec_entry}",

        // Call dispatch with pointer to SyscallArgs on stack
        // RDI = pointer to SyscallArgs (first arg in SysV ABI)
        "mov rdi, rsp",
        "call {dispatch}",

        // Result is in RAX — leave it there for sysret

        // Clean up SyscallArgs from stack (7 * 8 = 56 bytes)
        "add rsp, 56",

        // Restore RCX and R11 for sysret
        "pop r11",
        "pop rcx",

        // Restore callee-saved registers
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",

        // Return to userspace
        // sysretq loads RIP from RCX and RFLAGS from R11
        "sysretq",

        dispatch = sym smallaios_kernel::syscall::dispatch,
        spec_exec_entry = sym spec_exec_on_entry,
    );
}

/// Get the address of the syscall entry point (for testing/diagnostics).
pub fn entry_address() -> u64 {
    syscall_entry as *const () as usize as u64
}
