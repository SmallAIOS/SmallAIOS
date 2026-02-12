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

/// Low-level syscall entry point.
///
/// This is the target of IA32_LSTAR. It:
/// 1. Saves caller-saved registers
/// 2. Builds a SyscallArgs struct
/// 3. Calls the kernel dispatch function
/// 4. Restores registers
/// 5. Returns via `sysretq`
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
    );
}

/// Get the address of the syscall entry point (for testing/diagnostics).
pub fn entry_address() -> u64 {
    syscall_entry as *const () as usize as u64
}
