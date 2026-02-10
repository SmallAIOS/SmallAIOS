// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 SVC (Supervisor Call) handler for system calls.
//!
//! In VM mode, SmallAIOS uses the `svc` instruction for system calls.
//! The `svc #0` instruction causes a Synchronous exception at EL1.
//! The exception vector dispatches to the SVC handler.
//!
//! Calling convention:
//!   - Syscall number in X8
//!   - Arguments: X0-X5
//!   - Return value in X0
//!   - X0-X18 are caller-saved (corruptible)
//!   - X19-X29, SP, LR (X30) are callee-saved

/// SVC handler called from the exception vector table.
///
/// This function is called by the synchronous exception handler
/// when the ESR_EL1 indicates an SVC instruction (EC = 0x15).
///
/// # Arguments
/// * `regs` - Pointer to the saved register frame on the exception stack.
///
/// # Register frame layout (pushed by exception vector stub):
/// ```text
/// [sp + 0x00] X0     (arg0 / return value)
/// [sp + 0x08] X1     (arg1)
/// ...
/// [sp + 0x28] X5     (arg5)
/// [sp + 0x30] X6
/// ...
/// [sp + 0x40] X8     (syscall number)
/// ...
/// [sp + 0xF0] X30/LR
/// [sp + 0xF8] SP_EL0
/// [sp + 0x100] ELR_EL1 (return address)
/// [sp + 0x108] SPSR_EL1 (saved PSTATE)
/// ```
///
/// # Safety
/// Must only be called from the exception vector with a valid register frame.
#[no_mangle]
pub unsafe extern "C" fn svc_handler(regs: *mut u64) {
    // SAFETY: regs points to the exception frame saved by the vector stub.
    // The vector guarantees this pointer is valid and properly aligned.
    let regs = unsafe { core::slice::from_raw_parts_mut(regs, 34) };

    // Extract syscall number from X8 (index 8 in register frame)
    let syscall_number = regs[8] as usize;

    // Build SyscallArgs from saved registers
    let args = smallaios_kernel::syscall::SyscallArgs::new(
        syscall_number,
        [
            regs[0] as usize, // X0 = arg0
            regs[1] as usize, // X1 = arg1
            regs[2] as usize, // X2 = arg2
            regs[3] as usize, // X3 = arg3
            regs[4] as usize, // X4 = arg4
            regs[5] as usize, // X5 = arg5
        ],
    );

    // Dispatch to kernel syscall handler
    let result = smallaios_kernel::syscall::dispatch(&args);

    // Write return value back to X0 in the saved register frame
    regs[0] = result as u64;

    // Exception return (eret) will restore registers from the frame
}

/// ARM64 exception vector entry for SVC calls.
///
/// This is the synchronous exception handler stub that:
/// 1. Saves all general-purpose registers to the stack
/// 2. Checks ESR_EL1 for SVC (EC = 0x15)
/// 3. Calls svc_handler with pointer to saved registers
/// 4. Restores registers and returns via ERET
///
/// # Safety
/// This is a naked function that must be placed in the exception vector table.
#[unsafe(naked)]
pub extern "C" fn sync_exception_entry() {
    // SAFETY: Naked function implementing the exception entry/exit path.
    // All register saves/restores handled explicitly in assembly.
    unsafe {
        naked_asm!(
            // Save all general-purpose registers (X0-X30) + SP_EL0 + ELR_EL1 + SPSR_EL1
            // Sub stack for 34 * 8 = 272 bytes
            "sub sp, sp, #272",

            // Save X0-X29 (pairs)
            "stp x0, x1, [sp, #0x00]",
            "stp x2, x3, [sp, #0x10]",
            "stp x4, x5, [sp, #0x20]",
            "stp x6, x7, [sp, #0x30]",
            "stp x8, x9, [sp, #0x40]",
            "stp x10, x11, [sp, #0x50]",
            "stp x12, x13, [sp, #0x60]",
            "stp x14, x15, [sp, #0x70]",
            "stp x16, x17, [sp, #0x80]",
            "stp x18, x19, [sp, #0x90]",
            "stp x20, x21, [sp, #0xA0]",
            "stp x22, x23, [sp, #0xB0]",
            "stp x24, x25, [sp, #0xC0]",
            "stp x26, x27, [sp, #0xD0]",
            "stp x28, x29, [sp, #0xE0]",

            // Save X30 (LR) and SP_EL0
            "mrs x10, sp_el0",
            "stp x30, x10, [sp, #0xF0]",

            // Save ELR_EL1 and SPSR_EL1
            "mrs x10, elr_el1",
            "mrs x11, spsr_el1",
            "stp x10, x11, [sp, #0x100]",

            // Check ESR_EL1 for SVC exception class
            "mrs x10, esr_el1",
            "lsr x10, x10, #26",    // EC field is bits [31:26]
            "cmp x10, #0x15",       // EC 0x15 = SVC from AArch64
            "b.ne 1f",              // Not SVC — skip to generic handler

            // Call svc_handler(regs: *mut u64)
            "mov x0, sp",
            "bl {svc_handler}",
            "b 2f",

            // Generic exception handler (non-SVC) — halt for now
            "1:",
            "b 1b",

            // Restore and return
            "2:",
            // Restore ELR_EL1 and SPSR_EL1
            "ldp x10, x11, [sp, #0x100]",
            "msr elr_el1, x10",
            "msr spsr_el1, x11",

            // Restore X30 (LR) and SP_EL0
            "ldp x30, x10, [sp, #0xF0]",
            "msr sp_el0, x10",

            // Restore X0-X29
            "ldp x28, x29, [sp, #0xE0]",
            "ldp x26, x27, [sp, #0xD0]",
            "ldp x24, x25, [sp, #0xC0]",
            "ldp x22, x23, [sp, #0xB0]",
            "ldp x20, x21, [sp, #0xA0]",
            "ldp x18, x19, [sp, #0x90]",
            "ldp x16, x17, [sp, #0x80]",
            "ldp x14, x15, [sp, #0x70]",
            "ldp x12, x13, [sp, #0x60]",
            "ldp x10, x11, [sp, #0x50]",
            "ldp x8, x9, [sp, #0x40]",
            "ldp x6, x7, [sp, #0x30]",
            "ldp x4, x5, [sp, #0x20]",
            "ldp x2, x3, [sp, #0x10]",
            "ldp x0, x1, [sp, #0x00]",

            // Restore stack pointer
            "add sp, sp, #272",

            // Return from exception
            "eret",

            svc_handler = sym svc_handler,
        );
    }
}
