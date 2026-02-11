// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V trap/exception handler (23.7).
//!
//! All traps in S-mode (interrupts, exceptions, ecalls) enter via `trap_entry`.
//! The handler saves all registers, reads `scause` to classify the trap, and
//! dispatches to the appropriate handler:
//!
//! - Timer interrupt: acknowledge via SBI and reload
//! - External interrupt: PLIC claim/complete
//! - Ecall from U-mode: syscall dispatch
//! - Page faults: report (stub)
//! - Other exceptions: report (stub)

use core::arch::naked_asm;

/// Trap frame layout saved on the kernel stack.
///
/// 32 general-purpose registers (x0-x31) + sstatus + sepc + stval + scause = 36 slots.
/// x0 is hardwired zero but we save a slot for uniform indexing.
#[repr(C)]
pub struct TrapFrame {
    pub regs: [u64; 32], // x0-x31
    pub sstatus: u64,
    pub sepc: u64,
    pub stval: u64,
    pub scause: u64,
}

/// scause values for interrupts (bit 63 set).
pub const SCAUSE_INTERRUPT: u64 = 1 << 63;
pub const IRQ_S_SOFT: u64 = SCAUSE_INTERRUPT | 1;
pub const IRQ_S_TIMER: u64 = SCAUSE_INTERRUPT | 5;
pub const IRQ_S_EXTERNAL: u64 = SCAUSE_INTERRUPT | 9;

/// scause values for exceptions (bit 63 clear).
pub const EXC_INST_MISALIGNED: u64 = 0;
pub const EXC_INST_ACCESS_FAULT: u64 = 1;
pub const EXC_ILLEGAL_INST: u64 = 2;
pub const EXC_BREAKPOINT: u64 = 3;
pub const EXC_LOAD_ACCESS_FAULT: u64 = 5;
pub const EXC_STORE_MISALIGNED: u64 = 6;
pub const EXC_STORE_ACCESS_FAULT: u64 = 7;
pub const EXC_ECALL_FROM_U: u64 = 8;
pub const EXC_ECALL_FROM_S: u64 = 9;
pub const EXC_INST_PAGE_FAULT: u64 = 12;
pub const EXC_LOAD_PAGE_FAULT: u64 = 13;
pub const EXC_STORE_PAGE_FAULT: u64 = 15;

/// Naked trap entry point written to stvec.
///
/// Saves all 31 GP registers (x1-x31), plus CSRs (sstatus, sepc, stval, scause).
/// x0 is always zero so we skip writing it but reserve the slot.
///
/// Stack frame: 36 * 8 = 288 bytes.
///
/// # Safety
/// Must only be called as the stvec trap vector.
#[unsafe(naked)]
#[no_mangle]
#[link_section = ".text.trap"]
pub unsafe extern "C" fn trap_entry() {
    naked_asm!(
        // Allocate trap frame (36 * 8 = 288 bytes)
        "addi sp, sp, -288",

        // Save x1-x31 (skip x0 = zero)
        "sd x1,   1*8(sp)",
        "sd x2,   2*8(sp)",
        "sd x3,   3*8(sp)",
        "sd x4,   4*8(sp)",
        "sd x5,   5*8(sp)",
        "sd x6,   6*8(sp)",
        "sd x7,   7*8(sp)",
        "sd x8,   8*8(sp)",
        "sd x9,   9*8(sp)",
        "sd x10, 10*8(sp)",
        "sd x11, 11*8(sp)",
        "sd x12, 12*8(sp)",
        "sd x13, 13*8(sp)",
        "sd x14, 14*8(sp)",
        "sd x15, 15*8(sp)",
        "sd x16, 16*8(sp)",
        "sd x17, 17*8(sp)",
        "sd x18, 18*8(sp)",
        "sd x19, 19*8(sp)",
        "sd x20, 20*8(sp)",
        "sd x21, 21*8(sp)",
        "sd x22, 22*8(sp)",
        "sd x23, 23*8(sp)",
        "sd x24, 24*8(sp)",
        "sd x25, 25*8(sp)",
        "sd x26, 26*8(sp)",
        "sd x27, 27*8(sp)",
        "sd x28, 28*8(sp)",
        "sd x29, 29*8(sp)",
        "sd x30, 30*8(sp)",
        "sd x31, 31*8(sp)",

        // Save CSRs
        "csrr t0, sstatus",
        "sd t0, 32*8(sp)",
        "csrr t0, sepc",
        "sd t0, 33*8(sp)",
        "csrr t0, stval",
        "sd t0, 34*8(sp)",
        "csrr t0, scause",
        "sd t0, 35*8(sp)",

        // Call trap_handler(frame: *mut TrapFrame)
        "mv a0, sp",
        "call {trap_handler}",

        // Restore CSRs
        "ld t0, 32*8(sp)",
        "csrw sstatus, t0",
        "ld t0, 33*8(sp)",
        "csrw sepc, t0",

        // Restore x1-x31
        "ld x1,   1*8(sp)",
        "ld x2,   2*8(sp)",
        "ld x3,   3*8(sp)",
        "ld x4,   4*8(sp)",
        "ld x5,   5*8(sp)",
        "ld x6,   6*8(sp)",
        "ld x7,   7*8(sp)",
        "ld x8,   8*8(sp)",
        "ld x9,   9*8(sp)",
        "ld x10, 10*8(sp)",
        "ld x11, 11*8(sp)",
        "ld x12, 12*8(sp)",
        "ld x13, 13*8(sp)",
        "ld x14, 14*8(sp)",
        "ld x15, 15*8(sp)",
        "ld x16, 16*8(sp)",
        "ld x17, 17*8(sp)",
        "ld x18, 18*8(sp)",
        "ld x19, 19*8(sp)",
        "ld x20, 20*8(sp)",
        "ld x21, 21*8(sp)",
        "ld x22, 22*8(sp)",
        "ld x23, 23*8(sp)",
        "ld x24, 24*8(sp)",
        "ld x25, 25*8(sp)",
        "ld x26, 26*8(sp)",
        "ld x27, 27*8(sp)",
        "ld x28, 28*8(sp)",
        "ld x29, 29*8(sp)",
        "ld x30, 30*8(sp)",
        "ld x31, 31*8(sp)",

        // Deallocate frame and return
        "addi sp, sp, 288",
        "sret",

        trap_handler = sym trap_handler,
    )
}

/// High-level trap handler dispatching based on scause.
#[no_mangle]
pub unsafe extern "C" fn trap_handler(frame: *mut TrapFrame) {
    let frame = unsafe { &mut *frame };
    let scause = frame.scause;

    if scause & SCAUSE_INTERRUPT != 0 {
        // Interrupt
        match scause {
            IRQ_S_TIMER => handle_timer_irq(),
            IRQ_S_EXTERNAL => handle_external_irq(),
            IRQ_S_SOFT => handle_software_irq(),
            _ => {} // Unknown interrupt, ignore
        }
    } else {
        // Exception
        match scause {
            EXC_ECALL_FROM_U => {
                // Syscall: dispatch and advance sepc past the ecall instruction
                super::syscall::handle_ecall(frame);
                frame.sepc += 4; // ecall is 4 bytes (not compressed)
            }
            EXC_INST_PAGE_FAULT | EXC_LOAD_PAGE_FAULT | EXC_STORE_PAGE_FAULT => {
                handle_page_fault(scause, frame.stval, frame.sepc);
            }
            EXC_ILLEGAL_INST => {
                handle_illegal_instruction(frame.sepc, frame.stval);
            }
            _ => {
                handle_unknown_exception(scause, frame.sepc, frame.stval);
            }
        }
    }
}

/// Handle S-mode timer interrupt.
fn handle_timer_irq() {
    // Acknowledge timer by scheduling next tick via SBI.
    super::sbi::sbi_set_timer(u64::MAX);
}

/// Handle S-mode external interrupt (from PLIC).
fn handle_external_irq() {
    // In a full implementation, we would claim from PLIC,
    // dispatch to the device driver, then complete.
    // Stub: just return.
}

/// Handle S-mode software interrupt (IPI).
fn handle_software_irq() {
    // Clear SIP.SSIP via CSR to acknowledge.
    unsafe {
        core::arch::asm!(
            "csrc sip, {bit}",
            bit = in(reg) 1u64 << 1,
            options(nostack),
        );
    }
}

/// Handle page fault exceptions.
fn handle_page_fault(_cause: u64, _fault_addr: u64, _epc: u64) {
    // Stub: In a full implementation this would invoke the VM subsystem.
    // For now, halt.
}

/// Handle illegal instruction exception.
fn handle_illegal_instruction(_epc: u64, _instruction: u64) {
    // Stub: halt.
}

/// Handle unknown exception.
fn handle_unknown_exception(_cause: u64, _epc: u64, _tval: u64) {
    // Stub: halt.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scause_interrupt_bit() {
        assert_eq!(SCAUSE_INTERRUPT, 1 << 63);
        assert_ne!(IRQ_S_TIMER & SCAUSE_INTERRUPT, 0);
        assert_ne!(IRQ_S_EXTERNAL & SCAUSE_INTERRUPT, 0);
        assert_ne!(IRQ_S_SOFT & SCAUSE_INTERRUPT, 0);
    }

    #[test]
    fn test_scause_exception_codes() {
        assert_eq!(EXC_ECALL_FROM_U, 8);
        assert_eq!(EXC_ECALL_FROM_S, 9);
        assert_eq!(EXC_INST_PAGE_FAULT, 12);
        assert_eq!(EXC_LOAD_PAGE_FAULT, 13);
        assert_eq!(EXC_STORE_PAGE_FAULT, 15);
    }

    #[test]
    fn test_irq_codes() {
        assert_eq!(IRQ_S_SOFT, SCAUSE_INTERRUPT | 1);
        assert_eq!(IRQ_S_TIMER, SCAUSE_INTERRUPT | 5);
        assert_eq!(IRQ_S_EXTERNAL, SCAUSE_INTERRUPT | 9);
    }

    #[test]
    fn test_exception_vs_interrupt() {
        assert_eq!(EXC_ILLEGAL_INST & SCAUSE_INTERRUPT, 0);
        assert_eq!(EXC_BREAKPOINT & SCAUSE_INTERRUPT, 0);
        assert_ne!(IRQ_S_TIMER & SCAUSE_INTERRUPT, 0);
    }

    #[test]
    fn test_trap_frame_size() {
        // 36 u64 slots = 288 bytes
        assert_eq!(core::mem::size_of::<TrapFrame>(), 288);
    }
}
