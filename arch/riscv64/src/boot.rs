// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V boot entry point (23.1).
//!
//! This module provides the `_start` entry point that:
//! 1. Sets up the boot stack (64 KiB, 16-byte aligned)
//! 2. Clears BSS
//! 3. Writes the trap handler base address to stvec
//! 4. Calls kernel_main with hart ID (a0) and DTB pointer (a1)
//! 5. Includes a terminal spin loop as safety net

use core::arch::naked_asm;

extern "C" {
    static __bss_start: u8;
    static __bss_end: u8;
    static __stack_top: u8;
}

/// Entry point: jumped to by OpenSBI after M-mode initialization.
///
/// On entry:
///   - a0 = hart ID
///   - a1 = physical address of DTB (device tree blob)
///   - CPU is in S-mode
///   - MMU is off
#[unsafe(naked)]
#[no_mangle]
#[link_section = ".text.boot"]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // Save hart ID and DTB pointer
        "mv s0, a0",       // s0 = hart ID
        "mv s1, a1",       // s1 = DTB pointer

        // Park secondary harts: only hart 0 proceeds
        "bnez s0, 4f",

        // Set up stack pointer (16-byte aligned)
        "la sp, __stack_top",
        "andi sp, sp, -16",

        // Clear BSS section
        "la t0, __bss_start",
        "la t1, __bss_end",
        "1:",
        "bge t0, t1, 2f",
        "sd zero, 0(t0)",
        "addi t0, t0, 8",
        "j 1b",
        "2:",

        // Set up trap vector (stvec)
        "la t0, {trap_entry}",
        "csrw stvec, t0",

        // Call kernel_main(hart_id, dtb_addr)
        "mv a0, s0",
        "mv a1, s1",
        "call {kernel_main}",

        // Safety net: kernel_main must not return
        "3:",
        "wfi",
        "j 3b",

        // Secondary hart parking loop
        "4:",
        "wfi",
        "j 4b",

        trap_entry = sym super::trap::trap_entry,
        kernel_main = sym super::kernel_main,
    )
}
