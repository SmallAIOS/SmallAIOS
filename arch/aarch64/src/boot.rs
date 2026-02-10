// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AArch64 boot entry point.
//!
//! This module provides the `_start` entry point that:
//! 1. Parks secondary cores (only core 0 proceeds)
//! 2. Clears BSS
//! 3. Sets up the stack
//! 4. Calls kernel_main with the DTB pointer (x0 from firmware)

use core::arch::naked_asm;

extern "C" {
    static __bss_start: u8;
    static __bss_end: u8;
    static __stack_top: u8;
}

/// Entry point: jumped to by firmware/QEMU.
///
/// On entry (QEMU virt):
///   - x0 = physical address of DTB (device tree blob)
///   - CPU is in EL1 (or EL2, depending on firmware)
///   - MMU is off, caches may be off
#[unsafe(naked)]
#[no_mangle]
#[link_section = ".text.boot"]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // Save DTB pointer (x0) — we'll pass it to kernel_main
        "mov x19, x0",

        // Park secondary cores: only core 0 (MPIDR_EL1 Aff0 == 0) proceeds
        "mrs x1, mpidr_el1",
        "and x1, x1, #0xFF",
        "cbnz x1, 3f",

        // Clear BSS
        "adrp x0, __bss_start",
        "add x0, x0, :lo12:__bss_start",
        "adrp x1, __bss_end",
        "add x1, x1, :lo12:__bss_end",
        "1:",
        "cmp x0, x1",
        "b.ge 2f",
        "str xzr, [x0], #8",
        "b 1b",
        "2:",

        // Set up stack (aligned to 16 bytes)
        "adrp x0, __stack_top",
        "add x0, x0, :lo12:__stack_top",
        "and x0, x0, #-16",
        "mov sp, x0",

        // Pass DTB pointer as first argument
        "mov x0, x19",

        // Call Rust kernel_main
        "bl kernel_main",

        // Secondary core parking loop
        "3:",
        "wfi",
        "b 3b",
    )
}
