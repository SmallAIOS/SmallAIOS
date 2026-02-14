// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AArch64 boot entry point.
//!
//! This module provides the `_start` entry point that:
//! 1. Parks secondary cores (only core 0 proceeds)
//! 2. Detects current EL and drops from EL2 to EL1 if needed
//! 3. Clears BSS
//! 4. Sets up the stack
//! 5. Calls kernel_main with the DTB pointer (x0 from firmware)
//!
//! On Tegra X1, U-Boot/ATF may hand off at EL2, so the EL2→EL1
//! transition is necessary. On QEMU virt, firmware typically starts at EL1.

use core::arch::naked_asm;

extern "C" {
    static __bss_start: u8;
    static __bss_end: u8;
    static __stack_top: u8;
}

/// HCR_EL2: RW bit (bit 31) — EL1 executes in AArch64 mode.
pub const HCR_EL2_RW: u64 = 1 << 31;

/// SPSR_EL2 for returning to EL1h with DAIF masked.
/// EL1h (M[3:0] = 0b0101 = 5) + DAIF masked (bits 9:6 = 0b1111).
/// 0b1111_0000_0101 = 0x3C5
pub const SPSR_EL2_EL1H_DAIF: u64 = 0x3C5;

/// Entry point: jumped to by firmware/U-Boot/QEMU.
///
/// On entry:
///   - x0 = physical address of DTB (device tree blob)
///   - CPU may be in EL1 or EL2, depending on firmware
///   - MMU is off, caches may be off
///
/// # Safety
/// Must be called exactly once by the bootloader as the kernel entry point.
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
        "cbnz x1, 5f",
        // ── EL2 → EL1 transition (if needed) ────────────────────────
        // Read CurrentEL: bits [3:2] give the EL
        "mrs x0, CurrentEL",
        "lsr x0, x0, #2",
        "cmp x0, #2",
        "b.ne 1f",
        // We are at EL2 — configure and drop to EL1
        // HCR_EL2: set RW bit so EL1 executes in AArch64 mode
        "mov x0, #(1 << 31)",
        "msr hcr_el2, x0",
        // SCTLR_EL1: reset value (MMU/caches disabled)
        "mov x0, xzr",
        "msr sctlr_el1, x0",
        // SPSR_EL2: return to EL1h with DAIF masked
        "mov x0, #0x3C5",
        "msr spsr_el2, x0",
        // ELR_EL2: return address = label 1 (EL1 entry)
        "adr x0, 1f",
        "msr elr_el2, x0",
        // Perform the EL2 → EL1 transition
        "eret",
        // ── EL1 entry point ──────────────────────────────────────────
        "1:",
        // Clear BSS
        "adrp x0, __bss_start",
        "add x0, x0, :lo12:__bss_start",
        "adrp x1, __bss_end",
        "add x1, x1, :lo12:__bss_end",
        "2:",
        "cmp x0, x1",
        "b.ge 3f",
        "str xzr, [x0], #8",
        "b 2b",
        "3:",
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
        "5:",
        "wfi",
        "b 5b",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcr_el2_rw_bit() {
        // RW bit is bit 31 of HCR_EL2
        assert_eq!(HCR_EL2_RW, 1 << 31);
        assert_eq!(HCR_EL2_RW, 0x80000000);
    }

    #[test]
    fn test_spsr_el2_value() {
        // EL1h = M[3:0] = 5, DAIF masked = bits [9:6] all set
        assert_eq!(SPSR_EL2_EL1H_DAIF, 0x3C5);
        // M field (bits 3:0): 0b0101 = 5 (EL1h)
        assert_eq!(SPSR_EL2_EL1H_DAIF & 0xF, 5);
        // DAIF mask (bits 9:6): all set = 0b1111 << 6 = 0x3C0
        assert_eq!(SPSR_EL2_EL1H_DAIF & 0x3C0, 0x3C0);
    }
}
