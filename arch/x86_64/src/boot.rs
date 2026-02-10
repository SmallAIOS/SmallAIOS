// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 Multiboot2 boot entry point.
//!
//! This module provides the `_start` entry point that:
//! 1. Contains the Multiboot2 header
//! 2. Clears BSS
//! 3. Sets up the stack
//! 4. Calls kernel_main with the Multiboot2 info pointer

use core::arch::naked_asm;

// Multiboot2 constants
const MULTIBOOT2_MAGIC: u32 = 0xE85250D6;
const MULTIBOOT2_ARCH_X86: u32 = 0; // i386 (used for both 32 and 64-bit)
const MULTIBOOT2_HEADER_LENGTH: u32 = 24; // header + end tag (8+8+8)
const MULTIBOOT2_CHECKSUM: u32 = 0u32.wrapping_sub(
    MULTIBOOT2_MAGIC
        .wrapping_add(MULTIBOOT2_ARCH_X86)
        .wrapping_add(MULTIBOOT2_HEADER_LENGTH),
);

/// Multiboot2 header — must be in the first 32KiB of the binary.
/// Placed in the `.multiboot2` section which the linker script puts first.
#[link_section = ".multiboot2"]
#[used]
#[no_mangle]
static MULTIBOOT2_HEADER: [u32; 6] = [
    MULTIBOOT2_MAGIC,
    MULTIBOOT2_ARCH_X86,
    MULTIBOOT2_HEADER_LENGTH,
    MULTIBOOT2_CHECKSUM,
    // End tag: type=0, flags=0, size=8
    0, // type + flags
    8, // size
];

extern "C" {
    static __bss_start: u8;
    static __bss_end: u8;
    static __stack_top: u8;
}

/// Entry point: jumped to by the bootloader.
///
/// On entry (Multiboot2):
///   - EAX = 0x36D76289 (Multiboot2 magic)
///   - EBX = physical address of Multiboot2 info structure
///   - CPU is in 32-bit protected mode (if loaded by GRUB/Multiboot2)
///
/// Since we target QEMU `-kernel` which loads ELF directly in 64-bit mode,
/// we assume long mode is already active.
#[unsafe(naked)]
#[no_mangle]
#[link_section = ".text.boot"]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // Save Multiboot2 info pointer (RBX on x86-64 QEMU -kernel)
        "mov r12, rbx",

        // Clear BSS
        "lea rdi, [rip + __bss_start]",
        "lea rcx, [rip + __bss_end]",
        "sub rcx, rdi",
        "shr rcx, 3", // count in qwords
        "xor rax, rax",
        "rep stosq",

        // Set up stack
        "lea rsp, [rip + __stack_top]",

        // Align stack to 16 bytes (ABI requirement)
        "and rsp, -16",

        // Pass Multiboot2 info pointer as first argument
        "mov rdi, r12",

        // Call Rust kernel_main
        "call kernel_main",

        // Should never return, but just in case
        "2:",
        "hlt",
        "jmp 2b",
    )
}
