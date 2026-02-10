// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Global Descriptor Table (GDT) for x86-64.
//!
//! In long mode, segmentation is mostly disabled. We only need:
//! - Null descriptor (index 0)
//! - Kernel code segment (index 1): 64-bit, ring 0
//! - Kernel data segment (index 2): ring 0

use core::arch::asm;

/// GDT entry: 8 bytes each.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct GdtEntry {
    limit_low: u16,
    base_low: u16,
    base_mid: u8,
    access: u8,
    flags_limit_hi: u8,
    base_hi: u8,
}

impl GdtEntry {
    const fn null() -> Self {
        Self {
            limit_low: 0,
            base_low: 0,
            base_mid: 0,
            access: 0,
            flags_limit_hi: 0,
            base_hi: 0,
        }
    }

    /// Kernel 64-bit code segment: present, ring 0, code, readable, long mode.
    const fn kernel_code() -> Self {
        Self {
            limit_low: 0xFFFF,
            base_low: 0,
            base_mid: 0,
            access: 0x9A, // present | ring0 | code | readable
            flags_limit_hi: 0xAF, // long mode | limit[19:16]=0xF
            base_hi: 0,
        }
    }

    /// Kernel 64-bit data segment: present, ring 0, data, writable.
    const fn kernel_data() -> Self {
        Self {
            limit_low: 0xFFFF,
            base_low: 0,
            base_mid: 0,
            access: 0x92, // present | ring0 | data | writable
            flags_limit_hi: 0xCF, // 4KiB granularity | 32-bit | limit[19:16]=0xF
            base_hi: 0,
        }
    }
}

/// The GDT: null + kernel code + kernel data.
#[repr(C, align(16))]
struct Gdt {
    entries: [GdtEntry; 3],
}

static GDT: Gdt = Gdt {
    entries: [
        GdtEntry::null(),
        GdtEntry::kernel_code(),
        GdtEntry::kernel_data(),
    ],
};

/// GDT pointer structure for LGDT instruction.
#[repr(C, packed)]
struct GdtPtr {
    limit: u16,
    base: u64,
}

/// Load the GDT and reload segment registers.
///
/// # Safety
/// Must be called once during boot, before enabling interrupts.
pub unsafe fn init() {
    let gdt_ptr = GdtPtr {
        limit: (core::mem::size_of::<Gdt>() - 1) as u16,
        base: &GDT as *const Gdt as u64,
    };

    asm!(
        "lgdt [{gdt_ptr}]",
        // Reload CS via a far return
        "push 0x08",           // kernel code segment selector
        "lea {tmp}, [rip + 2f]",
        "push {tmp}",
        "retfq",
        "2:",
        // Reload data segment registers
        "mov ax, 0x10",        // kernel data segment selector
        "mov ds, ax",
        "mov es, ax",
        "mov fs, ax",
        "mov gs, ax",
        "mov ss, ax",
        gdt_ptr = in(reg) &gdt_ptr,
        tmp = lateout(reg) _,
        options(preserves_flags),
    );
}

/// Kernel code segment selector (offset 0x08 in GDT).
pub const KERNEL_CS: u16 = 0x08;

/// Kernel data segment selector (offset 0x10 in GDT).
pub const KERNEL_DS: u16 = 0x10;
