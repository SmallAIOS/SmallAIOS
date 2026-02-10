// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 Hardware Abstraction Layer
//!
//! Provides platform-specific initialization and hardware access:
//! - Multiboot2 boot entry point
//! - GDT setup (flat 64-bit segments)
//! - Serial console (COM1 at 0x3F8)
//! - BSS clearing and stack setup

#![no_std]

pub mod boot;
pub mod gdt;
pub mod interrupts;
pub mod paging;
pub mod serial;

use core::panic::PanicInfo;

/// Kernel entry point called from assembly boot code.
///
/// At this point we have:
/// - A valid 64-bit GDT loaded
/// - BSS zeroed
/// - Stack pointer set to __stack_top
/// - Multiboot2 info pointer in `mb_info_addr`
#[no_mangle]
pub extern "C" fn kernel_main(mb_info_addr: u64) -> ! {
    // Initialize serial console first for early diagnostics
    serial::init();

    serial::puts("[SmallAIOS] ");
    serial::puts(smallaios_kernel::NAME);
    serial::puts(" v");
    serial::puts(smallaios_kernel::VERSION);
    serial::puts(" booting on x86-64\n");

    serial::puts("[SmallAIOS] Serial console initialized (COM1 @ 0x3F8)\n");
    serial::puts("[SmallAIOS] Multiboot2 info at 0x");
    serial::put_hex(mb_info_addr);
    serial::putc(b'\n');

    serial::puts("[SmallAIOS] BSS cleared, stack initialized\n");

    // Parse physical memory map from Multiboot2 info
    serial::puts("[SmallAIOS] Parsing Multiboot2 memory map...\n");
    let mut phys_map = smallaios_kernel::mem::phys::PhysMemoryMap::new();
    unsafe {
        smallaios_kernel::mem::phys::parse_multiboot2(mb_info_addr as usize, &mut phys_map);
    }
    serial::puts("[SmallAIOS] Memory regions: ");
    serial::put_dec(phys_map.count() as u64);
    serial::puts(", usable: ");
    serial::put_dec((phys_map.total_usable() / 1024 / 1024) as u64);
    serial::puts(" MiB\n");

    serial::puts("[SmallAIOS] Boot complete. Halting.\n");

    halt_loop();
}

/// Halt the CPU in a loop, waiting for interrupts.
pub fn halt_loop() -> ! {
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial::puts("[SmallAIOS] PANIC: ");
    if let Some(location) = info.location() {
        serial::puts(location.file());
        serial::puts(":");
        serial::put_dec(location.line() as u64);
    }
    serial::putc(b'\n');
    halt_loop();
}
