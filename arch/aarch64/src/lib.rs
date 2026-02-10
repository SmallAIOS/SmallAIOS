// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 (AArch64) Hardware Abstraction Layer
//!
//! Provides platform-specific initialization and hardware access:
//! - DTB/FDT boot entry point
//! - PL011 UART console (QEMU virt @ 0x0900_0000)
//! - BSS clearing and stack setup

#![no_std]

pub mod boot;
pub mod interrupts;
pub mod paging;
pub mod syscall;
pub mod uart;

use core::panic::PanicInfo;

/// Kernel entry point called from assembly boot code.
///
/// At this point we have:
/// - BSS zeroed
/// - Stack pointer set to __stack_top
/// - DTB pointer in `dtb_addr` (x0 from firmware/QEMU)
#[no_mangle]
pub extern "C" fn kernel_main(dtb_addr: u64) -> ! {
    // Initialize PL011 UART for early diagnostics
    uart::init();

    uart::puts("[SmallAIOS] ");
    uart::puts(smallaios_kernel::NAME);
    uart::puts(" v");
    uart::puts(smallaios_kernel::VERSION);
    uart::puts(" booting on AArch64\n");

    uart::puts("[SmallAIOS] UART initialized (PL011 @ 0x09000000)\n");
    uart::puts("[SmallAIOS] DTB at 0x");
    uart::put_hex(dtb_addr);
    uart::putc(b'\n');

    uart::puts("[SmallAIOS] BSS cleared, stack initialized\n");

    // Parse physical memory map from DTB
    uart::puts("[SmallAIOS] Parsing DTB memory map...\n");
    let mut phys_map = smallaios_kernel::mem::phys::PhysMemoryMap::new();
    unsafe {
        smallaios_kernel::mem::phys::parse_dtb(dtb_addr as usize, &mut phys_map);
    }
    uart::puts("[SmallAIOS] Memory regions: ");
    uart::put_dec(phys_map.count() as u64);
    uart::puts(", usable: ");
    uart::put_dec((phys_map.total_usable() / 1024 / 1024) as u64);
    uart::puts(" MiB\n");

    uart::puts("[SmallAIOS] Boot complete. Halting.\n");

    halt_loop();
}

/// Halt the CPU using WFI (Wait For Interrupt).
pub fn halt_loop() -> ! {
    loop {
        unsafe {
            core::arch::asm!("wfi");
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    uart::puts("[SmallAIOS] PANIC: ");
    if let Some(location) = info.location() {
        uart::puts(location.file());
        uart::puts(":");
        uart::put_dec(location.line() as u64);
    }
    uart::putc(b'\n');
    halt_loop();
}
