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
