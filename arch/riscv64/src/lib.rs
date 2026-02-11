// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V RV64GC Hardware Abstraction Layer (23.1-23.10).
//!
//! Provides platform-specific initialization and hardware access:
//! - OpenSBI boot entry point (S-mode)
//! - NS16550A UART console (QEMU virt @ 0x10000000)
//! - SV48 page table management
//! - PLIC interrupt controller
//! - CLINT timer / IPI
//! - SBI firmware interface
//! - DTB parsing for hardware discovery
//! - ISA extension detection

#![no_std]
#![feature(naked_functions)]

pub mod boot;
pub mod clint;
pub mod dtb;
pub mod features;
pub mod paging;
pub mod plic;
pub mod sbi;
pub mod syscall;
pub mod trap;
pub mod uart;

use core::panic::PanicInfo;

/// Kernel entry point called from assembly boot code.
///
/// At this point we have:
/// - BSS zeroed
/// - Stack pointer set to __stack_top
/// - stvec set to trap_entry
/// - hart_id in a0, DTB pointer in a1
#[no_mangle]
pub extern "C" fn kernel_main(hart_id: u64, dtb_addr: u64) -> ! {
    // Initialize NS16550A UART for early diagnostics
    uart::init();

    uart::puts("[SmallAIOS] ");
    uart::puts(smallaios_kernel::NAME);
    uart::puts(" v");
    uart::puts(smallaios_kernel::VERSION);
    uart::puts(" booting on RISC-V RV64GC\n");

    uart::puts("[SmallAIOS] UART initialized (NS16550A @ 0x10000000)\n");
    uart::puts("[SmallAIOS] Hart ID: ");
    uart::put_dec(hart_id);
    uart::putc(b'\n');
    uart::puts("[SmallAIOS] DTB at 0x");
    uart::put_hex(dtb_addr);
    uart::putc(b'\n');

    // Parse DTB for hardware discovery
    uart::puts("[SmallAIOS] Parsing DTB...\n");
    let dtb_info = unsafe { dtb::parse_dtb(dtb_addr as usize) };
    if dtb_info.valid {
        uart::puts("[SmallAIOS] DTB valid, version ");
        uart::put_dec(dtb_info.dtb_version as u64);
        uart::puts(", size ");
        uart::put_dec(dtb_info.dtb_size as u64);
        uart::puts(" bytes\n");
        uart::puts("[SmallAIOS] CPUs: ");
        uart::put_dec(dtb_info.cpu_count as u64);
        uart::puts(", Memory regions: ");
        uart::put_dec(dtb_info.memory_count as u64);
        uart::puts(", Total: ");
        uart::put_dec(dtb_info.total_memory / 1024 / 1024);
        uart::puts(" MiB\n");
    } else {
        uart::puts("[SmallAIOS] WARNING: DTB invalid or not found\n");
    }

    // Query SBI implementation
    uart::puts("[SmallAIOS] SBI spec version: ");
    let spec = sbi::sbi_get_spec_version();
    if spec.is_success() {
        let major = (spec.value >> 24) & 0x7F;
        let minor = spec.value & 0xFF_FFFF;
        uart::put_dec(major as u64);
        uart::putc(b'.');
        uart::put_dec(minor as u64);
    } else {
        uart::puts("unknown");
    }
    uart::putc(b'\n');

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
