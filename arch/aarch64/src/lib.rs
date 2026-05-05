// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 (AArch64) Hardware Abstraction Layer
//!
//! Provides platform-specific initialization and hardware access:
//! - DTB/FDT boot entry point
//! - UART console (PL011 on QEMU virt, NS16550A on Tegra X1)
//! - BSS clearing and stack setup
//!
//! Platform selection via Cargo features:
//! - `qemu-virt` (default): QEMU virt machine
//! - `tegra-x1`: NVIDIA Jetson Nano

#![no_std]

// Bare-metal entry. References linker-defined `__bss_start` / `__bss_end` /
// `__stack_top` symbols that come from `arch/aarch64/linker*.ld`, so it only
// makes sense for `target_os = "none"` builds. The UEFI bin (`smallaios-uefi`,
// target `aarch64-unknown-uefi`) enters via `boot_uefi::efi_main` instead.
#[cfg(target_os = "none")]
pub mod boot;
pub mod console;
#[cfg(feature = "tegra-x1")]
pub mod fb_console;
#[cfg(feature = "tegra-x1")]
pub mod gicv2;
pub mod interrupts;
pub mod paging;
pub mod platform;
pub mod syscall;
pub mod uart;

#[cfg(feature = "tegra-x1")]
pub mod image_header;
#[cfg(feature = "tegra-x1")]
pub mod onnx_demo;
#[cfg(feature = "tegra-x1")]
pub mod tegra_dc;
#[cfg(feature = "tegra-x1")]
pub mod tegra_edid;
#[cfg(feature = "tegra-x1")]
pub mod tegra_pcie;
#[cfg(feature = "tegra-x1")]
pub mod tegra_sor;

// UEFI boot path for Tegra234. The types module compiles for any target as
// long as `tegra234` is on (lets host tests reach the GUID parsing code);
// `boot_uefi::efi_main` uses `extern "efiapi"` which is portable.
#[cfg(feature = "tegra234")]
pub mod boot_uefi;
#[cfg(feature = "tegra234")]
pub mod tegra234_uart;
#[cfg(feature = "tegra234")]
pub mod uefi;

/// Kernel entry point called from assembly boot code.
///
/// At this point we have:
/// - BSS zeroed
/// - Stack pointer set to __stack_top
/// - DTB pointer in `dtb_addr` (x0 from firmware/QEMU)
#[no_mangle]
pub extern "C" fn kernel_main(dtb_addr: u64) -> ! {
    // Initialize UART (no-op on Tegra where U-Boot pre-inits)
    uart::init();

    uart::puts("\n");
    uart::puts("========================================\n");
    uart::puts("  SmallAIOS ");
    uart::puts(smallaios_kernel::VERSION);
    uart::puts("\n");
    #[cfg(feature = "qemu-virt")]
    uart::puts("  Platform: AArch64 (QEMU virt)\n");
    #[cfg(feature = "tegra-x1")]
    uart::puts("  Platform: Tegra X1 (Jetson Nano)\n");
    #[cfg(feature = "tegra234")]
    uart::puts("  Platform: Tegra234 (Jetson Orin)\n");
    uart::puts("========================================\n");
    uart::puts("\n");

    // ── Stage 1: Early init ──────────────────────────────────────────
    uart::puts("[boot] Stage 1: Early initialization\n");

    #[cfg(feature = "qemu-virt")]
    uart::puts("[uart] PL011 @ 0x09000000 initialized\n");
    #[cfg(feature = "tegra-x1")]
    uart::puts("[uart] NS16550A @ 0x70006000 (pre-init by U-Boot)\n");

    uart::puts("[boot] BSS cleared, stack initialized\n");

    // Read and display current exception level
    let current_el: u64;
    unsafe {
        core::arch::asm!("mrs {}, CurrentEL", out(reg) current_el);
    }
    uart::puts("[boot] Running at EL");
    uart::put_dec((current_el >> 2) & 0x3);
    uart::puts("\n");

    uart::puts("[boot] DTB address: 0x");
    uart::put_hex(dtb_addr);
    uart::puts("\n");

    // ── Stage 2: Memory ──────────────────────────────────────────────
    uart::puts("\n[boot] Stage 2: Memory detection\n");

    let mut phys_map = smallaios_kernel::mem::phys::PhysMemoryMap::new();

    // Tegra234 with NVIDIA's UEFI doesn't expose `/memory@…` nodes in
    // the DTB (root holds only `/tegra-carveouts` and `/bus@0`). The
    // firmware-blessed source of truth for usable RAM is the EFI
    // memory map, harvested by `boot_uefi::harvest_efi_memory_map`
    // before `kernel_main` is called. Iterate the harvested regions
    // here and skip the FDT-based parser.
    //
    // For qemu-virt and tegra-x1 we still use the existing DTB parser
    // — those firmwares populate `/memory@…` normally.
    #[cfg(feature = "tegra234")]
    {
        let count = boot_uefi::efi_memory_region_count();
        for i in 0..count {
            if let Some(r) = boot_uefi::efi_memory_region(i) {
                let _ = phys_map.add_region(
                    smallaios_kernel::mem::PhysAddr::new(r.base as usize),
                    r.size as usize,
                    smallaios_kernel::mem::phys::RegionKind::Usable,
                );
            }
        }
    }
    #[cfg(not(feature = "tegra234"))]
    unsafe {
        smallaios_kernel::mem::phys::parse_dtb(dtb_addr as usize, &mut phys_map);
    }
    let region_count = phys_map.count();
    let usable_mib = phys_map.total_usable() / 1024 / 1024;
    uart::puts("[mem]  DTB parsed: ");
    uart::put_dec(region_count as u64);
    uart::puts(" region(s), ");
    uart::put_dec(usable_mib as u64);
    uart::puts(" MiB usable RAM\n");

    // ── Stage 2.5: Heap allocator ──────────────────────────────────────
    uart::puts("\n[boot] Stage 2.5: Heap allocator\n");
    if let Some(region) = phys_map.usable_regions().next() {
        let base = region.base.as_usize();
        let size = region.size;
        // Skip first 16 MiB for kernel image/stack/BSS
        let heap_offset = 16 * 1024 * 1024;
        if size > heap_offset {
            unsafe {
                smallaios_kernel::mem::global::global_allocator().init(
                    smallaios_kernel::mem::PhysAddr::new(base + heap_offset),
                    size - heap_offset,
                );
            }
            uart::puts("[heap] Initialized: ");
            uart::put_dec(((size - heap_offset) / 1024 / 1024) as u64);
            uart::puts(" MiB\n");
        } else {
            uart::puts("[heap] WARN: usable region too small for heap\n");
        }
    } else {
        uart::puts("[heap] WARN: no usable memory regions found\n");
    }

    // ── Stage 3: Interrupt controller ────────────────────────────────
    uart::puts("\n[boot] Stage 3: Interrupt controller\n");
    #[cfg(feature = "qemu-virt")]
    uart::puts("[irq]  GICv3 (QEMU virt)\n");
    #[cfg(feature = "tegra-x1")]
    {
        uart::puts("[irq]  GICv2 GICD @ 0x");
        uart::put_hex(platform::GICD_BASE as u64);
        uart::puts(", GICC @ 0x");
        uart::put_hex(platform::GICC_BASE as u64);
        uart::puts("\n");
        uart::puts("[irq]  Timer IRQ (PPI 30) enabled\n");
    }

    // ── Stage 4: PCIe bus (Tegra only) ───────────────────────────────
    #[cfg(feature = "tegra-x1")]
    {
        uart::puts("\n[boot] Stage 4: PCIe enumeration\n");
        uart::puts("[pcie] Tegra AFI controller @ 0x");
        uart::put_hex(platform::PCIE_AFI_BASE as u64);
        uart::puts("\n");
        uart::puts("[pcie] Enabling clocks and PHY...\n");

        let pcie_ok = unsafe { tegra_pcie::init() };
        if pcie_ok {
            uart::puts("[pcie] Link up, scanning bus 0...\n");
            let devices = unsafe { tegra_pcie::enumerate_bus(0) };
            let count = devices.count();
            uart::puts("[pcie] Found ");
            uart::put_dec(count as u64);
            uart::puts(" device(s):\n");

            for i in 0..count {
                if let Some(dev) = devices.get(i) {
                    uart::puts("[pcie]   ");
                    uart::put_dec(dev.bus as u64);
                    uart::puts(":");
                    uart::put_dec(dev.device as u64);
                    uart::puts(".");
                    uart::put_dec(dev.function as u64);
                    uart::puts("  vendor=0x");
                    uart::put_hex16(dev.vendor_id);
                    uart::puts(" device=0x");
                    uart::put_hex16(dev.device_id);
                    uart::puts(" class=0x");
                    uart::put_hex16(((dev.class_code as u16) << 8) | dev.subclass as u16);
                    // Identify known devices
                    if dev.vendor_id == 0x10EC
                        && (dev.device_id == 0x8168 || dev.device_id == 0x8169)
                    {
                        uart::puts("  <-- RTL8168/8169 GbE");
                    }
                    uart::puts("\n");

                    // Show active BARs
                    for bar_idx in 0..6 {
                        if dev.bar_sizes[bar_idx] > 0 {
                            uart::puts("[pcie]     BAR");
                            uart::put_dec(bar_idx as u64);
                            uart::puts(": 0x");
                            uart::put_hex(dev.bars[bar_idx]);
                            uart::puts("  size=");
                            let size = dev.bar_sizes[bar_idx];
                            if size >= 1024 * 1024 {
                                uart::put_dec(size / 1024 / 1024);
                                uart::puts(" MiB");
                            } else if size >= 1024 {
                                uart::put_dec(size / 1024);
                                uart::puts(" KiB");
                            } else {
                                uart::put_dec(size);
                                uart::puts(" B");
                            }
                            uart::puts("\n");
                        }
                    }
                }
            }
        } else {
            uart::puts("[pcie] Link training failed -- no PCIe devices\n");
        }
    }

    // ── Stage 5: Display init (Tegra only) ────────────────────────────
    #[cfg(feature = "tegra-x1")]
    {
        uart::puts("\n[boot] Stage 5: Display initialization\n");

        // Detect video mode from EDID (or fallback to 1080p)
        uart::puts("[hdmi] Reading EDID via DPAUX1/DDC...\n");
        unsafe { tegra_edid::dpaux_init() };
        let mode = unsafe { tegra_edid::detect_mode() };
        uart::puts("[hdmi] Mode: ");
        uart::put_dec(mode.width as u64);
        uart::puts("x");
        uart::put_dec(mode.height as u64);
        uart::puts(" @ ");
        uart::put_dec(mode.pixel_clock_khz as u64 / 1000);
        uart::puts(" MHz\n");

        // Initialize SOR0 (HDMI serializer)
        uart::puts("[hdmi] Initializing SOR0 (PLLD + TMDS PHY)...\n");
        let sor_ok = unsafe { tegra_sor::sor_init(&mode) };
        match sor_ok {
            Ok(()) => {
                uart::puts("[hdmi] SOR0 initialized, PLL locked\n");

                // Initialize DC0 (display controller)
                uart::puts("[hdmi] Initializing DC0 (framebuffer DMA)...\n");
                unsafe {
                    tegra_dc::dc_init(&mode, platform::FRAMEBUFFER_BASE);
                }
                uart::puts("[hdmi] DC0 enabled, framebuffer @ 0x");
                uart::put_hex(platform::FRAMEBUFFER_BASE as u64);
                uart::puts("\n");

                // Initialize framebuffer console
                let stride = tegra_dc::align_stride_64(
                    mode.width as usize * 4, // RGBA8888 = 4 bpp
                );
                unsafe {
                    console::init(
                        platform::FRAMEBUFFER_BASE,
                        mode.width as usize,
                        mode.height as usize,
                        stride,
                    );
                }
                uart::puts("[hdmi] Framebuffer console ready\n");
            }
            Err(_) => {
                uart::puts("[hdmi] WARN: SOR0 PLL lock failed, continuing without HDMI\n");
            }
        }
    }

    // ── Stage 6: ONNX inference demo (Tegra only) ──────────────────
    #[cfg(feature = "tegra-x1")]
    {
        uart::puts("\n[boot] Stage 6: ONNX inference demo\n");
        onnx_demo::run_cpu_inference_demo();
        onnx_demo::run_gpu_status_demo();
    }

    // ── Boot complete ────────────────────────────────────────────────
    console::puts("\n========================================\n");
    console::puts("  SmallAIOS ");
    console::puts(smallaios_kernel::VERSION);
    console::puts(" ready\n");
    #[cfg(feature = "tegra-x1")]
    console::puts("  Serial: ttyS0 @ 115200\n");
    console::puts("========================================\n");
    console::puts("\n");

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

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    uart::puts("[SmallAIOS] PANIC: ");
    if let Some(location) = info.location() {
        uart::puts(location.file());
        uart::puts(":");
        uart::put_dec(location.line() as u64);
    }
    uart::puts("\n");
    halt_loop();
}
