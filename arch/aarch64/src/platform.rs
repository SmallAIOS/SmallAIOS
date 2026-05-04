// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Platform-specific constants for AArch64 targets.
//!
//! Selected at compile time via Cargo feature flags:
//! - `qemu-virt` (default): QEMU virt machine (PL011, GICv3)
//! - `tegra-x1`: NVIDIA Jetson Nano / Tegra X1 (NS16550A, GICv2)
//! - `tegra234`: NVIDIA Jetson Orin family / Tegra234 (TCU UART, GICv3)

#[cfg(any(
    all(feature = "qemu-virt", feature = "tegra-x1"),
    all(feature = "qemu-virt", feature = "tegra234"),
    all(feature = "tegra-x1", feature = "tegra234"),
))]
compile_error!("Platform features are mutually exclusive — pick exactly one of `qemu-virt`, `tegra-x1`, `tegra234`");

#[cfg(not(any(feature = "qemu-virt", feature = "tegra-x1", feature = "tegra234")))]
compile_error!(
    "Exactly one platform feature must be enabled: `qemu-virt`, `tegra-x1`, or `tegra234`"
);

// ─── QEMU virt machine ──────────────────────────────────────────────────────

#[cfg(feature = "qemu-virt")]
pub const UART_BASE: usize = 0x0900_0000;

#[cfg(feature = "qemu-virt")]
pub const GICD_BASE: usize = 0x0800_0000;

#[cfg(feature = "qemu-virt")]
pub const GICR_BASE: usize = 0x080A_0000;

#[cfg(feature = "qemu-virt")]
pub const DRAM_BASE: usize = 0x4000_0000;

#[cfg(feature = "qemu-virt")]
pub const KERNEL_LOAD_ADDR: usize = 0x4008_0000;

// ─── NVIDIA Tegra X1 (Jetson Nano) ──────────────────────────────────────────

/// Tegra UART-A (NS16550A-compatible, reg-shift=2, pre-initialized by U-Boot).
#[cfg(feature = "tegra-x1")]
pub const UART_BASE: usize = 0x7000_6000;

/// Tegra UART register shift (registers at offset × 4).
#[cfg(feature = "tegra-x1")]
pub const UART_REG_SHIFT: u32 = 2;

/// GICv2 Distributor.
#[cfg(feature = "tegra-x1")]
pub const GICD_BASE: usize = 0x5004_1000;

/// GICv2 CPU Interface.
#[cfg(feature = "tegra-x1")]
pub const GICC_BASE: usize = 0x5004_2000;

/// DRAM base.
#[cfg(feature = "tegra-x1")]
pub const DRAM_BASE: usize = 0x8000_0000;

/// Kernel load address (U-Boot loads Image here).
#[cfg(feature = "tegra-x1")]
pub const KERNEL_LOAD_ADDR: usize = 0x8008_0000;

/// PCIe root complex (AFI controller).
#[cfg(feature = "tegra-x1")]
pub const PCIE_AFI_BASE: usize = 0x0100_3000;

/// PCIe root port configuration space window.
#[cfg(feature = "tegra-x1")]
pub const PCIE_RP_BASE: usize = 0x0100_0000;

/// Clock and Reset controller.
#[cfg(feature = "tegra-x1")]
pub const CAR_BASE: usize = 0x6000_6000;

/// Display Controller 0 (DC0) base address (TRM chapter 32).
#[cfg(feature = "tegra-x1")]
pub const DC0_BASE: usize = 0x5420_0000;

/// Serial Output Resource 0 (SOR0/HDMI) base address (TRM chapter 34).
#[cfg(feature = "tegra-x1")]
pub const SOR0_BASE: usize = 0x5454_0000;

/// DPAUX1 controller base address (DDC/I2C for HDMI EDID).
#[cfg(feature = "tegra-x1")]
pub const DPAUX1_BASE: usize = 0x545C_0000;

/// Power Management Controller base address.
#[cfg(feature = "tegra-x1")]
pub const PMC_BASE: usize = 0x7000_E000;

/// Framebuffer physical base address (fixed reservation in DRAM).
#[cfg(feature = "tegra-x1")]
pub const FRAMEBUFFER_BASE: usize = 0x8F00_0000;

/// Framebuffer size in bytes (8 MiB — enough for 1920x1080 RGBA8888 with margin).
#[cfg(feature = "tegra-x1")]
pub const FRAMEBUFFER_SIZE: usize = 8 * 1024 * 1024;

/// GPU BAR0 base address (GM20B control registers, 16 MiB).
#[cfg(feature = "tegra-x1")]
pub const GPU_BAR0_BASE: usize = 0x5700_0000;

/// GPU BAR1 base address (GPU memory/FIFO, 16 MiB).
#[cfg(feature = "tegra-x1")]
pub const GPU_BAR1_BASE: usize = 0x5800_0000;

/// GPU stall interrupt (GIC SPI 157, IRQ 189). Engine faults, semaphore, errors.
#[cfg(feature = "tegra-x1")]
pub const GPU_IRQ_STALL: u32 = 189;

/// GPU non-stall interrupt (GIC SPI 158, IRQ 190). Completion notifications.
#[cfg(feature = "tegra-x1")]
pub const GPU_IRQ_NONSTALL: u32 = 190;

// ─── NVIDIA Tegra234 (Jetson Orin family) ───────────────────────────────────
//
// Sources: NVIDIA Tegra234 Technical Reference Manual (DP-10465-001) and the
// upstream Linux DTS at `arch/arm64/boot/dts/nvidia/tegra234.dtsi`. These
// constants are factual hardware addresses (not copyrightable expression).
// The actual DTB is provided by the Orin's UEFI firmware at runtime via
// `EFI_DTB_TABLE_GUID` — see `boot_uefi.rs` (sub-PR 2c) — so the kernel
// doesn't bundle its own DTS.

/// Tegra Combined UART (TCU). NS16550-compatible register layout, but the
/// access pattern is the SoC-side mailbox interface used by NVIDIA's bring-up
/// firmware. Concrete driver lands in sub-PR 2d (`tegra234_uart.rs`).
#[cfg(feature = "tegra234")]
pub const UART_BASE: usize = 0x0C28_0000;

/// GICv3 Distributor base (Tegra234 puts the GIC at the Tegra-family
/// SCF address; same for Orin Nano / NX / AGX).
#[cfg(feature = "tegra234")]
pub const GICD_BASE: usize = 0x0F40_0000;

/// GICv3 Redistributor base (one frame per CPU; Tegra234 has up to 12 cores
/// total but Orin NX 16 GB exposes 8 — the redistributor span covers all of
/// them and is sized at runtime by `gicv3.rs` in sub-PR 2e).
#[cfg(feature = "tegra234")]
pub const GICR_BASE: usize = 0x0F44_0000;

/// DRAM base. Standard for the Tegra234 family.
#[cfg(feature = "tegra234")]
pub const DRAM_BASE: usize = 0x8000_0000;

/// Kernel load address — DRAM_BASE + 512 KiB. Matches the Linux ARM64 boot
/// protocol convention so the same artifact is bootable via U-Boot `booti`
/// from L4T's extlinux config. The UEFI boot path uses the
/// `aarch64-unknown-uefi` target instead (rust's built-in PE/COFF emission)
/// and the .efi loader chooses the load address at runtime — this constant
/// is unused on that path.
#[cfg(feature = "tegra234")]
pub const KERNEL_LOAD_ADDR: usize = 0x8008_0000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uart_base_is_valid() {
        assert_ne!(UART_BASE, 0);
    }

    #[test]
    fn test_gicd_base_is_valid() {
        assert_ne!(GICD_BASE, 0);
    }

    #[test]
    fn test_dram_base_is_valid() {
        assert_ne!(DRAM_BASE, 0);
    }

    #[test]
    fn test_kernel_load_after_dram() {
        assert!(KERNEL_LOAD_ADDR >= DRAM_BASE);
    }

    // Platform-specific address checks (run under the active feature)
    #[cfg(feature = "qemu-virt")]
    #[test]
    fn test_qemu_virt_addresses() {
        assert_eq!(UART_BASE, 0x0900_0000);
        assert_eq!(GICD_BASE, 0x0800_0000);
        assert_eq!(GICR_BASE, 0x080A_0000);
        assert_eq!(KERNEL_LOAD_ADDR, 0x4008_0000);
    }

    #[cfg(feature = "tegra-x1")]
    #[test]
    fn test_tegra_x1_addresses() {
        assert_eq!(UART_BASE, 0x7000_6000);
        assert_eq!(GICD_BASE, 0x5004_1000);
        assert_eq!(GICC_BASE, 0x5004_2000);
        assert_eq!(KERNEL_LOAD_ADDR, 0x8008_0000);
        assert_eq!(PCIE_AFI_BASE, 0x0100_3000);
        assert_eq!(CAR_BASE, 0x6000_6000);
        assert_eq!(UART_REG_SHIFT, 2);
    }

    #[cfg(feature = "tegra-x1")]
    #[test]
    fn test_tegra_x1_display_addresses() {
        assert_eq!(DC0_BASE, 0x5420_0000);
        assert_eq!(SOR0_BASE, 0x5454_0000);
        assert_eq!(DPAUX1_BASE, 0x545C_0000);
        assert_eq!(PMC_BASE, 0x7000_E000);
    }

    #[cfg(feature = "tegra-x1")]
    #[test]
    fn test_tegra_x1_framebuffer_constants() {
        assert_eq!(FRAMEBUFFER_BASE, 0x8F00_0000);
        assert_eq!(FRAMEBUFFER_SIZE, 8 * 1024 * 1024);
        // Framebuffer must be within DRAM region (above DRAM_BASE)
        assert!(FRAMEBUFFER_BASE >= DRAM_BASE);
        // Framebuffer end must not wrap around
        assert!(FRAMEBUFFER_BASE.checked_add(FRAMEBUFFER_SIZE).is_some());
    }

    #[cfg(feature = "tegra-x1")]
    #[test]
    fn test_tegra_x1_gpu_constants() {
        assert_eq!(GPU_BAR0_BASE, 0x5700_0000);
        assert_eq!(GPU_BAR1_BASE, 0x5800_0000);
        assert_eq!(GPU_IRQ_STALL, 189);
        assert_eq!(GPU_IRQ_NONSTALL, 190);
    }

    #[cfg(feature = "tegra234")]
    #[test]
    fn test_tegra234_addresses() {
        assert_eq!(UART_BASE, 0x0C28_0000);
        assert_eq!(GICD_BASE, 0x0F40_0000);
        assert_eq!(GICR_BASE, 0x0F44_0000);
        assert_eq!(DRAM_BASE, 0x8000_0000);
        assert_eq!(KERNEL_LOAD_ADDR, 0x8008_0000);
    }

    #[cfg(feature = "tegra234")]
    #[test]
    fn test_tegra234_load_offset_matches_linux_image_protocol() {
        // ARM64 Linux boot protocol convention: KERNEL_LOAD_ADDR is
        // DRAM_BASE + 512 KiB so U-Boot's `booti` lands the kernel at a
        // fixed offset above the start of DRAM. The Tegra X1 layout
        // follows the same convention; Tegra234 keeps it for symmetry.
        assert_eq!(KERNEL_LOAD_ADDR - DRAM_BASE, 512 * 1024);
    }
}
