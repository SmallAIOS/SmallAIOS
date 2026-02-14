// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Platform-specific constants for AArch64 targets.
//!
//! Selected at compile time via Cargo feature flags:
//! - `qemu-virt` (default): QEMU virt machine (PL011, GICv3)
//! - `tegra-x1`: NVIDIA Jetson Nano / Tegra X1 (NS16550A, GICv2)

#[cfg(all(feature = "qemu-virt", feature = "tegra-x1"))]
compile_error!("Features `qemu-virt` and `tegra-x1` are mutually exclusive");

#[cfg(not(any(feature = "qemu-virt", feature = "tegra-x1")))]
compile_error!("Exactly one platform feature must be enabled: `qemu-virt` or `tegra-x1`");

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
}
