// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Register addresses and bit field definitions in this file are derived from
// the NVIDIA nvgpu driver source code, which is licensed under the MIT License.
// Source: https://github.com/OE4T/linux-nvgpu (branch: rel-32-r7-opensource)
// Files referenced: include/nvgpu/hw/gm20b/*.h (MIT-licensed)
// See LICENSES/MIT-nvgpu.txt for the full MIT license text.
//
// The register addresses and bit positions are factual/functional information.
// All driver logic and code structure in this file is original Apache-2.0 work.

//! GM20B (Tegra X1 / Maxwell) register definitions.
//!
//! Constants for PMC power control, CAR clock/reset, GPCPLL configuration,
//! Falcon microcontrollers, GR engine, FIFO, and GMMU page tables.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// PMC registers (base 0x7000_E000)
// ---------------------------------------------------------------------------

/// GPU rail gate control register offset. Bit 0 = clamp enable.
pub const PMC_GPU_RG_CNTRL_OFFSET: u32 = 0x2D4;

/// Power gate toggle register offset.
pub const PMC_PWRGATE_TOGGLE_OFFSET: u32 = 0x30;

/// Power gate status register offset (read-only).
pub const PMC_PWRGATE_STATUS_OFFSET: u32 = 0x38;

/// GPU power partition ID for power gating.
pub const PMC_GPU_PARTITION: u32 = 14;

// ---------------------------------------------------------------------------
// CAR registers (base 0x6000_6000)
// ---------------------------------------------------------------------------

/// Clock enable set register for bank X.
pub const CAR_CLK_OUT_ENB_X_SET_OFFSET: u32 = 0x284;

/// Clock enable clear register for bank X.
pub const CAR_CLK_OUT_ENB_X_CLR_OFFSET: u32 = 0x288;

/// Reset device set register for bank X.
pub const CAR_RST_DEV_X_SET_OFFSET: u32 = 0x290;

/// Reset device clear register for bank X.
pub const CAR_RST_DEV_X_CLR_OFFSET: u32 = 0x294;

/// GPU bit position in CAR bank X registers.
pub const CAR_GPU_BIT: u32 = 1 << 24;

// ---------------------------------------------------------------------------
// GPU BAR0/BAR1 sizes
// ---------------------------------------------------------------------------

/// GPU BAR0 (register) aperture size: 16 MiB.
pub const GPU_BAR0_SIZE: usize = 0x100_0000;

/// GPU BAR1 (VRAM window) aperture size: 16 MiB.
pub const GPU_BAR1_SIZE: usize = 0x100_0000;

// ---------------------------------------------------------------------------
// GPCPLL registers (offsets from BAR0)
// ---------------------------------------------------------------------------

/// GPCPLL configuration register.
/// Bit 0: ENABLE, Bit 1: IDDQ, Bit 17: PLL_LOCK (read-only).
pub const GPCPLL_CFG: u32 = 0x13_7000;

/// GPCPLL coefficient register.
/// Bits [7:0]: M divider, [15:8]: N multiplier, [21:16]: P post-divider.
pub const GPCPLL_COEFF: u32 = 0x13_7004;

/// GPCPLL configuration register 2.
pub const GPCPLL_CFG2: u32 = 0x13_700C;

/// GPCPLL configuration register 3.
pub const GPCPLL_CFG3: u32 = 0x13_7018;

/// VCO select register.
pub const SEL_VCO: u32 = 0x13_7100;

/// GPC2CLK output divider register.
pub const GPC2CLK_OUT: u32 = 0x13_7250;

/// Bypass control register.
pub const BYPASSCTRL: u32 = 0x13_7340;

// GPCPLL CFG bit masks
/// PLL enable bit.
pub const GPCPLL_CFG_ENABLE: u32 = 1 << 0;

/// PLL IDDQ (power-down) bit.
pub const GPCPLL_CFG_IDDQ: u32 = 1 << 1;

/// PLL lock status bit (read-only).
pub const GPCPLL_CFG_LOCK: u32 = 1 << 17;

// GPCPLL coefficient masks and shifts
/// M divider mask (bits [7:0]).
pub const COEFF_M_MASK: u32 = 0xFF;

/// M divider shift.
pub const COEFF_M_SHIFT: u32 = 0;

/// N multiplier mask (bits [15:8]).
pub const COEFF_N_MASK: u32 = 0xFF << 8;

/// N multiplier shift.
pub const COEFF_N_SHIFT: u32 = 8;

/// P post-divider mask (bits [21:16]).
pub const COEFF_P_MASK: u32 = 0x3F << 16;

/// P post-divider shift.
pub const COEFF_P_SHIFT: u32 = 16;

// ---------------------------------------------------------------------------
// NV_PMC registers (offsets from BAR0)
// ---------------------------------------------------------------------------

/// GPU device ID / boot register.
pub const NV_PMC_BOOT_0: u32 = 0x0;

/// Engine enable register.
pub const NV_PMC_ENABLE: u32 = 0x200;

/// Top-level interrupt status register.
pub const NV_PMC_INTR_0: u32 = 0x100;

/// Top-level interrupt enable register.
pub const NV_PMC_INTR_EN_0: u32 = 0x140;

// ---------------------------------------------------------------------------
// Falcon registers (offsets from falcon base)
// ---------------------------------------------------------------------------

/// Falcon CPU control register.
pub const FALCON_CPUCTL: u32 = 0x100;

/// Falcon boot vector register.
pub const FALCON_BOOTVEC: u32 = 0x104;

/// Falcon DMA transfer base address register.
pub const FALCON_DMATRFBASE: u32 = 0x110;

/// Falcon DMA transfer memory offset register.
pub const FALCON_DMATRFMOFFS: u32 = 0x114;

/// Falcon DMA transfer command register.
pub const FALCON_DMATRFCMD: u32 = 0x118;

/// Falcon DMA transfer framebuffer offset register.
pub const FALCON_DMATRFFBOFFS: u32 = 0x11C;

/// Start CPU bit in FALCON_CPUCTL.
pub const FALCON_CPUCTL_STARTCPU: u32 = 1 << 1;

/// DMA transfer targets IMEM.
pub const FALCON_DMATRFCMD_IMEM: u32 = 1 << 4;

/// DMA transfer direction: write.
pub const FALCON_DMATRFCMD_WRITE: u32 = 1 << 5;

// ---------------------------------------------------------------------------
// Falcon base addresses (offsets from BAR0)
// ---------------------------------------------------------------------------

/// FECS (Front-End Context Switch) falcon base.
pub const FALCON_FECS_BASE: u32 = 0x40_9000;

/// GPCCS (GPC Context Switch) falcon base.
pub const FALCON_GPCCS_BASE: u32 = 0x41_A000;

/// PMU (Power Management Unit) falcon base.
pub const FALCON_PMU_BASE: u32 = 0x10_A000;

// ---------------------------------------------------------------------------
// GR registers (offsets from BAR0)
// ---------------------------------------------------------------------------

/// SM architecture register for GPC0/TPC0.
pub const GR_PRI_GPC0_TPC0_SM_ARCH: u32 = 0x50_4330;

/// Broadcast SM architecture register for all GPCs/TPCs.
pub const GR_PRI_GPCS_TPCS_SM_ARCH: u32 = 0x41_9E04;

// ---------------------------------------------------------------------------
// FIFO registers (offsets from BAR0)
// ---------------------------------------------------------------------------

/// BAR1 map base register.
pub const FIFO_BAR1_MAP_BASE: u32 = 0x2000;

/// Runlist base address register.
pub const FIFO_RUNLIST_BASE: u32 = 0x2270;

/// Engine runlist base address register.
pub const FIFO_ENG_RUNLIST_BASE: u32 = 0x2274;

// ---------------------------------------------------------------------------
// GMMU (GPU Memory Management Unit) constants
// ---------------------------------------------------------------------------

/// PDE aperture field mask (2 bits).
pub const GMMU_PDE_APERTURE_MASK: u64 = 0x3;

/// PDE/PTE address shift (4 KiB alignment).
pub const GMMU_PDE_ADDRESS_SHIFT: u32 = 12;

/// PTE valid bit.
pub const GMMU_PTE_VALID: u64 = 1 << 0;

/// PTE read permission bit.
pub const GMMU_PTE_READ: u64 = 1 << 1;

/// PTE write permission bit.
pub const GMMU_PTE_WRITE: u64 = 1 << 2;

/// Small page size (4 KiB).
pub const GMMU_PAGE_SIZE_SMALL: usize = 4096;

/// Big page size (64 KiB).
pub const GMMU_PAGE_SIZE_BIG: usize = 65536;

// ---------------------------------------------------------------------------
// GPCPLL frequency steps
// ---------------------------------------------------------------------------

/// A single GPCPLL frequency step with PLL coefficients.
///
/// The GPU clock frequency is: `(ref_clk * N) / (M * PL)` where ref_clk = 38.4 MHz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpcpllStep {
    /// Target frequency in MHz.
    pub freq_mhz: u32,
    /// M divider (reference divider).
    pub m: u8,
    /// N multiplier (feedback divider).
    pub n: u8,
    /// PL post-divider.
    pub pl: u8,
}

/// Pre-computed GPCPLL frequency steps for GM20B, from lowest to highest.
///
/// Steps span 76 MHz to 921 MHz, covering idle through max boost.
pub const GPCPLL_STEPS: [GpcpllStep; 12] = [
    GpcpllStep {
        freq_mhz: 76,
        m: 1,
        n: 8,
        pl: 4,
    },
    GpcpllStep {
        freq_mhz: 153,
        m: 1,
        n: 16,
        pl: 4,
    },
    GpcpllStep {
        freq_mhz: 230,
        m: 1,
        n: 24,
        pl: 4,
    },
    GpcpllStep {
        freq_mhz: 307,
        m: 1,
        n: 16,
        pl: 2,
    },
    GpcpllStep {
        freq_mhz: 384,
        m: 1,
        n: 20,
        pl: 2,
    },
    GpcpllStep {
        freq_mhz: 460,
        m: 1,
        n: 24,
        pl: 2,
    },
    GpcpllStep {
        freq_mhz: 537,
        m: 1,
        n: 28,
        pl: 2,
    },
    GpcpllStep {
        freq_mhz: 614,
        m: 1,
        n: 16,
        pl: 1,
    },
    GpcpllStep {
        freq_mhz: 691,
        m: 1,
        n: 18,
        pl: 1,
    },
    GpcpllStep {
        freq_mhz: 768,
        m: 1,
        n: 20,
        pl: 1,
    },
    GpcpllStep {
        freq_mhz: 844,
        m: 1,
        n: 22,
        pl: 1,
    },
    GpcpllStep {
        freq_mhz: 921,
        m: 1,
        n: 24,
        pl: 1,
    },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpcpll_steps_are_monotonically_increasing() {
        for i in 1..GPCPLL_STEPS.len() {
            assert!(
                GPCPLL_STEPS[i].freq_mhz > GPCPLL_STEPS[i - 1].freq_mhz,
                "Step {} ({} MHz) should be > step {} ({} MHz)",
                i,
                GPCPLL_STEPS[i].freq_mhz,
                i - 1,
                GPCPLL_STEPS[i - 1].freq_mhz,
            );
        }
    }

    #[test]
    fn gpcpll_steps_count() {
        assert_eq!(GPCPLL_STEPS.len(), 12);
    }

    #[test]
    fn gpcpll_steps_frequency_range() {
        assert_eq!(GPCPLL_STEPS[0].freq_mhz, 76);
        assert_eq!(GPCPLL_STEPS[11].freq_mhz, 921);
    }

    #[test]
    fn gpcpll_steps_m_divider_is_one() {
        // All GM20B steps use M=1 (no reference division)
        for step in &GPCPLL_STEPS {
            assert_eq!(
                step.m, 1,
                "M divider should be 1 for freq {} MHz",
                step.freq_mhz
            );
        }
    }

    #[test]
    fn gpcpll_steps_pl_values_valid() {
        // PL must be 1, 2, or 4 for GM20B
        for step in &GPCPLL_STEPS {
            assert!(
                step.pl == 1 || step.pl == 2 || step.pl == 4,
                "PL={} invalid for freq {} MHz",
                step.pl,
                step.freq_mhz,
            );
        }
    }

    #[test]
    fn coefficient_masks_do_not_overlap() {
        assert_eq!(COEFF_M_MASK & COEFF_N_MASK, 0);
        assert_eq!(COEFF_M_MASK & COEFF_P_MASK, 0);
        assert_eq!(COEFF_N_MASK & COEFF_P_MASK, 0);
    }

    #[test]
    fn coefficient_pack_unpack() {
        let m: u32 = 1;
        let n: u32 = 24;
        let pl: u32 = 2;

        let packed = (m << COEFF_M_SHIFT) | (n << COEFF_N_SHIFT) | (pl << COEFF_P_SHIFT);

        assert_eq!((packed & COEFF_M_MASK) >> COEFF_M_SHIFT, m);
        assert_eq!((packed & COEFF_N_MASK) >> COEFF_N_SHIFT, n);
        assert_eq!((packed & COEFF_P_MASK) >> COEFF_P_SHIFT, pl);
    }

    #[test]
    fn pmc_gpu_partition_bit() {
        // Bit 14 in PWRGATE_STATUS corresponds to GPU partition
        let mask = 1u32 << PMC_GPU_PARTITION;
        assert_eq!(mask, 0x4000);
    }

    #[test]
    fn car_gpu_bit_position() {
        assert_eq!(CAR_GPU_BIT, 0x0100_0000);
    }

    #[test]
    fn gpu_bar_sizes() {
        assert_eq!(GPU_BAR0_SIZE, 16 * 1024 * 1024);
        assert_eq!(GPU_BAR1_SIZE, 16 * 1024 * 1024);
    }

    #[test]
    fn gpcpll_cfg_bits_do_not_overlap() {
        assert_eq!(GPCPLL_CFG_ENABLE & GPCPLL_CFG_IDDQ, 0);
        assert_eq!(GPCPLL_CFG_ENABLE & GPCPLL_CFG_LOCK, 0);
        assert_eq!(GPCPLL_CFG_IDDQ & GPCPLL_CFG_LOCK, 0);
    }

    #[test]
    fn falcon_base_addresses_non_overlapping() {
        // Each falcon occupies at least 0x1000 bytes
        assert!(FALCON_FECS_BASE != FALCON_GPCCS_BASE);
        assert!(FALCON_FECS_BASE != FALCON_PMU_BASE);
        assert!(FALCON_GPCCS_BASE != FALCON_PMU_BASE);
    }

    #[test]
    fn gmmu_pte_bits_do_not_overlap() {
        assert_eq!(GMMU_PTE_VALID & GMMU_PTE_READ, 0);
        assert_eq!(GMMU_PTE_VALID & GMMU_PTE_WRITE, 0);
        assert_eq!(GMMU_PTE_READ & GMMU_PTE_WRITE, 0);
    }

    #[test]
    fn gmmu_page_sizes() {
        assert_eq!(GMMU_PAGE_SIZE_SMALL, 4096);
        assert_eq!(GMMU_PAGE_SIZE_BIG, 64 * 1024);
    }

    #[test]
    fn nv_pmc_boot_0_is_zero() {
        assert_eq!(NV_PMC_BOOT_0, 0);
    }
}
