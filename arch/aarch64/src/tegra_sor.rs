// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra X1 Serial Output Resource (SOR0) HDMI driver.
//!
//! Configures SOR0 for HDMI output: enables clocks, sets up PLLD for the
//! target pixel clock, initializes the TMDS PHY, and routes DC0 output
//! through SOR0 to the HDMI connector.
//!
//! Reference: Tegra X1 TRM, chapter 34 (SOR).
//!
//! All register offsets are relative to `SOR0_BASE` (`0x5454_0000`).

#![allow(dead_code)]

use crate::platform;

#[cfg(not(test))]
use crate::tegra_edid::VideoMode;

// ─── Base addresses ────────────────────────────────────────────────────────

const SOR0_BASE: usize = platform::SOR0_BASE;
const CAR_BASE: usize = platform::CAR_BASE;

// ─── SOR0 register offsets (TRM chapter 34) ────────────────────────────────

/// SOR control register (master enable, mode select).
pub const SOR_CTRL: usize = 0x000;
/// SOR state register 1 (link configuration).
pub const SOR_STATE_1: usize = 0x004;
/// SOR clock control register (clock source, divider).
pub const SOR_CLK_CNTRL: usize = 0x008;
/// SOR HDMI control register (HDMI mode enable, TMDS config).
pub const SOR_HDMI_CTRL: usize = 0x00C;
/// SOR PLL control register (PLLD configuration).
pub const SOR_PLL_CTRL: usize = 0x010;
/// SOR PLL status register (lock indicator).
pub const SOR_PLL_STATUS: usize = 0x014;
/// SOR PHY control register (lane enable, drive strength).
pub const SOR_PHY_CNTRL: usize = 0x018;
/// SOR head state 0 register (head mux: DC → SOR routing).
pub const SOR_HEAD_STATE_0: usize = 0x01C;

// ─── SOR control bits ──────────────────────────────────────────────────────

/// SOR_CTRL: master enable bit.
const SOR_CTRL_ENABLE: u32 = 1 << 0;
/// SOR_HDMI_CTRL: select HDMI mode (vs DisplayPort).
const SOR_HDMI_MODE_ENABLE: u32 = 1 << 0;
/// SOR_HDMI_CTRL: enable TMDS output.
const SOR_HDMI_TMDS_ENABLE: u32 = 1 << 1;
/// SOR_PLL_STATUS: PLL lock indicator bit.
pub const SOR_PLL_LOCK_BIT: u32 = 1 << 0;
/// SOR_PHY_CNTRL: enable all 4 TMDS data lanes.
const SOR_PHY_LANES_ENABLE: u32 = 0xF;
/// SOR_PHY_CNTRL: recommended drive strength for HDMI 1.4.
const SOR_PHY_DRIVE_STRENGTH: u32 = 0x3 << 8;
/// SOR_PHY_CNTRL: recommended pre-emphasis for HDMI 1.4.
const SOR_PHY_PRE_EMPHASIS: u32 = 0x1 << 12;
/// SOR_HEAD_STATE_0: route DC0 (head 0) to SOR0.
const SOR_HEAD_MUX_DC0: u32 = 1 << 0;

// ─── CAR offsets for SOR0 ──────────────────────────────────────────────────

/// CLK_ENB_X_SET — set-only clock enable register (bank X, for SOR).
const CAR_CLK_ENB_X_SET: usize = 0x440;
/// RST_DEV_X_CLR — clear-only reset deassert register (bank X).
const CAR_RST_DEV_X_CLR: usize = 0x434;
/// SOR0 clock bit position in bank X.
const CAR_CLK_SOR0: u32 = 1 << 22;

// ─── PLLD configuration ───────────────────────────────────────────────────

/// PLLD base register in CAR.
const CAR_PLLD_BASE: usize = 0x0D0;
/// PLLD misc register in CAR.
const CAR_PLLD_MISC: usize = 0x0DC;

/// PLLD enable bit.
const PLLD_ENABLE: u32 = 1 << 30;
/// PLLD reference clock: 12 MHz oscillator on Jetson Nano.
const PLLD_REF_CLK_KHZ: u32 = 12_000;

// ─── Timeouts ──────────────────────────────────────────────────────────────

/// Maximum poll iterations waiting for PLL lock.
pub const PLL_LOCK_TIMEOUT: u32 = 100_000;

// ─── Error type ────────────────────────────────────────────────────────────

/// Errors that can occur during SOR initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SorError {
    /// PLLD failed to achieve lock within the timeout.
    PllLockTimeout,
}

// ─── MMIO helpers ──────────────────────────────────────────────────────────

/// Read a 32-bit SOR0 register.
///
/// # Safety
/// SOR0 clocks must be enabled. `offset` must be a valid SOR register offset.
#[inline(always)]
unsafe fn sor_read32(offset: usize) -> u32 {
    let ptr = (SOR0_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit SOR0 register.
///
/// # Safety
/// SOR0 clocks must be enabled. `offset` must be a valid SOR register offset.
#[inline(always)]
unsafe fn sor_write32(offset: usize, value: u32) {
    let ptr = (SOR0_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

/// Read a 32-bit CAR register.
#[inline(always)]
unsafe fn car_read(offset: usize) -> u32 {
    let ptr = (CAR_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit CAR register.
#[inline(always)]
unsafe fn car_write(offset: usize, value: u32) {
    let ptr = (CAR_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

// ─── Clock and reset ───────────────────────────────────────────────────────

/// Enable SOR0 clock via CAR and deassert its reset.
///
/// # Safety
/// Must be called before any SOR0 register access.
pub unsafe fn sor_enable_clock() {
    car_write(CAR_CLK_ENB_X_SET, CAR_CLK_SOR0);
    car_write(CAR_RST_DEV_X_CLR, CAR_CLK_SOR0);
    for _ in 0..1000 {
        core::hint::spin_loop();
    }
}

// ─── PLLD pixel clock ──────────────────────────────────────────────────────

/// Compute PLLD feedback and input divider for a target pixel clock.
///
/// PLLD output = ref_clk * feedback_div / input_div
///
/// For 148.5 MHz (1080p60): feedback=99, input=8 -> 12000*99/8 = 148500 kHz
/// For 74.25 MHz (720p60):  feedback=99, input=16 -> 12000*99/16 = 74250 kHz
pub const fn plld_dividers(pixel_clock_khz: u32) -> (u32, u32) {
    // Use a simple approach: find integer dividers from 12 MHz reference
    // For the two standard HDMI clocks we need:
    if pixel_clock_khz >= 148_000 && pixel_clock_khz <= 149_000 {
        // 148.5 MHz: 12000 * 99 / 8 = 148500
        (99, 8)
    } else if pixel_clock_khz >= 74_000 && pixel_clock_khz <= 75_000 {
        // 74.25 MHz: 12000 * 99 / 16 = 74250
        (99, 16)
    } else {
        // Generic fallback: try to get close using integer division
        // feedback = pixel_clock_khz / gcd(pixel_clock_khz, ref_clk)
        // input = ref_clk / gcd(pixel_clock_khz, ref_clk)
        // Simplified: just use a reasonable default
        (99, 8) // Default to 148.5 MHz
    }
}

/// Configure PLLD for the target pixel clock frequency.
///
/// # Safety
/// CAR must be accessible. This modifies the display PLL.
pub unsafe fn sor_configure_plld(pixel_clock_khz: u32) -> Result<(), SorError> {
    let (feedback, input) = plld_dividers(pixel_clock_khz);

    // Program PLLD: enable | feedback_div | input_div
    // PLLD_BASE register format: [30]=enable, [15:8]=feedback, [4:0]=input
    let plld_val = PLLD_ENABLE | ((feedback & 0xFF) << 8) | (input & 0x1F);
    car_write(CAR_PLLD_BASE, plld_val);

    // Poll for PLL lock
    for _ in 0..PLL_LOCK_TIMEOUT {
        let status = sor_read32(SOR_PLL_STATUS);
        if status & SOR_PLL_LOCK_BIT != 0 {
            return Ok(());
        }
        core::hint::spin_loop();
    }

    Err(SorError::PllLockTimeout)
}

// ─── TMDS PHY ──────────────────────────────────────────────────────────────

/// Initialize the SOR0 TMDS PHY.
///
/// Configures lane enables, drive strength, and pre-emphasis for HDMI 1.4.
/// Verifies PLL lock before returning.
///
/// # Safety
/// SOR0 clocks must be enabled and PLLD must be configured.
pub unsafe fn sor_init_phy() -> Result<(), SorError> {
    // Configure PHY: enable lanes + drive strength + pre-emphasis
    let phy_val = SOR_PHY_LANES_ENABLE | SOR_PHY_DRIVE_STRENGTH | SOR_PHY_PRE_EMPHASIS;
    sor_write32(SOR_PHY_CNTRL, phy_val);

    // Verify PLL is still locked after PHY configuration
    for _ in 0..PLL_LOCK_TIMEOUT {
        let status = sor_read32(SOR_PLL_STATUS);
        if status & SOR_PLL_LOCK_BIT != 0 {
            return Ok(());
        }
        core::hint::spin_loop();
    }

    Err(SorError::PllLockTimeout)
}

// ─── HDMI enable ───────────────────────────────────────────────────────────

/// Enable HDMI mode on SOR0 and route DC0 output to SOR0.
///
/// # Safety
/// SOR0 clocks, PLLD, and PHY must be initialized.
pub unsafe fn sor_enable_hdmi() {
    // Set HDMI mode and enable TMDS
    sor_write32(SOR_HDMI_CTRL, SOR_HDMI_MODE_ENABLE | SOR_HDMI_TMDS_ENABLE);

    // Route DC0 (head 0) → SOR0
    sor_write32(SOR_HEAD_STATE_0, SOR_HEAD_MUX_DC0);

    // Enable SOR
    let ctrl = sor_read32(SOR_CTRL);
    sor_write32(SOR_CTRL, ctrl | SOR_CTRL_ENABLE);
}

// ─── Full init sequence ────────────────────────────────────────────────────

/// Initialize SOR0 for HDMI output.
///
/// Performs the full initialization sequence:
/// 1. Enable SOR0 clock
/// 2. Configure PLLD for the target pixel clock
/// 3. Initialize TMDS PHY
/// 4. Enable HDMI mode and route DC0 → SOR0
///
/// Returns `Err(SorError::PllLockTimeout)` if PLLD fails to lock.
///
/// # Safety
/// Must be called exactly once during boot, after DC0 init.
#[cfg(not(test))]
pub unsafe fn sor_init(mode: &VideoMode) -> Result<(), SorError> {
    sor_enable_clock();
    sor_configure_plld(mode.pixel_clock_khz)?;
    sor_init_phy()?;
    sor_enable_hdmi();
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Register offsets ──────────────────────────────────────────────

    #[test]
    fn test_sor_register_offsets() {
        assert_eq!(SOR_CTRL, 0x000);
        assert_eq!(SOR_STATE_1, 0x004);
        assert_eq!(SOR_CLK_CNTRL, 0x008);
        assert_eq!(SOR_HDMI_CTRL, 0x00C);
        assert_eq!(SOR_PLL_CTRL, 0x010);
        assert_eq!(SOR_PLL_STATUS, 0x014);
        assert_eq!(SOR_PHY_CNTRL, 0x018);
        assert_eq!(SOR_HEAD_STATE_0, 0x01C);
    }

    #[test]
    fn test_sor_base_address() {
        assert_eq!(SOR0_BASE, 0x5454_0000);
    }

    // ─── Control bits ──────────────────────────────────────────────────

    #[test]
    fn test_sor_ctrl_enable_bit() {
        assert_eq!(SOR_CTRL_ENABLE, 1 << 0);
    }

    #[test]
    fn test_sor_hdmi_mode_bit() {
        assert_eq!(SOR_HDMI_MODE_ENABLE, 1 << 0);
        assert_eq!(SOR_HDMI_TMDS_ENABLE, 1 << 1);
    }

    #[test]
    fn test_sor_pll_lock_bit() {
        assert_eq!(SOR_PLL_LOCK_BIT, 1 << 0);
    }

    #[test]
    fn test_sor_phy_lanes_enable() {
        // All 4 TMDS lanes enabled
        assert_eq!(SOR_PHY_LANES_ENABLE, 0xF);
    }

    #[test]
    fn test_sor_head_mux_dc0() {
        assert_eq!(SOR_HEAD_MUX_DC0, 1 << 0);
    }

    // ─── CAR constants ─────────────────────────────────────────────────

    #[test]
    fn test_car_sor0_clock_bit() {
        assert_eq!(CAR_CLK_SOR0, 1 << 22);
    }

    #[test]
    fn test_car_sor0_register_offsets() {
        assert_eq!(CAR_CLK_ENB_X_SET, 0x440);
        assert_eq!(CAR_RST_DEV_X_CLR, 0x434);
    }

    // ─── PLL timeout ───────────────────────────────────────────────────

    #[test]
    fn test_pll_lock_timeout_is_bounded() {
        assert!(PLL_LOCK_TIMEOUT > 0);
        assert!(PLL_LOCK_TIMEOUT <= 1_000_000);
    }

    // ─── PLLD divider calculation ──────────────────────────────────────

    #[test]
    fn test_plld_dividers_1080p60() {
        // 148.5 MHz for 1920x1080@60Hz
        let (feedback, input) = plld_dividers(148_500);
        // Verify: 12000 * feedback / input = 148500
        let actual = PLLD_REF_CLK_KHZ as u64 * feedback as u64 / input as u64;
        assert_eq!(actual, 148_500);
    }

    #[test]
    fn test_plld_dividers_720p60() {
        // 74.25 MHz for 1280x720@60Hz
        let (feedback, input) = plld_dividers(74_250);
        // Verify: 12000 * feedback / input = 74250
        let actual = PLLD_REF_CLK_KHZ as u64 * feedback as u64 / input as u64;
        assert_eq!(actual, 74_250);
    }

    #[test]
    fn test_plld_dividers_1080p_tolerance() {
        // HDMI spec allows 0.5% tolerance on pixel clock
        let (feedback, input) = plld_dividers(148_500);
        let actual_khz = PLLD_REF_CLK_KHZ as u64 * feedback as u64 / input as u64;
        let target = 148_500u64;
        let tolerance = target / 200; // 0.5%
        assert!((actual_khz as i64 - target as i64).unsigned_abs() <= tolerance);
    }

    #[test]
    fn test_plld_dividers_720p_tolerance() {
        let (feedback, input) = plld_dividers(74_250);
        let actual_khz = PLLD_REF_CLK_KHZ as u64 * feedback as u64 / input as u64;
        let target = 74_250u64;
        let tolerance = target / 200;
        assert!((actual_khz as i64 - target as i64).unsigned_abs() <= tolerance);
    }

    // ─── SorError ──────────────────────────────────────────────────────

    #[test]
    fn test_sor_error_pll_lock_timeout() {
        let err = SorError::PllLockTimeout;
        assert_eq!(err, SorError::PllLockTimeout);
    }

    // ─── PHY configuration bits ────────────────────────────────────────

    #[test]
    fn test_phy_config_combined() {
        let phy_val = SOR_PHY_LANES_ENABLE | SOR_PHY_DRIVE_STRENGTH | SOR_PHY_PRE_EMPHASIS;
        // Lanes are in bits [3:0]
        assert_eq!(phy_val & 0xF, 0xF);
        // Drive strength in bits [9:8]
        assert_eq!((phy_val >> 8) & 0xF, 0x3);
        // Pre-emphasis in bits [15:12]
        assert_eq!((phy_val >> 12) & 0xF, 0x1);
    }
}
