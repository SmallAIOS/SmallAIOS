// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Xilinx/AMD Zynq UltraScale+ MPSoC platform support package.
//!
//! Covers PS-PL interface management, FPGA fabric clock (FCLK) configuration
//! via CRL_APB registers, and PL reset control.
//!
//! The Zynq UltraScale+ has several PS-PL AXI interfaces:
//! - HPM0/HPM1 (High Performance Master): 128-bit AXI from PS to PL
//! - HPC0/HPC1 (High Performance Coherent): 128-bit AXI from PL to PS
//! - LPD (Low Power Domain): 64-bit AXI from PS LPD to PL

use core::fmt;

/// CRL_APB base address (Clock Reset LPD APB).
const CRL_APB_BASE: u64 = 0xFF5E_0000;

/// FPGA_RST_CTRL register offset within CRL_APB.
const FPGA_RST_CTRL_OFFSET: u32 = 0x0100;

/// PL clock control register offsets (PL0..PL3).
const PL_CLK_CTRL_OFFSETS: [u32; 4] = [0x00C0, 0x00C4, 0x00C8, 0x00CC];

/// PS-PL AXI interface type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsPlInterface {
    /// High Performance Master port 0 (PS -> PL, 128-bit).
    Hpm0,
    /// High Performance Master port 1 (PS -> PL, 128-bit).
    Hpm1,
    /// High Performance Coherent port 0 (PL -> PS, 128-bit).
    Hpc0,
    /// High Performance Coherent port 1 (PL -> PS, 128-bit).
    Hpc1,
    /// Low Power Domain port (PS LPD -> PL, 64-bit).
    Lpd,
}

/// FPGA fabric clock index (FCLK0..FCLK3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpgaClock {
    /// PL fabric clock 0.
    Fclk0,
    /// PL fabric clock 1.
    Fclk1,
    /// PL fabric clock 2.
    Fclk2,
    /// PL fabric clock 3.
    Fclk3,
}

impl FpgaClock {
    /// Clock index (0-3).
    pub fn index(&self) -> usize {
        match self {
            FpgaClock::Fclk0 => 0,
            FpgaClock::Fclk1 => 1,
            FpgaClock::Fclk2 => 2,
            FpgaClock::Fclk3 => 3,
        }
    }
}

/// PL reset line index (0-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlReset {
    /// PL reset 0.
    Reset0,
    /// PL reset 1.
    Reset1,
    /// PL reset 2.
    Reset2,
    /// PL reset 3.
    Reset3,
}

impl PlReset {
    /// Reset bit position in FPGA_RST_CTRL register.
    pub fn bit_position(&self) -> u32 {
        match self {
            PlReset::Reset0 => 0,
            PlReset::Reset1 => 1,
            PlReset::Reset2 => 2,
            PlReset::Reset3 => 3,
        }
    }
}

/// Zynq platform error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZynqError {
    /// CRL_APB register access failed.
    RegisterAccess,
    /// Requested frequency cannot be achieved within 1% tolerance.
    FrequencyOutOfRange {
        requested_mhz: u32,
        closest_mhz: u32,
    },
    /// Interface is not present in the DTB / not enabled.
    InterfaceNotPresent,
    /// Reset pulse duration too short.
    ResetTooShort,
}

impl fmt::Display for ZynqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZynqError::RegisterAccess => write!(f, "CRL_APB register access failed"),
            ZynqError::FrequencyOutOfRange {
                requested_mhz,
                closest_mhz,
            } => write!(
                f,
                "frequency {requested_mhz} MHz not achievable, closest {closest_mhz} MHz"
            ),
            ZynqError::InterfaceNotPresent => write!(f, "PS-PL interface not present"),
            ZynqError::ResetTooShort => write!(f, "reset pulse too short"),
        }
    }
}

/// PS-PL interface state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceState {
    /// Interface is disabled.
    Disabled,
    /// Interface is enabled and operational.
    Enabled,
}

/// Clock configuration state.
#[derive(Debug, Clone, Copy)]
pub struct ClockConfig {
    /// Requested frequency in MHz.
    pub requested_mhz: u32,
    /// Actual configured frequency in MHz.
    pub actual_mhz: u32,
    /// Clock divider value.
    pub divider: u16,
    /// Whether the clock is enabled.
    pub enabled: bool,
}

/// Zynq UltraScale+ platform support package.
///
/// Manages PS-PL interfaces, FPGA fabric clocks, and PL resets.
/// All register access goes through the CRL_APB AXI-Lite region.
#[derive(Debug)]
pub struct ZynqPlatform {
    /// PS-PL interface states.
    interfaces: [InterfaceState; 5],
    /// FPGA clock configurations.
    clocks: [ClockConfig; 4],
    /// PL reset states (true = in reset).
    resets: [bool; 4],
    /// Reference clock frequency in MHz (PS_REF_CLK, typically 33.33 MHz).
    ref_clk_mhz: u32,
}

impl Default for ZynqPlatform {
    fn default() -> Self {
        Self::new(33)
    }
}

impl ZynqPlatform {
    /// CRL_APB base address.
    pub const CRL_APB_BASE: u64 = CRL_APB_BASE;

    /// Create a new Zynq platform support package.
    ///
    /// `ref_clk_mhz` is the PS reference clock frequency (typically 33 MHz).
    pub fn new(ref_clk_mhz: u32) -> Self {
        Self {
            interfaces: [InterfaceState::Disabled; 5],
            clocks: [ClockConfig {
                requested_mhz: 0,
                actual_mhz: 0,
                divider: 0,
                enabled: false,
            }; 4],
            resets: [true; 4], // All in reset initially
            ref_clk_mhz,
        }
    }

    /// Reference clock frequency.
    pub fn ref_clk_mhz(&self) -> u32 {
        self.ref_clk_mhz
    }

    /// Enable a PS-PL AXI interface.
    pub fn enable_interface(&mut self, iface: PsPlInterface) -> Result<(), ZynqError> {
        let idx = iface as usize;
        self.interfaces[idx] = InterfaceState::Enabled;
        Ok(())
    }

    /// Disable a PS-PL AXI interface.
    pub fn disable_interface(&mut self, iface: PsPlInterface) -> Result<(), ZynqError> {
        let idx = iface as usize;
        self.interfaces[idx] = InterfaceState::Disabled;
        Ok(())
    }

    /// Get the state of a PS-PL interface.
    pub fn interface_state(&self, iface: PsPlInterface) -> InterfaceState {
        self.interfaces[iface as usize]
    }

    /// Configure an FPGA fabric clock.
    ///
    /// Calculates the closest achievable frequency using PLL dividers.
    /// The actual frequency must be within 1% of the requested value.
    pub fn configure_clock(
        &mut self,
        clock: FpgaClock,
        requested_mhz: u32,
    ) -> Result<ClockConfig, ZynqError> {
        if requested_mhz == 0 || self.ref_clk_mhz == 0 {
            return Err(ZynqError::FrequencyOutOfRange {
                requested_mhz,
                closest_mhz: 0,
            });
        }

        // Simplified PLL model: PLL output = ref_clk * PLL_MULT
        // FCLK = PLL_output / divider
        // For this stub, assume PLL output is 1500 MHz (typical for Zynq US+)
        let pll_output_mhz: u32 = 1500;

        // Find best divider
        let ideal_divider = pll_output_mhz / requested_mhz;
        let divider = ideal_divider.clamp(1, 63) as u16;
        let actual_mhz = pll_output_mhz / divider as u32;

        // Check 1% tolerance
        let diff = actual_mhz.abs_diff(requested_mhz);
        let tolerance = requested_mhz / 100; // 1%
        if diff > tolerance.max(1) {
            return Err(ZynqError::FrequencyOutOfRange {
                requested_mhz,
                closest_mhz: actual_mhz,
            });
        }

        let config = ClockConfig {
            requested_mhz,
            actual_mhz,
            divider,
            enabled: true,
        };
        self.clocks[clock.index()] = config;

        // In real hardware: write PL_CLK_CTRL register
        let _offset = PL_CLK_CTRL_OFFSETS[clock.index()];

        Ok(config)
    }

    /// Enable a previously configured FPGA clock.
    pub fn enable_clock(&mut self, clock: FpgaClock) -> Result<(), ZynqError> {
        self.clocks[clock.index()].enabled = true;
        Ok(())
    }

    /// Disable an FPGA clock.
    pub fn disable_clock(&mut self, clock: FpgaClock) -> Result<(), ZynqError> {
        self.clocks[clock.index()].enabled = false;
        Ok(())
    }

    /// Get clock configuration.
    pub fn clock_config(&self, clock: FpgaClock) -> &ClockConfig {
        &self.clocks[clock.index()]
    }

    /// Assert a PL reset line (hold peripheral in reset).
    pub fn assert_reset(&mut self, reset: PlReset) -> Result<(), ZynqError> {
        self.resets[reset.bit_position() as usize] = true;
        // In real hardware: set bit in FPGA_RST_CTRL
        let _bit = 1u32 << reset.bit_position();
        Ok(())
    }

    /// Deassert a PL reset line (release peripheral from reset).
    pub fn deassert_reset(&mut self, reset: PlReset) -> Result<(), ZynqError> {
        self.resets[reset.bit_position() as usize] = false;
        Ok(())
    }

    /// Assert and deassert reset (pulse).
    ///
    /// `hold_cycles` is the minimum number of clock cycles to hold reset.
    /// Must be at least 16 for most Zynq peripherals.
    pub fn reset_pulse(&mut self, reset: PlReset, hold_cycles: u32) -> Result<(), ZynqError> {
        if hold_cycles < 16 {
            return Err(ZynqError::ResetTooShort);
        }
        self.assert_reset(reset)?;
        // In real hardware: busy-wait or timer delay for hold_cycles
        self.deassert_reset(reset)?;
        Ok(())
    }

    /// Check if a PL reset line is asserted.
    pub fn is_in_reset(&self, reset: PlReset) -> bool {
        self.resets[reset.bit_position() as usize]
    }

    /// CRL_APB register offset for FPGA_RST_CTRL.
    pub fn rst_ctrl_offset() -> u32 {
        FPGA_RST_CTRL_OFFSET
    }

    /// CRL_APB register offset for a PL clock control register.
    pub fn clk_ctrl_offset(clock: FpgaClock) -> u32 {
        PL_CLK_CTRL_OFFSETS[clock.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_new_default() {
        let platform = ZynqPlatform::default();
        assert_eq!(platform.ref_clk_mhz(), 33);
        assert_eq!(
            platform.interface_state(PsPlInterface::Hpm0),
            InterfaceState::Disabled
        );
        assert!(platform.is_in_reset(PlReset::Reset0));
    }

    #[test]
    fn test_enable_disable_interface() {
        let mut platform = ZynqPlatform::new(33);
        platform.enable_interface(PsPlInterface::Hpm0).unwrap();
        assert_eq!(
            platform.interface_state(PsPlInterface::Hpm0),
            InterfaceState::Enabled
        );
        platform.disable_interface(PsPlInterface::Hpm0).unwrap();
        assert_eq!(
            platform.interface_state(PsPlInterface::Hpm0),
            InterfaceState::Disabled
        );
    }

    #[test]
    fn test_all_interfaces() {
        let mut platform = ZynqPlatform::new(33);
        platform.enable_interface(PsPlInterface::Hpm0).unwrap();
        platform.enable_interface(PsPlInterface::Hpm1).unwrap();
        platform.enable_interface(PsPlInterface::Hpc0).unwrap();
        platform.enable_interface(PsPlInterface::Hpc1).unwrap();
        platform.enable_interface(PsPlInterface::Lpd).unwrap();
        for iface in [
            PsPlInterface::Hpm0,
            PsPlInterface::Hpm1,
            PsPlInterface::Hpc0,
            PsPlInterface::Hpc1,
            PsPlInterface::Lpd,
        ] {
            assert_eq!(platform.interface_state(iface), InterfaceState::Enabled);
        }
    }

    #[test]
    fn test_configure_clock_100mhz() {
        let mut platform = ZynqPlatform::new(33);
        let config = platform.configure_clock(FpgaClock::Fclk0, 100).unwrap();
        assert!(config.enabled);
        // PLL=1500, divider=15, actual=100
        assert_eq!(config.actual_mhz, 100);
        assert_eq!(config.divider, 15);
    }

    #[test]
    fn test_configure_clock_250mhz() {
        let mut platform = ZynqPlatform::new(33);
        let config = platform.configure_clock(FpgaClock::Fclk1, 250).unwrap();
        // PLL=1500, divider=6, actual=250
        assert_eq!(config.actual_mhz, 250);
    }

    #[test]
    fn test_configure_clock_zero() {
        let mut platform = ZynqPlatform::new(33);
        assert!(platform.configure_clock(FpgaClock::Fclk0, 0).is_err());
    }

    #[test]
    fn test_enable_disable_clock() {
        let mut platform = ZynqPlatform::new(33);
        platform.configure_clock(FpgaClock::Fclk0, 100).unwrap();
        assert!(platform.clock_config(FpgaClock::Fclk0).enabled);

        platform.disable_clock(FpgaClock::Fclk0).unwrap();
        assert!(!platform.clock_config(FpgaClock::Fclk0).enabled);

        platform.enable_clock(FpgaClock::Fclk0).unwrap();
        assert!(platform.clock_config(FpgaClock::Fclk0).enabled);
    }

    #[test]
    fn test_assert_deassert_reset() {
        let mut platform = ZynqPlatform::new(33);
        assert!(platform.is_in_reset(PlReset::Reset0)); // Initially in reset

        platform.deassert_reset(PlReset::Reset0).unwrap();
        assert!(!platform.is_in_reset(PlReset::Reset0));

        platform.assert_reset(PlReset::Reset0).unwrap();
        assert!(platform.is_in_reset(PlReset::Reset0));
    }

    #[test]
    fn test_reset_pulse() {
        let mut platform = ZynqPlatform::new(33);
        platform.deassert_reset(PlReset::Reset1).unwrap();
        platform.reset_pulse(PlReset::Reset1, 32).unwrap();
        assert!(!platform.is_in_reset(PlReset::Reset1)); // Deasserted after pulse
    }

    #[test]
    fn test_reset_pulse_too_short() {
        let mut platform = ZynqPlatform::new(33);
        assert_eq!(
            platform.reset_pulse(PlReset::Reset0, 8),
            Err(ZynqError::ResetTooShort)
        );
    }

    #[test]
    fn test_all_resets() {
        let mut platform = ZynqPlatform::new(33);
        for reset in [PlReset::Reset0, PlReset::Reset1, PlReset::Reset2, PlReset::Reset3] {
            assert!(platform.is_in_reset(reset));
            platform.deassert_reset(reset).unwrap();
            assert!(!platform.is_in_reset(reset));
        }
    }

    #[test]
    fn test_clock_index() {
        assert_eq!(FpgaClock::Fclk0.index(), 0);
        assert_eq!(FpgaClock::Fclk1.index(), 1);
        assert_eq!(FpgaClock::Fclk2.index(), 2);
        assert_eq!(FpgaClock::Fclk3.index(), 3);
    }

    #[test]
    fn test_reset_bit_position() {
        assert_eq!(PlReset::Reset0.bit_position(), 0);
        assert_eq!(PlReset::Reset1.bit_position(), 1);
        assert_eq!(PlReset::Reset2.bit_position(), 2);
        assert_eq!(PlReset::Reset3.bit_position(), 3);
    }

    #[test]
    fn test_register_offsets() {
        assert_eq!(ZynqPlatform::rst_ctrl_offset(), 0x0100);
        assert_eq!(ZynqPlatform::clk_ctrl_offset(FpgaClock::Fclk0), 0x00C0);
        assert_eq!(ZynqPlatform::clk_ctrl_offset(FpgaClock::Fclk3), 0x00CC);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", ZynqError::RegisterAccess),
            "CRL_APB register access failed"
        );
        assert_eq!(
            format!(
                "{}",
                ZynqError::FrequencyOutOfRange {
                    requested_mhz: 300,
                    closest_mhz: 250,
                }
            ),
            "frequency 300 MHz not achievable, closest 250 MHz"
        );
        assert_eq!(
            format!("{}", ZynqError::InterfaceNotPresent),
            "PS-PL interface not present"
        );
        assert_eq!(format!("{}", ZynqError::ResetTooShort), "reset pulse too short");
    }

    #[test]
    fn test_interface_state_variants() {
        assert_ne!(InterfaceState::Disabled, InterfaceState::Enabled);
    }
}
