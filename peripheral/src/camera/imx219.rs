// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Sony IMX219 image sensor driver.
//!
//! The IMX219 is an 8-megapixel (3280x2464) back-illuminated CMOS sensor
//! used in the Raspberry Pi Camera Module v2. It supports MIPI CSI-2
//! with 2 data lanes at up to 912 Mbps/lane.
//!
//! This module provides register definitions, initialization sequences,
//! and resolution mode tables for configuring the sensor via I2C.
//!
//! # References
//!
//! - Sony IMX219PQH5-C datasheet
//! - Raspberry Pi Camera Module v2 schematic

#![allow(dead_code)]

use super::sensor::{self, IMX219_I2C_ADDR};
use smallaios_kernel::hal::{HalError, I2cController};

// ---------------------------------------------------------------------------
// Register addresses (16-bit, big-endian)
// ---------------------------------------------------------------------------

/// Chip ID (model ID).
const MODEL_ID_H: u16 = 0x0000;
const MODEL_ID_L: u16 = 0x0001;

/// Frame format descriptors.
const FRAME_LENGTH_H: u16 = 0x0160;
const FRAME_LENGTH_L: u16 = 0x0161;
const LINE_LENGTH_H: u16 = 0x0162;
const LINE_LENGTH_L: u16 = 0x0163;

/// X/Y output size.
const X_OUTPUT_SIZE_H: u16 = 0x016C;
const X_OUTPUT_SIZE_L: u16 = 0x016D;
const Y_OUTPUT_SIZE_H: u16 = 0x016E;
const Y_OUTPUT_SIZE_L: u16 = 0x016F;

/// Binning mode.
const BINNING_MODE_H: u16 = 0x0174;
const BINNING_MODE_V: u16 = 0x0175;

/// Crop region.
const X_ADDR_START_H: u16 = 0x0164;
const X_ADDR_START_L: u16 = 0x0165;
const Y_ADDR_START_H: u16 = 0x0166;
const Y_ADDR_START_L: u16 = 0x0167;
const X_ADDR_END_H: u16 = 0x0168;
const X_ADDR_END_L: u16 = 0x0169;
const Y_ADDR_END_H: u16 = 0x016A;
const Y_ADDR_END_L: u16 = 0x016B;

/// Exposure and gain.
const COARSE_INTEGRATION_TIME_H: u16 = 0x015A;
const COARSE_INTEGRATION_TIME_L: u16 = 0x015B;
const ANALOG_GAIN: u16 = 0x0157;
const DIGITAL_GAIN_H: u16 = 0x0158;
const DIGITAL_GAIN_L: u16 = 0x0159;

/// Mode select: 0=standby, 1=streaming.
const MODE_SELECT: u16 = 0x0100;

/// Software reset.
const SOFTWARE_RESET: u16 = 0x0103;

/// CSI data format (bits per pixel).
const CSI_DATA_FORMAT_H: u16 = 0x018C;
const CSI_DATA_FORMAT_L: u16 = 0x018D;

/// CSI lane mode: 0x01 = 2 lanes, 0x03 = 4 lanes (IMX219 only supports 2).
const CSI_LANE_MODE: u16 = 0x0114;

/// DPHY control.
const DPHY_CTRL: u16 = 0x0128;

/// EXCK frequency (external clock, in MHz * 256).
const EXCK_FREQ_H: u16 = 0x012A;
const EXCK_FREQ_L: u16 = 0x012B;

/// PLL dividers.
const VTPXCK_DIV: u16 = 0x0301;
const VTSYCK_DIV: u16 = 0x0303;
const PREPLLCK_VT_DIV: u16 = 0x0304;
const PREPLLCK_OP_DIV: u16 = 0x0305;
const PLL_VT_MPY_H: u16 = 0x0306;
const PLL_VT_MPY_L: u16 = 0x0307;
const OPPXCK_DIV: u16 = 0x0309;
const OPSYCK_DIV: u16 = 0x030B;
const PLL_OP_MPY_H: u16 = 0x030C;
const PLL_OP_MPY_L: u16 = 0x030D;

/// Test pattern mode: 0=off, 1=solid color, 2=color bars, 3=grey ramp, 4=PN9.
const TEST_PATTERN_H: u16 = 0x0600;
const TEST_PATTERN_L: u16 = 0x0601;

// ---------------------------------------------------------------------------
// Resolution modes
// ---------------------------------------------------------------------------

/// Supported IMX219 output resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imx219Resolution {
    /// 640x480 (VGA, 2x2 binned).
    Vga,
    /// 1640x1232 (2x2 binned).
    Binned2x2,
    /// 1920x1080 (1080p, cropped from center).
    Hd1080,
    /// 3280x2464 (full resolution).
    Full8mp,
}

impl Imx219Resolution {
    /// Return (width, height) for this mode.
    pub fn dimensions(self) -> (u16, u16) {
        match self {
            Self::Vga => (640, 480),
            Self::Binned2x2 => (1640, 1232),
            Self::Hd1080 => (1920, 1080),
            Self::Full8mp => (3280, 2464),
        }
    }

    /// Return the line length (pixels per line including blanking).
    pub fn line_length(self) -> u16 {
        match self {
            Self::Vga => 3448,
            Self::Binned2x2 => 3448,
            Self::Hd1080 => 3448,
            Self::Full8mp => 3448,
        }
    }

    /// Return the frame length (lines per frame including blanking).
    pub fn frame_length(self) -> u16 {
        match self {
            Self::Vga => 600,
            Self::Binned2x2 => 1320,
            Self::Hd1080 => 1200,
            Self::Full8mp => 2528,
        }
    }

    /// Whether this mode uses binning.
    pub fn uses_binning(self) -> bool {
        matches!(self, Self::Vga | Self::Binned2x2)
    }
}

/// A (register_address, value) pair for sensor configuration.
pub type RegEntry = (u16, u8);

/// Build the output size register writes for a given resolution.
pub fn build_output_size_regs(res: Imx219Resolution) -> [RegEntry; 4] {
    let (w, h) = res.dimensions();
    [
        (X_OUTPUT_SIZE_H, (w >> 8) as u8),
        (X_OUTPUT_SIZE_L, (w & 0xFF) as u8),
        (Y_OUTPUT_SIZE_H, (h >> 8) as u8),
        (Y_OUTPUT_SIZE_L, (h & 0xFF) as u8),
    ]
}

/// Build the frame/line length register writes.
pub fn build_frame_regs(res: Imx219Resolution) -> [RegEntry; 4] {
    let fl = res.frame_length();
    let ll = res.line_length();
    [
        (FRAME_LENGTH_H, (fl >> 8) as u8),
        (FRAME_LENGTH_L, (fl & 0xFF) as u8),
        (LINE_LENGTH_H, (ll >> 8) as u8),
        (LINE_LENGTH_L, (ll & 0xFF) as u8),
    ]
}

/// Build the binning mode register writes.
pub fn build_binning_regs(res: Imx219Resolution) -> [RegEntry; 2] {
    let bin_val = if res.uses_binning() { 0x02 } else { 0x00 };
    [(BINNING_MODE_H, bin_val), (BINNING_MODE_V, bin_val)]
}

/// Build the crop region register writes for 1080p (center crop from 3280x2464).
pub fn build_crop_regs_1080p() -> [RegEntry; 8] {
    // Center crop: (680, 692) to (2599, 1771) for 1920x1080.
    [
        (X_ADDR_START_H, 0x02),
        (X_ADDR_START_L, 0xA8),
        (Y_ADDR_START_H, 0x02),
        (Y_ADDR_START_L, 0xB4),
        (X_ADDR_END_H, 0x0A),
        (X_ADDR_END_L, 0x27),
        (Y_ADDR_END_H, 0x06),
        (Y_ADDR_END_L, 0xEB),
    ]
}

/// IMX219 data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imx219DataFormat {
    /// 8-bit RAW.
    Raw8,
    /// 10-bit RAW.
    Raw10,
}

/// Build CSI data format register writes.
pub fn build_data_format_regs(fmt: Imx219DataFormat) -> [RegEntry; 2] {
    let (hi, lo) = match fmt {
        Imx219DataFormat::Raw8 => (0x08, 0x08),
        Imx219DataFormat::Raw10 => (0x0A, 0x0A),
    };
    [(CSI_DATA_FORMAT_H, hi), (CSI_DATA_FORMAT_L, lo)]
}

// ---------------------------------------------------------------------------
// I2C sensor control
// ---------------------------------------------------------------------------

/// Write a sequence of register entries to the IMX219 via I2C.
pub fn write_regs<I: I2cController>(i2c: &mut I, regs: &[RegEntry]) -> Result<(), HalError> {
    for &(addr, val) in regs {
        sensor::write_reg8(i2c, IMX219_I2C_ADDR, addr, val)?;
    }
    Ok(())
}

/// Perform a software reset.
pub fn soft_reset<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, IMX219_I2C_ADDR, SOFTWARE_RESET, 0x01)
}

/// Put the IMX219 into standby mode.
pub fn standby<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, IMX219_I2C_ADDR, MODE_SELECT, 0x00)
}

/// Start streaming.
pub fn start_streaming<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, IMX219_I2C_ADDR, MODE_SELECT, 0x01)
}

/// Read the IMX219 model ID (should be 0x0219).
pub fn read_model_id<I: I2cController>(i2c: &mut I) -> Result<u16, HalError> {
    let high = sensor::read_reg8(i2c, IMX219_I2C_ADDR, MODEL_ID_H)?;
    let low = sensor::read_reg8(i2c, IMX219_I2C_ADDR, MODEL_ID_L)?;
    Ok((high as u16) << 8 | low as u16)
}

/// Set the test pattern mode.
///
/// - 0: Off
/// - 1: Solid color
/// - 2: Color bars
/// - 3: Grey ramp
/// - 4: PN9
pub fn set_test_pattern<I: I2cController>(i2c: &mut I, pattern: u8) -> Result<(), HalError> {
    if pattern > 4 {
        return Err(HalError::InvalidConfig);
    }
    sensor::write_reg8(i2c, IMX219_I2C_ADDR, TEST_PATTERN_H, 0x00)?;
    sensor::write_reg8(i2c, IMX219_I2C_ADDR, TEST_PATTERN_L, pattern)
}

/// Set the analog gain (0-232 range, where gain = 256/(256-value)).
pub fn set_analog_gain<I: I2cController>(i2c: &mut I, gain: u8) -> Result<(), HalError> {
    if gain > 232 {
        return Err(HalError::InvalidConfig);
    }
    sensor::write_reg8(i2c, IMX219_I2C_ADDR, ANALOG_GAIN, gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_dimensions() {
        assert_eq!(Imx219Resolution::Vga.dimensions(), (640, 480));
        assert_eq!(Imx219Resolution::Binned2x2.dimensions(), (1640, 1232));
        assert_eq!(Imx219Resolution::Hd1080.dimensions(), (1920, 1080));
        assert_eq!(Imx219Resolution::Full8mp.dimensions(), (3280, 2464));
    }

    #[test]
    fn test_resolution_frame_length_gte_height() {
        for res in [
            Imx219Resolution::Vga,
            Imx219Resolution::Binned2x2,
            Imx219Resolution::Hd1080,
            Imx219Resolution::Full8mp,
        ] {
            let (_, h) = res.dimensions();
            assert!(
                res.frame_length() >= h,
                "{:?}: frame_length {} < height {}",
                res,
                res.frame_length(),
                h
            );
        }
    }

    #[test]
    fn test_resolution_uses_binning() {
        assert!(Imx219Resolution::Vga.uses_binning());
        assert!(Imx219Resolution::Binned2x2.uses_binning());
        assert!(!Imx219Resolution::Hd1080.uses_binning());
        assert!(!Imx219Resolution::Full8mp.uses_binning());
    }

    #[test]
    fn test_build_output_size_regs_1080p() {
        let regs = build_output_size_regs(Imx219Resolution::Hd1080);
        // 1920 = 0x0780
        assert_eq!(regs[0], (X_OUTPUT_SIZE_H, 0x07));
        assert_eq!(regs[1], (X_OUTPUT_SIZE_L, 0x80));
        // 1080 = 0x0438
        assert_eq!(regs[2], (Y_OUTPUT_SIZE_H, 0x04));
        assert_eq!(regs[3], (Y_OUTPUT_SIZE_L, 0x38));
    }

    #[test]
    fn test_build_frame_regs_full() {
        let regs = build_frame_regs(Imx219Resolution::Full8mp);
        let fl = Imx219Resolution::Full8mp.frame_length(); // 2528
        let ll = Imx219Resolution::Full8mp.line_length(); // 3448
        assert_eq!(regs[0].1, (fl >> 8) as u8);
        assert_eq!(regs[1].1, (fl & 0xFF) as u8);
        assert_eq!(regs[2].1, (ll >> 8) as u8);
        assert_eq!(regs[3].1, (ll & 0xFF) as u8);
    }

    #[test]
    fn test_build_binning_regs() {
        let binned = build_binning_regs(Imx219Resolution::Vga);
        assert_eq!(binned[0].1, 0x02);
        assert_eq!(binned[1].1, 0x02);

        let no_bin = build_binning_regs(Imx219Resolution::Full8mp);
        assert_eq!(no_bin[0].1, 0x00);
        assert_eq!(no_bin[1].1, 0x00);
    }

    #[test]
    fn test_build_data_format_regs_raw10() {
        let regs = build_data_format_regs(Imx219DataFormat::Raw10);
        assert_eq!(regs[0], (CSI_DATA_FORMAT_H, 0x0A));
        assert_eq!(regs[1], (CSI_DATA_FORMAT_L, 0x0A));
    }

    #[test]
    fn test_build_crop_regs_1080p() {
        let regs = build_crop_regs_1080p();
        // Should have 8 entries for X/Y start/end.
        assert_eq!(regs.len(), 8);
        // Start X should be > 0 (center crop offset).
        let x_start = (regs[0].1 as u16) << 8 | regs[1].1 as u16;
        assert!(x_start > 0);
    }
}
