// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Sony IMX477 image sensor driver.
//!
//! The IMX477 is a 12.3-megapixel (4056x3040) back-illuminated stacked
//! CMOS sensor used in the Raspberry Pi HQ Camera. It supports MIPI CSI-2
//! with 2 data lanes at up to 912 Mbps/lane and features full 12-bit ADC.
//!
//! This module provides register definitions, resolution mode tables,
//! and I2C configuration helpers.
//!
//! # References
//!
//! - Sony IMX477 datasheet (under NDA — register map derived from Linux driver)
//! - Raspberry Pi HQ Camera documentation

#![allow(dead_code)]

use super::sensor::{self, IMX477_I2C_ADDR};
use smallaios_kernel::hal::{HalError, I2cController};

// ---------------------------------------------------------------------------
// Register addresses (16-bit, big-endian)
// ---------------------------------------------------------------------------

/// Model ID.
const MODEL_ID_H: u16 = 0x0016;
const MODEL_ID_L: u16 = 0x0017;

/// Mode select: 0=standby, 1=streaming.
const MODE_SELECT: u16 = 0x0100;

/// Software reset.
const SOFTWARE_RESET: u16 = 0x0103;

/// Frame/line length.
const FRAME_LENGTH_H: u16 = 0x0340;
const FRAME_LENGTH_L: u16 = 0x0341;
const LINE_LENGTH_H: u16 = 0x0342;
const LINE_LENGTH_L: u16 = 0x0343;

/// Output size.
const X_OUTPUT_SIZE_H: u16 = 0x034C;
const X_OUTPUT_SIZE_L: u16 = 0x034D;
const Y_OUTPUT_SIZE_H: u16 = 0x034E;
const Y_OUTPUT_SIZE_L: u16 = 0x034F;

/// Crop region.
const X_ADDR_START_H: u16 = 0x0344;
const X_ADDR_START_L: u16 = 0x0345;
const Y_ADDR_START_H: u16 = 0x0346;
const Y_ADDR_START_L: u16 = 0x0347;
const X_ADDR_END_H: u16 = 0x0348;
const X_ADDR_END_L: u16 = 0x0349;
const Y_ADDR_END_H: u16 = 0x034A;
const Y_ADDR_END_L: u16 = 0x034B;

/// Exposure.
const COARSE_INTEGRATION_TIME_H: u16 = 0x0202;
const COARSE_INTEGRATION_TIME_L: u16 = 0x0203;

/// Analog gain.
const ANALOG_GAIN_H: u16 = 0x0204;
const ANALOG_GAIN_L: u16 = 0x0205;

/// Digital gain.
const DIGITAL_GAIN_H: u16 = 0x020E;
const DIGITAL_GAIN_L: u16 = 0x020F;

/// CSI data format (bits per pixel).
const CSI_DATA_FORMAT_H: u16 = 0x0112;
const CSI_DATA_FORMAT_L: u16 = 0x0113;

/// CSI lane mode.
const CSI_LANE_MODE: u16 = 0x0114;

/// Binning mode.
const BINNING_MODE: u16 = 0x0900;
const BINNING_TYPE_H: u16 = 0x0901;
const BINNING_TYPE_V: u16 = 0x0902;

/// PLL configuration.
const IVT_PXCK_DIV: u16 = 0x0301;
const IVT_SYCK_DIV: u16 = 0x0303;
const IVT_PREPLLCK_DIV: u16 = 0x0305;
const IVT_PLL_MPY_H: u16 = 0x0306;
const IVT_PLL_MPY_L: u16 = 0x0307;
const IOP_PXCK_DIV: u16 = 0x0309;
const IOP_SYCK_DIV: u16 = 0x030B;
const IOP_PREPLLCK_DIV: u16 = 0x030D;
const IOP_PLL_MPY_H: u16 = 0x030E;
const IOP_PLL_MPY_L: u16 = 0x030F;

/// Test pattern.
const TEST_PATTERN_H: u16 = 0x0600;
const TEST_PATTERN_L: u16 = 0x0601;

// ---------------------------------------------------------------------------
// Resolution modes
// ---------------------------------------------------------------------------

/// Supported IMX477 output resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imx477Resolution {
    /// 1332x990 (2x2 binned from center).
    Binned2x2,
    /// 2028x1080 (cropped for widescreen).
    Wide1080,
    /// 2028x1520 (2x2 binned full FOV).
    BinnedFull,
    /// 4056x3040 (full 12.3 MP).
    Full12mp,
}

impl Imx477Resolution {
    /// Return (width, height) for this mode.
    pub fn dimensions(self) -> (u16, u16) {
        match self {
            Self::Binned2x2 => (1332, 990),
            Self::Wide1080 => (2028, 1080),
            Self::BinnedFull => (2028, 1520),
            Self::Full12mp => (4056, 3040),
        }
    }

    /// Return the line length (pixels per line including blanking).
    pub fn line_length(self) -> u16 {
        match self {
            Self::Binned2x2 => 5400,
            Self::Wide1080 => 5400,
            Self::BinnedFull => 5400,
            Self::Full12mp => 5400,
        }
    }

    /// Return the frame length (lines per frame including blanking).
    pub fn frame_length(self) -> u16 {
        match self {
            Self::Binned2x2 => 1058,
            Self::Wide1080 => 1150,
            Self::BinnedFull => 1598,
            Self::Full12mp => 3118,
        }
    }

    /// Whether this mode uses binning.
    pub fn uses_binning(self) -> bool {
        matches!(self, Self::Binned2x2 | Self::BinnedFull)
    }
}

/// A (register_address, value) pair for sensor configuration.
pub type RegEntry = (u16, u8);

/// Build the output size register writes for a given resolution.
pub fn build_output_size_regs(res: Imx477Resolution) -> [RegEntry; 4] {
    let (w, h) = res.dimensions();
    [
        (X_OUTPUT_SIZE_H, (w >> 8) as u8),
        (X_OUTPUT_SIZE_L, (w & 0xFF) as u8),
        (Y_OUTPUT_SIZE_H, (h >> 8) as u8),
        (Y_OUTPUT_SIZE_L, (h & 0xFF) as u8),
    ]
}

/// Build the frame/line length register writes.
pub fn build_frame_regs(res: Imx477Resolution) -> [RegEntry; 4] {
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
pub fn build_binning_regs(res: Imx477Resolution) -> [RegEntry; 3] {
    if res.uses_binning() {
        [
            (BINNING_MODE, 0x01),   // Enable binning.
            (BINNING_TYPE_H, 0x02), // 2x2 horizontal.
            (BINNING_TYPE_V, 0x02), // 2x2 vertical.
        ]
    } else {
        [
            (BINNING_MODE, 0x00),   // Disable binning.
            (BINNING_TYPE_H, 0x01), // No binning.
            (BINNING_TYPE_V, 0x01), // No binning.
        ]
    }
}

/// IMX477 data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imx477DataFormat {
    /// 10-bit RAW.
    Raw10,
    /// 12-bit RAW.
    Raw12,
}

/// Build CSI data format register writes.
pub fn build_data_format_regs(fmt: Imx477DataFormat) -> [RegEntry; 2] {
    let (hi, lo) = match fmt {
        Imx477DataFormat::Raw10 => (0x0A, 0x0A),
        Imx477DataFormat::Raw12 => (0x0C, 0x0C),
    };
    [(CSI_DATA_FORMAT_H, hi), (CSI_DATA_FORMAT_L, lo)]
}

/// Build crop region register writes for a given resolution.
///
/// Full sensor is 4056x3040. Crop is centered.
pub fn build_crop_regs(res: Imx477Resolution) -> [RegEntry; 8] {
    let (xs, ys, xe, ye) = match res {
        Imx477Resolution::Binned2x2 => {
            // Binned 2x2: use central 2664x1980 of the sensor.
            let xs = (4056 - 2664) / 2;
            let ys = (3040 - 1980) / 2;
            let xe = xs + 2664 - 1;
            let ye = ys + 1980 - 1;
            (xs, ys, xe, ye)
        }
        Imx477Resolution::Wide1080 => {
            // Widescreen crop: full width, center height.
            let ys = (3040 - 1080) / 2;
            (0u16, ys, 4055, ys + 1080 - 1)
        }
        Imx477Resolution::BinnedFull => {
            // Full FOV binned: use entire sensor area.
            (0, 0, 4055, 3039)
        }
        Imx477Resolution::Full12mp => {
            // Full resolution: entire sensor.
            (0, 0, 4055, 3039)
        }
    };
    // xs, ys, xe, ye are already bound from the match above.
    [
        (X_ADDR_START_H, (xs >> 8) as u8),
        (X_ADDR_START_L, (xs & 0xFF) as u8),
        (Y_ADDR_START_H, (ys >> 8) as u8),
        (Y_ADDR_START_L, (ys & 0xFF) as u8),
        (X_ADDR_END_H, (xe >> 8) as u8),
        (X_ADDR_END_L, (xe & 0xFF) as u8),
        (Y_ADDR_END_H, (ye >> 8) as u8),
        (Y_ADDR_END_L, (ye & 0xFF) as u8),
    ]
}

// ---------------------------------------------------------------------------
// I2C sensor control
// ---------------------------------------------------------------------------

/// Write a sequence of register entries to the IMX477 via I2C.
pub fn write_regs<I: I2cController>(i2c: &mut I, regs: &[RegEntry]) -> Result<(), HalError> {
    for &(addr, val) in regs {
        sensor::write_reg8(i2c, IMX477_I2C_ADDR, addr, val)?;
    }
    Ok(())
}

/// Perform a software reset.
pub fn soft_reset<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, IMX477_I2C_ADDR, SOFTWARE_RESET, 0x01)
}

/// Put the IMX477 into standby mode.
pub fn standby<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, IMX477_I2C_ADDR, MODE_SELECT, 0x00)
}

/// Start streaming.
pub fn start_streaming<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, IMX477_I2C_ADDR, MODE_SELECT, 0x01)
}

/// Read the IMX477 model ID (should be 0x0477).
pub fn read_model_id<I: I2cController>(i2c: &mut I) -> Result<u16, HalError> {
    let high = sensor::read_reg8(i2c, IMX477_I2C_ADDR, MODEL_ID_H)?;
    let low = sensor::read_reg8(i2c, IMX477_I2C_ADDR, MODEL_ID_L)?;
    Ok((high as u16) << 8 | low as u16)
}

/// Set analog gain (code 0-978, gain = 1024 / (1024 - code)).
pub fn set_analog_gain<I: I2cController>(i2c: &mut I, gain_code: u16) -> Result<(), HalError> {
    if gain_code > 978 {
        return Err(HalError::InvalidConfig);
    }
    sensor::write_reg8(i2c, IMX477_I2C_ADDR, ANALOG_GAIN_H, (gain_code >> 8) as u8)?;
    sensor::write_reg8(
        i2c,
        IMX477_I2C_ADDR,
        ANALOG_GAIN_L,
        (gain_code & 0xFF) as u8,
    )
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
    sensor::write_reg8(i2c, IMX477_I2C_ADDR, TEST_PATTERN_H, 0x00)?;
    sensor::write_reg8(i2c, IMX477_I2C_ADDR, TEST_PATTERN_L, pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_dimensions() {
        assert_eq!(Imx477Resolution::Binned2x2.dimensions(), (1332, 990));
        assert_eq!(Imx477Resolution::Wide1080.dimensions(), (2028, 1080));
        assert_eq!(Imx477Resolution::BinnedFull.dimensions(), (2028, 1520));
        assert_eq!(Imx477Resolution::Full12mp.dimensions(), (4056, 3040));
    }

    #[test]
    fn test_resolution_frame_length_gte_height() {
        for res in [
            Imx477Resolution::Binned2x2,
            Imx477Resolution::Wide1080,
            Imx477Resolution::BinnedFull,
            Imx477Resolution::Full12mp,
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
        assert!(Imx477Resolution::Binned2x2.uses_binning());
        assert!(Imx477Resolution::BinnedFull.uses_binning());
        assert!(!Imx477Resolution::Wide1080.uses_binning());
        assert!(!Imx477Resolution::Full12mp.uses_binning());
    }

    #[test]
    fn test_build_output_size_regs_full() {
        let regs = build_output_size_regs(Imx477Resolution::Full12mp);
        // 4056 = 0x0FD8
        assert_eq!(regs[0], (X_OUTPUT_SIZE_H, 0x0F));
        assert_eq!(regs[1], (X_OUTPUT_SIZE_L, 0xD8));
        // 3040 = 0x0BE0
        assert_eq!(regs[2], (Y_OUTPUT_SIZE_H, 0x0B));
        assert_eq!(regs[3], (Y_OUTPUT_SIZE_L, 0xE0));
    }

    #[test]
    fn test_build_binning_regs_enabled() {
        let regs = build_binning_regs(Imx477Resolution::Binned2x2);
        assert_eq!(regs[0], (BINNING_MODE, 0x01));
        assert_eq!(regs[1], (BINNING_TYPE_H, 0x02));
        assert_eq!(regs[2], (BINNING_TYPE_V, 0x02));
    }

    #[test]
    fn test_build_binning_regs_disabled() {
        let regs = build_binning_regs(Imx477Resolution::Full12mp);
        assert_eq!(regs[0], (BINNING_MODE, 0x00));
    }

    #[test]
    fn test_build_data_format_regs_raw12() {
        let regs = build_data_format_regs(Imx477DataFormat::Raw12);
        assert_eq!(regs[0], (CSI_DATA_FORMAT_H, 0x0C));
        assert_eq!(regs[1], (CSI_DATA_FORMAT_L, 0x0C));
    }

    #[test]
    fn test_build_crop_regs_full() {
        let regs = build_crop_regs(Imx477Resolution::Full12mp);
        // Full resolution: start at (0,0), end at (4055, 3039).
        let xs = (regs[0].1 as u16) << 8 | regs[1].1 as u16;
        let ys = (regs[2].1 as u16) << 8 | regs[3].1 as u16;
        let xe = (regs[4].1 as u16) << 8 | regs[5].1 as u16;
        let ye = (regs[6].1 as u16) << 8 | regs[7].1 as u16;
        assert_eq!(xs, 0);
        assert_eq!(ys, 0);
        assert_eq!(xe, 4055);
        assert_eq!(ye, 3039);
    }

    #[test]
    fn test_build_frame_regs() {
        let regs = build_frame_regs(Imx477Resolution::Full12mp);
        let fl = Imx477Resolution::Full12mp.frame_length();
        assert_eq!(regs[0].1, (fl >> 8) as u8);
        assert_eq!(regs[1].1, (fl & 0xFF) as u8);
    }
}
