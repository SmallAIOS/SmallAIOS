// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! OmniVision OV5640 image sensor driver.
//!
//! The OV5640 is a 5-megapixel (2592x1944) CMOS image sensor supporting
//! both MIPI CSI-2 and parallel DVP output interfaces. It is commonly
//! found on evaluation boards and embedded platforms.
//!
//! This module provides register definitions, initialization sequences,
//! and resolution mode tables for configuring the sensor via I2C.
//!
//! # References
//!
//! - OV5640 datasheet (OmniVision)
//! - OV5640 Application Note (register programming guide)

#![allow(dead_code)]

use super::sensor::{self, OV5640_I2C_ADDR};
use smallaios_kernel::hal::{HalError, I2cController};

// ---------------------------------------------------------------------------
// Register addresses (16-bit)
// ---------------------------------------------------------------------------

/// System control registers.
const SYSTEM_RESET00: u16 = 0x3000;
const SYSTEM_RESET02: u16 = 0x3002;
const SYSTEM_CTROL0: u16 = 0x3008;
const CHIP_ID_HIGH: u16 = 0x300A;
const CHIP_ID_LOW: u16 = 0x300B;
const PAD_OUTPUT_EN01: u16 = 0x3017;
const PAD_OUTPUT_EN02: u16 = 0x3018;

/// PLL control.
const SC_PLL_CTRL0: u16 = 0x3034;
const SC_PLL_CTRL1: u16 = 0x3035;
const SC_PLL_CTRL2: u16 = 0x3036;
const SC_PLL_CTRL3: u16 = 0x3037;

/// Timing control.
const TIMING_HS_H: u16 = 0x3800;
const TIMING_HS_L: u16 = 0x3801;
const TIMING_VS_H: u16 = 0x3802;
const TIMING_VS_L: u16 = 0x3803;
const TIMING_HW_H: u16 = 0x3804;
const TIMING_HW_L: u16 = 0x3805;
const TIMING_VH_H: u16 = 0x3806;
const TIMING_VH_L: u16 = 0x3807;
const TIMING_DVPHO_H: u16 = 0x3808;
const TIMING_DVPHO_L: u16 = 0x3809;
const TIMING_DVPVO_H: u16 = 0x380A;
const TIMING_DVPVO_L: u16 = 0x380B;
const TIMING_HTS_H: u16 = 0x380C;
const TIMING_HTS_L: u16 = 0x380D;
const TIMING_VTS_H: u16 = 0x380E;
const TIMING_VTS_L: u16 = 0x380F;
const TIMING_TC_REG20: u16 = 0x3820;
const TIMING_TC_REG21: u16 = 0x3821;

/// AEC/AGC control.
const AEC_CTRL00: u16 = 0x3503;
const AEC_PK_EXPOSURE_H: u16 = 0x3500;
const AEC_PK_EXPOSURE_M: u16 = 0x3501;
const AEC_PK_EXPOSURE_L: u16 = 0x3502;
const AEC_PK_MANUAL: u16 = 0x3503;
const AEC_PK_REAL_GAIN: u16 = 0x350A;

/// ISP control.
const ISP_CTRL00: u16 = 0x5000;
const ISP_CTRL01: u16 = 0x5001;
const ISP_MUX_CTRL: u16 = 0x501F;

/// Format control.
const FORMAT_CTRL00: u16 = 0x4300;

/// MIPI control.
const MIPI_CTRL00: u16 = 0x4800;
const MIPI_LP_GPIO: u16 = 0x4805;
const MIPI_LANE_SEL: u16 = 0x3621;

/// Software standby/streaming.
const SW_STANDBY: u8 = 0x42;
const SW_STREAMING: u8 = 0x02;

// ---------------------------------------------------------------------------
// Resolution modes
// ---------------------------------------------------------------------------

/// Supported OV5640 output resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ov5640Resolution {
    /// 320x240 (QVGA).
    Qvga,
    /// 640x480 (VGA).
    Vga,
    /// 1280x720 (720p).
    Hd720,
    /// 1920x1080 (1080p).
    Hd1080,
    /// 2592x1944 (Full 5MP).
    Full5mp,
}

impl Ov5640Resolution {
    /// Return (width, height) for this mode.
    pub fn dimensions(self) -> (u16, u16) {
        match self {
            Self::Qvga => (320, 240),
            Self::Vga => (640, 480),
            Self::Hd720 => (1280, 720),
            Self::Hd1080 => (1920, 1080),
            Self::Full5mp => (2592, 1944),
        }
    }

    /// Return the default total horizontal pixel count (HTS) for this mode.
    pub fn hts(self) -> u16 {
        match self {
            Self::Qvga => 1896,
            Self::Vga => 1896,
            Self::Hd720 => 1892,
            Self::Hd1080 => 2500,
            Self::Full5mp => 2844,
        }
    }

    /// Return the default total vertical line count (VTS) for this mode.
    pub fn vts(self) -> u16 {
        match self {
            Self::Qvga => 984,
            Self::Vga => 984,
            Self::Hd720 => 740,
            Self::Hd1080 => 1120,
            Self::Full5mp => 1968,
        }
    }
}

/// A (register_address, value) pair for sensor initialization.
pub type RegEntry = (u16, u8);

/// Build the output size register writes for a given resolution.
pub fn build_output_size_regs(res: Ov5640Resolution) -> [RegEntry; 4] {
    let (w, h) = res.dimensions();
    [
        (TIMING_DVPHO_H, (w >> 8) as u8),
        (TIMING_DVPHO_L, (w & 0xFF) as u8),
        (TIMING_DVPVO_H, (h >> 8) as u8),
        (TIMING_DVPVO_L, (h & 0xFF) as u8),
    ]
}

/// Build the timing register writes (HTS and VTS) for a given resolution.
pub fn build_timing_regs(res: Ov5640Resolution) -> [RegEntry; 4] {
    let hts = res.hts();
    let vts = res.vts();
    [
        (TIMING_HTS_H, (hts >> 8) as u8),
        (TIMING_HTS_L, (hts & 0xFF) as u8),
        (TIMING_VTS_H, (vts >> 8) as u8),
        (TIMING_VTS_L, (vts & 0xFF) as u8),
    ]
}

/// Build the PLL configuration for MIPI CSI-2 output.
///
/// The OV5640 PLL generates the MIPI clock from a 24 MHz input:
/// `MIPI_CLK = (XVCLK / pll_pre_div) * pll_mult / pll_sys_div / mipi_div`
///
/// `pll_pre_div` is bits [2:0] of 0x3037, `pll_mult` is register 0x3036.
pub fn build_pll_regs_24mhz(target_lane_mbps: u16) -> [RegEntry; 4] {
    // Simplified PLL settings for common lane rates from 24 MHz XVCLK.
    let (ctrl0, ctrl1, ctrl2, ctrl3) = match target_lane_mbps {
        0..=300 => (0x08, 0x41, 0x48, 0x13),   // ~200 Mbps/lane
        301..=500 => (0x08, 0x21, 0x54, 0x13), // ~400 Mbps/lane
        501..=700 => (0x08, 0x11, 0x60, 0x13), // ~600 Mbps/lane
        _ => (0x08, 0x11, 0x70, 0x13),         // ~800 Mbps/lane
    };
    [
        (SC_PLL_CTRL0, ctrl0),
        (SC_PLL_CTRL1, ctrl1),
        (SC_PLL_CTRL2, ctrl2),
        (SC_PLL_CTRL3, ctrl3),
    ]
}

/// OV5640 output pixel format (ISP output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ov5640PixelFormat {
    /// YUV422 (YUYV).
    Yuv422,
    /// RGB565.
    Rgb565,
    /// Raw Bayer (10-bit).
    Raw10,
}

/// Build format control register writes.
pub fn build_format_regs(fmt: Ov5640PixelFormat) -> [RegEntry; 2] {
    let (fmt_ctrl, isp_mux) = match fmt {
        Ov5640PixelFormat::Yuv422 => (0x30, 0x00), // YUV422, ISP to YUV
        Ov5640PixelFormat::Rgb565 => (0x61, 0x01), // RGB565, ISP to RGB
        Ov5640PixelFormat::Raw10 => (0x00, 0x03),  // RAW, ISP bypass
    };
    [(FORMAT_CTRL00, fmt_ctrl), (ISP_MUX_CTRL, isp_mux)]
}

// ---------------------------------------------------------------------------
// Sensor configuration via I2C
// ---------------------------------------------------------------------------

/// Write a sequence of register entries to the OV5640 via I2C.
pub fn write_regs<I: I2cController>(i2c: &mut I, regs: &[RegEntry]) -> Result<(), HalError> {
    for &(addr, val) in regs {
        sensor::write_reg8(i2c, OV5640_I2C_ADDR, addr, val)?;
    }
    Ok(())
}

/// Put the OV5640 into software standby.
pub fn standby<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, OV5640_I2C_ADDR, SYSTEM_CTROL0, SW_STANDBY)
}

/// Start streaming (exit software standby).
pub fn start_streaming<I: I2cController>(i2c: &mut I) -> Result<(), HalError> {
    sensor::write_reg8(i2c, OV5640_I2C_ADDR, SYSTEM_CTROL0, SW_STREAMING)
}

/// Read the OV5640 chip ID (should be 0x5640).
pub fn read_chip_id<I: I2cController>(i2c: &mut I) -> Result<u16, HalError> {
    let high = sensor::read_reg8(i2c, OV5640_I2C_ADDR, CHIP_ID_HIGH)?;
    let low = sensor::read_reg8(i2c, OV5640_I2C_ADDR, CHIP_ID_LOW)?;
    Ok((high as u16) << 8 | low as u16)
}

/// Configure MIPI CSI-2 lane count (1 or 2 lanes).
pub fn set_mipi_lanes<I: I2cController>(i2c: &mut I, lanes: u8) -> Result<(), HalError> {
    if lanes != 1 && lanes != 2 {
        return Err(HalError::InvalidConfig);
    }
    // Bit 5 of MIPI_CTRL00: 0 = 2 lanes, 1 = 1 lane.
    let val = if lanes == 1 { 0x24 } else { 0x04 };
    sensor::write_reg8(i2c, OV5640_I2C_ADDR, MIPI_CTRL00, val)
}

/// Enable or disable the test pattern generator.
///
/// Pattern 0 = off, 1 = color bars, 2 = color bars with fade-to-gray.
pub fn set_test_pattern<I: I2cController>(i2c: &mut I, pattern: u8) -> Result<(), HalError> {
    let val = match pattern {
        0 => 0x00,
        1 => 0x80,
        2 => 0x82,
        _ => return Err(HalError::InvalidConfig),
    };
    sensor::write_reg8(i2c, OV5640_I2C_ADDR, 0x503D, val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_dimensions() {
        assert_eq!(Ov5640Resolution::Qvga.dimensions(), (320, 240));
        assert_eq!(Ov5640Resolution::Vga.dimensions(), (640, 480));
        assert_eq!(Ov5640Resolution::Hd720.dimensions(), (1280, 720));
        assert_eq!(Ov5640Resolution::Hd1080.dimensions(), (1920, 1080));
        assert_eq!(Ov5640Resolution::Full5mp.dimensions(), (2592, 1944));
    }

    #[test]
    fn test_resolution_hts_vts() {
        // HTS must be >= width, VTS must be >= height.
        for res in [
            Ov5640Resolution::Qvga,
            Ov5640Resolution::Vga,
            Ov5640Resolution::Hd720,
            Ov5640Resolution::Hd1080,
            Ov5640Resolution::Full5mp,
        ] {
            let (w, h) = res.dimensions();
            assert!(res.hts() >= w, "{:?}: HTS {} < width {}", res, res.hts(), w);
            assert!(
                res.vts() >= h,
                "{:?}: VTS {} < height {}",
                res,
                res.vts(),
                h
            );
        }
    }

    #[test]
    fn test_build_output_size_regs() {
        let regs = build_output_size_regs(Ov5640Resolution::Hd1080);
        // 1920 = 0x0780, 1080 = 0x0438
        assert_eq!(regs[0], (TIMING_DVPHO_H, 0x07));
        assert_eq!(regs[1], (TIMING_DVPHO_L, 0x80));
        assert_eq!(regs[2], (TIMING_DVPVO_H, 0x04));
        assert_eq!(regs[3], (TIMING_DVPVO_L, 0x38));
    }

    #[test]
    fn test_build_timing_regs_vga() {
        let regs = build_timing_regs(Ov5640Resolution::Vga);
        let hts = Ov5640Resolution::Vga.hts();
        let vts = Ov5640Resolution::Vga.vts();
        assert_eq!(regs[0].1, (hts >> 8) as u8);
        assert_eq!(regs[1].1, (hts & 0xFF) as u8);
        assert_eq!(regs[2].1, (vts >> 8) as u8);
        assert_eq!(regs[3].1, (vts & 0xFF) as u8);
    }

    #[test]
    fn test_build_pll_regs_ranges() {
        // Each lane speed range should produce valid PLL registers.
        let r200 = build_pll_regs_24mhz(200);
        let r400 = build_pll_regs_24mhz(400);
        let r600 = build_pll_regs_24mhz(600);
        let r800 = build_pll_regs_24mhz(800);
        // All should write to the same 4 PLL registers.
        assert_eq!(r200[0].0, SC_PLL_CTRL0);
        assert_eq!(r400[0].0, SC_PLL_CTRL0);
        assert_eq!(r600[0].0, SC_PLL_CTRL0);
        assert_eq!(r800[0].0, SC_PLL_CTRL0);
        // Higher speed should have higher multiplier.
        assert!(r800[2].1 >= r200[2].1);
    }

    #[test]
    fn test_build_format_regs_yuv422() {
        let regs = build_format_regs(Ov5640PixelFormat::Yuv422);
        assert_eq!(regs[0], (FORMAT_CTRL00, 0x30));
        assert_eq!(regs[1], (ISP_MUX_CTRL, 0x00));
    }

    #[test]
    fn test_build_format_regs_raw10() {
        let regs = build_format_regs(Ov5640PixelFormat::Raw10);
        assert_eq!(regs[0], (FORMAT_CTRL00, 0x00));
        assert_eq!(regs[1], (ISP_MUX_CTRL, 0x03));
    }

    #[test]
    fn test_build_output_size_qvga() {
        let regs = build_output_size_regs(Ov5640Resolution::Qvga);
        // 320 = 0x0140, 240 = 0x00F0
        assert_eq!(regs[0], (TIMING_DVPHO_H, 0x01));
        assert_eq!(regs[1], (TIMING_DVPHO_L, 0x40));
        assert_eq!(regs[2], (TIMING_DVPVO_H, 0x00));
        assert_eq!(regs[3], (TIMING_DVPVO_L, 0xF0));
    }
}
