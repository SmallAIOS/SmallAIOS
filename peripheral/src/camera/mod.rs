// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIPI CSI-2 camera peripheral drivers.
//!
//! Provides platform-specific CSI-2 receiver implementations for:
//! - NVIDIA Tegra VI (ARM MMIO)
//! - Broadcom Unicam (RPi CSI-2, ARM MMIO)
//! - FPGA CSI-2 receiver IP (generic over [`FpgaFabric`])
//!
//! Plus image sensor configuration drivers (OV5640, IMX219, IMX477)
//! configured via I2C, and a preprocessing module for format conversion.
//!
//! All receiver drivers implement the [`CsiReceiver`] trait from `smallaios_kernel::hal`.

#![allow(dead_code)]

pub mod broadcom_unicam;
pub mod fpga_csi;
pub mod imx219;
pub mod imx477;
pub mod ov5640;
pub mod preprocess;
pub mod sensor;
pub mod tegra_vi;

pub use smallaios_kernel::hal::{
    CsiConfig, CsiFrame, CsiIrqSource, CsiPixelFormat, CsiReceiver, HalError,
};

// ---------------------------------------------------------------------------
// Shared CSI-2 helpers
// ---------------------------------------------------------------------------

/// Maximum number of CSI-2 data lanes.
pub const MAX_CSI_LANES: u8 = 4;

/// Maximum number of frame buffers.
pub const MAX_FRAME_BUFFERS: u8 = 4;

/// Minimum frame width.
pub const MIN_FRAME_WIDTH: u16 = 16;

/// Maximum frame width.
pub const MAX_FRAME_WIDTH: u16 = 4096;

/// Minimum frame height.
pub const MIN_FRAME_HEIGHT: u16 = 16;

/// Maximum frame height.
pub const MAX_FRAME_HEIGHT: u16 = 4096;

/// Validate a CSI-2 configuration.
///
/// Checks that lanes, resolution, FPS, and buffer count are within valid ranges.
pub fn validate_csi_config(config: &CsiConfig) -> Result<(), HalError> {
    // Validate lane count: 1, 2, or 4.
    if config.lanes != 1 && config.lanes != 2 && config.lanes != 4 {
        return Err(HalError::InvalidConfig);
    }
    // Validate resolution.
    if config.width < MIN_FRAME_WIDTH || config.width > MAX_FRAME_WIDTH {
        return Err(HalError::InvalidConfig);
    }
    if config.height < MIN_FRAME_HEIGHT || config.height > MAX_FRAME_HEIGHT {
        return Err(HalError::InvalidConfig);
    }
    // Validate FPS.
    if config.fps == 0 || config.fps > 120 {
        return Err(HalError::InvalidConfig);
    }
    // Validate buffer count.
    if config.num_buffers == 0 || config.num_buffers > MAX_FRAME_BUFFERS {
        return Err(HalError::InvalidConfig);
    }
    Ok(())
}

/// Compute the byte size of a single frame buffer for the given config.
///
/// This accounts for the pixel format and resolution.
pub fn compute_frame_size(config: &CsiConfig) -> usize {
    let pixels = config.width as usize * config.height as usize;
    match config.pixel_format {
        CsiPixelFormat::Raw8 => pixels,
        CsiPixelFormat::Raw10 => (pixels * 10).div_ceil(8), // 10 bits per pixel, packed
        CsiPixelFormat::Raw12 => (pixels * 12).div_ceil(8), // 12 bits per pixel, packed
        CsiPixelFormat::Yuv422 => pixels * 2,               // 16 bits per pixel
        CsiPixelFormat::Rgb888 => pixels * 3,               // 24 bits per pixel
    }
}

/// Compute the required lane bandwidth in Mbps for a given config.
///
/// This helps determine if the requested configuration is feasible
/// for the configured lane speed.
pub fn required_bandwidth_mbps(config: &CsiConfig) -> u32 {
    let bits_per_pixel = match config.pixel_format {
        CsiPixelFormat::Raw8 => 8u32,
        CsiPixelFormat::Raw10 => 10,
        CsiPixelFormat::Raw12 => 12,
        CsiPixelFormat::Yuv422 => 16,
        CsiPixelFormat::Rgb888 => 24,
    };
    let pixels_per_frame = config.width as u32 * config.height as u32;
    let bits_per_second = pixels_per_frame as u64 * bits_per_pixel as u64 * config.fps as u64;
    // Include ~20% overhead for CSI-2 protocol (headers, blanking, etc.).
    let total_bits = bits_per_second * 120 / 100;
    let per_lane = total_bits / config.lanes as u64;
    (per_lane / 1_000_000) as u32
}

/// Validate that the config's bandwidth fits within the configured lane speed.
pub fn validate_bandwidth(config: &CsiConfig) -> Result<(), HalError> {
    let required = required_bandwidth_mbps(config);
    if required > config.lane_speed_mbps as u32 {
        return Err(HalError::InvalidConfig);
    }
    Ok(())
}

/// Validate a buffer index against the configured buffer count.
pub fn validate_buffer_index(index: u8, num_buffers: u8) -> Result<(), HalError> {
    if index >= num_buffers {
        return Err(HalError::OutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> CsiConfig {
        CsiConfig {
            lanes: 2,
            lane_speed_mbps: 800,
            width: 1920,
            height: 1080,
            pixel_format: CsiPixelFormat::Raw10,
            fps: 30,
            num_buffers: 3,
        }
    }

    #[test]
    fn test_validate_csi_config_valid() {
        assert!(validate_csi_config(&make_config()).is_ok());
    }

    #[test]
    fn test_validate_csi_config_invalid_lanes() {
        let mut cfg = make_config();
        cfg.lanes = 3; // Only 1, 2, 4 are valid.
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_validate_csi_config_invalid_width() {
        let mut cfg = make_config();
        cfg.width = 0;
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
        cfg.width = 5000;
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_validate_csi_config_invalid_fps() {
        let mut cfg = make_config();
        cfg.fps = 0;
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
        cfg.fps = 121;
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_validate_csi_config_invalid_buffers() {
        let mut cfg = make_config();
        cfg.num_buffers = 0;
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
        cfg.num_buffers = 5;
        assert_eq!(validate_csi_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_compute_frame_size_raw8() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw8;
        cfg.width = 640;
        cfg.height = 480;
        assert_eq!(compute_frame_size(&cfg), 640 * 480);
    }

    #[test]
    fn test_compute_frame_size_raw10() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw10;
        cfg.width = 1920;
        cfg.height = 1080;
        // 1920*1080*10/8 = 2,592,000
        assert_eq!(compute_frame_size(&cfg), (1920 * 1080 * 10 + 7) / 8);
    }

    #[test]
    fn test_compute_frame_size_yuv422() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Yuv422;
        cfg.width = 640;
        cfg.height = 480;
        assert_eq!(compute_frame_size(&cfg), 640 * 480 * 2);
    }

    #[test]
    fn test_compute_frame_size_rgb888() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Rgb888;
        cfg.width = 320;
        cfg.height = 240;
        assert_eq!(compute_frame_size(&cfg), 320 * 240 * 3);
    }

    #[test]
    fn test_required_bandwidth_mbps() {
        let cfg = make_config();
        let bw = required_bandwidth_mbps(&cfg);
        // 1920*1080*10*30 = 622,080,000 bits/s = 622 Mbps
        // With 20% overhead: ~746 Mbps, per 2 lanes: ~373 Mbps
        assert!(bw > 300 && bw < 500);
    }

    #[test]
    fn test_validate_bandwidth_ok() {
        let cfg = make_config();
        assert!(validate_bandwidth(&cfg).is_ok());
    }

    #[test]
    fn test_validate_bandwidth_too_slow() {
        let mut cfg = make_config();
        cfg.lane_speed_mbps = 100; // Way too slow for 1080p30.
        assert_eq!(validate_bandwidth(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_validate_buffer_index() {
        assert!(validate_buffer_index(0, 3).is_ok());
        assert!(validate_buffer_index(2, 3).is_ok());
        assert_eq!(validate_buffer_index(3, 3), Err(HalError::OutOfRange));
    }
}
