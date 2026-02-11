// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! I2S/TDM audio peripheral drivers.
//!
//! Provides platform-specific I2S controller implementations for:
//! - ARM MMIO I2S (e.g., BCM2835 PCM/I2S, STM32 SAI)
//! - RISC-V MMIO I2S (e.g., SiFive I2S)
//! - FPGA I2S IP (generic over [`FpgaFabric`])
//!
//! Plus audio codec configuration drivers (TLV320AIC3x, WM8960, ES8388)
//! configured via I2C, and a preprocessing module for audio feature extraction.
//!
//! All I2S controller drivers implement the [`I2sController`] trait from
//! `smallaios_kernel::hal`.

#![allow(dead_code)]

pub mod codec;
pub mod tlv320aic3x;
pub mod wm8960;
pub mod es8388;
pub mod arm_i2s;
pub mod riscv_i2s;
pub mod fpga_i2s;
pub mod preprocess;

pub use smallaios_kernel::hal::{
    AudioBuffer, HalError, I2sBitDepth, I2sConfig, I2sController, I2sIrqSource, I2sMode, I2sRole,
};

// ---------------------------------------------------------------------------
// Shared I2S helpers
// ---------------------------------------------------------------------------

/// Maximum number of audio DMA ring buffers.
pub const MAX_AUDIO_BUFFERS: u8 = 4;

/// Maximum number of channels (8 for TDM).
pub const MAX_CHANNELS: u8 = 8;

/// Minimum sample rate in Hz.
pub const MIN_SAMPLE_RATE: u32 = 8_000;

/// Maximum sample rate in Hz.
pub const MAX_SAMPLE_RATE: u32 = 192_000;

/// Validate an I2S configuration.
///
/// Checks sample rate, channels, buffer count, and bit depth constraints.
pub fn validate_i2s_config(config: &I2sConfig) -> Result<(), HalError> {
    // Validate sample rate.
    if config.sample_rate_hz < MIN_SAMPLE_RATE || config.sample_rate_hz > MAX_SAMPLE_RATE {
        return Err(HalError::InvalidConfig);
    }
    // Validate channels.
    if config.channels == 0 || config.channels > MAX_CHANNELS {
        return Err(HalError::InvalidConfig);
    }
    // Standard I2S modes only support 1 or 2 channels.
    match config.mode {
        I2sMode::Standard | I2sMode::LeftJustified | I2sMode::RightJustified => {
            if config.channels > 2 {
                return Err(HalError::InvalidConfig);
            }
        }
        I2sMode::Tdm { num_slots } => {
            if num_slots == 0 || config.channels > num_slots {
                return Err(HalError::InvalidConfig);
            }
        }
    }
    // Validate buffer count.
    if config.num_buffers < 2 || config.num_buffers > MAX_AUDIO_BUFFERS {
        return Err(HalError::InvalidConfig);
    }
    // Validate buffer frames.
    if config.buffer_frames == 0 {
        return Err(HalError::InvalidConfig);
    }
    Ok(())
}

/// Get the number of bytes per sample for a given bit depth.
pub fn bytes_per_sample(bit_depth: I2sBitDepth) -> usize {
    match bit_depth {
        I2sBitDepth::Bits16 => 2,
        I2sBitDepth::Bits24 => 3,
        I2sBitDepth::Bits32 => 4,
    }
}

/// Get the number of bits per sample slot in the I2S frame.
///
/// The slot width is always a power-of-2 aligned container:
/// 16-bit -> 16, 24-bit -> 32, 32-bit -> 32.
pub fn slot_width_bits(bit_depth: I2sBitDepth) -> u32 {
    match bit_depth {
        I2sBitDepth::Bits16 => 16,
        I2sBitDepth::Bits24 => 32, // 24-bit samples in 32-bit container.
        I2sBitDepth::Bits32 => 32,
    }
}

/// Compute the required BCLK (bit clock) frequency for a given config.
///
/// `BCLK = sample_rate * channels * slot_width`
///
/// For standard I2S, channels is always 2 (left + right).
pub fn compute_bclk(config: &I2sConfig) -> u32 {
    let ch = match config.mode {
        I2sMode::Tdm { num_slots } => num_slots as u32,
        _ => 2, // Standard I2S always has 2 slots.
    };
    config.sample_rate_hz * ch * slot_width_bits(config.bit_depth)
}

/// Compute the byte size of a single audio DMA buffer.
///
/// `size = buffer_frames * channels * bytes_per_sample`
pub fn compute_buffer_size(config: &I2sConfig) -> usize {
    config.buffer_frames as usize
        * config.channels as usize
        * bytes_per_sample(config.bit_depth)
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

    fn make_config() -> I2sConfig {
        I2sConfig {
            mode: I2sMode::Standard,
            role: I2sRole::Master,
            sample_rate_hz: 48000,
            bit_depth: I2sBitDepth::Bits16,
            channels: 2,
            buffer_frames: 1024,
            num_buffers: 3,
        }
    }

    #[test]
    fn test_validate_i2s_config_valid() {
        assert!(validate_i2s_config(&make_config()).is_ok());
    }

    #[test]
    fn test_validate_i2s_config_invalid_sample_rate() {
        let mut cfg = make_config();
        cfg.sample_rate_hz = 0;
        assert_eq!(validate_i2s_config(&cfg), Err(HalError::InvalidConfig));
        cfg.sample_rate_hz = 200_000;
        assert_eq!(validate_i2s_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_validate_i2s_config_invalid_channels() {
        let mut cfg = make_config();
        cfg.channels = 0;
        assert_eq!(validate_i2s_config(&cfg), Err(HalError::InvalidConfig));
        // Standard I2S only supports 1-2 channels.
        cfg.channels = 4;
        assert_eq!(validate_i2s_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_validate_i2s_config_tdm_channels() {
        let mut cfg = make_config();
        cfg.mode = I2sMode::Tdm { num_slots: 8 };
        cfg.channels = 8;
        assert!(validate_i2s_config(&cfg).is_ok());
        // More channels than slots.
        cfg.channels = 9;
        assert!(validate_i2s_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_i2s_config_invalid_buffers() {
        let mut cfg = make_config();
        cfg.num_buffers = 1; // Minimum is 2.
        assert_eq!(validate_i2s_config(&cfg), Err(HalError::InvalidConfig));
        cfg.num_buffers = 5;
        assert_eq!(validate_i2s_config(&cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_bytes_per_sample() {
        assert_eq!(bytes_per_sample(I2sBitDepth::Bits16), 2);
        assert_eq!(bytes_per_sample(I2sBitDepth::Bits24), 3);
        assert_eq!(bytes_per_sample(I2sBitDepth::Bits32), 4);
    }

    #[test]
    fn test_slot_width_bits() {
        assert_eq!(slot_width_bits(I2sBitDepth::Bits16), 16);
        assert_eq!(slot_width_bits(I2sBitDepth::Bits24), 32);
        assert_eq!(slot_width_bits(I2sBitDepth::Bits32), 32);
    }

    #[test]
    fn test_compute_bclk_standard_16bit_48khz() {
        let cfg = make_config();
        let bclk = compute_bclk(&cfg);
        // 48000 * 2 * 16 = 1,536,000
        assert_eq!(bclk, 1_536_000);
    }

    #[test]
    fn test_compute_bclk_tdm_8slot() {
        let mut cfg = make_config();
        cfg.mode = I2sMode::Tdm { num_slots: 8 };
        cfg.channels = 8;
        cfg.bit_depth = I2sBitDepth::Bits32;
        let bclk = compute_bclk(&cfg);
        // 48000 * 8 * 32 = 12,288,000
        assert_eq!(bclk, 12_288_000);
    }

    #[test]
    fn test_compute_buffer_size() {
        let cfg = make_config();
        let size = compute_buffer_size(&cfg);
        // 1024 frames * 2 channels * 2 bytes = 4096
        assert_eq!(size, 4096);
    }

    #[test]
    fn test_validate_buffer_index() {
        assert!(validate_buffer_index(0, 3).is_ok());
        assert!(validate_buffer_index(2, 3).is_ok());
        assert_eq!(validate_buffer_index(3, 3), Err(HalError::OutOfRange));
    }
}
