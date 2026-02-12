// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AXI4 full memory-mapped burst transfer driver.
//!
//! Supports INCR burst reads/writes with configurable burst length (1-256 beats),
//! data width, and error handling per AXI4 protocol spec.

use core::fmt;
use core::sync::atomic::{compiler_fence, Ordering};

/// Maximum AXI4 burst length (256 beats).
pub const MAX_BURST_LEN: u16 = 256;

/// AXI4 burst type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurstType {
    /// Fixed address — all beats access the same address (FIFO-like).
    Fixed,
    /// Incrementing address — each beat increments by data width.
    Incr,
    /// Wrapping burst — wraps at aligned boundary.
    Wrap,
}

/// AXI4 response code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxiResponse {
    /// OKAY — normal access success.
    Okay,
    /// EXOKAY — exclusive access success.
    ExclusiveOkay,
    /// SLVERR — slave error (peripheral rejected the access).
    SlaveError,
    /// DECERR — decode error (no slave at this address).
    DecodeError,
}

/// Error type for AXI4 burst operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxiFullError {
    /// Base address is null.
    InvalidBaseAddress,
    /// Burst length is zero or exceeds 256 beats.
    InvalidBurstLength,
    /// Data width is not a power of 2 or exceeds 128 bytes (1024 bits).
    InvalidDataWidth,
    /// Transfer received an error response from the slave.
    TransferError {
        /// AXI response code.
        response: AxiResponse,
        /// Beat index at which the error occurred.
        beat_index: u16,
    },
    /// Buffer size doesn't match burst_len * data_width.
    BufferSizeMismatch,
    /// Address is not aligned to data width.
    UnalignedAddress,
}

impl fmt::Display for AxiFullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxiFullError::InvalidBaseAddress => write!(f, "invalid base address"),
            AxiFullError::InvalidBurstLength => write!(f, "invalid burst length"),
            AxiFullError::InvalidDataWidth => write!(f, "invalid data width"),
            AxiFullError::TransferError {
                response,
                beat_index,
            } => {
                write!(f, "transfer error: {response:?} at beat {beat_index}")
            }
            AxiFullError::BufferSizeMismatch => write!(f, "buffer size mismatch"),
            AxiFullError::UnalignedAddress => write!(f, "unaligned address"),
        }
    }
}

/// AXI4 burst transfer configuration.
#[derive(Debug, Clone, Copy)]
pub struct BurstConfig {
    /// Starting address (must be aligned to data_width_bytes).
    pub address: u64,
    /// Number of beats in the burst (1-256).
    pub burst_len: u16,
    /// Data width per beat in bytes (must be power of 2, max 128).
    pub data_width_bytes: u8,
    /// Burst type (INCR is the standard choice).
    pub burst_type: BurstType,
}

impl BurstConfig {
    /// Total transfer size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.burst_len as usize * self.data_width_bytes as usize
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), AxiFullError> {
        if self.address == 0 {
            return Err(AxiFullError::InvalidBaseAddress);
        }
        if self.burst_len == 0 || self.burst_len > MAX_BURST_LEN {
            return Err(AxiFullError::InvalidBurstLength);
        }
        if self.data_width_bytes == 0
            || !self.data_width_bytes.is_power_of_two()
            || self.data_width_bytes > 128
        {
            return Err(AxiFullError::InvalidDataWidth);
        }
        if !self.address.is_multiple_of(self.data_width_bytes as u64) {
            return Err(AxiFullError::UnalignedAddress);
        }
        Ok(())
    }
}

/// AXI4 full burst transfer result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurstResult {
    /// Number of beats successfully completed.
    pub beats_completed: u16,
    /// Total bytes transferred.
    pub bytes_transferred: u32,
    /// Final response code.
    pub response: AxiResponse,
}

/// AXI4 full bus driver.
///
/// Manages burst read/write operations against an AXI4 peripheral.
/// In a real implementation, this would program AXI master interface
/// registers. This stub provides the validation and protocol logic.
#[derive(Debug, Clone, Copy)]
pub struct AxiFullDevice {
    /// MMIO base address of the AXI master controller.
    base_addr: u64,
    /// Maximum supported data width in bytes.
    max_data_width: u8,
}

impl AxiFullDevice {
    /// Create a new AXI4 full bus driver.
    pub fn new(base_addr: u64, max_data_width: u8) -> Result<Self, AxiFullError> {
        if base_addr == 0 {
            return Err(AxiFullError::InvalidBaseAddress);
        }
        if max_data_width == 0 || !max_data_width.is_power_of_two() || max_data_width > 128 {
            return Err(AxiFullError::InvalidDataWidth);
        }
        Ok(Self {
            base_addr,
            max_data_width,
        })
    }

    /// Base address of the AXI master controller.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Maximum supported data width.
    pub fn max_data_width(&self) -> u8 {
        self.max_data_width
    }

    /// Initiate a burst read.
    ///
    /// Reads `config.burst_len` beats of `config.data_width_bytes` each
    /// starting from `config.address`. The result is written into `buffer`.
    pub fn burst_read(
        &self,
        config: &BurstConfig,
        buffer: &mut [u8],
    ) -> Result<BurstResult, AxiFullError> {
        config.validate()?;
        if config.data_width_bytes > self.max_data_width {
            return Err(AxiFullError::InvalidDataWidth);
        }
        let total = config.total_bytes();
        if buffer.len() < total {
            return Err(AxiFullError::BufferSizeMismatch);
        }

        // Program AXI read address channel (stub)
        compiler_fence(Ordering::SeqCst);
        let _ar_addr = config.address;
        let _ar_len = config.burst_len - 1; // AXI ARLEN = burst_len - 1
        let _ar_size = config.data_width_bytes.trailing_zeros() as u8;

        // In a real implementation: wait for read data channel beats
        // Stub: fill buffer with zeros and report success
        for byte in buffer[..total].iter_mut() {
            *byte = 0;
        }
        compiler_fence(Ordering::Acquire);

        Ok(BurstResult {
            beats_completed: config.burst_len,
            bytes_transferred: total as u32,
            response: AxiResponse::Okay,
        })
    }

    /// Initiate a burst write.
    ///
    /// Writes `config.burst_len` beats of `config.data_width_bytes` each
    /// to `config.address` from `buffer`.
    pub fn burst_write(
        &self,
        config: &BurstConfig,
        buffer: &[u8],
    ) -> Result<BurstResult, AxiFullError> {
        config.validate()?;
        if config.data_width_bytes > self.max_data_width {
            return Err(AxiFullError::InvalidDataWidth);
        }
        let total = config.total_bytes();
        if buffer.len() < total {
            return Err(AxiFullError::BufferSizeMismatch);
        }

        // Program AXI write address channel (stub)
        compiler_fence(Ordering::Release);
        let _aw_addr = config.address;
        let _aw_len = config.burst_len - 1;
        let _aw_size = config.data_width_bytes.trailing_zeros() as u8;

        // Write data beats (stub)
        // In real implementation: write each beat to the write data channel
        compiler_fence(Ordering::SeqCst);

        Ok(BurstResult {
            beats_completed: config.burst_len,
            bytes_transferred: total as u32,
            response: AxiResponse::Okay,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_burst_config_validate_valid() {
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 16,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.total_bytes(), 64);
    }

    #[test]
    fn test_burst_config_zero_address() {
        let cfg = BurstConfig {
            address: 0,
            burst_len: 1,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        assert_eq!(cfg.validate(), Err(AxiFullError::InvalidBaseAddress));
    }

    #[test]
    fn test_burst_config_zero_length() {
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 0,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        assert_eq!(cfg.validate(), Err(AxiFullError::InvalidBurstLength));
    }

    #[test]
    fn test_burst_config_excess_length() {
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 257,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        assert_eq!(cfg.validate(), Err(AxiFullError::InvalidBurstLength));
    }

    #[test]
    fn test_burst_config_max_length() {
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 256,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.total_bytes(), 1024);
    }

    #[test]
    fn test_burst_config_invalid_data_width() {
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 1,
            data_width_bytes: 3, // not power of 2
            burst_type: BurstType::Incr,
        };
        assert_eq!(cfg.validate(), Err(AxiFullError::InvalidDataWidth));
    }

    #[test]
    fn test_burst_config_unaligned_address() {
        let cfg = BurstConfig {
            address: 0x1001, // not aligned to 4
            burst_len: 1,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        assert_eq!(cfg.validate(), Err(AxiFullError::UnalignedAddress));
    }

    #[test]
    fn test_device_new_valid() {
        let dev = AxiFullDevice::new(0x4000_0000, 8).unwrap();
        assert_eq!(dev.base_addr(), 0x4000_0000);
        assert_eq!(dev.max_data_width(), 8);
    }

    #[test]
    fn test_device_new_invalid_base() {
        let err = AxiFullDevice::new(0, 4).unwrap_err();
        assert_eq!(err, AxiFullError::InvalidBaseAddress);
    }

    #[test]
    fn test_device_new_invalid_width() {
        let err = AxiFullDevice::new(0x1000, 3).unwrap_err();
        assert_eq!(err, AxiFullError::InvalidDataWidth);
    }

    #[test]
    fn test_burst_read_success() {
        let dev = AxiFullDevice::new(0x4000_0000, 8).unwrap();
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 4,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        let mut buf = [0xFFu8; 16];
        let result = dev.burst_read(&cfg, &mut buf).unwrap();
        assert_eq!(result.beats_completed, 4);
        assert_eq!(result.bytes_transferred, 16);
        assert_eq!(result.response, AxiResponse::Okay);
    }

    #[test]
    fn test_burst_read_buffer_too_small() {
        let dev = AxiFullDevice::new(0x4000_0000, 8).unwrap();
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 4,
            data_width_bytes: 4,
            burst_type: BurstType::Incr,
        };
        let mut buf = [0u8; 8]; // only 8 bytes, need 16
        let err = dev.burst_read(&cfg, &mut buf).unwrap_err();
        assert_eq!(err, AxiFullError::BufferSizeMismatch);
    }

    #[test]
    fn test_burst_read_width_exceeds_device() {
        let dev = AxiFullDevice::new(0x4000_0000, 4).unwrap();
        let cfg = BurstConfig {
            address: 0x1000,
            burst_len: 1,
            data_width_bytes: 8, // exceeds max of 4
            burst_type: BurstType::Incr,
        };
        let mut buf = [0u8; 8];
        let err = dev.burst_read(&cfg, &mut buf).unwrap_err();
        assert_eq!(err, AxiFullError::InvalidDataWidth);
    }

    #[test]
    fn test_burst_write_success() {
        let dev = AxiFullDevice::new(0x4000_0000, 8).unwrap();
        let cfg = BurstConfig {
            address: 0x2000,
            burst_len: 2,
            data_width_bytes: 8,
            burst_type: BurstType::Incr,
        };
        let buf = [0xABu8; 16];
        let result = dev.burst_write(&cfg, &buf).unwrap();
        assert_eq!(result.beats_completed, 2);
        assert_eq!(result.bytes_transferred, 16);
        assert_eq!(result.response, AxiResponse::Okay);
    }

    #[test]
    fn test_burst_write_buffer_too_small() {
        let dev = AxiFullDevice::new(0x4000_0000, 8).unwrap();
        let cfg = BurstConfig {
            address: 0x2000,
            burst_len: 2,
            data_width_bytes: 8,
            burst_type: BurstType::Incr,
        };
        let buf = [0u8; 4]; // need 16
        let err = dev.burst_write(&cfg, &buf).unwrap_err();
        assert_eq!(err, AxiFullError::BufferSizeMismatch);
    }

    #[test]
    fn test_burst_type_variants() {
        assert_ne!(BurstType::Fixed, BurstType::Incr);
        assert_ne!(BurstType::Incr, BurstType::Wrap);
    }

    #[test]
    fn test_axi_response_variants() {
        assert_ne!(AxiResponse::Okay, AxiResponse::SlaveError);
        assert_ne!(AxiResponse::DecodeError, AxiResponse::ExclusiveOkay);
    }

    #[test]
    fn test_error_display() {
        let err = AxiFullError::TransferError {
            response: AxiResponse::SlaveError,
            beat_index: 5,
        };
        assert_eq!(format!("{err}"), "transfer error: SlaveError at beat 5");
        assert_eq!(
            format!("{}", AxiFullError::InvalidBurstLength),
            "invalid burst length"
        );
        assert_eq!(
            format!("{}", AxiFullError::BufferSizeMismatch),
            "buffer size mismatch"
        );
        assert_eq!(
            format!("{}", AxiFullError::UnalignedAddress),
            "unaligned address"
        );
    }
}
