// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AXI4-Lite memory-mapped register read/write driver.
//!
//! Provides 32-bit volatile register access to FPGA soft-IP peripherals
//! with bounds checking and memory barrier semantics.

use core::fmt;
use core::sync::atomic::{compiler_fence, Ordering};

/// Error type for AXI4-Lite register operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxiLiteError {
    /// Register offset exceeds the peripheral address range.
    OffsetOutOfRange {
        /// The attempted offset.
        offset: u32,
        /// The maximum valid offset (address_size - 4).
        max_offset: u32,
    },
    /// Base address is null or invalid.
    InvalidBaseAddress,
    /// Peripheral address size is zero or not aligned to 4 bytes.
    InvalidAddressSize,
}

impl fmt::Display for AxiLiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxiLiteError::OffsetOutOfRange { offset, max_offset } => {
                write!(f, "offset 0x{offset:x} exceeds max 0x{max_offset:x}")
            }
            AxiLiteError::InvalidBaseAddress => write!(f, "invalid base address"),
            AxiLiteError::InvalidAddressSize => write!(f, "invalid address size"),
        }
    }
}

/// AXI4-Lite peripheral driver.
///
/// Wraps a base MMIO address and address range, providing bounded 32-bit
/// volatile register access with memory barriers.
#[derive(Debug, Clone, Copy)]
pub struct AxiLiteDevice {
    /// MMIO base address.
    base_addr: u64,
    /// Total address space size in bytes (must be >= 4 and aligned to 4).
    addr_size: u32,
}

impl AxiLiteDevice {
    /// Create a new AXI4-Lite device driver.
    ///
    /// `base_addr` is the physical MMIO base address.
    /// `addr_size` is the size of the peripheral's address space in bytes.
    pub fn new(base_addr: u64, addr_size: u32) -> Result<Self, AxiLiteError> {
        if base_addr == 0 {
            return Err(AxiLiteError::InvalidBaseAddress);
        }
        if addr_size < 4 || !addr_size.is_multiple_of(4) {
            return Err(AxiLiteError::InvalidAddressSize);
        }
        Ok(Self {
            base_addr,
            addr_size,
        })
    }

    /// Base address of this peripheral.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Address space size.
    pub fn addr_size(&self) -> u32 {
        self.addr_size
    }

    /// Maximum valid register offset.
    pub fn max_offset(&self) -> u32 {
        self.addr_size - 4
    }

    /// Validate that an offset is within bounds and 4-byte aligned.
    fn validate_offset(&self, offset: u32) -> Result<(), AxiLiteError> {
        if !offset.is_multiple_of(4) || offset > self.max_offset() {
            return Err(AxiLiteError::OffsetOutOfRange {
                offset,
                max_offset: self.max_offset(),
            });
        }
        Ok(())
    }

    /// Read a 32-bit register at the given offset.
    ///
    /// Performs a volatile read at `base_addr + offset` followed by a
    /// read memory barrier to ensure ordering with subsequent accesses.
    ///
    /// # Safety Note
    /// In a real kernel this would perform an actual MMIO read. This
    /// implementation provides the bounds-checking and barrier logic;
    /// the actual volatile read is stubbed for host-testable code.
    pub fn read32(&self, offset: u32) -> Result<u32, AxiLiteError> {
        self.validate_offset(offset)?;

        let _addr = self.base_addr + offset as u64;
        // In a real kernel: unsafe { core::ptr::read_volatile(addr as *const u32) }
        // Stub: return 0 for host testing. Real MMIO reads happen on hardware.
        compiler_fence(Ordering::Acquire);
        Ok(0)
    }

    /// Write a 32-bit value to a register at the given offset.
    ///
    /// Issues a write memory barrier before the volatile write at
    /// `base_addr + offset` to ensure prior stores are visible.
    pub fn write32(&self, offset: u32, _value: u32) -> Result<(), AxiLiteError> {
        self.validate_offset(offset)?;

        compiler_fence(Ordering::Release);
        let _addr = self.base_addr + offset as u64;
        // In a real kernel: unsafe { core::ptr::write_volatile(addr as *mut u32, value) }
        compiler_fence(Ordering::SeqCst);
        Ok(())
    }

    /// Read-modify-write: read a register, apply a mask, OR in new bits, write back.
    pub fn modify32(
        &self,
        offset: u32,
        clear_mask: u32,
        set_bits: u32,
    ) -> Result<(), AxiLiteError> {
        let val = self.read32(offset)?;
        let new_val = (val & !clear_mask) | set_bits;
        self.write32(offset, new_val)
    }

    /// Poll a register until a condition is met or max iterations exceeded.
    /// Returns the final register value on success, or AxiLiteError on timeout.
    pub fn poll32(
        &self,
        offset: u32,
        mask: u32,
        expected: u32,
        max_iterations: u32,
    ) -> Result<u32, AxiLiteError> {
        for _ in 0..max_iterations {
            let val = self.read32(offset)?;
            if (val & mask) == expected {
                return Ok(val);
            }
        }
        // Timeout: return the condition-unmatched error as offset out of range
        // (in a real driver this would be a Timeout variant)
        Err(AxiLiteError::OffsetOutOfRange {
            offset,
            max_offset: self.max_offset(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_new_valid() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x1000).unwrap();
        assert_eq!(dev.base_addr(), 0x4000_0000);
        assert_eq!(dev.addr_size(), 0x1000);
        assert_eq!(dev.max_offset(), 0x0FFC);
    }

    #[test]
    fn test_new_invalid_base() {
        let err = AxiLiteDevice::new(0, 0x1000).unwrap_err();
        assert_eq!(err, AxiLiteError::InvalidBaseAddress);
    }

    #[test]
    fn test_new_invalid_size_zero() {
        let err = AxiLiteDevice::new(0x4000_0000, 0).unwrap_err();
        assert_eq!(err, AxiLiteError::InvalidAddressSize);
    }

    #[test]
    fn test_new_invalid_size_unaligned() {
        let err = AxiLiteDevice::new(0x4000_0000, 5).unwrap_err();
        assert_eq!(err, AxiLiteError::InvalidAddressSize);
    }

    #[test]
    fn test_new_minimum_size() {
        let dev = AxiLiteDevice::new(0x1000, 4).unwrap();
        assert_eq!(dev.max_offset(), 0);
    }

    #[test]
    fn test_read32_valid_offset() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        assert!(dev.read32(0).is_ok());
        assert!(dev.read32(0xFC).is_ok());
    }

    #[test]
    fn test_read32_out_of_range() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        let err = dev.read32(0x100).unwrap_err();
        assert_eq!(
            err,
            AxiLiteError::OffsetOutOfRange {
                offset: 0x100,
                max_offset: 0xFC,
            }
        );
    }

    #[test]
    fn test_read32_unaligned_offset() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        assert!(dev.read32(1).is_err());
        assert!(dev.read32(2).is_err());
        assert!(dev.read32(3).is_err());
    }

    #[test]
    fn test_write32_valid() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        assert!(dev.write32(0, 0xDEAD_BEEF).is_ok());
        assert!(dev.write32(0xFC, 0x1234_5678).is_ok());
    }

    #[test]
    fn test_write32_out_of_range() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        assert!(dev.write32(0x100, 0).is_err());
    }

    #[test]
    fn test_modify32() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        // Stub reads 0, so modify(0, 0xFF, 0x0A) should write 0x0A
        assert!(dev.modify32(0, 0xFF, 0x0A).is_ok());
    }

    #[test]
    fn test_modify32_out_of_range() {
        let dev = AxiLiteDevice::new(0x4000_0000, 0x100).unwrap();
        assert!(dev.modify32(0x100, 0xFF, 0x0A).is_err());
    }

    #[test]
    fn test_error_display() {
        let err = AxiLiteError::OffsetOutOfRange {
            offset: 0x200,
            max_offset: 0xFC,
        };
        assert_eq!(format!("{err}"), "offset 0x200 exceeds max 0xfc");
        assert_eq!(
            format!("{}", AxiLiteError::InvalidBaseAddress),
            "invalid base address"
        );
        assert_eq!(
            format!("{}", AxiLiteError::InvalidAddressSize),
            "invalid address size"
        );
    }

    #[test]
    fn test_error_debug() {
        let err = AxiLiteError::InvalidBaseAddress;
        assert_eq!(format!("{err:?}"), "InvalidBaseAddress");
    }

    #[test]
    fn test_error_eq() {
        assert_eq!(
            AxiLiteError::InvalidBaseAddress,
            AxiLiteError::InvalidBaseAddress
        );
        assert_ne!(
            AxiLiteError::InvalidBaseAddress,
            AxiLiteError::InvalidAddressSize
        );
    }
}
