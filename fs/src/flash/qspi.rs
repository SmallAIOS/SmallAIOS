// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QSPI NOR flash driver scaffolding.
//!
//! Phase 1 of `embedded-flash-fs-v1` ships this module as a documented
//! stub. The actual QSPI controller binding lives in the per-arch crate
//! (`arch/aarch64/src/flash/qspi.rs`, `arch/riscv64/src/flash/qspi.rs`)
//! and lands when the first MCU/FPGA target arrives. The `fs` crate
//! exposes the `QspiNorDevice` shape so downstream code (littlefs
//! mount, BBT loader, format-on-first-boot path) can reference the
//! type without a hardware-conditional compile.
//!
//! Defaults per `fs-flash-device` spec:
//!
//! * `block_size_bytes = 4096` (typical QSPI NOR sector size).
//! * `page_size_bytes = 256` (typical QSPI NOR page size).
//!
//! All operations return [`FlashError::NotPresent`] until the per-arch
//! controller binds.

use super::device::{FlashDevice, FlashError};

/// Stub QSPI NOR flash device.
///
/// Until the per-arch QSPI controller binds, every operation surfaces
/// [`FlashError::NotPresent`]. This is the documented Phase 1 contract.
#[derive(Debug, Default)]
pub struct QspiNorDevice {
    block_size: u32,
    page_size: u32,
    block_count: u64,
}

impl QspiNorDevice {
    /// QSPI NOR default block size (sector erase): 4 KiB.
    pub const DEFAULT_BLOCK_SIZE_BYTES: u32 = 4096;
    /// QSPI NOR default page size (program unit): 256 bytes.
    pub const DEFAULT_PAGE_SIZE_BYTES: u32 = 256;

    /// Construct a QSPI NOR device descriptor with default sector/page
    /// sizes and an operator-supplied block count.
    pub fn new(block_count: u64) -> Self {
        Self {
            block_size: Self::DEFAULT_BLOCK_SIZE_BYTES,
            page_size: Self::DEFAULT_PAGE_SIZE_BYTES,
            block_count,
        }
    }

    /// Construct a QSPI NOR device descriptor with custom geometry.
    pub fn with_geometry(block_count: u64, block_size: u32, page_size: u32) -> Self {
        Self {
            block_size,
            page_size,
            block_count,
        }
    }
}

impl FlashDevice for QspiNorDevice {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FlashError> {
        Err(FlashError::NotPresent)
    }

    fn program(&mut self, _offset: u64, _buf: &[u8]) -> Result<(), FlashError> {
        Err(FlashError::NotPresent)
    }

    fn erase(&mut self, _block: u64) -> Result<(), FlashError> {
        Err(FlashError::NotPresent)
    }

    fn block_size_bytes(&self) -> u32 {
        self.block_size
    }

    fn page_size_bytes(&self) -> u32 {
        self.page_size
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }

    fn is_bad(&self, _block: u64) -> bool {
        false
    }

    fn mark_bad(&mut self, _block: u64) -> Result<(), FlashError> {
        Err(FlashError::NotPresent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let dev = QspiNorDevice::new(64);
        assert_eq!(dev.block_size_bytes(), 4096);
        assert_eq!(dev.page_size_bytes(), 256);
        assert_eq!(dev.block_count(), 64);
    }

    #[test]
    fn stub_returns_not_present() {
        let mut dev = QspiNorDevice::new(64);
        let mut buf = [0u8; 16];
        assert_eq!(dev.read(0, &mut buf), Err(FlashError::NotPresent));
        assert_eq!(dev.program(0, &[0u8; 256]), Err(FlashError::NotPresent));
        assert_eq!(dev.erase(0), Err(FlashError::NotPresent));
        assert_eq!(dev.mark_bad(0), Err(FlashError::NotPresent));
        assert!(!dev.is_bad(0));
    }

    #[test]
    fn custom_geometry() {
        let dev = QspiNorDevice::with_geometry(128, 8192, 512);
        assert_eq!(dev.block_size_bytes(), 8192);
        assert_eq!(dev.page_size_bytes(), 512);
        assert_eq!(dev.block_count(), 128);
    }
}
