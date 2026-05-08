// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONFI NAND flash driver scaffolding.
//!
//! Phase 1 of `embedded-flash-fs-v1` ships this module as a documented
//! stub. The actual ONFI NAND controller binding lives in the per-arch
//! crate (`arch/aarch64/src/flash/onfi.rs`,
//! `arch/riscv64/src/flash/onfi.rs`) and lands when the first MCU/FPGA
//! target arrives. ECC is handled by the controller; the `FlashDevice`
//! surface is post-ECC.
//!
//! Defaults per `fs-flash-device` spec:
//!
//! * `block_size_bytes = 131072` (128 KiB — typical small-page MLC
//!   NAND erase-block).
//! * `page_size_bytes = 4096` (4 KiB — typical NAND page).
//!
//! All operations return [`FlashError::NotPresent`] until the per-arch
//! controller binds.

use super::device::{FlashDevice, FlashError};

/// Stub ONFI NAND flash device.
///
/// Until the per-arch ONFI controller binds, every operation surfaces
/// [`FlashError::NotPresent`].
#[derive(Debug, Default)]
pub struct OnfiNandDevice {
    block_size: u32,
    page_size: u32,
    block_count: u64,
}

impl OnfiNandDevice {
    /// ONFI NAND default erase-block size: 128 KiB.
    pub const DEFAULT_BLOCK_SIZE_BYTES: u32 = 131072;
    /// ONFI NAND default page size: 4 KiB.
    pub const DEFAULT_PAGE_SIZE_BYTES: u32 = 4096;

    /// Construct an ONFI NAND device descriptor with default geometry.
    pub fn new(block_count: u64) -> Self {
        Self {
            block_size: Self::DEFAULT_BLOCK_SIZE_BYTES,
            page_size: Self::DEFAULT_PAGE_SIZE_BYTES,
            block_count,
        }
    }

    /// Construct with custom geometry (e.g. once the per-target ONFI
    /// parameter-page probe lands and reports a different layout).
    pub fn with_geometry(block_count: u64, block_size: u32, page_size: u32) -> Self {
        Self {
            block_size,
            page_size,
            block_count,
        }
    }
}

impl FlashDevice for OnfiNandDevice {
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
        let dev = OnfiNandDevice::new(1024);
        assert_eq!(dev.block_size_bytes(), 131072);
        assert_eq!(dev.page_size_bytes(), 4096);
        assert_eq!(dev.block_count(), 1024);
    }

    #[test]
    fn stub_returns_not_present() {
        let mut dev = OnfiNandDevice::new(1024);
        let mut buf = [0u8; 16];
        assert_eq!(dev.read(0, &mut buf), Err(FlashError::NotPresent));
        assert_eq!(dev.program(0, &[0u8; 4096]), Err(FlashError::NotPresent));
        assert_eq!(dev.erase(0), Err(FlashError::NotPresent));
        assert_eq!(dev.mark_bad(0), Err(FlashError::NotPresent));
    }
}
