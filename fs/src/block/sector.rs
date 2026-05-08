// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! 4 KiB-native FS access over devices that report 4 KiB or 512 B
//! sectors.
//!
//! The SmallAIOS `fs` crate operates natively at 4 KiB
//! ([`crate::block::FS_BLOCK_SIZE_BYTES`]). Underlying block devices,
//! however, fall into one of three categories per spec Q23:
//!
//! 1. **4 KiB native** (`logical == physical == 4096`). FS reads and
//!    writes pass through directly.
//! 2. **Advanced Format / 512e** (`logical == 512`, `physical == 4096`).
//!    The device exposes 512-byte LBAs but the underlying media works
//!    in 4 KiB physical sectors. FS reads issue 8-LBA-aligned 4 KiB
//!    I/O so we never split a logical FS block across two physical
//!    sectors.
//! 3. **Legacy 512 / 512** (`logical == physical == 512`). The
//!    emulation layer bundles 8 consecutive logical sectors per FS
//!    block. A one-time warning is logged at mount because this is
//!    the slow path on modern hardware.
//!
//! All three modes are wrapped uniformly by [`SectorAdapter`], which
//! itself implements [`BlockDevice`] with a fixed
//! [`FS_BLOCK_SIZE_BYTES`] block size. Code above the adapter
//! (squashfs, F2FS, GPT parser, A/B boot) never sees 512-byte LBAs.

use super::{
    check_buf_alignment, check_lba_in_range, BlockDevice, BlockError, FS_BLOCK_SIZE_BYTES,
};

/// LBA-multiplier from a 512 B logical sector to a 4 KiB FS block.
pub const LEGACY_SECTORS_PER_FS_BLOCK: u32 = 8;

/// Sector mode reported by [`SectorAdapter::mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectorMode {
    /// Logical and physical sector are both 4 KiB. Direct passthrough.
    Native4K,
    /// 512 B logical / 4 KiB physical. FS issues 4 KiB-aligned 8-LBA I/O.
    AdvancedFormat512e,
    /// 512 B logical / 512 B physical. Slow-path emulation: bundle 8
    /// logical sectors per FS block. Warned about at mount.
    Legacy512,
}

impl SectorMode {
    /// Pick the appropriate mode from the device-reported `logical`
    /// and `physical` sector sizes.
    ///
    /// Rules:
    /// - `logical == 4096`, `physical == 4096` → `Native4K`
    /// - `logical == 512`, `physical == 4096` → `AdvancedFormat512e`
    /// - `logical == 512`, `physical == 512`  → `Legacy512`
    /// - any other combination → `None` (unsupported)
    pub fn from_sizes(logical: u32, physical: u32) -> Option<Self> {
        match (logical, physical) {
            (4096, 4096) => Some(Self::Native4K),
            (512, 4096) => Some(Self::AdvancedFormat512e),
            (512, 512) => Some(Self::Legacy512),
            _ => None,
        }
    }

    /// `true` when the FS block size requires bundling several
    /// logical sectors per FS block.
    pub fn needs_bundling(self) -> bool {
        matches!(self, Self::AdvancedFormat512e | Self::Legacy512)
    }

    /// `true` when this mode is the legacy 512/512 slow-path that
    /// SHOULD log a one-time warning at mount per the spec.
    pub fn is_legacy_slow_path(self) -> bool {
        matches!(self, Self::Legacy512)
    }
}

/// Wraps any [`BlockDevice`] and presents a 4 KiB FS-block interface.
///
/// Use [`SectorAdapter::new`] to construct from an underlying device
/// plus the device-reported physical sector size.
pub struct SectorAdapter<D: BlockDevice> {
    inner: D,
    mode: SectorMode,
    /// FS-level block count (capacity_bytes / 4 KiB). Pre-computed so
    /// `block_count` is O(1) and we never overflow inside the trait
    /// bounds checks.
    fs_block_count: u64,
}

impl<D: BlockDevice> SectorAdapter<D> {
    /// Construct a new adapter.
    ///
    /// `physical_sector_bytes` is the device-reported *physical* sector
    /// size (which may differ from `inner.block_size_bytes()`, the
    /// logical sector size, for Advanced Format devices).
    ///
    /// Returns `None` if the (logical, physical) combination is not
    /// one of the three supported modes.
    pub fn new(inner: D, physical_sector_bytes: u32) -> Option<Self> {
        let logical = inner.block_size_bytes();
        let mode = SectorMode::from_sizes(logical, physical_sector_bytes)?;
        let inner_capacity_blocks = inner.block_count();
        let fs_block_count = match mode {
            SectorMode::Native4K => inner_capacity_blocks,
            SectorMode::AdvancedFormat512e | SectorMode::Legacy512 => {
                inner_capacity_blocks / LEGACY_SECTORS_PER_FS_BLOCK as u64
            }
        };
        Some(Self {
            inner,
            mode,
            fs_block_count,
        })
    }

    /// Return the operating mode of this adapter.
    pub fn mode(&self) -> SectorMode {
        self.mode
    }

    /// Borrow the underlying device.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Mutable-borrow the underlying device.
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.inner
    }

    /// Translate an FS-level LBA to the underlying-device LBA.
    fn fs_lba_to_inner_lba(&self, fs_lba: u64) -> u64 {
        match self.mode {
            SectorMode::Native4K => fs_lba,
            SectorMode::AdvancedFormat512e | SectorMode::Legacy512 => {
                fs_lba.saturating_mul(LEGACY_SECTORS_PER_FS_BLOCK as u64)
            }
        }
    }
}

impl<D: BlockDevice> BlockDevice for SectorAdapter<D> {
    fn read_block(&self, fs_lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        check_buf_alignment(buf.len(), FS_BLOCK_SIZE_BYTES)?;
        check_lba_in_range(fs_lba, buf.len(), FS_BLOCK_SIZE_BYTES, self.fs_block_count)?;
        let inner_lba = self.fs_lba_to_inner_lba(fs_lba);
        // Buffer length is already a multiple of 4 KiB, which is also
        // a multiple of 512 — so passes the inner alignment check.
        self.inner.read_block(inner_lba, buf)
    }

    fn write_block(&mut self, fs_lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        check_buf_alignment(buf.len(), FS_BLOCK_SIZE_BYTES)?;
        check_lba_in_range(fs_lba, buf.len(), FS_BLOCK_SIZE_BYTES, self.fs_block_count)?;
        let inner_lba = self.fs_lba_to_inner_lba(fs_lba);
        self.inner.write_block(inner_lba, buf)
    }

    fn block_size_bytes(&self) -> u32 {
        FS_BLOCK_SIZE_BYTES
    }

    fn block_count(&self) -> u64 {
        self.fs_block_count
    }

    fn flush(&mut self) -> Result<(), BlockError> {
        self.inner.flush()
    }
}

/// One-time warning hook for the 512/512 legacy slow path.
///
/// Wave 0 has no logging crate yet; this is a thin abstraction so the
/// mount code can register a callback once the audit-ring lands.
/// Default impl is a no-op.
pub trait LegacyPathWarner {
    /// Called exactly once when a 512/512 legacy device is mounted.
    fn warn_legacy_slow_path(&self) {}
}

/// Helper that wraps the adapter constructor and emits the one-time
/// legacy-path warning if applicable.
pub fn new_with_warning<D: BlockDevice, W: LegacyPathWarner>(
    inner: D,
    physical_sector_bytes: u32,
    warner: &W,
) -> Option<SectorAdapter<D>> {
    let adapter = SectorAdapter::new(inner, physical_sector_bytes)?;
    if adapter.mode().is_legacy_slow_path() {
        warner.warn_legacy_slow_path();
    }
    Some(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::mock::MockBlockDevice;

    #[test]
    fn sector_mode_from_sizes() {
        assert_eq!(
            SectorMode::from_sizes(4096, 4096),
            Some(SectorMode::Native4K)
        );
        assert_eq!(
            SectorMode::from_sizes(512, 4096),
            Some(SectorMode::AdvancedFormat512e)
        );
        assert_eq!(
            SectorMode::from_sizes(512, 512),
            Some(SectorMode::Legacy512)
        );
        assert_eq!(SectorMode::from_sizes(1024, 4096), None);
        assert_eq!(SectorMode::from_sizes(4096, 512), None);
    }

    #[test]
    fn native_4k_passthrough() {
        // 16 blocks * 4 KiB = 64 KiB device.
        let mock = MockBlockDevice::new(4096, 16);
        let adapter = SectorAdapter::new(mock, 4096).unwrap();
        assert_eq!(adapter.mode(), SectorMode::Native4K);
        assert_eq!(adapter.block_size_bytes(), 4096);
        assert_eq!(adapter.block_count(), 16);
    }

    #[test]
    fn advanced_format_512e_aligns_to_8_lbas() {
        // 128 logical sectors of 512 = 64 KiB device, exposes 16 FS blocks.
        let mock = MockBlockDevice::new(512, 128);
        let adapter = SectorAdapter::new(mock, 4096).unwrap();
        assert_eq!(adapter.mode(), SectorMode::AdvancedFormat512e);
        assert_eq!(adapter.block_size_bytes(), 4096);
        assert_eq!(adapter.block_count(), 16);
        // Internal LBA mapping multiplies by 8.
        assert_eq!(adapter.fs_lba_to_inner_lba(0), 0);
        assert_eq!(adapter.fs_lba_to_inner_lba(1), 8);
        assert_eq!(adapter.fs_lba_to_inner_lba(15), 120);
    }

    #[test]
    fn legacy_512_marks_slow_path() {
        let mock = MockBlockDevice::new(512, 128);
        let adapter = SectorAdapter::new(mock, 512).unwrap();
        assert_eq!(adapter.mode(), SectorMode::Legacy512);
        assert!(adapter.mode().is_legacy_slow_path());
        assert_eq!(adapter.block_count(), 16);
    }

    #[test]
    fn unaligned_buf_rejected_at_adapter() {
        let mock = MockBlockDevice::new(4096, 16);
        let adapter = SectorAdapter::new(mock, 4096).unwrap();
        let mut buf = alloc::vec![0u8; 4097];
        assert_eq!(adapter.read_block(0, &mut buf), Err(BlockError::Unaligned));
    }

    #[test]
    fn out_of_range_rejected_at_adapter() {
        let mock = MockBlockDevice::new(4096, 16);
        let adapter = SectorAdapter::new(mock, 4096).unwrap();
        let mut buf = alloc::vec![0u8; 4096];
        assert_eq!(
            adapter.read_block(16, &mut buf),
            Err(BlockError::OutOfRange)
        );
    }

    #[test]
    fn legacy_path_warning_fires_once() {
        use core::cell::Cell;

        struct CountingWarner(Cell<u32>);
        impl LegacyPathWarner for CountingWarner {
            fn warn_legacy_slow_path(&self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let warner = CountingWarner(Cell::new(0));
        let _adapter = new_with_warning(MockBlockDevice::new(512, 128), 512, &warner).unwrap();
        assert_eq!(warner.0.get(), 1);
        // No second adapter constructed → no second warning. The "once"
        // contract is at the call site (mount only fires this once); the
        // helper itself emits at construction.
    }

    #[test]
    fn non_legacy_path_no_warning() {
        use core::cell::Cell;

        struct CountingWarner(Cell<u32>);
        impl LegacyPathWarner for CountingWarner {
            fn warn_legacy_slow_path(&self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let warner = CountingWarner(Cell::new(0));
        let _adapter = new_with_warning(MockBlockDevice::new(4096, 16), 4096, &warner).unwrap();
        assert_eq!(warner.0.get(), 0);
        let _adapter = new_with_warning(MockBlockDevice::new(512, 128), 4096, &warner).unwrap();
        assert_eq!(warner.0.get(), 0);
    }

    #[test]
    fn read_then_write_via_adapter_roundtrips() {
        let mock = MockBlockDevice::new(512, 128);
        let mut adapter = SectorAdapter::new(mock, 4096).unwrap();
        let pattern: alloc::vec::Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        adapter.write_block(3, &pattern).unwrap();
        let mut readback = alloc::vec![0u8; 4096];
        adapter.read_block(3, &mut readback).unwrap();
        assert_eq!(readback, pattern);
    }
}
