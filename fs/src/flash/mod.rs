// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Raw-flash filesystem layer for SmallAIOS (`embedded-flash-fs-v1`).
//!
//! This module is the home for the raw-NOR / raw-NAND filesystem stack
//! that turns a [`FlashDevice`](device::FlashDevice) into a usable
//! POSIX-shaped tree mounted under `/flash/`. It is intentionally
//! separate from the block-device + on-disk filesystem code in
//! [`crate::block`] because raw flash needs explicit erase-block
//! accounting, bad-block management, and asymmetric program/erase
//! semantics that block devices abstract away.
//!
//! ## Submodules
//!
//! * [`device`] — `FlashDevice` trait + `FlashError` enum + POSIX
//!   errno conversion. The single abstraction layer between any
//!   filesystem here and the underlying medium.
//! * [`mock`] — In-memory simulator behind the `fs-flash-mock`
//!   feature, used by every conformance test and by fixture
//!   generators. NOT compiled into release builds.
//! * [`qspi`] — QSPI NOR driver scaffolding (Phase 1 stub).
//! * [`onfi`] — ONFI NAND driver scaffolding (Phase 1 stub).
//! * [`littlefs`] — clean-room `#![no_std]` reader for the littlefs
//!   v2.x format. The write path lives behind a feature flag and
//!   lands in Phase 2.
//!
//! ## Phase 1 scope
//!
//! Phase 1 (this crate revision) lands:
//!
//! * `FlashDevice` trait + typed `FlashError` enum.
//! * `MockFlashDevice` with bit-flip and bad-block-on-erase injection.
//! * littlefs v2.x **read-only** reader (superblock, metadata pairs,
//!   CTZ-skip-list file reads, directory iteration, CRC32C).
//! * QSPI / ONFI per-arch driver stubs (`FlashError::NotPresent` for
//!   every operation until the per-arch controller binds).
//!
//! Subsequent phases land the write path, fsync semantics, wear-
//! leveling / BBT integration, and the `/flash/` VFS mount.

pub mod device;
pub mod littlefs;
pub mod onfi;
pub mod presence;
pub mod qspi;

#[cfg(feature = "fs-flash-mock")]
pub mod mock;

pub use device::{flash_error_to_errno, flash_error_to_negative_errno, FlashDevice, FlashError};
pub use presence::{KernelPresenceProvider, PhysicalPresenceProvider, TestPresenceProvider};

use littlefs::{LittleFs, LittleFsConfig, LittleFsError};

/// Reasons the `/flash/` boot sequencer may decline to mount.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlashMountFailure {
    /// The medium is unformatted AND no physical-presence signal is
    /// asserted. The kernel SHALL halt and surface a recovery message
    /// rather than format silently.
    UnformattedNoPresence,
    /// The medium is formatted but mounting failed (e.g. CRC mismatch
    /// on both halves). The wrapped error gives the proximate cause.
    MountFailed(LittleFsError),
    /// The medium was unformatted, presence was asserted, and the
    /// `format()` step itself failed (typically because the medium has
    /// physical I/O errors).
    FormatFailed(LittleFsError),
}

/// Boot-time `/flash/` mount sequencer.
///
/// Decision tree:
///
/// 1. If the medium already passes [`LittleFs::is_formatted`], skip to
///    step 4.
/// 2. Otherwise, ask `presence` whether the operator has asserted the
///    physical-presence signal.
/// 3. If presence is asserted, run [`LittleFs::format`] and continue to
///    step 4. Otherwise, return [`FlashMountFailure::UnformattedNoPresence`]
///    so the kernel halts with a recovery message.
/// 4. Mount the filesystem and return the live handle.
///
/// Phase 3 of `embedded-flash-fs-v1` exposes this orchestrator behind
/// the `fs-flash` cargo feature; per-arch boot code calls it from the
/// platform bring-up sequence (after the per-arch QSPI/ONFI controller
/// has bound to the device).
pub fn mount_or_format<'a, D: FlashDevice, P: PhysicalPresenceProvider>(
    device: &'a mut D,
    config: LittleFsConfig,
    presence: &P,
) -> Result<LittleFs<'a, D>, FlashMountFailure> {
    if !LittleFs::is_formatted(device, config) {
        if !presence.asserted() {
            return Err(FlashMountFailure::UnformattedNoPresence);
        }
        LittleFs::format(device, config).map_err(FlashMountFailure::FormatFailed)?;
    }
    LittleFs::mount(device, config).map_err(FlashMountFailure::MountFailed)
}

/// Diagnostic record of which on-disk filesystems are currently
/// mounted.
///
/// Used by boot logs and `/proc/mounts`-style introspection to confirm
/// the F2FS `/data/` and littlefs `/flash/` substrates are coexisting
/// as designed. `embedded-filesystem-v1` Phase 6 will populate the F2FS
/// side; v1 here only reports the flash side.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MountTable {
    /// Whether `/flash/` (littlefs) is currently mounted.
    pub flash_mounted: bool,
    /// Whether `/data/` (F2FS) is currently mounted.
    pub data_mounted: bool,
}

impl MountTable {
    /// Construct an empty mount table (nothing mounted).
    pub const fn new() -> Self {
        Self {
            flash_mounted: false,
            data_mounted: false,
        }
    }

    /// `true` iff *both* `/flash/` (littlefs) and `/data/` (F2FS) are
    /// mounted side-by-side. Targets with both eMMC and QSPI NOR (e.g.
    /// the planned Jetson-class units) SHALL report `true` here once
    /// boot completes; minimal MCU/FPGA targets without an eMMC stay at
    /// `flash_mounted = true, data_mounted = false`.
    pub const fn has_both(&self) -> bool {
        self.flash_mounted && self.data_mounted
    }
}

#[cfg(test)]
#[cfg(feature = "fs-flash-mock")]
mod tests {
    use super::*;
    use crate::flash::mock::MockFlashDevice;

    #[test]
    fn mount_or_format_refuses_unformatted_without_presence() {
        let mut dev = MockFlashDevice::new(8, 4096, 256);
        let cfg = LittleFsConfig::for_device(&dev);
        let p = TestPresenceProvider::asserted(false);
        let res = mount_or_format(&mut dev, cfg, &p);
        assert_eq!(res.err().unwrap(), FlashMountFailure::UnformattedNoPresence);
    }

    #[test]
    fn mount_or_format_formats_when_presence_asserted() {
        let mut dev = MockFlashDevice::new(8, 4096, 256);
        let cfg = LittleFsConfig::for_device(&dev);
        let p = TestPresenceProvider::asserted(true);
        let fs = mount_or_format(&mut dev, cfg, &p).expect("format-on-presence");
        assert_eq!(fs.config().block_size, 4096);
    }

    #[test]
    fn is_formatted_round_trips_after_format() {
        let mut dev = MockFlashDevice::new(8, 4096, 256);
        let cfg = LittleFsConfig::for_device(&dev);
        assert!(!LittleFs::is_formatted(&mut dev, cfg));
        LittleFs::format(&mut dev, cfg).unwrap();
        assert!(LittleFs::is_formatted(&mut dev, cfg));
    }

    #[test]
    fn mount_or_format_skips_format_on_already_formatted_medium() {
        let mut dev = MockFlashDevice::new(8, 4096, 256);
        let cfg = LittleFsConfig::for_device(&dev);
        // Pre-format the device.
        LittleFs::format(&mut dev, cfg).unwrap();
        // Even with presence not asserted, mount-only should succeed.
        let p = TestPresenceProvider::asserted(false);
        let fs = mount_or_format(&mut dev, cfg, &p).expect("mount existing image");
        assert_eq!(fs.superblock().block_size, 4096);
    }

    #[test]
    fn mount_table_has_both_only_when_both_mounted() {
        assert!(!MountTable::new().has_both());
        let only_flash = MountTable {
            flash_mounted: true,
            data_mounted: false,
        };
        assert!(!only_flash.has_both());
        let only_data = MountTable {
            flash_mounted: false,
            data_mounted: true,
        };
        assert!(!only_data.has_both());
        let both = MountTable {
            flash_mounted: true,
            data_mounted: true,
        };
        assert!(both.has_both());
    }
}
