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
pub mod qspi;

#[cfg(feature = "fs-flash-mock")]
pub mod mock;

pub use device::{flash_error_to_errno, flash_error_to_negative_errno, FlashDevice, FlashError};
