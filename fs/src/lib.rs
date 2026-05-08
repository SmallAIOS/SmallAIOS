// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! On-disk and raw-flash filesystem stack for SmallAIOS.
//!
//! Three orthogonal storage stacks live in this crate, each behind its
//! own cargo feature so targets pay only for what they use:
//!
//! 1. **Block-device on-disk filesystems** (`fs-on-disk-mounts`) —
//!    GPT partition tables, squashfs (read-only `/models/`, A/B slots),
//!    F2FS (read-write `/data/`), A/B boot pointer, bsdiff delta-update
//!    pipeline. Owned by `embedded-filesystem-v1`.
//!
//! 2. **Overlay layer** (`fs-overlay-mounts`) — OverlayFS-style
//!    writable layer on top of `/models/` so operators can drop in
//!    custom models without re-imaging. Owned by
//!    `embedded-overlay-v1`.
//!
//! 3. **Raw-flash filesystem** (`fs-flash`) — littlefs v2.x for
//!    MCU/FPGA targets where there is no block device (raw NOR/NAND
//!    flash on a memory-mapped bus). Owned by
//!    `embedded-flash-fs-v1`.
//!
//! ## Layer
//!
//! `fs` sits at **Layer 1**, peer to `auth`, `mgmt`, `ipc`, `net`,
//! `posix`, `onnx-rt`, `usb`. It depends on `kernel` (Layer 0),
//! `security` (Layer 0, for SHA-3-256 + ML-DSA-65), and `mgmt`
//! (Layer 1 peer, for audit ring + config fields).
//!
//! ## Status
//!
//! Wave 0 **skeleton**. Implementation phases land per the agent-team
//! plan; see each owning OpenSpec change for the phase breakdown.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

// ─── Module skeleton ─────────────────────────────────────────────────────────
//
// Each `pub mod` is gated by its owning cargo feature. Modules contain
// no code in Wave 0; the implementation agents fill them.

/// Block-device abstraction (`BlockDevice` trait, `BlockError` enum,
/// per-arch impls, retry/timeout policy, sector-size emulation).
///
/// Filled by `embedded-filesystem-v1` Phase 1.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod block;

/// GPT partition table parser + protective-MBR writer + v1 layout
/// enforcement.
///
/// Filled by `embedded-filesystem-v1` Phase 2.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod gpt {}

/// Squashfs 4.0 read-only reader with zstd/xz/gzip/lz4 decoders and
/// the appended SmallAIOS manifest-footer verifier.
///
/// Filled by `embedded-filesystem-v1` Phase 3.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod squashfs {}

/// F2FS read+write implementation (Linux 6.6 LTS feature set):
/// superblock, checkpoint, NAT, SIT, SSA, inode, dir, data, fsync,
/// garbage collection.
///
/// Filled by `embedded-filesystem-v1` Phases 5 → 8.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod f2fs {}

/// A/B boot pointer (8 MiB partition, double-buffered records,
/// monotonic generation counter, UEFI variable mirror).
///
/// Filled by `embedded-filesystem-v1` Phase 4.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod boot_config {}

/// bsdiff delta-update applier with ML-DSA-65 signed pre/post checks.
///
/// Filled by `embedded-filesystem-v1` Phase 4.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod delta {}

/// Per-mount LRU block cache (16 MiB models, 4 MiB data) +
/// 256 KiB write-coalescing buffer.
///
/// Filled by `embedded-filesystem-v1` Phase 6.
#[cfg(feature = "fs-on-disk-mounts")]
pub mod cache {}

/// OverlayFS-style merged `/models/` view (upper = F2FS, lower =
/// squashfs A/B slot), file-based whiteouts, capacity-cap, optional
/// ML-DSA-65 signed-policy enforcement.
///
/// Filled by `embedded-overlay-v1` Phases 1 → 6.
#[cfg(feature = "fs-overlay-mounts")]
pub mod overlay {}

/// Raw-flash filesystem (littlefs v2.x) for MCU/FPGA targets.
/// Includes `FlashDevice` trait, per-bus drivers (QSPI NOR, ONFI NAND),
/// in-memory mock for CI, and the littlefs reader+writer.
///
/// Filled by `embedded-flash-fs-v1` Phases 1 → N.
#[cfg(feature = "fs-flash")]
pub mod flash {}
