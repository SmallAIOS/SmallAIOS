// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! A/B boot pointer (8 MiB partition, double-buffered records,
//! monotonic generation counter, UEFI variable mirror).
//!
//! Implements the on-disk format documented in
//! `openspec/changes/embedded-filesystem-v1/specs/fs-ab-boot/spec.md`.
//!
//! The boot-config partition (GPT type
//! `A3F7C2E0-FACE-4FFF-BBBB-000000000000`, partition #5) holds two
//! 4 KiB double-buffered records at fixed offsets `0x000` and `0x1000`.
//! On every update the kernel:
//!
//! 1. Picks the **inactive** slot (lower `generation`).
//! 2. Writes the new record bytes into the inactive slot.
//! 3. Calls `flush()` on the underlying [`crate::block::BlockDevice`].
//!
//! Power loss before step 3 leaves the previously-active slot intact;
//! the bootloader's "highest-valid-generation" rule means it picks the
//! old slot.
//!
//! Owned by `embedded-filesystem-v1` Phase 4.

extern crate alloc;

use core::fmt;

use crate::block::{BlockDevice, BlockError, FS_BLOCK_SIZE_BYTES};

pub mod uefi_mirror;
pub mod watchdog;

pub use uefi_mirror::{KernelUefiMirror, TestUefiMirror, UefiMirror, UEFI_VARIABLE_NAME};
pub use watchdog::{boot_success, KernelWatchdog, MockWatchdog, Watchdog};

/// Magic bytes at offset 0 of every [`BootConfigRecord`].
//
// lgtm[rust/hard-coded-cryptographic-value]
pub const BOOT_CONFIG_MAGIC: [u8; 8] = *b"SmAIOSBC";

/// Current boot-config record version.
pub const BOOT_CONFIG_VERSION: u8 = 1;

/// Size of one [`BootConfigRecord`] on disk, in bytes.
///
/// One record fills exactly one 4 KiB filesystem block so that the two
/// double-buffered slots sit at LBA-aligned offsets 0x000 and 0x1000
/// of the partition, and so that each slot is a single atomic write
/// from the underlying transport's perspective.
pub const BOOT_CONFIG_RECORD_SIZE: usize = FS_BLOCK_SIZE_BYTES as usize;

/// On-disk offset of slot A within the boot-config partition.
pub const SLOT_A_OFFSET: u64 = 0x0000;

/// On-disk offset of slot B within the boot-config partition.
pub const SLOT_B_OFFSET: u64 = 0x1000;

/// LBA delta from `partition_lba` to slot A (always 0).
pub const SLOT_A_LBA_DELTA: u64 = 0;

/// LBA delta from `partition_lba` to slot B (one 4 KiB block in).
pub const SLOT_B_LBA_DELTA: u64 = 1;

/// Length of the SHA-3-256 record hash trailer.
const RECORD_HASH_LEN: usize = 32;

/// Length of the variable-payload region preceding `record_hash`.
///
/// `record_size - 32` bytes are hashed; the trailing 32 bytes hold
/// the SHA-3-256 of the preceding bytes.
pub const RECORD_BODY_LEN: usize = BOOT_CONFIG_RECORD_SIZE - RECORD_HASH_LEN;

/// Which of the two double-buffered records this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotPosition {
    /// On-disk slot at offset `0x000`.
    SlotX,
    /// On-disk slot at offset `0x1000`.
    SlotY,
}

impl SlotPosition {
    /// LBA delta within the boot-config partition for this slot.
    pub const fn lba_delta(&self) -> u64 {
        match self {
            Self::SlotX => SLOT_A_LBA_DELTA,
            Self::SlotY => SLOT_B_LBA_DELTA,
        }
    }

    /// The opposite slot (inactive ↔ active).
    pub const fn other(&self) -> Self {
        match self {
            Self::SlotX => Self::SlotY,
            Self::SlotY => Self::SlotX,
        }
    }
}

/// Which squashfs slot is currently the boot target (A or B).
///
/// This is `active_slot` in the on-disk record. Note: this is the
/// **logical** A/B target and is independent of which physical
/// `SlotPosition` (X or Y) the record itself was written to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActiveSquashfsSlot {
    /// Logical squashfs slot A.
    A,
    /// Logical squashfs slot B.
    B,
}

impl ActiveSquashfsSlot {
    /// Decode from on-disk byte (0 = A, 1 = B). Anything else is a
    /// corruption signal.
    pub const fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::A),
            1 => Some(Self::B),
            _ => None,
        }
    }

    /// Encode to on-disk byte.
    pub const fn as_u8(&self) -> u8 {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    /// Flip A ↔ B.
    pub const fn flip(&self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

/// One double-buffered boot-config record.
///
/// On-disk layout (4 KiB total, little-endian fields):
///
/// ```text
///   off  size  field
///   0    8     magic = "SmAIOSBC"
///   8    1     version
///   9    1     valid          (1 = valid, 0 = invalid)
///   10   1     active_slot    (0 = A, 1 = B)
///   11   1     tentative      (1 = updated, awaiting boot_success)
///   12   8     generation     (u64 LE)
///   20   1     boot_success
///   21   N     pad (zeros)
///   E-32 32    record_hash    SHA-3-256 of bytes [0..E-32]
/// ```
///
/// The record is a Rust struct (not `#[repr(C, packed)]`) — the
/// on-disk encoder/decoder lives in [`encode_record`] / [`decode_record`].
/// Avoiding `#[repr(C, packed)]` keeps `forbid(unsafe_code)` clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootConfigRecord {
    /// Constant: [`BOOT_CONFIG_MAGIC`]. Carried so callers can verify
    /// after decode if they want to.
    pub magic: [u8; 8],
    /// Constant: [`BOOT_CONFIG_VERSION`].
    pub version: u8,
    /// `true` iff the bootloader should consider this record at all.
    pub valid: bool,
    /// Which squashfs slot to boot from.
    pub active_slot: ActiveSquashfsSlot,
    /// `true` while the kernel waits for the watchdog `boot_success`.
    pub tentative: bool,
    /// Monotonically increasing generation counter. Higher wins.
    pub generation: u64,
    /// `true` once the kernel called the `SYS_BOOT_SUCCESS` syscall.
    pub boot_success: bool,
    /// SHA-3-256 of bytes `[0..record_size - 32]`. Filled by
    /// [`encode_record`].
    pub record_hash: [u8; 32],
}

impl BootConfigRecord {
    /// Construct a fresh in-RAM record with default flags. The
    /// `record_hash` is zero until [`encode_record`] is called; in
    /// callers, `update_atomic` recomputes the hash before write.
    pub fn new(active_slot: ActiveSquashfsSlot, generation: u64) -> Self {
        Self {
            magic: BOOT_CONFIG_MAGIC,
            version: BOOT_CONFIG_VERSION,
            valid: true,
            active_slot,
            tentative: false,
            generation,
            boot_success: false,
            record_hash: [0u8; 32],
        }
    }
}

/// Errors surfaced by the boot-config layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootConfigError {
    /// Both records' magic / version / record_hash failed validation.
    BothSlotsInvalid,
    /// On-disk magic is not [`BOOT_CONFIG_MAGIC`].
    BadMagic,
    /// On-disk version is not [`BOOT_CONFIG_VERSION`].
    UnsupportedVersion(u8),
    /// `record_hash` did not match the recomputed SHA-3-256.
    BadRecordHash,
    /// `active_slot` byte was not 0 or 1.
    BadActiveSlot,
    /// Caller passed a buffer whose length is not
    /// [`BOOT_CONFIG_RECORD_SIZE`].
    BadBufferLen,
    /// New record's generation is not strictly greater than the
    /// current maximum (would violate the "monotonic, never reused"
    /// invariant of the spec).
    GenerationNotMonotonic,
    /// The underlying block device returned an error.
    Block(BlockError),
}

impl From<BlockError> for BootConfigError {
    fn from(e: BlockError) -> Self {
        Self::Block(e)
    }
}

impl fmt::Display for BootConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BothSlotsInvalid => f.write_str("boot config corrupt, both slots"),
            Self::BadMagic => f.write_str("boot config bad magic"),
            Self::UnsupportedVersion(v) => write!(f, "boot config unsupported version {v}"),
            Self::BadRecordHash => f.write_str("boot config record_hash mismatch"),
            Self::BadActiveSlot => f.write_str("boot config active_slot byte invalid"),
            Self::BadBufferLen => f.write_str("boot config buffer length wrong"),
            Self::GenerationNotMonotonic => {
                f.write_str("boot config generation not monotonically increasing")
            }
            Self::Block(e) => write!(f, "boot config block error: {e}"),
        }
    }
}

/// Encode a [`BootConfigRecord`] into its [`BOOT_CONFIG_RECORD_SIZE`]
/// on-disk byte form, including the trailing SHA-3-256 hash.
pub fn encode_record(rec: &BootConfigRecord) -> [u8; BOOT_CONFIG_RECORD_SIZE] {
    let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
    buf[0..8].copy_from_slice(&rec.magic);
    buf[8] = rec.version;
    buf[9] = rec.valid as u8;
    buf[10] = rec.active_slot.as_u8();
    buf[11] = rec.tentative as u8;
    buf[12..20].copy_from_slice(&rec.generation.to_le_bytes());
    buf[20] = rec.boot_success as u8;
    // [21..RECORD_BODY_LEN] stays zero.
    let hash = compute_record_hash(&buf[..RECORD_BODY_LEN]);
    buf[RECORD_BODY_LEN..].copy_from_slice(&hash);
    buf
}

/// Decode an on-disk record. Validates magic, version, and the
/// SHA-3-256 record hash. Returns `Err` on any mismatch.
///
/// The decoder is strict: a record with `record_hash` mismatch is
/// rejected as corrupt regardless of how plausible its other fields
/// look. This keeps the "valid" predicate equivalent to "intact-on-
/// disk", which is what the bootloader's pick-highest-valid rule
/// needs to be safe.
pub fn decode_record(buf: &[u8]) -> Result<BootConfigRecord, BootConfigError> {
    if buf.len() != BOOT_CONFIG_RECORD_SIZE {
        return Err(BootConfigError::BadBufferLen);
    }
    if buf[0..8] != BOOT_CONFIG_MAGIC {
        return Err(BootConfigError::BadMagic);
    }
    let version = buf[8];
    if version != BOOT_CONFIG_VERSION {
        return Err(BootConfigError::UnsupportedVersion(version));
    }
    // Recompute the hash of the body and compare to the trailer.
    let hash = compute_record_hash(&buf[..RECORD_BODY_LEN]);
    if hash != buf[RECORD_BODY_LEN..] {
        return Err(BootConfigError::BadRecordHash);
    }
    let valid = buf[9] != 0;
    let active_slot = ActiveSquashfsSlot::from_u8(buf[10]).ok_or(BootConfigError::BadActiveSlot)?;
    let tentative = buf[11] != 0;
    let mut gen_bytes = [0u8; 8];
    gen_bytes.copy_from_slice(&buf[12..20]);
    let generation = u64::from_le_bytes(gen_bytes);
    let boot_success = buf[20] != 0;
    let mut record_hash = [0u8; 32];
    record_hash.copy_from_slice(&buf[RECORD_BODY_LEN..]);
    Ok(BootConfigRecord {
        magic: BOOT_CONFIG_MAGIC,
        version,
        valid,
        active_slot,
        tentative,
        generation,
        boot_success,
        record_hash,
    })
}

/// SHA-3-256 of `buf` returned as a 32-byte array.
fn compute_record_hash(buf: &[u8]) -> [u8; 32] {
    use smallaios_security::crypto::sha3::sha3_256;
    *sha3_256(buf).as_bytes()
}

/// Read the active boot record from the boot-config partition.
///
/// `partition_lba` is the LBA of the boot-config partition on the
/// underlying [`BlockDevice`]. Reads both slots, validates each, and
/// picks the valid one with the highest `generation`. If both fail
/// validation returns [`BootConfigError::BothSlotsInvalid`].
pub fn read_active<D: BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
) -> Result<(BootConfigRecord, SlotPosition), BootConfigError> {
    let bs = device.block_size_bytes() as usize;
    if bs != BOOT_CONFIG_RECORD_SIZE {
        // We require the underlying device to expose 4 KiB blocks.
        // Devices with a different native sector size MUST be wrapped
        // by [`crate::block::sector::SectorAdapter`] before being
        // handed to the boot-config layer.
        return Err(BootConfigError::Block(BlockError::Unaligned));
    }
    let slot_x = read_slot(device, partition_lba, SlotPosition::SlotX);
    let slot_y = read_slot(device, partition_lba, SlotPosition::SlotY);
    pick_active(slot_x, slot_y)
}

fn read_slot<D: BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    slot: SlotPosition,
) -> Result<BootConfigRecord, BootConfigError> {
    let lba = partition_lba
        .checked_add(slot.lba_delta())
        .ok_or(BootConfigError::Block(BlockError::OutOfRange))?;
    let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
    device.read_block(lba, &mut buf)?;
    decode_record(&buf)
}

/// Decide which of two read attempts is "the active record".
fn pick_active(
    slot_x: Result<BootConfigRecord, BootConfigError>,
    slot_y: Result<BootConfigRecord, BootConfigError>,
) -> Result<(BootConfigRecord, SlotPosition), BootConfigError> {
    let x_valid = slot_x.as_ref().map(|r| r.valid).unwrap_or(false);
    let y_valid = slot_y.as_ref().map(|r| r.valid).unwrap_or(false);
    match (x_valid, y_valid) {
        (false, false) => Err(BootConfigError::BothSlotsInvalid),
        (true, false) => Ok((slot_x.unwrap(), SlotPosition::SlotX)),
        (false, true) => Ok((slot_y.unwrap(), SlotPosition::SlotY)),
        (true, true) => {
            let rx = slot_x.unwrap();
            let ry = slot_y.unwrap();
            if rx.generation >= ry.generation {
                Ok((rx, SlotPosition::SlotX))
            } else {
                Ok((ry, SlotPosition::SlotY))
            }
        }
    }
}

/// Atomically update the boot-config partition with a new record.
///
/// The new record is encoded with `record_hash` recomputed, then
/// written to the **inactive** slot (the slot whose decoded record
/// has the lower `generation`, or the slot that does not validate at
/// all). After `flush()`, the new slot wins the highest-valid-
/// generation rule and becomes the active record on next boot.
///
/// Per the spec, `new_record.generation` MUST be strictly greater
/// than the current max generation across both slots; otherwise
/// returns [`BootConfigError::GenerationNotMonotonic`].
///
/// If neither slot is currently valid, the new record is written to
/// [`SlotPosition::SlotX`].
pub fn update_atomic<D: BlockDevice + ?Sized>(
    device: &mut D,
    partition_lba: u64,
    new_record: BootConfigRecord,
) -> Result<SlotPosition, BootConfigError> {
    let bs = device.block_size_bytes() as usize;
    if bs != BOOT_CONFIG_RECORD_SIZE {
        return Err(BootConfigError::Block(BlockError::Unaligned));
    }

    let slot_x = read_slot(device, partition_lba, SlotPosition::SlotX);
    let slot_y = read_slot(device, partition_lba, SlotPosition::SlotY);

    let (x_gen, x_valid) = match &slot_x {
        Ok(r) => (Some(r.generation), r.valid),
        Err(_) => (None, false),
    };
    let (y_gen, y_valid) = match &slot_y {
        Ok(r) => (Some(r.generation), r.valid),
        Err(_) => (None, false),
    };

    let max_gen = match (x_gen, y_gen) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    };
    if let Some(g) = max_gen {
        if new_record.generation <= g {
            return Err(BootConfigError::GenerationNotMonotonic);
        }
    }

    // Determine target slot — the one with the lower (or absent)
    // generation. If neither validates, default to SlotX.
    let target = match (x_valid, y_valid) {
        (true, true) => {
            // Both valid — write to the lower-gen slot.
            if x_gen.unwrap_or(0) <= y_gen.unwrap_or(0) {
                SlotPosition::SlotX
            } else {
                SlotPosition::SlotY
            }
        }
        (true, false) => SlotPosition::SlotY, // overwrite the broken one
        (false, true) => SlotPosition::SlotX,
        (false, false) => SlotPosition::SlotX,
    };

    let buf = encode_record(&new_record);
    let lba = partition_lba
        .checked_add(target.lba_delta())
        .ok_or(BootConfigError::Block(BlockError::OutOfRange))?;
    device.write_block(lba, &buf)?;
    device.flush()?;
    Ok(target)
}

/// Mark the record at `slot` as `valid = 0`.
///
/// Used after a failed delta-update post-apply check, after watchdog
/// rollback, or by recovery tooling. Reads the existing record (if it
/// decodes), flips `valid`, recomputes `record_hash`, and writes it
/// back to the same physical slot.
///
/// If the slot does not currently decode (already corrupt), this
/// returns `Ok(())` — the record is already not "valid" in the
/// bootloader's sense.
pub fn mark_invalid<D: BlockDevice + ?Sized>(
    device: &mut D,
    partition_lba: u64,
    slot: SlotPosition,
) -> Result<(), BootConfigError> {
    let bs = device.block_size_bytes() as usize;
    if bs != BOOT_CONFIG_RECORD_SIZE {
        return Err(BootConfigError::Block(BlockError::Unaligned));
    }
    let lba = partition_lba
        .checked_add(slot.lba_delta())
        .ok_or(BootConfigError::Block(BlockError::OutOfRange))?;
    let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
    device.read_block(lba, &mut buf)?;
    match decode_record(&buf) {
        Ok(mut rec) => {
            rec.valid = false;
            let new_buf = encode_record(&rec);
            device.write_block(lba, &new_buf)?;
            device.flush()?;
            Ok(())
        }
        Err(_) => {
            // Already corrupt — bootloader will skip it. Nothing to do.
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::mock::MockBlockDevice;

    fn fresh_partition(blocks: u64) -> MockBlockDevice {
        // Boot-config partition is 8 MiB == 2048 4 KiB blocks. Tests
        // use a smaller partition rooted at LBA 0 for clarity.
        MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, blocks)
    }

    #[test]
    fn round_trip_record() {
        let r = BootConfigRecord {
            magic: BOOT_CONFIG_MAGIC,
            version: BOOT_CONFIG_VERSION,
            valid: true,
            active_slot: ActiveSquashfsSlot::B,
            tentative: true,
            generation: 0xdead_beef_cafe_babe,
            boot_success: false,
            record_hash: [0u8; 32],
        };
        let buf = encode_record(&r);
        let decoded = decode_record(&buf).unwrap();
        assert_eq!(decoded.active_slot, ActiveSquashfsSlot::B);
        assert!(decoded.tentative);
        assert_eq!(decoded.generation, 0xdead_beef_cafe_babe);
        assert!(decoded.valid);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let r = BootConfigRecord::new(ActiveSquashfsSlot::A, 1);
        let mut buf = encode_record(&r);
        buf[0] ^= 0xFF;
        assert!(matches!(
            decode_record(&buf),
            Err(BootConfigError::BadMagic)
        ));
    }

    #[test]
    fn decode_rejects_bad_version() {
        let r = BootConfigRecord::new(ActiveSquashfsSlot::A, 1);
        let mut buf = encode_record(&r);
        buf[8] = 0xFF;
        // re-hash so version is the only check that fails
        let h = compute_record_hash(&buf[..RECORD_BODY_LEN]);
        buf[RECORD_BODY_LEN..].copy_from_slice(&h);
        assert!(matches!(
            decode_record(&buf),
            Err(BootConfigError::UnsupportedVersion(0xFF))
        ));
    }

    #[test]
    fn decode_rejects_bad_record_hash() {
        let r = BootConfigRecord::new(ActiveSquashfsSlot::A, 1);
        let mut buf = encode_record(&r);
        // Flip a body byte without re-hashing.
        buf[12] ^= 0xFF;
        assert!(matches!(
            decode_record(&buf),
            Err(BootConfigError::BadRecordHash)
        ));
    }

    #[test]
    fn decode_rejects_bad_active_slot() {
        let r = BootConfigRecord::new(ActiveSquashfsSlot::A, 1);
        let mut buf = encode_record(&r);
        buf[10] = 7;
        // re-hash so the bogus byte survives the trailer check
        let h = compute_record_hash(&buf[..RECORD_BODY_LEN]);
        buf[RECORD_BODY_LEN..].copy_from_slice(&h);
        assert!(matches!(
            decode_record(&buf),
            Err(BootConfigError::BadActiveSlot)
        ));
    }

    #[test]
    fn decode_rejects_wrong_buffer_len() {
        let r = BootConfigRecord::new(ActiveSquashfsSlot::A, 1);
        let mut buf = encode_record(&r).to_vec();
        buf.pop();
        assert!(matches!(
            decode_record(&buf),
            Err(BootConfigError::BadBufferLen)
        ));
    }

    #[test]
    fn read_active_picks_highest_generation() {
        let mut dev = fresh_partition(4);
        // Write SlotX with gen=4, SlotY with gen=5 — Y wins.
        let rx = BootConfigRecord {
            generation: 4,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 4)
        };
        let ry = BootConfigRecord {
            generation: 5,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 5)
        };
        dev.write_block(0, &encode_record(&rx)).unwrap();
        dev.write_block(1, &encode_record(&ry)).unwrap();
        let (rec, slot) = read_active(&dev, 0).unwrap();
        assert_eq!(slot, SlotPosition::SlotY);
        assert_eq!(rec.generation, 5);
        assert_eq!(rec.active_slot, ActiveSquashfsSlot::B);
    }

    #[test]
    fn read_active_skips_corrupt_slot() {
        let mut dev = fresh_partition(4);
        // SlotX has bad magic; SlotY is valid at gen=4.
        let mut bad = encode_record(&BootConfigRecord::new(ActiveSquashfsSlot::A, 9));
        bad[0] = 0;
        dev.write_block(0, &bad).unwrap();
        let ry = BootConfigRecord {
            generation: 4,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 4)
        };
        dev.write_block(1, &encode_record(&ry)).unwrap();
        let (rec, slot) = read_active(&dev, 0).unwrap();
        assert_eq!(slot, SlotPosition::SlotY);
        assert_eq!(rec.generation, 4);
    }

    #[test]
    fn read_active_both_invalid_errors() {
        let mut dev = fresh_partition(4);
        // Both slots all-zero: bad magic.
        dev.write_block(0, &[0u8; BOOT_CONFIG_RECORD_SIZE]).unwrap();
        dev.write_block(1, &[0u8; BOOT_CONFIG_RECORD_SIZE]).unwrap();
        assert!(matches!(
            read_active(&dev, 0),
            Err(BootConfigError::BothSlotsInvalid)
        ));
    }

    #[test]
    fn read_active_skips_explicit_invalid_flag() {
        // valid=0 records should NOT be picked even if their hash is fine.
        let mut dev = fresh_partition(4);
        let rx = BootConfigRecord {
            valid: false,
            generation: 100,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 100)
        };
        let ry = BootConfigRecord {
            generation: 4,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 4)
        };
        dev.write_block(0, &encode_record(&rx)).unwrap();
        dev.write_block(1, &encode_record(&ry)).unwrap();
        let (rec, slot) = read_active(&dev, 0).unwrap();
        assert_eq!(slot, SlotPosition::SlotY);
        assert_eq!(rec.generation, 4);
    }

    #[test]
    fn update_atomic_writes_inactive_slot() {
        let mut dev = fresh_partition(4);
        // Seed with rx@1, ry@2 — rx is the inactive one.
        let rx = BootConfigRecord {
            generation: 1,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 1)
        };
        let ry = BootConfigRecord {
            generation: 2,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 2)
        };
        dev.write_block(0, &encode_record(&rx)).unwrap();
        dev.write_block(1, &encode_record(&ry)).unwrap();
        let pre_flush = dev.flush_count();
        let new_rec = BootConfigRecord {
            generation: 3,
            tentative: true,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 3)
        };
        let written = update_atomic(&mut dev, 0, new_rec).unwrap();
        assert_eq!(written, SlotPosition::SlotX);
        assert!(dev.flush_count() > pre_flush, "must flush after write");
        // SlotY (gen=2) is untouched.
        let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
        dev.read_block(1, &mut buf).unwrap();
        let still_y = decode_record(&buf).unwrap();
        assert_eq!(still_y.generation, 2);
        // SlotX is now gen=3.
        dev.read_block(0, &mut buf).unwrap();
        let new_x = decode_record(&buf).unwrap();
        assert_eq!(new_x.generation, 3);
        assert!(new_x.tentative);
    }

    #[test]
    fn update_atomic_rejects_non_monotonic_generation() {
        let mut dev = fresh_partition(4);
        let ry = BootConfigRecord {
            generation: 5,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 5)
        };
        dev.write_block(1, &encode_record(&ry)).unwrap();
        // Same generation — must be rejected.
        let bad = BootConfigRecord {
            generation: 5,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 5)
        };
        assert!(matches!(
            update_atomic(&mut dev, 0, bad),
            Err(BootConfigError::GenerationNotMonotonic)
        ));
    }

    #[test]
    fn update_atomic_initial_partition_writes_slot_x() {
        // Both slots start corrupt (zero magic). Writing a brand-new
        // record should land in SlotX.
        let mut dev = fresh_partition(4);
        let new_rec = BootConfigRecord {
            generation: 1,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 1)
        };
        let written = update_atomic(&mut dev, 0, new_rec).unwrap();
        assert_eq!(written, SlotPosition::SlotX);
        let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
        dev.read_block(0, &mut buf).unwrap();
        assert_eq!(decode_record(&buf).unwrap().generation, 1);
    }

    #[test]
    fn update_atomic_overwrites_corrupt_slot_first() {
        // SlotX is corrupt, SlotY at gen=2 is valid. Update should
        // land in SlotX (the broken one), preserving SlotY.
        let mut dev = fresh_partition(4);
        dev.write_block(0, &[0u8; BOOT_CONFIG_RECORD_SIZE]).unwrap();
        let ry = BootConfigRecord {
            generation: 2,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 2)
        };
        dev.write_block(1, &encode_record(&ry)).unwrap();
        let new_rec = BootConfigRecord {
            generation: 3,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 3)
        };
        let written = update_atomic(&mut dev, 0, new_rec).unwrap();
        assert_eq!(written, SlotPosition::SlotX);
        // SlotY untouched.
        let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
        dev.read_block(1, &mut buf).unwrap();
        assert_eq!(decode_record(&buf).unwrap().generation, 2);
    }

    #[test]
    fn power_loss_before_flush_leaves_old_slot_intact() {
        // Simulate: a write to the inactive slot fails (permanent
        // device failure) — old slot must still be readable.
        let mut dev = fresh_partition(4);
        let rx = BootConfigRecord {
            generation: 5,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 5)
        };
        let ry = BootConfigRecord {
            generation: 6,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 6)
        };
        dev.write_block(0, &encode_record(&rx)).unwrap();
        dev.write_block(1, &encode_record(&ry)).unwrap();
        // Inject a permanent write failure mid-update.
        dev.set_permanent_failure(BlockError::MediaError);
        let new_rec = BootConfigRecord {
            generation: 7,
            tentative: true,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 7)
        };
        assert!(update_atomic(&mut dev, 0, new_rec).is_err());
        // Failure injection blocks reads too while it's set; lift it
        // and confirm prior slot Y is intact.
        dev.clear_permanent_failure();
        let (rec, slot) = read_active(&dev, 0).unwrap();
        assert_eq!(slot, SlotPosition::SlotY);
        assert_eq!(rec.generation, 6);
    }

    #[test]
    fn mark_invalid_clears_valid_flag() {
        let mut dev = fresh_partition(4);
        let rx = BootConfigRecord {
            generation: 1,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 1)
        };
        let ry = BootConfigRecord {
            generation: 2,
            ..BootConfigRecord::new(ActiveSquashfsSlot::B, 2)
        };
        dev.write_block(0, &encode_record(&rx)).unwrap();
        dev.write_block(1, &encode_record(&ry)).unwrap();
        mark_invalid(&mut dev, 0, SlotPosition::SlotY).unwrap();
        // After marking SlotY invalid, the active selector must pick X.
        let (rec, slot) = read_active(&dev, 0).unwrap();
        assert_eq!(slot, SlotPosition::SlotX);
        assert_eq!(rec.generation, 1);
        // Reading SlotY directly: its `valid` flag is false, but the
        // record itself still decodes (hash recomputed for the valid=0 form).
        let mut buf = [0u8; BOOT_CONFIG_RECORD_SIZE];
        dev.read_block(1, &mut buf).unwrap();
        let dec_y = decode_record(&buf).unwrap();
        assert!(!dec_y.valid);
        assert_eq!(dec_y.generation, 2);
    }

    #[test]
    fn mark_invalid_on_corrupt_is_noop() {
        let mut dev = fresh_partition(4);
        // SlotY is just zeros (bad magic). mark_invalid succeeds without
        // returning an error since the record was already not-bootable.
        assert!(mark_invalid(&mut dev, 0, SlotPosition::SlotY).is_ok());
    }

    #[test]
    fn slot_position_helpers() {
        assert_eq!(SlotPosition::SlotX.lba_delta(), 0);
        assert_eq!(SlotPosition::SlotY.lba_delta(), 1);
        assert_eq!(SlotPosition::SlotX.other(), SlotPosition::SlotY);
        assert_eq!(SlotPosition::SlotY.other(), SlotPosition::SlotX);
    }

    #[test]
    fn active_squashfs_slot_helpers() {
        assert_eq!(ActiveSquashfsSlot::A.as_u8(), 0);
        assert_eq!(ActiveSquashfsSlot::B.as_u8(), 1);
        assert_eq!(ActiveSquashfsSlot::A.flip(), ActiveSquashfsSlot::B);
        assert_eq!(ActiveSquashfsSlot::B.flip(), ActiveSquashfsSlot::A);
        assert_eq!(ActiveSquashfsSlot::from_u8(0), Some(ActiveSquashfsSlot::A));
        assert_eq!(ActiveSquashfsSlot::from_u8(1), Some(ActiveSquashfsSlot::B));
        assert_eq!(ActiveSquashfsSlot::from_u8(2), None);
    }

    #[test]
    fn boot_config_error_display() {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        write!(s, "{}", BootConfigError::BothSlotsInvalid).unwrap();
        assert!(s.contains("corrupt"));
        s.clear();
        write!(s, "{}", BootConfigError::UnsupportedVersion(2)).unwrap();
        assert!(s.contains("unsupported version"));
        s.clear();
        write!(s, "{}", BootConfigError::Block(BlockError::MediaError)).unwrap();
        assert!(s.contains("media"));
    }

    #[test]
    fn read_active_requires_4kib_block_size() {
        // 512 B device — boot-config layer rejects.
        let dev = MockBlockDevice::new(512, 16);
        let r = read_active(&dev, 0);
        assert!(matches!(
            r,
            Err(BootConfigError::Block(BlockError::Unaligned))
        ));
    }

    #[test]
    fn record_size_constants_consistent() {
        assert_eq!(BOOT_CONFIG_RECORD_SIZE, 4096);
        assert_eq!(RECORD_BODY_LEN + RECORD_HASH_LEN, BOOT_CONFIG_RECORD_SIZE);
    }

    #[test]
    fn atomicity_invariant_random_partial_writes() {
        // Hand-rolled fuzz: walk through 64 random partial-write
        // sequences and assert that after each, at least one slot
        // remains valid with a generation in the two most-recent set.
        // Random source: a small LCG seeded for determinism.
        let mut dev = fresh_partition(4);
        // Start with a known state.
        let r0 = BootConfigRecord {
            generation: 1,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 1)
        };
        update_atomic(&mut dev, 0, r0).unwrap();

        let mut rng_state: u64 = 0x1234_5678_9abc_def0;
        let next = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        };

        let mut last_two: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
        last_two.push(1);

        for round in 2..=64u64 {
            let new_rec = BootConfigRecord {
                generation: round,
                ..BootConfigRecord::new(ActiveSquashfsSlot::A, round)
            };
            // 25% of the time, simulate "power loss before flush" by
            // injecting a permanent failure. The rest of the time, the
            // write succeeds normally.
            if next(&mut rng_state) % 4 == 0 {
                dev.set_permanent_failure(BlockError::MediaError);
                let _ = update_atomic(&mut dev, 0, new_rec);
                dev.clear_permanent_failure();
                // generation didn't actually advance — last_two stays.
            } else {
                update_atomic(&mut dev, 0, new_rec).unwrap();
                last_two.push(round);
                if last_two.len() > 2 {
                    last_two.remove(0);
                }
            }
            // Invariant: at least one slot is valid AND its generation
            // is in `last_two`.
            let (rec, _) = read_active(&dev, 0).expect("at least one valid slot");
            assert!(
                last_two.contains(&rec.generation),
                "round {round}: read gen {} not in {:?}",
                rec.generation,
                last_two
            );
            assert!(rec.valid);
        }
    }
}
