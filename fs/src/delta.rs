// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! bsdiff delta-update applier.
//!
//! Clean-room `#![no_std]` Rust applier for a SmallAIOS-flavoured
//! bsdiff format. The streaming algorithm is identical to Colin
//! Percival's bsdiff/bspatch (control/diff/extra triple streams; old
//! file is read with bytewise add of the diff block then a seek).
//!
//! ## Wire format (`bsdiff format v1`)
//!
//! No new external Rust crate dependencies, so the bzip2 framing of
//! upstream bsdiff is replaced with raw streams. The format is:
//!
//! ```text
//!   magic[16]  =  b"SmallAIOSBSD\x01\0\0\0"
//!   ref_hash[32]   SHA-3-256 of reference file (LE)
//!   new_size[8]    u64 LE — size of the produced output
//!   ctrl_len[8]    u64 LE — length of control block, in bytes
//!   diff_len[8]    u64 LE — length of diff block, in bytes
//!   extra_len[8]   u64 LE — length of extra block, in bytes
//!   ctrl[ctrl_len]   sequence of control entries, each 24 bytes
//!     i64 LE diff_len  (must be ≥ 0)
//!     i64 LE extra_len (must be ≥ 0)
//!     i64 LE seek      (signed)
//!   diff[diff_len]    raw bytes (length sum across ctrl)
//!   extra[extra_len]  raw bytes (length sum across ctrl)
//! ```
//!
//! ML-DSA-65 signs `magic || ref_hash || new_size || ctrl_len ||
//! diff_len || extra_len || ctrl || diff || extra` (i.e. the entire
//! patch payload). The signature is supplied separately so the
//! production tooling can ship the signature and the patch as two
//! files; the applier checks the signature before any byte is
//! written.
//!
//! ## Streaming behavior
//!
//! For each control entry `(diff_len_i, extra_len_i, seek_i)`:
//!
//! 1. For `j in 0..diff_len_i`: write `old[old_ptr + j] +
//!    diff_block[diff_ptr + j]` (Wrapping add) to `new[new_ptr + j]`.
//! 2. For `j in 0..extra_len_i`: write `extra_block[extra_ptr + j]`
//!    to `new[new_ptr + diff_len_i + j]`.
//! 3. Advance `old_ptr += diff_len_i + seek_i` (signed addition).
//! 4. Advance `new_ptr += diff_len_i + extra_len_i`.
//! 5. Advance `diff_ptr += diff_len_i`, `extra_ptr += extra_len_i`.
//!
//! At the end, `new_ptr` MUST equal `new_size`, `diff_ptr` MUST
//! equal `diff_len`, and `extra_ptr` MUST equal `extra_len`.
//!
//! Owned by `embedded-filesystem-v1` Phase 4.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use smallaios_security::crypto::ml_dsa::{
    ml_dsa_65_verify, MlDsaError, MlDsaPublicKey, MlDsaSignature, ML_DSA_65_PK_LEN,
    ML_DSA_65_SIG_LEN,
};
use smallaios_security::crypto::sha3::sha3_256;

use crate::block::{BlockDevice, BlockError, FS_BLOCK_SIZE_BYTES};
use crate::boot_config::{self, ActiveSquashfsSlot, BootConfigError, SlotPosition};
use crate::squashfs::{Squashfs, SquashfsError};

/// Magic prefix at offset 0 of every patch payload.
//
// lgtm[rust/hard-coded-cryptographic-value]
pub const PATCH_MAGIC: [u8; 16] = *b"SmallAIOSBSD\x01\0\0\0";

/// Length of the patch header, in bytes (magic + ref_hash + 4× u64).
pub const PATCH_HEADER_LEN: usize = 16 + 32 + 8 * 4;

/// Size of one control-block entry in the patch.
pub const CTRL_ENTRY_LEN: usize = 24;

/// Errors surfaced by the delta applier and the high-level
/// pre/apply/post update flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaError {
    /// Patch payload too short to contain even the header.
    HeaderTruncated,
    /// `magic` field at the start of the patch did not match
    /// [`PATCH_MAGIC`].
    BadMagic,
    /// One of the length fields in the header is too large for the
    /// host pointer width or for the patch payload.
    HeaderInconsistent,
    /// Patch length does not match `header.ctrl_len + header.diff_len
    /// + header.extra_len + PATCH_HEADER_LEN`.
    PatchTruncated,
    /// `ctrl_len` is not a multiple of [`CTRL_ENTRY_LEN`].
    BadCtrlLength,
    /// One of the entries in the control block is malformed
    /// (`diff_len < 0` or `extra_len < 0`).
    BadControlEntry,
    /// Reference-blob `seek_i` would step `old_ptr` outside
    /// `[0, reference.len()]`.
    OffsetOutOfRange,
    /// Reference-blob diff range would read past `reference.len()`.
    ReferencePastEof,
    /// `extra_ptr + extra_len_i` walks past the end of the extra
    /// block.
    ExtraTruncated,
    /// `diff_ptr + diff_len_i` walks past the end of the diff block.
    DiffTruncated,
    /// Sum of `(diff_len_i + extra_len_i)` does not match
    /// `header.new_size`.
    NewSizeMismatch,
    /// `reference_hash` in the patch header did not match
    /// `SHA-3-256(reference)`.
    ReferenceHashMismatch,
    /// Caller passed a `public_key` that is not [`ML_DSA_65_PK_LEN`]
    /// bytes long, or the inner ML-DSA-65 verify failed.
    SignatureInvalid,
    /// `delta_signature.len() != ML_DSA_65_SIG_LEN`.
    BadSignatureLength,
    /// New blob written to inactive partition does not validate as a
    /// signed squashfs (e.g., manifest-footer hash mismatch, ML-DSA
    /// verify on the manifest array failed, or per-block hash
    /// mismatch).
    PostApplyVerificationFailed(SquashfsError),
    /// Boot-config update after a successful apply failed.
    BootConfig(BootConfigError),
    /// Block-device error.
    Block(BlockError),
    /// Inactive partition LBA range is too small for the new blob.
    InactivePartitionTooSmall,
    /// `inactive_len_bytes` is not a multiple of the underlying block
    /// size.
    InactiveLenUnaligned,
}

impl From<BlockError> for DeltaError {
    fn from(e: BlockError) -> Self {
        Self::Block(e)
    }
}

impl From<BootConfigError> for DeltaError {
    fn from(e: BootConfigError) -> Self {
        match e {
            BootConfigError::Block(b) => Self::Block(b),
            other => Self::BootConfig(other),
        }
    }
}

impl fmt::Display for DeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderTruncated => f.write_str("bsdiff patch header truncated"),
            Self::BadMagic => f.write_str("bsdiff patch bad magic"),
            Self::HeaderInconsistent => f.write_str("bsdiff patch header lengths inconsistent"),
            Self::PatchTruncated => f.write_str("bsdiff patch truncated"),
            Self::BadCtrlLength => f.write_str("bsdiff patch ctrl_len not a multiple of 24 bytes"),
            Self::BadControlEntry => f.write_str("bsdiff patch control entry has negative length"),
            Self::OffsetOutOfRange => f.write_str("bsdiff offset out of range"),
            Self::ReferencePastEof => f.write_str("bsdiff reference past EOF"),
            Self::ExtraTruncated => f.write_str("bsdiff patch extra block truncated"),
            Self::DiffTruncated => f.write_str("bsdiff patch diff block truncated"),
            Self::NewSizeMismatch => {
                f.write_str("bsdiff patch new_size does not match control block")
            }
            Self::ReferenceHashMismatch => {
                f.write_str("delta reference mismatch (reference hash differs)")
            }
            Self::SignatureInvalid => f.write_str("delta verification failed (signature)"),
            Self::BadSignatureLength => f.write_str("delta signature length wrong"),
            Self::PostApplyVerificationFailed(e) => {
                write!(f, "post-apply verification failed: {e}")
            }
            Self::BootConfig(e) => write!(f, "boot config update failed: {e}"),
            Self::Block(e) => write!(f, "block error: {e}"),
            Self::InactivePartitionTooSmall => {
                f.write_str("inactive partition too small for staged image")
            }
            Self::InactiveLenUnaligned => {
                f.write_str("inactive partition length not block-aligned")
            }
        }
    }
}

/// Parsed patch header.
#[derive(Debug, Clone, Copy)]
pub struct PatchHeader {
    /// 32-byte SHA-3-256 of the reference file the patch was built
    /// against.
    pub reference_hash: [u8; 32],
    /// Size, in bytes, of the produced output.
    pub new_size: u64,
    /// Length, in bytes, of the control block.
    pub ctrl_len: u64,
    /// Length, in bytes, of the diff block.
    pub diff_len: u64,
    /// Length, in bytes, of the extra block.
    pub extra_len: u64,
}

impl PatchHeader {
    /// Total expected patch length (header + ctrl + diff + extra).
    pub fn total_len(&self) -> Result<u64, DeltaError> {
        let total = (PATCH_HEADER_LEN as u64)
            .checked_add(self.ctrl_len)
            .and_then(|s| s.checked_add(self.diff_len))
            .and_then(|s| s.checked_add(self.extra_len))
            .ok_or(DeltaError::HeaderInconsistent)?;
        Ok(total)
    }
}

/// Parse the patch header at the start of `patch`. Strict
/// length checks; does not authenticate.
pub fn parse_header(patch: &[u8]) -> Result<PatchHeader, DeltaError> {
    if patch.len() < PATCH_HEADER_LEN {
        return Err(DeltaError::HeaderTruncated);
    }
    if patch[0..16] != PATCH_MAGIC {
        return Err(DeltaError::BadMagic);
    }
    let mut ref_hash = [0u8; 32];
    ref_hash.copy_from_slice(&patch[16..48]);
    let take_u64 = |off: usize| -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&patch[off..off + 8]);
        u64::from_le_bytes(b)
    };
    let new_size = take_u64(48);
    let ctrl_len = take_u64(56);
    let diff_len = take_u64(64);
    let extra_len = take_u64(72);
    let header = PatchHeader {
        reference_hash: ref_hash,
        new_size,
        ctrl_len,
        diff_len,
        extra_len,
    };
    let total = header.total_len()?;
    if patch.len() as u64 != total {
        return Err(DeltaError::PatchTruncated);
    }
    if !ctrl_len.is_multiple_of(CTRL_ENTRY_LEN as u64) {
        return Err(DeltaError::BadCtrlLength);
    }
    Ok(header)
}

/// Apply a patch in memory.
///
/// `reference` is the input blob (the active squashfs partition for
/// production; an arbitrary blob for unit tests). `patch` is the full
/// bsdiff-format-v1 payload **including** the header.
///
/// The applier does NOT verify the ML-DSA-65 signature here — that
/// must already have happened upstream in
/// [`apply_delta_to_partition`]. Call this in a verified context only.
pub fn apply_delta(reference: &[u8], patch: &[u8]) -> Result<Vec<u8>, DeltaError> {
    let header = parse_header(patch)?;
    let new_size_usize =
        usize::try_from(header.new_size).map_err(|_| DeltaError::HeaderInconsistent)?;
    let ctrl_off = PATCH_HEADER_LEN;
    let ctrl_len_usize =
        usize::try_from(header.ctrl_len).map_err(|_| DeltaError::HeaderInconsistent)?;
    let diff_len_usize =
        usize::try_from(header.diff_len).map_err(|_| DeltaError::HeaderInconsistent)?;
    let extra_len_usize =
        usize::try_from(header.extra_len).map_err(|_| DeltaError::HeaderInconsistent)?;
    let ctrl = &patch[ctrl_off..ctrl_off + ctrl_len_usize];
    let diff_off = ctrl_off + ctrl_len_usize;
    let diff = &patch[diff_off..diff_off + diff_len_usize];
    let extra_off = diff_off + diff_len_usize;
    let extra = &patch[extra_off..extra_off + extra_len_usize];

    let mut out: Vec<u8> = alloc::vec![0u8; new_size_usize];
    let mut new_ptr: usize = 0;
    let mut old_ptr: i64 = 0;
    let mut diff_ptr: usize = 0;
    let mut extra_ptr: usize = 0;
    let n_ctrl = ctrl.len() / CTRL_ENTRY_LEN;
    for i in 0..n_ctrl {
        let off = i * CTRL_ENTRY_LEN;
        let dlen = read_i64(&ctrl[off..off + 8]);
        let elen = read_i64(&ctrl[off + 8..off + 16]);
        let seek = read_i64(&ctrl[off + 16..off + 24]);
        if dlen < 0 || elen < 0 {
            return Err(DeltaError::BadControlEntry);
        }
        let dlen_usize = usize::try_from(dlen).map_err(|_| DeltaError::BadControlEntry)?;
        let elen_usize = usize::try_from(elen).map_err(|_| DeltaError::BadControlEntry)?;
        if diff_ptr
            .checked_add(dlen_usize)
            .ok_or(DeltaError::DiffTruncated)?
            > diff.len()
        {
            return Err(DeltaError::DiffTruncated);
        }
        if extra_ptr
            .checked_add(elen_usize)
            .ok_or(DeltaError::ExtraTruncated)?
            > extra.len()
        {
            return Err(DeltaError::ExtraTruncated);
        }
        if new_ptr
            .checked_add(dlen_usize)
            .and_then(|s| s.checked_add(elen_usize))
            .ok_or(DeltaError::NewSizeMismatch)?
            > new_size_usize
        {
            return Err(DeltaError::NewSizeMismatch);
        }
        // 1) wrapping-add diff block to old slice
        if old_ptr < 0 {
            return Err(DeltaError::OffsetOutOfRange);
        }
        let old_start = usize::try_from(old_ptr).map_err(|_| DeltaError::OffsetOutOfRange)?;
        let old_end = old_start
            .checked_add(dlen_usize)
            .ok_or(DeltaError::ReferencePastEof)?;
        if old_end > reference.len() {
            return Err(DeltaError::ReferencePastEof);
        }
        for j in 0..dlen_usize {
            out[new_ptr + j] = reference[old_start + j].wrapping_add(diff[diff_ptr + j]);
        }
        new_ptr += dlen_usize;
        diff_ptr += dlen_usize;
        // 2) copy extra block
        out[new_ptr..new_ptr + elen_usize]
            .copy_from_slice(&extra[extra_ptr..extra_ptr + elen_usize]);
        new_ptr += elen_usize;
        extra_ptr += elen_usize;
        // 3) advance old_ptr
        let advance = (dlen)
            .checked_add(seek)
            .ok_or(DeltaError::OffsetOutOfRange)?;
        old_ptr = old_ptr
            .checked_add(advance)
            .ok_or(DeltaError::OffsetOutOfRange)?;
        if old_ptr < 0 || (old_ptr as u64) > reference.len() as u64 {
            return Err(DeltaError::OffsetOutOfRange);
        }
    }
    if new_ptr != new_size_usize {
        return Err(DeltaError::NewSizeMismatch);
    }
    if diff_ptr != diff.len() {
        return Err(DeltaError::DiffTruncated);
    }
    if extra_ptr != extra.len() {
        return Err(DeltaError::ExtraTruncated);
    }
    Ok(out)
}

#[inline]
fn read_i64(b: &[u8]) -> i64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    i64::from_le_bytes(a)
}

/// Verify the patch's ML-DSA-65 signature and the
/// `reference_hash` field against `reference`.
///
/// On success, returns the parsed [`PatchHeader`] so the caller does
/// not have to re-parse it. Both checks happen before this function
/// returns; if either fails, no caller-visible side effects occur.
fn verify_patch(
    reference: &[u8],
    patch: &[u8],
    delta_signature: &[u8],
    public_key: &[u8],
) -> Result<PatchHeader, DeltaError> {
    if public_key.len() != ML_DSA_65_PK_LEN {
        return Err(DeltaError::SignatureInvalid);
    }
    if delta_signature.len() != ML_DSA_65_SIG_LEN {
        return Err(DeltaError::BadSignatureLength);
    }
    let header = parse_header(patch)?;
    let mut pk_arr = [0u8; ML_DSA_65_PK_LEN];
    pk_arr.copy_from_slice(public_key);
    let pk = MlDsaPublicKey::from_bytes(pk_arr);
    let mut sig_arr = [0u8; ML_DSA_65_SIG_LEN];
    sig_arr.copy_from_slice(delta_signature);
    let sig = MlDsaSignature::from_bytes(sig_arr);
    match ml_dsa_65_verify(&pk, patch, &sig) {
        Ok(()) => {}
        Err(MlDsaError::VerificationFailed) => return Err(DeltaError::SignatureInvalid),
        Err(_) => return Err(DeltaError::SignatureInvalid),
    }
    let actual = *sha3_256(reference).as_bytes();
    if actual != header.reference_hash {
        return Err(DeltaError::ReferenceHashMismatch);
    }
    Ok(header)
}

/// One staged-update event the FS delta-update layer emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateEvent {
    /// Pre-apply or post-apply integrity check failed. The boot
    /// config was NOT updated; the active slot continues serving.
    DeltaSignatureInvalid,
    /// Reference hash mismatch — patch was built against a different
    /// version than what the active partition currently holds.
    DeltaReferenceMismatch,
    /// Post-apply hash check on the inactive partition failed; the
    /// inactive slot has been marked `valid=0`.
    PostApplyFailed,
    /// Successful staging. `new_slot` and `generation` are the
    /// boot-config record's `active_slot` and `generation`. The flow
    /// did NOT initiate a reboot; that is the remote-update layer's
    /// responsibility.
    UpdateStaged {
        new_slot: ActiveSquashfsSlot,
        generation: u64,
    },
}

/// Sink for [`UpdateEvent`] notifications.
///
/// `remote-update-v1` registers an implementation that forwards into
/// the audit ring. The default `NoopSink` discards events; tests use
/// a `Vec`-backed mock.
pub trait UpdateEventSink {
    /// Called once per emitted event. MUST NOT panic; errors are
    /// expected to be logged internally.
    fn emit(&mut self, event: UpdateEvent);
}

/// Discarding sink for callers that don't care about events.
#[derive(Debug, Default)]
pub struct NoopSink;

impl UpdateEventSink for NoopSink {
    fn emit(&mut self, _event: UpdateEvent) {}
}

/// In-memory test sink that records every event.
#[derive(Debug, Default)]
pub struct VecSink {
    /// All events seen, in order.
    pub events: Vec<UpdateEvent>,
}

impl VecSink {
    /// Construct an empty sink.
    pub fn new() -> Self {
        Self::default()
    }
}

impl UpdateEventSink for VecSink {
    fn emit(&mut self, event: UpdateEvent) {
        self.events.push(event);
    }
}

/// Top-level delta-update entry point.
///
/// Pre-apply: ML-DSA-65 signature check on the patch and
/// reference-blob SHA-3-256 check against the active partition.
///
/// Apply: streamed bsdiff applier writes the new blob to the
/// inactive partition, **starting at `inactive_lba`**. The applier
/// builds the new blob in RAM, then writes it 4 KiB at a time. (For
/// embedded targets where RAM is tight, the streaming-to-disk path
/// will land in a follow-up phase; the v1 API is in-RAM-staged.)
///
/// Post-apply: the inactive partition is re-mounted via
/// [`Squashfs::mount`], which validates the manifest footer and the
/// per-block SHA-3-256 hash array. On any failure the inactive
/// slot's boot record is `mark_invalid`'d and the active slot
/// continues serving.
///
/// Hand-off: `update_atomic` writes a new boot record with
/// `active_slot` flipped, `tentative=1`, `boot_success=0`, and
/// `generation = max + 1`. An [`UpdateEvent::UpdateStaged`] is
/// emitted via `sink`.
///
/// `boot_config_lba` is the LBA of the GPT partition #5 (boot config).
/// `active_lba` and `inactive_lba` are the LBAs of squashfs partitions
/// #2 and #3 (slot A and slot B); the caller picks them based on the
/// current `active_slot`.
#[allow(clippy::too_many_arguments)]
pub fn apply_delta_to_partition<D: BlockDevice, S: UpdateEventSink + ?Sized>(
    device: &mut D,
    boot_config_lba: u64,
    active_lba: u64,
    active_len_bytes: u64,
    inactive_lba: u64,
    inactive_len_bytes: u64,
    new_active_slot: ActiveSquashfsSlot,
    patch: &[u8],
    delta_signature: &[u8],
    public_key: &[u8],
    sink: &mut S,
) -> Result<SlotPosition, DeltaError> {
    let bs = device.block_size_bytes() as u64;
    if bs != FS_BLOCK_SIZE_BYTES as u64 {
        return Err(DeltaError::Block(BlockError::Unaligned));
    }
    if !inactive_len_bytes.is_multiple_of(bs) {
        return Err(DeltaError::InactiveLenUnaligned);
    }
    if !active_len_bytes.is_multiple_of(bs) {
        return Err(DeltaError::InactiveLenUnaligned);
    }

    // Read the active partition fully (this matches the squashfs
    // mount path, which already does an in-RAM mount).
    let active_blocks = active_len_bytes / bs;
    let mut reference: Vec<u8> = alloc::vec![0u8; active_len_bytes as usize];
    let mut scratch: Vec<u8> = alloc::vec![0u8; bs as usize];
    for i in 0..active_blocks {
        device.read_block(active_lba + i, &mut scratch)?;
        let off = (i as usize) * (bs as usize);
        reference[off..off + bs as usize].copy_from_slice(&scratch);
    }

    // Pre-apply: signature + reference-hash check.
    let header = match verify_patch(&reference, patch, delta_signature, public_key) {
        Ok(h) => h,
        Err(DeltaError::SignatureInvalid) | Err(DeltaError::BadSignatureLength) => {
            sink.emit(UpdateEvent::DeltaSignatureInvalid);
            return Err(DeltaError::SignatureInvalid);
        }
        Err(DeltaError::ReferenceHashMismatch) => {
            sink.emit(UpdateEvent::DeltaReferenceMismatch);
            return Err(DeltaError::ReferenceHashMismatch);
        }
        Err(e) => return Err(e),
    };

    // Apply: build new blob from patch.
    let new_blob = apply_delta(&reference, patch)?;
    // Pad to inactive_len_bytes (squashfs partitions are
    // 4 KiB-multiple; trailing zeros are part of the manifest).
    let inactive_size = inactive_len_bytes as usize;
    if new_blob.len() > inactive_size {
        return Err(DeltaError::InactivePartitionTooSmall);
    }
    let mut staged = new_blob;
    staged.resize(inactive_size, 0);

    // Post-apply: write to inactive partition.
    let inactive_blocks = inactive_len_bytes / bs;
    for i in 0..inactive_blocks {
        let off = (i as usize) * (bs as usize);
        device.write_block(inactive_lba + i, &staged[off..off + bs as usize])?;
    }
    device.flush()?;

    // Post-apply: re-mount the staged partition. The mount path
    // validates the manifest footer (ML-DSA-65 + per-block hash
    // array). On failure, mark the inactive slot's boot record
    // invalid and surface PostApplyVerificationFailed.
    match Squashfs::mount(device, inactive_lba, inactive_blocks, public_key) {
        Ok(_fs) => { /* success */ }
        Err(e) => {
            sink.emit(UpdateEvent::PostApplyFailed);
            // Best-effort: mark the inactive slot's boot record
            // invalid. We do not yet know which physical slot
            // corresponds — we have to read the boot config to find
            // the active position, then mark the OTHER one invalid.
            if let Ok((_rec, active_pos)) = boot_config::read_active(device, boot_config_lba) {
                let _ = boot_config::mark_invalid(device, boot_config_lba, active_pos.other());
            }
            return Err(DeltaError::PostApplyVerificationFailed(e));
        }
    }
    let _ = header; // header used for diagnostics; future audit detail will name new_size

    // Hand-off: write the new boot-config record.
    let (new_record, pos) =
        boot_config::watchdog::stage_new_active_slot(device, boot_config_lba, new_active_slot)?;
    sink.emit(UpdateEvent::UpdateStaged {
        new_slot: new_record.active_slot,
        generation: new_record.generation,
    });
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::mock::MockBlockDevice;
    use alloc::vec;
    use smallaios_security::crypto::ml_dsa::{ml_dsa_65_keygen, ml_dsa_65_sign};

    /// Build a minimal patch payload that turns `reference` into
    /// `target` via a single control-block entry. Returns
    /// `(patch_bytes, signature)` signed with `seed`.
    fn build_signed_patch(
        reference: &[u8],
        target: &[u8],
        seed: &[u8; 32],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // diff_len covers the overlap region; extra_len covers the
        // tail of target beyond reference.len(). Here we use the
        // simplest construction: diff_len = min(ref, tgt), extra_len
        // = remaining tgt bytes, seek = 0.
        let diff_len = core::cmp::min(reference.len(), target.len());
        let extra_len = target.len() - diff_len;
        let mut diff = alloc::vec![0u8; diff_len];
        for j in 0..diff_len {
            diff[j] = target[j].wrapping_sub(reference[j]);
        }
        let extra: Vec<u8> = target[diff_len..].to_vec();
        let mut ctrl = Vec::with_capacity(CTRL_ENTRY_LEN);
        ctrl.extend_from_slice(&(diff_len as i64).to_le_bytes());
        ctrl.extend_from_slice(&(extra_len as i64).to_le_bytes());
        ctrl.extend_from_slice(&0i64.to_le_bytes()); // seek = 0
        let mut patch =
            Vec::with_capacity(PATCH_HEADER_LEN + ctrl.len() + diff.len() + extra.len());
        patch.extend_from_slice(&PATCH_MAGIC);
        let ref_hash = *sha3_256(reference).as_bytes();
        patch.extend_from_slice(&ref_hash);
        patch.extend_from_slice(&(target.len() as u64).to_le_bytes());
        patch.extend_from_slice(&(ctrl.len() as u64).to_le_bytes());
        patch.extend_from_slice(&(diff.len() as u64).to_le_bytes());
        patch.extend_from_slice(&(extra.len() as u64).to_le_bytes());
        patch.extend_from_slice(&ctrl);
        patch.extend_from_slice(&diff);
        patch.extend_from_slice(&extra);
        let kp = ml_dsa_65_keygen(seed).unwrap();
        let pk = kp.public_key.as_bytes().to_vec();
        let nonce = [0xA5u8; 32];
        let sig = ml_dsa_65_sign(&kp.secret_key, &patch, &nonce).unwrap();
        (patch, sig.as_bytes().to_vec(), pk)
    }

    #[test]
    fn parse_header_rejects_short_buffer() {
        assert!(matches!(
            parse_header(&[0u8; 8]),
            Err(DeltaError::HeaderTruncated)
        ));
    }

    #[test]
    fn parse_header_rejects_bad_magic() {
        let mut buf = alloc::vec![0u8; PATCH_HEADER_LEN];
        buf[0] = 1; // not PATCH_MAGIC
        assert!(matches!(parse_header(&buf), Err(DeltaError::BadMagic)));
    }

    #[test]
    fn apply_delta_round_trip_simple() {
        let reference = b"hello world!".to_vec();
        let target = b"HELLO world!!".to_vec();
        let (patch, _sig, _pk) = build_signed_patch(&reference, &target, &[3u8; 32]);
        let out = apply_delta(&reference, &patch).unwrap();
        assert_eq!(out, target);
    }

    #[test]
    fn apply_delta_pure_copy_round_trip() {
        let reference = vec![0xAAu8; 256];
        let target = reference.clone();
        let (patch, _sig, _pk) = build_signed_patch(&reference, &target, &[7u8; 32]);
        let out = apply_delta(&reference, &patch).unwrap();
        assert_eq!(out, target);
    }

    #[test]
    fn apply_delta_grow_target_round_trip() {
        let reference = vec![0u8; 4];
        let target: Vec<u8> = (0u8..32).collect();
        let (patch, _sig, _pk) = build_signed_patch(&reference, &target, &[11u8; 32]);
        let out = apply_delta(&reference, &patch).unwrap();
        assert_eq!(out, target);
    }

    #[test]
    fn apply_delta_shrink_target_round_trip() {
        let reference: Vec<u8> = (0u8..64).collect();
        let target: Vec<u8> = (0u8..16).collect();
        let (patch, _sig, _pk) = build_signed_patch(&reference, &target, &[13u8; 32]);
        let out = apply_delta(&reference, &patch).unwrap();
        assert_eq!(out, target);
    }

    #[test]
    fn apply_delta_truncated_patch_rejected() {
        let reference = b"abcdef".to_vec();
        let target = b"ABCDEF".to_vec();
        let (mut patch, _, _) = build_signed_patch(&reference, &target, &[2u8; 32]);
        patch.truncate(patch.len() - 1);
        let r = apply_delta(&reference, &patch);
        assert!(matches!(r, Err(DeltaError::PatchTruncated)));
    }

    #[test]
    fn apply_delta_bad_ctrl_length_rejected() {
        let reference = b"abcdef".to_vec();
        let target = b"ABCDEF".to_vec();
        let (mut patch, _, _) = build_signed_patch(&reference, &target, &[4u8; 32]);
        // Set ctrl_len to non-multiple of 24, and pad/cut accordingly so
        // patch length still matches header.
        let ctrl_len_off = 16 + 32 + 8; // after magic + ref_hash + new_size
        patch[ctrl_len_off..ctrl_len_off + 8].copy_from_slice(&5u64.to_le_bytes());
        // To keep patch length consistent we'd have to reshape it; the
        // simplest assertion is that parse_header rejects the bad
        // ctrl_len once total_len matches.
        // For simplicity, we just verify the BadCtrlLength path triggers
        // on a constructed buffer of exactly header+5+0+0:
        let mut crafted = alloc::vec![0u8; PATCH_HEADER_LEN + 5];
        crafted[0..16].copy_from_slice(&PATCH_MAGIC);
        crafted[ctrl_len_off..ctrl_len_off + 8].copy_from_slice(&5u64.to_le_bytes());
        let r = parse_header(&crafted);
        assert!(matches!(r, Err(DeltaError::BadCtrlLength)));
    }

    #[test]
    fn apply_delta_offset_out_of_range_rejected() {
        // Build a hand-crafted patch whose seek pushes old_ptr past
        // reference.len().
        let reference = vec![1u8; 8];
        let mut patch = Vec::new();
        patch.extend_from_slice(&PATCH_MAGIC);
        let ref_hash = *sha3_256(&reference).as_bytes();
        patch.extend_from_slice(&ref_hash);
        // new_size = 0 — only ctrl runs, no body
        patch.extend_from_slice(&0u64.to_le_bytes()); // new_size
        patch.extend_from_slice(&(CTRL_ENTRY_LEN as u64).to_le_bytes()); // ctrl_len
        patch.extend_from_slice(&0u64.to_le_bytes()); // diff_len
        patch.extend_from_slice(&0u64.to_le_bytes()); // extra_len
                                                      // ctrl entry: diff=0, extra=0, seek=999 (too far)
        patch.extend_from_slice(&0i64.to_le_bytes());
        patch.extend_from_slice(&0i64.to_le_bytes());
        patch.extend_from_slice(&999i64.to_le_bytes());
        let r = apply_delta(&reference, &patch);
        assert!(matches!(r, Err(DeltaError::OffsetOutOfRange)));
    }

    #[test]
    fn apply_delta_negative_diff_len_rejected() {
        let reference = vec![1u8; 8];
        let mut patch = Vec::new();
        patch.extend_from_slice(&PATCH_MAGIC);
        let ref_hash = *sha3_256(&reference).as_bytes();
        patch.extend_from_slice(&ref_hash);
        patch.extend_from_slice(&0u64.to_le_bytes());
        patch.extend_from_slice(&(CTRL_ENTRY_LEN as u64).to_le_bytes());
        patch.extend_from_slice(&0u64.to_le_bytes());
        patch.extend_from_slice(&0u64.to_le_bytes());
        // ctrl: diff_len = -1
        patch.extend_from_slice(&(-1i64).to_le_bytes());
        patch.extend_from_slice(&0i64.to_le_bytes());
        patch.extend_from_slice(&0i64.to_le_bytes());
        let r = apply_delta(&reference, &patch);
        assert!(matches!(r, Err(DeltaError::BadControlEntry)));
    }

    #[test]
    fn verify_patch_accepts_valid_signature() {
        let reference = b"hello".to_vec();
        let target = b"hellz".to_vec();
        let (patch, sig, pk) = build_signed_patch(&reference, &target, &[42u8; 32]);
        verify_patch(&reference, &patch, &sig, &pk).unwrap();
    }

    #[test]
    fn verify_patch_rejects_forged_signature() {
        let reference = b"hello".to_vec();
        let target = b"hellz".to_vec();
        let (patch, mut sig, pk) = build_signed_patch(&reference, &target, &[1u8; 32]);
        sig[0] ^= 0xFF;
        let r = verify_patch(&reference, &patch, &sig, &pk);
        assert!(matches!(r, Err(DeltaError::SignatureInvalid)));
    }

    #[test]
    fn verify_patch_rejects_wrong_pubkey() {
        let reference = b"hello".to_vec();
        let target = b"hellz".to_vec();
        let (patch, sig, _pk) = build_signed_patch(&reference, &target, &[1u8; 32]);
        let other = ml_dsa_65_keygen(&[88u8; 32]).unwrap();
        let other_pk = other.public_key.as_bytes().to_vec();
        let r = verify_patch(&reference, &patch, &sig, &other_pk);
        assert!(matches!(r, Err(DeltaError::SignatureInvalid)));
    }

    #[test]
    fn verify_patch_rejects_wrong_reference() {
        let reference = b"hello".to_vec();
        let other_reference = b"world".to_vec();
        let target = b"hellz".to_vec();
        let (patch, sig, pk) = build_signed_patch(&reference, &target, &[1u8; 32]);
        let r = verify_patch(&other_reference, &patch, &sig, &pk);
        assert!(matches!(r, Err(DeltaError::ReferenceHashMismatch)));
    }

    #[test]
    fn verify_patch_rejects_bad_signature_length() {
        let reference = b"hello".to_vec();
        let target = b"hellz".to_vec();
        let (patch, _sig, pk) = build_signed_patch(&reference, &target, &[1u8; 32]);
        let r = verify_patch(&reference, &patch, &[0u8; 100], &pk);
        assert!(matches!(r, Err(DeltaError::BadSignatureLength)));
    }

    #[test]
    fn verify_patch_rejects_bad_pubkey_length() {
        let reference = b"hello".to_vec();
        let target = b"hellz".to_vec();
        let (patch, sig, _pk) = build_signed_patch(&reference, &target, &[1u8; 32]);
        let r = verify_patch(&reference, &patch, &sig, &[0u8; 16]);
        assert!(matches!(r, Err(DeltaError::SignatureInvalid)));
    }

    #[test]
    fn delta_error_display() {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        write!(s, "{}", DeltaError::OffsetOutOfRange).unwrap();
        assert!(s.contains("offset out of range"));
        s.clear();
        write!(s, "{}", DeltaError::PatchTruncated).unwrap();
        assert!(s.contains("truncated"));
        s.clear();
        write!(s, "{}", DeltaError::SignatureInvalid).unwrap();
        assert!(s.contains("signature"));
    }

    #[test]
    fn vec_sink_records_events() {
        let mut sink = VecSink::new();
        sink.emit(UpdateEvent::DeltaSignatureInvalid);
        sink.emit(UpdateEvent::UpdateStaged {
            new_slot: ActiveSquashfsSlot::B,
            generation: 7,
        });
        assert_eq!(sink.events.len(), 2);
        assert_eq!(sink.events[0], UpdateEvent::DeltaSignatureInvalid);
    }

    #[test]
    fn noop_sink_compiles() {
        let mut sink = NoopSink;
        sink.emit(UpdateEvent::PostApplyFailed);
    }

    #[test]
    fn apply_delta_to_partition_rejects_bad_block_size() {
        let mut dev = MockBlockDevice::new(512, 16);
        let mut sink = NoopSink;
        let r = apply_delta_to_partition(
            &mut dev,
            0,
            0,
            0,
            0,
            0,
            ActiveSquashfsSlot::A,
            &[],
            &[],
            &[],
            &mut sink,
        );
        assert!(matches!(r, Err(DeltaError::Block(BlockError::Unaligned))));
    }

    #[test]
    fn apply_delta_to_partition_rejects_unaligned_lengths() {
        let mut dev = MockBlockDevice::new(4096, 16);
        let mut sink = NoopSink;
        let r = apply_delta_to_partition(
            &mut dev,
            0,
            1,
            4097, // not a multiple of 4096
            5,
            8192,
            ActiveSquashfsSlot::A,
            &[],
            &[],
            &[],
            &mut sink,
        );
        assert!(matches!(r, Err(DeltaError::InactiveLenUnaligned)));
    }

    #[test]
    fn apply_delta_to_partition_emits_signature_invalid_event() {
        let mut dev = MockBlockDevice::new(4096, 16);
        // Pre-fill an "active" partition with bytes whose hash will
        // match the patch we're about to build.
        let active = vec![0xAAu8; 4096 * 4];
        for i in 0..4 {
            let off = i * 4096;
            dev.write_block(i as u64, &active[off..off + 4096]).unwrap();
        }
        // Build a real signed patch against `active`, then corrupt the
        // signature so verify_patch falls into the signature path
        // (instead of header truncation).
        let target = vec![0xBBu8; 4096];
        let (patch, mut sig, pk) = build_signed_patch(&active, &target, &[42u8; 32]);
        sig[0] ^= 0xFF;
        let mut sink = VecSink::new();
        let r = apply_delta_to_partition(
            &mut dev,
            8,
            0,
            (4096 * 4) as u64,
            12,
            (4096 * 4) as u64,
            ActiveSquashfsSlot::B,
            &patch,
            &sig,
            &pk,
            &mut sink,
        );
        assert!(matches!(r, Err(DeltaError::SignatureInvalid)));
        assert!(sink.events.contains(&UpdateEvent::DeltaSignatureInvalid));
    }

    #[test]
    fn apply_delta_to_partition_emits_reference_mismatch_event() {
        let mut dev = MockBlockDevice::new(4096, 16);
        let active_size = 4096 * 4;
        // Fill the active partition with a known pattern.
        let active = vec![0x11u8; active_size];
        for i in 0..4 {
            let off = i * 4096;
            dev.write_block(i as u64, &active[off..off + 4096]).unwrap();
        }
        // Build a patch whose reference_hash is a different version.
        let other_reference = vec![0x22u8; 64];
        let target = vec![0x33u8; 64];
        let (patch, sig, pk) = build_signed_patch(&other_reference, &target, &[42u8; 32]);
        let mut sink = VecSink::new();
        let r = apply_delta_to_partition(
            &mut dev,
            8,
            0,
            active_size as u64,
            12,
            active_size as u64,
            ActiveSquashfsSlot::B,
            &patch,
            &sig,
            &pk,
            &mut sink,
        );
        assert!(matches!(r, Err(DeltaError::ReferenceHashMismatch)));
        assert!(sink.events.contains(&UpdateEvent::DeltaReferenceMismatch));
    }
}
