// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS GC relocation-staging journal records (Phase 9).
//!
//! Phase 9 GC relocates live blocks from a victim segment to a fresh
//! segment, then updates inode address pointers + NAT + SIT, and finally
//! frees the victim segment. A power loss between any two of those
//! steps must NOT lose data: the journal STAGES every relocation as a
//! `(inode, old_block, new_block)` triple BEFORE any of the in-place
//! mutations happen. On recovery, replay the staged triples that were
//! committed to the journal and skip the ones that weren't.
//!
//! Layout per journal entry (24 bytes packed little-endian):
//!
//! | offset | size | field             | description                    |
//! |--------|------|-------------------|--------------------------------|
//! |   0    |  4   | magic             | `GCJ_MAGIC` = 0x47434A52 ("GCJR") |
//! |   4    |  4   | nid               | owning node-id                  |
//! |   8    |  4   | ofs_in_node       | block offset within inode       |
//! |  12    |  4   | old_block_addr    | source physical block addr      |
//! |  16    |  4   | new_block_addr    | destination physical block addr |
//! |  20    |  2   | flags             | bit 0 = committed; bit 1 = age  |
//! |  22    |  2   | crc16             | CRC-16/IBM-3740 over [0..22]    |
//!
//! Records are written to the F2FS journal block alongside rename
//! records (see [`super::journal`]). Replay walks the buffer in order;
//! a record with `committed=0` MAY be skipped (the relocation never
//! finished) but a `committed=1` record MUST be replayed.
//!
//! Owned by `embedded-filesystem-v1` Phase 9.

extern crate alloc;

use alloc::vec::Vec;

use super::error::F2fsError;

/// Magic for a GC-journal-record (`GCJR` little-endian).
pub const GCJ_MAGIC: u32 = 0x4A434752; // "RGCJ" big-endian; LE bytes are 'G','C','J','R' starting from 0x52.

/// On-disk size of one GC journal record (bytes).
pub const GC_JOURNAL_RECORD_SIZE: usize = 24;

/// Bit flag: this record's relocation was successfully committed.
pub const GCJ_FLAG_COMMITTED: u16 = 0x0001;

/// Bit flag: this record describes a *cold* block (age heuristic).
pub const GCJ_FLAG_COLD: u16 = 0x0002;

/// One staged relocation entry (in-memory form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcJournalRecord {
    /// Owning node-id (from SSA reverse mapping).
    pub nid: u32,
    /// Offset within the inode's data array (block index).
    pub ofs_in_node: u32,
    /// Source block address (the victim segment).
    pub old_block_addr: u32,
    /// Destination block address (the fresh segment).
    pub new_block_addr: u32,
    /// Flags: see [`GCJ_FLAG_COMMITTED`], [`GCJ_FLAG_COLD`].
    pub flags: u16,
}

impl GcJournalRecord {
    /// Construct a fresh (uncommitted) staging record.
    #[inline]
    pub fn new(nid: u32, ofs_in_node: u32, old: u32, new: u32, cold: bool) -> Self {
        let mut flags = 0u16;
        if cold {
            flags |= GCJ_FLAG_COLD;
        }
        Self {
            nid,
            ofs_in_node,
            old_block_addr: old,
            new_block_addr: new,
            flags,
        }
    }

    /// True if the relocation was successfully committed to disk.
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.flags & GCJ_FLAG_COMMITTED != 0
    }

    /// True if the record is for a cold block.
    #[inline]
    pub fn is_cold(&self) -> bool {
        self.flags & GCJ_FLAG_COLD != 0
    }

    /// Mark the record committed (returns a new record with the bit
    /// set).
    #[inline]
    pub fn committed(mut self) -> Self {
        self.flags |= GCJ_FLAG_COMMITTED;
        self
    }

    /// Encode to a 24-byte little-endian buffer with CRC-16.
    pub fn encode(&self) -> [u8; GC_JOURNAL_RECORD_SIZE] {
        let mut buf = [0u8; GC_JOURNAL_RECORD_SIZE];
        buf[0..4].copy_from_slice(&GCJ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&self.nid.to_le_bytes());
        buf[8..12].copy_from_slice(&self.ofs_in_node.to_le_bytes());
        buf[12..16].copy_from_slice(&self.old_block_addr.to_le_bytes());
        buf[16..20].copy_from_slice(&self.new_block_addr.to_le_bytes());
        buf[20..22].copy_from_slice(&self.flags.to_le_bytes());
        let crc = crc16_ibm_3740(&buf[..22]);
        buf[22..24].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Decode one record from a buffer. Returns `Err` if magic or CRC
    /// mismatch — the caller MUST surface this as journal corruption.
    pub fn decode(buf: &[u8]) -> Result<Self, F2fsError> {
        if buf.len() < GC_JOURNAL_RECORD_SIZE {
            return Err(F2fsError::InodeCorrupt);
        }
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != GCJ_MAGIC {
            return Err(F2fsError::InodeCorrupt);
        }
        let crc_actual = u16::from_le_bytes([buf[22], buf[23]]);
        let crc_expect = crc16_ibm_3740(&buf[..22]);
        if crc_actual != crc_expect {
            return Err(F2fsError::InodeCorrupt);
        }
        Ok(Self {
            nid: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            ofs_in_node: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            old_block_addr: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            new_block_addr: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            flags: u16::from_le_bytes([buf[20], buf[21]]),
        })
    }
}

/// Encode a list of GC-journal records back-to-back. Caller-side
/// helper; the on-disk journal block format prefixes a u32 count which
/// is the responsibility of [`encode_block`].
pub fn encode_records(records: &[GcJournalRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * GC_JOURNAL_RECORD_SIZE);
    for r in records {
        out.extend_from_slice(&r.encode());
    }
    out
}

/// Decode a sequence of GC-journal records. Stops at the first record
/// that fails to decode (returns the records parsed so far + the error).
///
/// On replay, callers walk the returned vector and apply `committed=1`
/// records' new addresses to the relevant inode/NAT/SIT.
pub fn decode_records(buf: &[u8]) -> (Vec<GcJournalRecord>, Option<F2fsError>) {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + GC_JOURNAL_RECORD_SIZE <= buf.len() {
        match GcJournalRecord::decode(&buf[off..off + GC_JOURNAL_RECORD_SIZE]) {
            Ok(r) => out.push(r),
            Err(e) => return (out, Some(e)),
        }
        off += GC_JOURNAL_RECORD_SIZE;
    }
    (out, None)
}

/// Encode a full GC-journal block: a u32 record-count header, then the
/// records, then a trailing CRC-32 over all of the above.
///
/// Layout:
///
/// | offset | size      | field            |
/// |--------|-----------|------------------|
/// |   0    |    4      | record_count     |
/// |   4    | 24*count  | records          |
/// |  4+24n |    4      | block_crc32      |
pub fn encode_block(records: &[GcJournalRecord]) -> Vec<u8> {
    let count = records.len() as u32;
    let body = encode_records(records);
    let mut out = Vec::with_capacity(4 + body.len() + 4);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&body);
    let crc = crc32_jamcrc(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// Decode a full GC-journal block produced by [`encode_block`].
pub fn decode_block(buf: &[u8]) -> Result<Vec<GcJournalRecord>, F2fsError> {
    if buf.len() < 8 {
        return Err(F2fsError::InodeCorrupt);
    }
    let count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let body_len = count
        .checked_mul(GC_JOURNAL_RECORD_SIZE)
        .ok_or(F2fsError::SizeOverflow)?;
    let total = body_len.checked_add(8).ok_or(F2fsError::SizeOverflow)?;
    if buf.len() < total {
        return Err(F2fsError::InodeCorrupt);
    }
    let crc_actual = u32::from_le_bytes([
        buf[4 + body_len],
        buf[4 + body_len + 1],
        buf[4 + body_len + 2],
        buf[4 + body_len + 3],
    ]);
    let crc_expect = crc32_jamcrc(&buf[..4 + body_len]);
    if crc_actual != crc_expect {
        return Err(F2fsError::InodeCorrupt);
    }
    let body = &buf[4..4 + body_len];
    let (records, err) = decode_records(body);
    if let Some(e) = err {
        return Err(e);
    }
    if records.len() != count {
        return Err(F2fsError::InodeCorrupt);
    }
    Ok(records)
}

// ── Pure-functional CRC implementations ─────────────────────────────────────
//
// These are vendored as constants and a small computation routine so
// gc_journal.rs has zero dependencies on the existing
// [`super::crc::f2fs_crc32`] (which uses the F2FS-flavored polynomial
// and is owned by Phase 5). The Phase 9 journal record CRC uses
// CRC-16/IBM-3740 (polynomial 0x1021, init 0xFFFF, no reflect), and
// the block CRC uses JAMCRC (the inverted CRC-32, polynomial 0x04C11DB7
// reflected, init 0xFFFF_FFFF). Different polynomials guard against
// mis-typed F2FS-crc usage in an unrelated layout.

/// CRC-16/IBM-3740 (init=0xFFFF, poly=0x1021, no reflection).
pub fn crc16_ibm_3740(buf: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in buf {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// JAMCRC = !CRC-32 (reflected, init 0xFFFFFFFF, no XOR-out, then
/// inverted by definition). We simply compute the standard reflected
/// CRC-32 with polynomial 0xEDB88320 and final XOR=0 to produce the
/// JAMCRC variant.
pub fn crc32_jamcrc(buf: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in buf {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trip() {
        let r = GcJournalRecord::new(7, 3, 100, 200, false);
        let bytes = r.encode();
        let r2 = GcJournalRecord::decode(&bytes).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn record_committed_flag_round_trip() {
        let r = GcJournalRecord::new(11, 0, 50, 75, true).committed();
        assert!(r.is_committed());
        assert!(r.is_cold());
        let bytes = r.encode();
        let r2 = GcJournalRecord::decode(&bytes).unwrap();
        assert!(r2.is_committed());
        assert!(r2.is_cold());
    }

    #[test]
    fn record_uncommitted_default() {
        let r = GcJournalRecord::new(1, 0, 0, 0, false);
        assert!(!r.is_committed());
        assert!(!r.is_cold());
    }

    #[test]
    fn record_decode_rejects_bad_magic() {
        let mut bytes = GcJournalRecord::new(1, 0, 0, 1, false).encode();
        bytes[0] ^= 0xFF;
        assert!(GcJournalRecord::decode(&bytes).is_err());
    }

    #[test]
    fn record_decode_rejects_bad_crc() {
        let mut bytes = GcJournalRecord::new(1, 0, 0, 1, false).encode();
        bytes[22] ^= 0xFF;
        assert!(GcJournalRecord::decode(&bytes).is_err());
    }

    #[test]
    fn record_decode_truncated_buffer() {
        let bytes = GcJournalRecord::new(1, 0, 0, 1, false).encode();
        assert!(GcJournalRecord::decode(&bytes[..16]).is_err());
    }

    #[test]
    fn encode_records_concatenates() {
        let rs = [
            GcJournalRecord::new(1, 0, 100, 200, false),
            GcJournalRecord::new(2, 1, 101, 201, true),
        ];
        let bytes = encode_records(&rs);
        assert_eq!(bytes.len(), 2 * GC_JOURNAL_RECORD_SIZE);
        let (decoded, err) = decode_records(&bytes);
        assert!(err.is_none());
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], rs[0]);
        assert_eq!(decoded[1], rs[1]);
    }

    #[test]
    fn decode_records_stops_at_corruption() {
        let mut bytes = encode_records(&[
            GcJournalRecord::new(1, 0, 100, 200, false),
            GcJournalRecord::new(2, 1, 101, 201, false),
        ]);
        // Corrupt the second record's magic.
        bytes[GC_JOURNAL_RECORD_SIZE] ^= 0xFF;
        let (decoded, err) = decode_records(&bytes);
        assert_eq!(decoded.len(), 1);
        assert!(err.is_some());
    }

    #[test]
    fn block_round_trip_empty() {
        let bytes = encode_block(&[]);
        let decoded = decode_block(&bytes).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn block_round_trip_many() {
        let rs: alloc::vec::Vec<GcJournalRecord> = (0..16u32)
            .map(|i| GcJournalRecord::new(i, i, 100 + i, 200 + i, i % 2 == 0).committed())
            .collect();
        let bytes = encode_block(&rs);
        let decoded = decode_block(&bytes).unwrap();
        assert_eq!(decoded.len(), 16);
        assert_eq!(decoded[0], rs[0]);
        assert_eq!(decoded[15], rs[15]);
    }

    #[test]
    fn block_decode_rejects_corrupt_block_crc() {
        let rs = [GcJournalRecord::new(1, 0, 100, 200, false)];
        let mut bytes = encode_block(&rs);
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        assert!(decode_block(&bytes).is_err());
    }

    #[test]
    fn block_decode_rejects_truncated() {
        let rs = [GcJournalRecord::new(1, 0, 100, 200, false)];
        let bytes = encode_block(&rs);
        assert!(decode_block(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn block_decode_rejects_count_mismatch() {
        // Hand-craft a block claiming 5 records but only contains 1.
        let mut buf = alloc::vec![0u8; 4 + GC_JOURNAL_RECORD_SIZE];
        buf[0..4].copy_from_slice(&5u32.to_le_bytes());
        let r = GcJournalRecord::new(1, 0, 100, 200, false);
        buf[4..4 + GC_JOURNAL_RECORD_SIZE].copy_from_slice(&r.encode());
        // Append a fake CRC.
        let crc = crc32_jamcrc(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        // The decode routine still reads count*RECORD_SIZE bytes; if
        // the buffer is too short this returns InodeCorrupt.
        let r = decode_block(&buf);
        assert!(r.is_err());
    }

    #[test]
    fn crc16_known_vector() {
        // KAT: CRC-16/IBM-3740 of "123456789" = 0x29B1.
        assert_eq!(crc16_ibm_3740(b"123456789"), 0x29B1);
    }

    #[test]
    fn crc32_jamcrc_known_vector() {
        // KAT: standard CRC-32/JAMCRC reflected of "123456789" = 0x340BC6D9.
        assert_eq!(crc32_jamcrc(b"123456789"), 0x340BC6D9);
    }

    #[test]
    fn record_size_constant() {
        let r = GcJournalRecord::new(0, 0, 0, 0, false);
        assert_eq!(r.encode().len(), GC_JOURNAL_RECORD_SIZE);
    }

    #[test]
    fn flags_orthogonal() {
        let r = GcJournalRecord::new(1, 0, 0, 1, true).committed();
        assert!(r.is_cold());
        assert!(r.is_committed());
        assert_eq!(r.flags & !(GCJ_FLAG_COLD | GCJ_FLAG_COMMITTED), 0);
    }
}
