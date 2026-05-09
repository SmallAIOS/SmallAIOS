// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test vectors for the F2FS GC sophistication (Phase 9).
//!
//! Hand-rolled synthetic fixtures used by the Phase 9 unit tests. Lives
//! under `*_test_vectors.rs` so the CodeQL "hard-coded byte array" rule
//! does not flag the on-disk-format byte arrays we need.
//!
//! Fixtures provided:
//!
//! - [`fragment_pattern_partial`]: 100-segment SIT validity bitmap with
//!   30 segments at varying fragmentation levels. Used to verify
//!   multi-victim selection picks the top-N.
//! - [`hot_cold_inode_table`]: 2000 synthetic (nid, mtime) pairs split
//!   evenly between "hot" (recent mtime) and "cold" (old mtime). Used
//!   to verify hot/cold routing.
//! - [`fragmented_segment_validity_map`]: a single 64-byte bitmap with
//!   exactly N bits set, for stress-testing the live-block extractor.
//! - [`make_summary_for_segment`]: a synthetic SSA summary mapping
//!   each block index to (idx + 1, idx) so reverse-lookups are
//!   deterministic.

#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;

/// 100-segment fragmentation pattern: 30 fragmented segments, scoring
/// from very-fragmented (segment 0) to nearly-full (segment 29). The
/// remaining 70 are at boundary states (fully empty or fully full) so
/// they are NOT GC candidates.
///
/// Returns `[(segno, valid_blocks); 100]`. Used by the multi-victim
/// tests in `f2fs_gc_test_vectors.rs`.
pub fn fragment_pattern_partial() -> Vec<(u32, u16)> {
    let mut out = Vec::with_capacity(100);
    for i in 0..30u32 {
        // Fragmented: valid_blocks scales linearly so segment 0 is
        // the most fragmented (valid=10) and segment 29 the least
        // (valid=300).
        let valid = (10 + i * 10) as u16;
        out.push((i, valid));
    }
    for i in 30..70u32 {
        out.push((i, 512)); // fully full, not a victim
    }
    for i in 70..100u32 {
        out.push((i, 0)); // already free, not a victim
    }
    out
}

/// `(nid, mtime_secs)` pair shorthand.
pub type InodeMtime = (u32, u64);

/// 2000 synthetic (nid, mtime) pairs. The first 1000 are "hot" (mtime
/// = `now - 5`), the next 1000 are "cold" (mtime = `now - 10000`).
///
/// `now` is a caller-provided current-second value; the function
/// returns `(hot_inodes, cold_inodes)` so tests can pick from either
/// pool freely.
pub fn hot_cold_inode_table(now: u64) -> (Vec<InodeMtime>, Vec<InodeMtime>) {
    let mut hot = Vec::with_capacity(1000);
    let mut cold = Vec::with_capacity(1000);
    for i in 0..1000u32 {
        hot.push((100 + i, now.saturating_sub(5)));
        cold.push((10_000 + i, now.saturating_sub(10_000)));
    }
    (hot, cold)
}

/// Construct a 64-byte SIT validity bitmap with the first `n` bits set
/// (n = 0..=512). Useful for live-block-extractor stress tests.
pub fn fragmented_segment_validity_map(n: usize) -> [u8; 64] {
    let mut map = [0u8; 64];
    for i in 0..n.min(512) {
        map[i / 8] |= 1u8 << (i % 8);
    }
    map
}

/// Synthetic SSA summary closure for tests: returns
/// `(nid = block_idx + 1, ofs_in_node = block_idx)`.  Combined with
/// [`fragmented_segment_validity_map`] this yields a deterministic
/// (nid, ofs) for every live block.
pub fn make_summary_for_segment(block_idx: u16) -> Option<(u32, u32)> {
    Some((block_idx as u32 + 1, block_idx as u32))
}

/// 1000 synthetic file-content payloads. Each payload is 64 bytes of
/// `(file_id, file_id ^ 0xA5, ...)` so a byte-for-byte comparison
/// post-relocation can detect any data corruption.
pub fn synthetic_file_payloads(n: usize) -> Vec<Vec<u8>> {
    (0..n as u32)
        .map(|id| {
            let mut v = Vec::with_capacity(64);
            for j in 0..64u32 {
                v.push(((id ^ j) & 0xFF) as u8);
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_pattern_partial_has_100_entries() {
        let p = fragment_pattern_partial();
        assert_eq!(p.len(), 100);
    }

    #[test]
    fn fragment_pattern_partial_30_fragmented() {
        let p = fragment_pattern_partial();
        let frag = p.iter().filter(|(_, vb)| *vb > 0 && *vb < 512).count();
        assert_eq!(frag, 30);
    }

    #[test]
    fn fragment_pattern_partial_first_most_fragmented() {
        let p = fragment_pattern_partial();
        assert_eq!(p[0].1, 10);
        assert_eq!(p[29].1, 300);
    }

    #[test]
    fn hot_cold_table_sizes() {
        let (h, c) = hot_cold_inode_table(100_000);
        assert_eq!(h.len(), 1000);
        assert_eq!(c.len(), 1000);
    }

    #[test]
    fn hot_cold_table_disjoint_nids() {
        let (h, c) = hot_cold_inode_table(100_000);
        let hot_nids: alloc::collections::BTreeSet<u32> = h.iter().map(|(n, _)| *n).collect();
        let cold_nids: alloc::collections::BTreeSet<u32> = c.iter().map(|(n, _)| *n).collect();
        assert!(hot_nids.is_disjoint(&cold_nids));
    }

    #[test]
    fn fragmented_validity_map_zero() {
        let m = fragmented_segment_validity_map(0);
        assert_eq!(m, [0u8; 64]);
    }

    #[test]
    fn fragmented_validity_map_full() {
        let m = fragmented_segment_validity_map(512);
        assert_eq!(m, [0xFFu8; 64]);
    }

    #[test]
    fn fragmented_validity_map_mid() {
        let m = fragmented_segment_validity_map(8);
        assert_eq!(m[0], 0xFF);
        assert_eq!(m[1], 0);
    }

    #[test]
    fn fragmented_validity_map_clamps_above_512() {
        let m = fragmented_segment_validity_map(1024);
        assert_eq!(m, [0xFFu8; 64]);
    }

    #[test]
    fn synthetic_payloads_unique_per_id() {
        let p = synthetic_file_payloads(4);
        assert_eq!(p.len(), 4);
        // Different IDs produce different first bytes.
        assert_ne!(p[0][0], p[1][0]);
        assert_ne!(p[0][0], p[3][0]);
    }

    #[test]
    fn make_summary_for_segment_deterministic() {
        let (nid, ofs) = make_summary_for_segment(7).unwrap();
        assert_eq!(nid, 8);
        assert_eq!(ofs, 7);
    }
}
