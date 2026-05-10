// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic GC fixtures for Phase 9 conformance tests.
//!
//! Mirrors `fs/src/f2fs/gc_test_vectors.rs` but lives outside the
//! library crate so integration tests (which compile as separate
//! crates) can include it via `#[path]`.
//!
//! The directory in this file is paths-ignored by the codeql config
//! (`**/*_test_vectors.rs`).

#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;

/// 100-segment fragmentation pattern: 30 fragmented segments, scoring
/// from very-fragmented (segment 0) to nearly-full (segment 29). The
/// remaining 70 are at boundary states (fully empty or fully full).
pub fn fragment_pattern_partial() -> Vec<(u32, u16)> {
    let mut out = Vec::with_capacity(100);
    for i in 0..30u32 {
        let valid = (10 + i * 10) as u16;
        out.push((i, valid));
    }
    for i in 30..70u32 {
        out.push((i, 512));
    }
    for i in 70..100u32 {
        out.push((i, 0));
    }
    out
}

/// `(nid, mtime_secs)` pair shorthand.
pub type InodeMtime = (u32, u64);

/// 2000 synthetic (nid, mtime) pairs split evenly between hot
/// (mtime = now-5) and cold (mtime = now-10000).
pub fn hot_cold_inode_table(now: u64) -> (Vec<InodeMtime>, Vec<InodeMtime>) {
    let mut hot = Vec::with_capacity(1000);
    let mut cold = Vec::with_capacity(1000);
    for i in 0..1000u32 {
        hot.push((100 + i, now.saturating_sub(5)));
        cold.push((10_000 + i, now.saturating_sub(10_000)));
    }
    (hot, cold)
}

/// 64-byte SIT validity bitmap with the first `n` bits set.
pub fn fragmented_segment_validity_map(n: usize) -> [u8; 64] {
    let mut map = [0u8; 64];
    for i in 0..n.min(512) {
        map[i / 8] |= 1u8 << (i % 8);
    }
    map
}

/// 64-byte payload generated from a file_id; deterministic.
pub fn synthetic_payload(id: u32) -> [u8; 64] {
    let mut v = [0u8; 64];
    for j in 0..64u32 {
        v[j as usize] = ((id ^ j) & 0xFF) as u8;
    }
    v
}

/// 64-byte overwrite payload (XOR'd with 0xA5 from the original).
pub fn synthetic_overwrite_payload(id: u32) -> [u8; 64] {
    let mut v = [0u8; 64];
    for j in 0..64u32 {
        v[j as usize] = ((id ^ j ^ 0xA5) & 0xFF) as u8;
    }
    v
}
