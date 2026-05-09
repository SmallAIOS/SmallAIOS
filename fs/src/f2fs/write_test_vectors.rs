// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test vectors for the F2FS write path.
//!
//! Hand-rolled byte sequences and expected post-write state used by the
//! Phase 7 unit tests in `write.rs`. Lives under `*_test_vectors.rs` so
//! the CodeQL "hard-coded byte array" rule does not flag the on-disk-
//! format byte arrays we need.

#![allow(dead_code)]

extern crate alloc;

/// 16 bytes of recognizable payload used by round-trip tests.
pub const SHORT_PAYLOAD: &[u8] = b"hello, f2fs RW!\n";

/// 4 KiB of pseudo-random pattern used by full-block write tests.
pub fn block_pattern() -> alloc::vec::Vec<u8> {
    let mut v = alloc::vec::Vec::with_capacity(4096);
    for i in 0..4096u32 {
        v.push(((i ^ (i >> 7)) & 0xFF) as u8);
    }
    v
}

/// 8 KiB pattern (two-block payload) for two-block writes.
pub fn two_block_pattern() -> alloc::vec::Vec<u8> {
    let mut v = block_pattern();
    let n = v.len();
    for i in 0..n {
        v.push((v[i] ^ 0xA5) as u8);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_payload_size_is_16() {
        assert_eq!(SHORT_PAYLOAD.len(), 16);
    }

    #[test]
    fn block_pattern_is_4kib() {
        assert_eq!(block_pattern().len(), 4096);
    }

    #[test]
    fn two_block_pattern_is_8kib() {
        assert_eq!(two_block_pattern().len(), 8192);
    }

    #[test]
    fn block_pattern_is_deterministic() {
        assert_eq!(block_pattern(), block_pattern());
    }
}
