// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room LZ4 block-format decompressor (squashfs convention).
//!
//! Squashfs `mksquashfs -comp lz4` writes each block as a raw LZ4
//! block (NOT the LZ4-frame format with magic `0x184d2204`). The block
//! is a sequence of *sequences*, each with:
//!
//! ```text
//!   ┌──────────────┬───────────────┐
//!   │ token (1 B)  │ literal/match │
//!   └──────────────┴───────────────┘
//!     │  high 4 bits = literal length (extended via 0xFF runs)
//!     │  low  4 bits = match length minus 4 (extended via 0xFF runs)
//!     │
//!     │  literals (literal_len bytes)
//!     │  if not last sequence:
//!     │    offset (u16 LE)
//!     │    extended match-length bytes
//! ```
//!
//! The last sequence in a block has 0 in its low nibble and no offset
//! field. See <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>
//! for the canonical reference; this implementation matches that
//! description byte-for-byte.

extern crate alloc;

use alloc::vec::Vec;

use super::DecompressError;

/// Decompress a single LZ4 block.
pub fn decompress(input: &[u8], max_out: usize) -> Result<Vec<u8>, DecompressError> {
    let mut out: Vec<u8> = Vec::new();
    let mut ip: usize = 0;
    while ip < input.len() {
        let token = input[ip];
        ip += 1;
        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            loop {
                if ip >= input.len() {
                    return Err(DecompressError::Truncated);
                }
                let b = input[ip];
                ip += 1;
                lit_len = lit_len
                    .checked_add(b as usize)
                    .ok_or(DecompressError::Corrupt)?;
                if b != 0xFF {
                    break;
                }
            }
        }
        if ip + lit_len > input.len() {
            return Err(DecompressError::Truncated);
        }
        if out.len().saturating_add(lit_len) > max_out {
            return Err(DecompressError::OutputTooLarge);
        }
        out.extend_from_slice(&input[ip..ip + lit_len]);
        ip += lit_len;

        // End of block? Remaining input is exactly 0 if this was the
        // last sequence; otherwise we expect a 2-byte offset.
        if ip == input.len() {
            // Validate the 5-byte rule (per LZ4 block-format rules):
            // last 5 bytes are always literals. We don't enforce that
            // here strictly because some encoders are lenient, but we
            // do accept it.
            return Ok(out);
        }
        if ip + 2 > input.len() {
            return Err(DecompressError::Truncated);
        }
        let offset = u16::from_le_bytes([input[ip], input[ip + 1]]) as usize;
        ip += 2;
        if offset == 0 || offset > out.len() {
            return Err(DecompressError::Corrupt);
        }
        let mut match_len = (token & 0x0F) as usize + 4;
        if (token & 0x0F) == 15 {
            loop {
                if ip >= input.len() {
                    return Err(DecompressError::Truncated);
                }
                let b = input[ip];
                ip += 1;
                match_len = match_len
                    .checked_add(b as usize)
                    .ok_or(DecompressError::Corrupt)?;
                if b != 0xFF {
                    break;
                }
            }
        }
        if out.len().saturating_add(match_len) > max_out {
            return Err(DecompressError::OutputTooLarge);
        }
        let start = out.len() - offset;
        // Forward copy that handles overlap (LZ4 self-references like
        // RLE — `offset = 1` repeats the last byte for `match_len`
        // iterations).
        for i in 0..match_len {
            let b = out[start + i];
            out.push(b);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a single LZ4 sequence and append it to `out`. Helper for
    /// tests; not a general-purpose encoder.
    fn encode_token(literals: &[u8], match_len: Option<usize>, offset: u16, out: &mut Vec<u8>) {
        let lit_len = literals.len();
        let lit_token = if lit_len < 15 { lit_len } else { 15 };
        let match_part = match match_len {
            Some(m) => {
                let m4 = m.saturating_sub(4);
                if m4 < 15 {
                    m4
                } else {
                    15
                }
            }
            None => 0,
        };
        let token = ((lit_token as u8) << 4) | (match_part as u8);
        out.push(token);
        if lit_len >= 15 {
            let mut rem = lit_len - 15;
            while rem >= 0xFF {
                out.push(0xFF);
                rem -= 0xFF;
            }
            out.push(rem as u8);
        }
        out.extend_from_slice(literals);
        if let Some(m) = match_len {
            out.extend_from_slice(&offset.to_le_bytes());
            let m4 = m - 4;
            if m4 >= 15 {
                let mut rem = m4 - 15;
                while rem >= 0xFF {
                    out.push(0xFF);
                    rem -= 0xFF;
                }
                out.push(rem as u8);
            }
        }
    }

    #[test]
    fn pure_literal_block() {
        let lit: &[u8] = b"abcdef";
        let mut buf = Vec::new();
        encode_token(lit, None, 0, &mut buf);
        let out = decompress(&buf, 64).unwrap();
        assert_eq!(out, b"abcdef");
    }

    #[test]
    fn literal_then_match() {
        let mut buf = Vec::new();
        encode_token(b"ab", Some(4), 2, &mut buf);
        encode_token(b"", None, 0, &mut buf);
        let out = decompress(&buf, 64).unwrap();
        // After literals "ab", match offset 2 length 4 → repeat "ab" twice.
        assert_eq!(out, b"ababab");
    }

    #[test]
    fn rle_offset_one() {
        let mut buf = Vec::new();
        encode_token(b"a", Some(5), 1, &mut buf);
        encode_token(b"", None, 0, &mut buf);
        let out = decompress(&buf, 64).unwrap();
        assert_eq!(out, b"aaaaaa");
    }

    #[test]
    fn extended_literal_run() {
        let big: Vec<u8> = (0..50).map(|i| i as u8).collect();
        let mut buf = Vec::new();
        encode_token(&big, None, 0, &mut buf);
        let out = decompress(&buf, 1024).unwrap();
        assert_eq!(out, big);
    }

    #[test]
    fn extended_match_run() {
        let lit: &[u8] = b"a";
        let mut buf = Vec::new();
        encode_token(lit, Some(50), 1, &mut buf);
        encode_token(b"", None, 0, &mut buf);
        let out = decompress(&buf, 1024).unwrap();
        assert_eq!(out, alloc::vec![b'a'; 51]);
    }

    #[test]
    fn rejects_offset_past_buffer() {
        // Try to reference offset 2 with no preceding literals.
        let mut buf = Vec::new();
        encode_token(b"", Some(4), 2, &mut buf);
        encode_token(b"", None, 0, &mut buf);
        assert_eq!(decompress(&buf, 64), Err(DecompressError::Corrupt));
    }

    #[test]
    fn rejects_offset_zero() {
        // Token says 4-byte literal then a match; manually craft an
        // offset of 0 (illegal in LZ4).
        let bad = alloc::vec![0x40, b'a', b'b', b'c', b'd', 0x00, 0x00];
        assert_eq!(decompress(&bad, 64), Err(DecompressError::Corrupt));
    }

    #[test]
    fn rejects_truncated_offset() {
        let bad = alloc::vec![0x10, b'a', 0x01];
        assert_eq!(decompress(&bad, 64), Err(DecompressError::Truncated));
    }

    #[test]
    fn rejects_output_overflow() {
        let lit: &[u8] = b"a";
        let mut buf = Vec::new();
        encode_token(lit, Some(50), 1, &mut buf);
        encode_token(b"", None, 0, &mut buf);
        assert_eq!(decompress(&buf, 4), Err(DecompressError::OutputTooLarge));
    }
}
