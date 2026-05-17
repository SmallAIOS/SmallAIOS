// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Poly1305 one-time MAC (RFC 8439 § 2.5).
//!
//! Used as the authentication half of ChaCha20-Poly1305 AEAD.
//! The key MUST be derived per-message; reusing a Poly1305 key
//! across two messages catastrophically breaks the MAC. The
//! AEAD wrapper in `chacha20_poly1305.rs` enforces this by
//! deriving the Poly1305 key from `ChaCha20(key, nonce, 0)`.

#![allow(clippy::many_single_char_names)]

/// 32-byte Poly1305 key (16-byte r + 16-byte s).
pub const KEY_LEN: usize = 32;

/// 16-byte tag.
pub const TAG_LEN: usize = 16;

/// Compute the Poly1305 tag of `msg` under `key`.
///
/// Implementation follows RFC 8439 § 2.5.1 directly. The
/// arithmetic is done in 130-bit precision over `u64` limbs
/// with a 130-bit modulus reduction at each block.
pub fn poly1305(key: &[u8; KEY_LEN], msg: &[u8]) -> [u8; TAG_LEN] {
    // Clamp r per § 2.5.
    let mut r = [0u8; 16];
    r.copy_from_slice(&key[..16]);
    r[3] &= 0x0f;
    r[7] &= 0x0f;
    r[11] &= 0x0f;
    r[15] &= 0x0f;
    r[4] &= 0xfc;
    r[8] &= 0xfc;
    r[12] &= 0xfc;

    // r as five 26-bit limbs (radix 2^26).
    let r0 = u32::from_le_bytes([r[0], r[1], r[2], r[3]]) & 0x3ff_ffff;
    let r1 = (u32::from_le_bytes([r[3], r[4], r[5], r[6]]) >> 2) & 0x3ff_ff03;
    let r2 = (u32::from_le_bytes([r[6], r[7], r[8], r[9]]) >> 4) & 0x3ff_c0ff;
    let r3 = (u32::from_le_bytes([r[9], r[10], r[11], r[12]]) >> 6) & 0x3f0_3fff;
    let r4 = (u32::from_le_bytes([r[12], r[13], r[14], r[15]]) >> 8) & 0x00f_ffff;
    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

    let mut h0: u32 = 0;
    let mut h1: u32 = 0;
    let mut h2: u32 = 0;
    let mut h3: u32 = 0;
    let mut h4: u32 = 0;

    let mut offset = 0;
    while offset < msg.len() {
        let take = (msg.len() - offset).min(16);
        let mut block = [0u8; 17];
        block[..take].copy_from_slice(&msg[offset..offset + take]);
        block[take] = 1;

        let m0 = u32::from_le_bytes([block[0], block[1], block[2], block[3]]) & 0x3ff_ffff;
        let m1 = (u32::from_le_bytes([block[3], block[4], block[5], block[6]]) >> 2) & 0x3ff_ffff;
        let m2 = (u32::from_le_bytes([block[6], block[7], block[8], block[9]]) >> 4) & 0x3ff_ffff;
        let m3 =
            (u32::from_le_bytes([block[9], block[10], block[11], block[12]]) >> 6) & 0x3ff_ffff;
        let m4 = u32::from_le_bytes([block[12], block[13], block[14], block[15]]) >> 8
            | (block[16] as u32) << 24;

        h0 = h0.wrapping_add(m0);
        h1 = h1.wrapping_add(m1);
        h2 = h2.wrapping_add(m2);
        h3 = h3.wrapping_add(m3);
        h4 = h4.wrapping_add(m4);

        // h *= r mod (2^130 - 5)
        let d0 = (h0 as u64) * (r0 as u64)
            + (h1 as u64) * (s4 as u64)
            + (h2 as u64) * (s3 as u64)
            + (h3 as u64) * (s2 as u64)
            + (h4 as u64) * (s1 as u64);
        let d1 = (h0 as u64) * (r1 as u64)
            + (h1 as u64) * (r0 as u64)
            + (h2 as u64) * (s4 as u64)
            + (h3 as u64) * (s3 as u64)
            + (h4 as u64) * (s2 as u64);
        let d2 = (h0 as u64) * (r2 as u64)
            + (h1 as u64) * (r1 as u64)
            + (h2 as u64) * (r0 as u64)
            + (h3 as u64) * (s4 as u64)
            + (h4 as u64) * (s3 as u64);
        let d3 = (h0 as u64) * (r3 as u64)
            + (h1 as u64) * (r2 as u64)
            + (h2 as u64) * (r1 as u64)
            + (h3 as u64) * (r0 as u64)
            + (h4 as u64) * (s4 as u64);
        let d4 = (h0 as u64) * (r4 as u64)
            + (h1 as u64) * (r3 as u64)
            + (h2 as u64) * (r2 as u64)
            + (h3 as u64) * (r1 as u64)
            + (h4 as u64) * (r0 as u64);

        let c = (d0 >> 26) as u32;
        h0 = (d0 as u32) & 0x3ff_ffff;
        let d1 = d1 + c as u64;
        let c = (d1 >> 26) as u32;
        h1 = (d1 as u32) & 0x3ff_ffff;
        let d2 = d2 + c as u64;
        let c = (d2 >> 26) as u32;
        h2 = (d2 as u32) & 0x3ff_ffff;
        let d3 = d3 + c as u64;
        let c = (d3 >> 26) as u32;
        h3 = (d3 as u32) & 0x3ff_ffff;
        let d4 = d4 + c as u64;
        let c = (d4 >> 26) as u32;
        h4 = (d4 as u32) & 0x3ff_ffff;
        h0 = h0.wrapping_add(c.wrapping_mul(5));
        let c = h0 >> 26;
        h0 &= 0x3ff_ffff;
        h1 = h1.wrapping_add(c);

        offset += take;
    }

    // Final carry chain.
    let c = h1 >> 26;
    h1 &= 0x3ff_ffff;
    h2 = h2.wrapping_add(c);
    let c = h2 >> 26;
    h2 &= 0x3ff_ffff;
    h3 = h3.wrapping_add(c);
    let c = h3 >> 26;
    h3 &= 0x3ff_ffff;
    h4 = h4.wrapping_add(c);
    let c = h4 >> 26;
    h4 &= 0x3ff_ffff;
    h0 = h0.wrapping_add(c.wrapping_mul(5));
    let c = h0 >> 26;
    h0 &= 0x3ff_ffff;
    h1 = h1.wrapping_add(c);

    // Compute h - p where p = 2^130 - 5.
    let mut g0 = h0.wrapping_add(5);
    let c = g0 >> 26;
    g0 &= 0x3ff_ffff;
    let mut g1 = h1.wrapping_add(c);
    let c = g1 >> 26;
    g1 &= 0x3ff_ffff;
    let mut g2 = h2.wrapping_add(c);
    let c = g2 >> 26;
    g2 &= 0x3ff_ffff;
    let mut g3 = h3.wrapping_add(c);
    let c = g3 >> 26;
    g3 &= 0x3ff_ffff;
    let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    // Mask = 0xFFFFFFFF if h >= p else 0.
    let mask = (g4 >> 31).wrapping_sub(1);
    h0 = (h0 & !mask) | (g0 & mask);
    h1 = (h1 & !mask) | (g1 & mask);
    h2 = (h2 & !mask) | (g2 & mask);
    h3 = (h3 & !mask) | (g3 & mask);
    h4 = (h4 & !mask) | (g4 & mask);

    // Repack as 128 bits.
    let h0 = h0 | (h1 << 26);
    let h1 = (h1 >> 6) | (h2 << 20);
    let h2 = (h2 >> 12) | (h3 << 14);
    let h3 = (h3 >> 18) | (h4 << 8);

    // Add s.
    let s0 = u32::from_le_bytes([key[16], key[17], key[18], key[19]]);
    let s1 = u32::from_le_bytes([key[20], key[21], key[22], key[23]]);
    let s2 = u32::from_le_bytes([key[24], key[25], key[26], key[27]]);
    let s3 = u32::from_le_bytes([key[28], key[29], key[30], key[31]]);

    let t0 = h0 as u64 + s0 as u64;
    let t1 = h1 as u64 + s1 as u64 + (t0 >> 32);
    let t2 = h2 as u64 + s2 as u64 + (t1 >> 32);
    let t3 = h3 as u64 + s3 as u64 + (t2 >> 32);

    let mut tag = [0u8; 16];
    tag[..4].copy_from_slice(&(t0 as u32).to_le_bytes());
    tag[4..8].copy_from_slice(&(t1 as u32).to_le_bytes());
    tag[8..12].copy_from_slice(&(t2 as u32).to_le_bytes());
    tag[12..16].copy_from_slice(&(t3 as u32).to_le_bytes());
    tag
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 8439 § 2.5.2 test vector.
    #[test]
    fn rfc_8439_2_5_2() {
        let key: [u8; 32] = [
            0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5,
            0x06, 0xa8, 0x01, 0x03, 0x80, 0x8a, 0xfb, 0x0d, 0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf,
            0x41, 0x49, 0xf5, 0x1b,
        ];
        let msg = b"Cryptographic Forum Research Group";
        let tag = poly1305(&key, msg);
        let expected = [
            0xa8, 0x06, 0x1d, 0xc1, 0x30, 0x51, 0x36, 0xc6, 0xc2, 0x2b, 0x8b, 0xaf, 0x0c, 0x01,
            0x27, 0xa9,
        ];
        assert_eq!(tag, expected);
    }

    #[test]
    fn empty_message_round_trip() {
        // Empty message: tag is just s after the clamp-multiply step.
        let key = [0u8; 32];
        let tag = poly1305(&key, &[]);
        assert_eq!(tag, [0u8; 16]);
    }
}
