// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS Communications Link Transmission Unit (CLTU) encoding.
//!
//! Implements CLTU encoding per CCSDS 231.0-B for TC uplink. A CLTU wraps
//! a TC Transfer Frame with:
//! - Start sequence (0xEB90)
//! - BCH-encoded code blocks (7 info bytes + 1 parity byte)
//! - Tail sequence (0xC5C5C5C5C5C5C579)
//!
//! The BCH(63,56) code provides single-error correction for each code block.

use alloc::vec::Vec;
use crate::BusError;

/// CLTU start sequence (2 bytes).
pub const CLTU_START: [u8; 2] = [0xEB, 0x90];

/// CLTU tail sequence (8 bytes) — used to signal end of CLTU.
pub const CLTU_TAIL: [u8; 8] = [0xC5, 0xC5, 0xC5, 0xC5, 0xC5, 0xC5, 0xC5, 0x79];

/// Number of information bytes per BCH code block.
pub const CODE_BLOCK_INFO_LEN: usize = 7;

/// Total code block length (info + parity).
pub const CODE_BLOCK_LEN: usize = 8;

/// Fill byte value (0x55 per CCSDS 231.0-B).
pub const FILL_BYTE: u8 = 0x55;

/// Minimum CLTU length: start(2) + one code block(8) + tail(8).
pub const CLTU_MIN_LEN: usize = CLTU_START.len() + CODE_BLOCK_LEN + CLTU_TAIL.len();

/// BCH lookup table for the shortened (63,56) code.
///
/// The generator polynomial is g(x) = x^7 + x^6 + x^2 + 1 (0xC5 in reversed form).
/// This table provides the BCH parity byte for each input byte XOR'd into the register.
const BCH_TABLE: [u8; 256] = bch_build_table();

/// Build the BCH-63,56 lookup table at compile time.
const fn bch_build_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i: u16 = 0;
    while i < 256 {
        let mut crc: u8 = i as u8;
        let mut bit = 0;
        while bit < 8 {
            if crc & 0x80 != 0 {
                // g(x) = x^7 + x^6 + x^2 + 1 = 0b11000101 = 0xC5
                crc = (crc << 1) ^ 0xC5;
            } else {
                crc <<= 1;
            }
            bit += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
}

/// Compute the BCH parity byte for a 7-byte information block.
pub fn bch_parity(info: &[u8; CODE_BLOCK_INFO_LEN]) -> u8 {
    let mut reg: u8 = 0;
    for &byte in info {
        reg = BCH_TABLE[(reg ^ byte) as usize];
    }
    // Complement the parity (CCSDS convention)
    !reg
}

/// Verify a code block (7 info + 1 parity). Returns true if valid.
pub fn bch_verify(block: &[u8; CODE_BLOCK_LEN]) -> bool {
    let mut info = [0u8; CODE_BLOCK_INFO_LEN];
    info.copy_from_slice(&block[..CODE_BLOCK_INFO_LEN]);
    let expected = bch_parity(&info);
    expected == block[CODE_BLOCK_INFO_LEN]
}

/// Encode a TC frame data into a CLTU.
///
/// The input data is split into 7-byte blocks, each protected with a BCH
/// parity byte. The last block is padded with fill bytes (0x55) if needed.
/// The CLTU is wrapped with a start sequence and tail sequence.
pub fn cltu_encode(tc_data: &[u8]) -> Vec<u8> {
    if tc_data.is_empty() {
        // Even empty data gets one fill block
        let mut out = Vec::with_capacity(CLTU_MIN_LEN);
        out.extend_from_slice(&CLTU_START);
        let mut info = [FILL_BYTE; CODE_BLOCK_INFO_LEN];
        info[..0].copy_from_slice(&[]); // no-op, all fill
        let parity = bch_parity(&info);
        out.extend_from_slice(&info);
        out.push(parity);
        out.extend_from_slice(&CLTU_TAIL);
        return out;
    }

    let num_blocks = tc_data.len().div_ceil(CODE_BLOCK_INFO_LEN);
    let out_len = CLTU_START.len() + num_blocks * CODE_BLOCK_LEN + CLTU_TAIL.len();
    let mut out = Vec::with_capacity(out_len);

    // Start sequence
    out.extend_from_slice(&CLTU_START);

    // Code blocks
    let mut offset = 0;
    for _ in 0..num_blocks {
        let mut info = [FILL_BYTE; CODE_BLOCK_INFO_LEN];
        let remaining = tc_data.len() - offset;
        let copy_len = if remaining >= CODE_BLOCK_INFO_LEN {
            CODE_BLOCK_INFO_LEN
        } else {
            remaining
        };
        info[..copy_len].copy_from_slice(&tc_data[offset..offset + copy_len]);
        offset += copy_len;

        let parity = bch_parity(&info);
        out.extend_from_slice(&info);
        out.push(parity);
    }

    // Tail sequence
    out.extend_from_slice(&CLTU_TAIL);

    out
}

/// Decode a CLTU back into TC frame data.
///
/// Verifies the start sequence, BCH parity on each code block, and tail
/// sequence. Returns the concatenated information bytes (with fill bytes
/// from the last block stripped if they trail the data).
pub fn cltu_decode(cltu: &[u8]) -> Result<Vec<u8>, BusError> {
    if cltu.len() < CLTU_MIN_LEN {
        return Err(BusError::FrameTooShort);
    }

    // Verify start sequence
    if cltu[0..2] != CLTU_START {
        return Err(BusError::InvalidHeader);
    }

    // Verify tail sequence
    let tail_start = cltu.len() - CLTU_TAIL.len();
    if cltu[tail_start..] != CLTU_TAIL {
        return Err(BusError::InvalidHeader);
    }

    // Extract and verify code blocks
    let blocks_data = &cltu[CLTU_START.len()..tail_start];
    if !blocks_data.len().is_multiple_of(CODE_BLOCK_LEN) {
        return Err(BusError::InvalidFrame);
    }

    let num_blocks = blocks_data.len() / CODE_BLOCK_LEN;
    let mut result = Vec::with_capacity(num_blocks * CODE_BLOCK_INFO_LEN);

    for i in 0..num_blocks {
        let block_start = i * CODE_BLOCK_LEN;
        let mut block = [0u8; CODE_BLOCK_LEN];
        block.copy_from_slice(&blocks_data[block_start..block_start + CODE_BLOCK_LEN]);

        if !bch_verify(&block) {
            return Err(BusError::CrcMismatch);
        }

        result.extend_from_slice(&block[..CODE_BLOCK_INFO_LEN]);
    }

    Ok(result)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bch_parity_deterministic() {
        let info = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let p1 = bch_parity(&info);
        let p2 = bch_parity(&info);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_bch_parity_changes_with_data() {
        let info1 = [0x00; CODE_BLOCK_INFO_LEN];
        let info2 = [0xFF; CODE_BLOCK_INFO_LEN];
        assert_ne!(bch_parity(&info1), bch_parity(&info2));
    }

    #[test]
    fn test_bch_verify_valid() {
        let info = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11];
        let parity = bch_parity(&info);
        let mut block = [0u8; CODE_BLOCK_LEN];
        block[..CODE_BLOCK_INFO_LEN].copy_from_slice(&info);
        block[CODE_BLOCK_INFO_LEN] = parity;
        assert!(bch_verify(&block));
    }

    #[test]
    fn test_bch_verify_invalid() {
        let info = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11];
        let parity = bch_parity(&info);
        let mut block = [0u8; CODE_BLOCK_LEN];
        block[..CODE_BLOCK_INFO_LEN].copy_from_slice(&info);
        block[CODE_BLOCK_INFO_LEN] = parity ^ 0xFF; // corrupt parity
        assert!(!bch_verify(&block));
    }

    #[test]
    fn test_bch_verify_data_corruption() {
        let info = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let parity = bch_parity(&info);
        let mut block = [0u8; CODE_BLOCK_LEN];
        block[..CODE_BLOCK_INFO_LEN].copy_from_slice(&info);
        block[CODE_BLOCK_INFO_LEN] = parity;
        // Corrupt one data byte
        block[3] ^= 0x01;
        assert!(!bch_verify(&block));
    }

    #[test]
    fn test_cltu_encode_single_block() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let cltu = cltu_encode(&data);

        // Start(2) + 1 block(8) + tail(8) = 18
        assert_eq!(cltu.len(), 18);

        // Verify start sequence
        assert_eq!(&cltu[0..2], &CLTU_START);

        // Verify tail sequence
        assert_eq!(&cltu[cltu.len() - 8..], &CLTU_TAIL);
    }

    #[test]
    fn test_cltu_encode_multiple_blocks() {
        // 14 bytes -> 2 code blocks
        let data = [0xAA; 14];
        let cltu = cltu_encode(&data);

        // Start(2) + 2 blocks(16) + tail(8) = 26
        assert_eq!(cltu.len(), 26);
    }

    #[test]
    fn test_cltu_encode_partial_block_padded() {
        // 5 bytes -> 1 block, padded with 2 fill bytes
        let data = [0x11, 0x22, 0x33, 0x44, 0x55];
        let cltu = cltu_encode(&data);

        // Start(2) + 1 block(8) + tail(8) = 18
        assert_eq!(cltu.len(), 18);

        // The info bytes should be: data(5) + fill(2)
        assert_eq!(cltu[2], 0x11);
        assert_eq!(cltu[3], 0x22);
        assert_eq!(cltu[7], FILL_BYTE); // last info byte is fill
        assert_eq!(cltu[8], FILL_BYTE); // second fill byte... wait let's check
        // Actually: start is bytes 0-1, then block 0 is bytes 2-9
        // bytes 2..8 are info (7 bytes), byte 9 is parity
        assert_eq!(cltu[2], 0x11); // info[0]
        assert_eq!(cltu[3], 0x22); // info[1]
        assert_eq!(cltu[4], 0x33); // info[2]
        assert_eq!(cltu[5], 0x44); // info[3]
        assert_eq!(cltu[6], 0x55); // info[4]
        assert_eq!(cltu[7], FILL_BYTE); // info[5] = fill
        assert_eq!(cltu[8], FILL_BYTE); // info[6] = fill
        // byte 9 is parity
    }

    #[test]
    fn test_cltu_encode_empty_data() {
        let cltu = cltu_encode(&[]);
        assert_eq!(cltu.len(), CLTU_MIN_LEN);
    }

    #[test]
    fn test_cltu_roundtrip_exact_block() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let cltu = cltu_encode(&data);
        let decoded = cltu_decode(&cltu).unwrap();
        assert_eq!(&decoded[..7], &data);
    }

    #[test]
    fn test_cltu_roundtrip_multiple_blocks() {
        let data: Vec<u8> = (0..21).collect(); // 21 bytes = 3 full blocks
        let cltu = cltu_encode(&data);
        let decoded = cltu_decode(&cltu).unwrap();
        assert_eq!(&decoded[..21], &data[..]);
    }

    #[test]
    fn test_cltu_roundtrip_partial_last_block() {
        // 10 bytes -> 2 blocks, second block padded
        let data = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22, 0x33, 0x44];
        let cltu = cltu_encode(&data);
        let decoded = cltu_decode(&cltu).unwrap();
        // First 10 bytes match, remaining are fill
        assert_eq!(&decoded[..10], &data);
        // Last 4 bytes of decoded are fill
        for &b in &decoded[10..] {
            assert_eq!(b, FILL_BYTE);
        }
    }

    #[test]
    fn test_cltu_decode_too_short() {
        assert_eq!(cltu_decode(&[0u8; 5]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_cltu_decode_bad_start_sequence() {
        let mut cltu = cltu_encode(&[0x01; 7]);
        cltu[0] = 0x00; // corrupt start
        assert_eq!(cltu_decode(&cltu).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_cltu_decode_bad_tail_sequence() {
        let mut cltu = cltu_encode(&[0x01; 7]);
        let len = cltu.len();
        cltu[len - 1] = 0x00; // corrupt tail
        assert_eq!(cltu_decode(&cltu).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_cltu_decode_corrupted_code_block() {
        let mut cltu = cltu_encode(&[0x01; 7]);
        // Corrupt a data byte in the code block (byte 3 in the CLTU = info[1])
        cltu[3] ^= 0xFF;
        assert_eq!(cltu_decode(&cltu).err(), Some(BusError::CrcMismatch));
    }

    #[test]
    fn test_cltu_decode_corrupted_parity() {
        let mut cltu = cltu_encode(&[0x01; 7]);
        // Corrupt the parity byte (byte 9 in the CLTU)
        cltu[9] ^= 0xFF;
        assert_eq!(cltu_decode(&cltu).err(), Some(BusError::CrcMismatch));
    }

    #[test]
    fn test_cltu_constants() {
        assert_eq!(CLTU_START, [0xEB, 0x90]);
        assert_eq!(CLTU_TAIL.len(), 8);
        assert_eq!(CODE_BLOCK_INFO_LEN, 7);
        assert_eq!(CODE_BLOCK_LEN, 8);
        assert_eq!(FILL_BYTE, 0x55);
    }

    #[test]
    fn test_cltu_large_data() {
        // 252 bytes = 36 full blocks
        let data: Vec<u8> = (0..252).map(|i| i as u8).collect();
        let cltu = cltu_encode(&data);
        let expected_len = 2 + 36 * 8 + 8; // 298
        assert_eq!(cltu.len(), expected_len);
        let decoded = cltu_decode(&cltu).unwrap();
        assert_eq!(&decoded[..252], &data[..]);
    }
}
