// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC packet number encoding/decoding (RFC 9000 Section 17.1, Appendix A).
//!
//! Packet numbers are encoded using variable-length encoding (1-4 bytes).
//! Decoding requires the largest acknowledged packet number to resolve
//! the full packet number from the truncated representation.

use crate::NetError;

/// Minimum packet number length.
pub const MIN_PN_LENGTH: u8 = 1;

/// Maximum packet number length.
pub const MAX_PN_LENGTH: u8 = 4;

/// Encode a packet number using the minimum number of bytes needed.
///
/// The length is determined by how many bits differ between `pn` and
/// `largest_acked`. Per RFC 9000 Appendix A, the sender uses enough
/// bytes so the receiver can unambiguously recover the full PN.
///
/// Returns (encoded_bytes, pn_length).
pub fn encode_packet_number(
    pn: u64,
    largest_acked: u64,
    buf: &mut [u8],
) -> Result<(usize, u8), NetError> {
    let pn_len = packet_number_length(pn, largest_acked);
    if buf.len() < pn_len as usize {
        return Err(NetError::BufferTooSmall);
    }
    let truncated = pn & ((1u64 << (pn_len as u64 * 8)) - 1);
    let bytes = truncated.to_be_bytes();
    let start = 8 - pn_len as usize;
    buf[..pn_len as usize].copy_from_slice(&bytes[start..]);
    Ok((pn_len as usize, pn_len))
}

/// Decode a truncated packet number using the largest acknowledged PN.
///
/// Implements the algorithm from RFC 9000 Appendix A.
pub fn decode_packet_number(truncated_pn: u64, pn_length: u8, largest_pn: u64) -> u64 {
    let pn_nbits = pn_length as u64 * 8;
    let pn_win = 1u64 << pn_nbits;
    let pn_hwin = pn_win / 2;
    let pn_mask = pn_win - 1;

    // Expected packet number is one more than the largest received
    let expected_pn = largest_pn + 1;

    // Candidate = high bits from expected + low bits from truncated
    let candidate = (expected_pn & !pn_mask) | truncated_pn;

    // Adjust if candidate is too far from expected
    // Use saturating subtraction to avoid wrapping issues with u64
    if candidate + pn_hwin <= expected_pn && candidate + pn_win < (1u64 << 62) {
        candidate + pn_win
    } else if candidate > expected_pn + pn_hwin && candidate >= pn_win {
        candidate - pn_win
    } else {
        candidate
    }
}

/// Determine the minimum packet number encoding length.
fn packet_number_length(pn: u64, largest_acked: u64) -> u8 {
    // Number of contiguous unacked packets
    let num_unacked = if pn > largest_acked {
        pn - largest_acked
    } else {
        1
    };

    // Need enough bytes so the truncated PN can uniquely identify
    // the full PN within a window of 2 * num_unacked
    let range = 2 * num_unacked;

    if range <= 0x80 {
        1
    } else if range <= 0x8000 {
        2
    } else if range <= 0x800000 {
        3
    } else {
        4
    }
}

/// Read a truncated packet number from a buffer.
pub fn read_packet_number(data: &[u8], pn_length: u8) -> Result<u64, NetError> {
    let len = pn_length as usize;
    if data.len() < len {
        return Err(NetError::PacketTooShort);
    }
    let mut pn = 0u64;
    for &byte in &data[..len] {
        pn = (pn << 8) | byte as u64;
    }
    Ok(pn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_1byte() {
        let mut buf = [0u8; 4];
        let (len, pn_len) = encode_packet_number(10, 9, &mut buf).unwrap();
        assert_eq!(pn_len, 1);
        assert_eq!(len, 1);
        assert_eq!(buf[0], 10);
    }

    #[test]
    fn test_encode_2byte() {
        let mut buf = [0u8; 4];
        let (len, pn_len) = encode_packet_number(300, 0, &mut buf).unwrap();
        assert_eq!(pn_len, 2);
        assert_eq!(len, 2);
        let truncated = u16::from_be_bytes([buf[0], buf[1]]);
        assert_eq!(truncated, 300);
    }

    #[test]
    fn test_encode_buffer_too_small() {
        let mut buf = [0u8; 0];
        assert_eq!(
            encode_packet_number(10, 0, &mut buf).err(),
            Some(NetError::BufferTooSmall)
        );
    }

    #[test]
    fn test_decode_1byte_simple() {
        let decoded = decode_packet_number(10, 1, 9);
        assert_eq!(decoded, 10);
    }

    #[test]
    fn test_decode_2byte() {
        let decoded = decode_packet_number(300, 2, 200);
        assert_eq!(decoded, 300);
    }

    #[test]
    fn test_decode_wrapping() {
        // PN wraps around: largest_pn = 0xFE, truncated = 0x02, 1-byte
        let decoded = decode_packet_number(0x02, 1, 0xFE);
        assert_eq!(decoded, 0x102);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        for (pn, largest) in [
            (0, 0),
            (1, 0),
            (100, 99),
            (256, 200),
            (1000, 500),
            (70000, 69000),
        ] {
            let mut buf = [0u8; 4];
            let (_, pn_len) = encode_packet_number(pn, largest, &mut buf).unwrap();
            let truncated = read_packet_number(&buf, pn_len).unwrap();
            let decoded = decode_packet_number(truncated, pn_len, largest);
            assert_eq!(decoded, pn, "Failed for pn={pn}, largest={largest}");
        }
    }

    #[test]
    fn test_read_packet_number() {
        assert_eq!(read_packet_number(&[0x42], 1).unwrap(), 0x42);
        assert_eq!(read_packet_number(&[0x01, 0x00], 2).unwrap(), 256);
        assert_eq!(read_packet_number(&[0x00, 0x01, 0x00], 3).unwrap(), 256);
        assert_eq!(
            read_packet_number(&[0x00, 0x00, 0x01, 0x00], 4).unwrap(),
            256
        );
    }

    #[test]
    fn test_read_packet_number_too_short() {
        assert_eq!(
            read_packet_number(&[0x42], 2).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_packet_number_length() {
        assert_eq!(packet_number_length(1, 0), 1);
        assert_eq!(packet_number_length(64, 0), 1);
        assert_eq!(packet_number_length(200, 0), 2);
        assert_eq!(packet_number_length(100_000, 0), 3);
        assert_eq!(packet_number_length(10_000_000, 0), 4);
    }

    #[test]
    fn test_packet_number_length_close() {
        // When pn and largest_acked are close, 1 byte suffices
        assert_eq!(packet_number_length(1000, 999), 1);
        assert_eq!(packet_number_length(1000, 998), 1);
    }

    #[test]
    fn test_decode_large_gap() {
        // 4-byte encoding for large numbers
        let mut buf = [0u8; 4];
        let pn = 0x00FFFFFF;
        let largest = 0x00FF0000;
        let (_, pn_len) = encode_packet_number(pn, largest, &mut buf).unwrap();
        let truncated = read_packet_number(&buf, pn_len).unwrap();
        let decoded = decode_packet_number(truncated, pn_len, largest);
        assert_eq!(decoded, pn);
    }
}
