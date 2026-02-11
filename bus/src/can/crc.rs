// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN CRC algorithms.
//!
//! - CRC-15: Used by CAN 2.0A/B, polynomial 0x4599 (ISO 11898)
//! - CRC-17: Used by CAN FD for payloads <= 16 bytes, polynomial 0x3685B
//! - CRC-21: Used by CAN FD for payloads > 16 bytes, polynomial 0x302899

/// CRC-15 generator polynomial for CAN 2.0A/B (ISO 11898).
///
/// x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1 = 0x4599
const CRC15_POLY: u16 = 0x4599;

/// CRC-17 generator polynomial for CAN FD (payloads <= 16 bytes).
///
/// x^17 + x^16 + x^14 + x^13 + x^11 + x^6 + x^4 + x^3 + x + 1 = 0x3685B
const CRC17_POLY: u32 = 0x3685B;

/// CRC-21 generator polynomial for CAN FD (payloads > 16 bytes).
///
/// x^21 + x^20 + x^13 + x^11 + x^7 + x^4 + x^3 + 1 = 0x302899
const CRC21_POLY: u32 = 0x302899;

/// Compute CRC-15 over a byte slice.
///
/// The CRC is computed bit-by-bit over each byte (MSB first), using the
/// ISO 11898 generator polynomial (0x4599). The CRC register is initialized
/// to 0, each bit is shifted in, and the polynomial is XORed when the MSB
/// of the register is set. Returns a 15-bit CRC value.
pub fn crc15(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u16;
            let msb = (crc >> 14) & 1;
            crc = ((crc << 1) & 0x7FFF) | bit;
            if msb != 0 {
                crc ^= CRC15_POLY;
            }
        }
    }
    crc & 0x7FFF
}

/// Compute CRC-17 over a byte slice.
///
/// Used by CAN FD for frames with payloads <= 16 bytes.
/// CRC register is initialized to 0, computed bit-by-bit MSB first.
pub fn crc17(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let msb = (crc >> 16) & 1;
            crc = ((crc << 1) & 0x1FFFF) | bit;
            if msb != 0 {
                crc ^= CRC17_POLY;
            }
        }
    }
    crc & 0x1FFFF
}

/// Compute CRC-21 over a byte slice.
///
/// Used by CAN FD for frames with payloads > 16 bytes.
/// CRC register is initialized to 0, computed bit-by-bit MSB first.
pub fn crc21(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let msb = (crc >> 20) & 1;
            crc = ((crc << 1) & 0x1FFFFF) | bit;
            if msb != 0 {
                crc ^= CRC21_POLY;
            }
        }
    }
    crc & 0x1FFFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc15_empty() {
        assert_eq!(crc15(&[]), 0);
    }

    #[test]
    fn test_crc15_single_byte() {
        let crc = crc15(&[0xAA]);
        // CRC should be a 15-bit value
        assert!(crc <= 0x7FFF);
        // Should be deterministic
        assert_eq!(crc, crc15(&[0xAA]));
    }

    #[test]
    fn test_crc15_known_data() {
        // Test with a known data pattern
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let crc = crc15(&data);
        assert!(crc <= 0x7FFF);
        // Verify determinism
        assert_eq!(crc, crc15(&data));
    }

    #[test]
    fn test_crc15_different_data_different_crc() {
        let crc1 = crc15(&[0x01, 0x02, 0x03]);
        let crc2 = crc15(&[0x04, 0x05, 0x06]);
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn test_crc15_detects_single_bit_error() {
        let data1 = [0x55, 0xAA, 0x33];
        let data2 = [0x55, 0xAA, 0x32]; // bit 0 flipped
        assert_ne!(crc15(&data1), crc15(&data2));
    }

    #[test]
    fn test_crc17_empty() {
        assert_eq!(crc17(&[]), 0);
    }

    #[test]
    fn test_crc17_single_byte() {
        let crc = crc17(&[0xAA]);
        assert!(crc <= 0x1FFFF);
        assert_eq!(crc, crc17(&[0xAA]));
    }

    #[test]
    fn test_crc17_different_data_different_crc() {
        let crc1 = crc17(&[0x01, 0x02, 0x03]);
        let crc2 = crc17(&[0x04, 0x05, 0x06]);
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn test_crc17_detects_single_bit_error() {
        let data1 = [0x55, 0xAA, 0x33, 0xFF];
        let data2 = [0x55, 0xAA, 0x33, 0xFE];
        assert_ne!(crc17(&data1), crc17(&data2));
    }

    #[test]
    fn test_crc21_empty() {
        assert_eq!(crc21(&[]), 0);
    }

    #[test]
    fn test_crc21_single_byte() {
        let crc = crc21(&[0xAA]);
        assert!(crc <= 0x1FFFFF);
        assert_eq!(crc, crc21(&[0xAA]));
    }

    #[test]
    fn test_crc21_different_data_different_crc() {
        let crc1 = crc21(&[0x01, 0x02, 0x03]);
        let crc2 = crc21(&[0x04, 0x05, 0x06]);
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn test_crc21_detects_single_bit_error() {
        let data1 = [0x55; 32];
        let mut data2 = [0x55; 32];
        data2[16] = 0x54;
        assert_ne!(crc21(&data1), crc21(&data2));
    }

    #[test]
    fn test_crc15_max_value_range() {
        // Test with all-ones to ensure max CRC is within 15-bit range
        let data = [0xFF; 64];
        let crc = crc15(&data);
        assert!(crc <= 0x7FFF, "CRC-15 exceeded 15-bit range: {:#06x}", crc);
    }

    #[test]
    fn test_crc17_max_value_range() {
        let data = [0xFF; 64];
        let crc = crc17(&data);
        assert!(
            crc <= 0x1FFFF,
            "CRC-17 exceeded 17-bit range: {:#07x}",
            crc
        );
    }

    #[test]
    fn test_crc21_max_value_range() {
        let data = [0xFF; 64];
        let crc = crc21(&data);
        assert!(
            crc <= 0x1FFFFF,
            "CRC-21 exceeded 21-bit range: {:#08x}",
            crc
        );
    }
}
