// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 BCD (Binary Coded Decimal) data format.
//!
//! BCD encodes decimal digits into 4-bit nibbles packed into the 19-bit
//! data field (bits 29:11). Up to 4 full BCD digits fit in the data field
//! (16 bits), with 3 remaining bits available for sub-digit resolution or
//! discrete flags.
//!
//! The sign is encoded in the SSM field using BCD convention:
//! - SSM = 00: Plus / North / East / Right / To / Above
//! - SSM = 11: Minus / South / West / Left / From / Below
//!
//! BCD digits are packed MSB-first: the most significant digit occupies
//! the highest bits of the data field.

use super::word::{Arinc429Word, Label, Sdi, Ssm};
use crate::BusError;

/// Maximum number of BCD digits in the 19-bit data field.
/// 4 digits = 16 bits, leaving 3 bits for sub-digit info.
pub const MAX_BCD_DIGITS: usize = 4;

/// BCD codec for encoding/decoding decimal values.
///
/// Configurable number of digits (1-4) and optional fractional scaling.
#[derive(Debug, Clone, Copy)]
pub struct BcdCodec {
    /// Number of BCD digits to encode (1-4).
    num_digits: u8,
}

impl BcdCodec {
    /// Create a new BCD codec with the given number of digits.
    ///
    /// `num_digits` must be 1-4.
    pub fn new(num_digits: u8) -> Result<Self, BusError> {
        if num_digits == 0 || num_digits > MAX_BCD_DIGITS as u8 {
            return Err(BusError::InvalidDlc);
        }
        Ok(BcdCodec { num_digits })
    }

    /// Returns the number of BCD digits.
    pub fn num_digits(&self) -> u8 {
        self.num_digits
    }

    /// Maximum decimal value for this codec (e.g., 9999 for 4 digits).
    pub fn max_value(&self) -> u32 {
        10u32.pow(self.num_digits as u32) - 1
    }

    /// Encode a decimal value into the 19-bit data field as BCD.
    ///
    /// The digits are packed MSB-first into the upper bits of the data field.
    /// Returns [`BusError::OutOfRange`] if the value exceeds the max for the
    /// configured number of digits.
    pub fn encode(&self, value: u32) -> Result<u32, BusError> {
        if value > self.max_value() {
            return Err(BusError::OutOfRange);
        }

        let mut data: u32 = 0;
        let mut remaining = value;

        // Pack digits MSB-first into the data field.
        // Digit positions from MSB: digit 0 at highest nibble position.
        for i in 0..self.num_digits {
            let digit_pos = self.num_digits - 1 - i;
            let divisor = 10u32.pow(digit_pos as u32);
            let digit = remaining / divisor;
            remaining %= divisor;

            // Each digit occupies 4 bits. MSB digit at highest bit position.
            // In the 19-bit data field, we place digits starting from bit 18.
            let shift = (self.num_digits - 1 - i) as u32 * 4;
            data |= (digit & 0xF) << shift;
        }

        Ok(data)
    }

    /// Decode BCD digits from the 19-bit data field.
    ///
    /// Returns [`BusError::InvalidBcd`] if any nibble contains an invalid
    /// BCD digit (value > 9).
    pub fn decode(&self, data: u32) -> Result<u32, BusError> {
        let mut value: u32 = 0;

        for i in 0..self.num_digits {
            let shift = (self.num_digits - 1 - i) as u32 * 4;
            let nibble = (data >> shift) & 0xF;

            if nibble > 9 {
                return Err(BusError::InvalidBcd);
            }

            let digit_pos = self.num_digits - 1 - i;
            value += nibble * 10u32.pow(digit_pos as u32);
        }

        Ok(value)
    }

    /// Encode a BCD value into a complete ARINC 429 word.
    ///
    /// The sign is encoded using BCD SSM convention.
    pub fn encode_word(
        &self,
        label: Label,
        sdi: Sdi,
        value: u32,
        ssm: Ssm,
    ) -> Result<u32, BusError> {
        let data = self.encode(value)?;
        let word = Arinc429Word {
            label,
            sdi,
            data,
            ssm,
        };
        Ok(word.encode_with_ssm_raw(ssm.to_bcd_raw()))
    }

    /// Decode a BCD value from a complete ARINC 429 word.
    ///
    /// Returns the decoded decimal value and SSM status.
    pub fn decode_word(&self, raw: u32) -> Result<(u32, Ssm), BusError> {
        let word = Arinc429Word::decode_bcd(raw)?;
        let value = self.decode(word.data)?;
        Ok((value, word.ssm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BcdCodec creation
    // -----------------------------------------------------------------------

    #[test]
    fn test_bcd_codec_new_valid() {
        for digits in 1..=4u8 {
            let codec = BcdCodec::new(digits).unwrap();
            assert_eq!(codec.num_digits(), digits);
        }
    }

    #[test]
    fn test_bcd_codec_new_zero_digits() {
        assert!(BcdCodec::new(0).is_err());
    }

    #[test]
    fn test_bcd_codec_new_too_many_digits() {
        assert!(BcdCodec::new(5).is_err());
    }

    #[test]
    fn test_bcd_max_value() {
        assert_eq!(BcdCodec::new(1).unwrap().max_value(), 9);
        assert_eq!(BcdCodec::new(2).unwrap().max_value(), 99);
        assert_eq!(BcdCodec::new(3).unwrap().max_value(), 999);
        assert_eq!(BcdCodec::new(4).unwrap().max_value(), 9999);
    }

    // -----------------------------------------------------------------------
    // BCD encode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bcd_encode_zero() {
        let codec = BcdCodec::new(4).unwrap();
        let data = codec.encode(0).unwrap();
        assert_eq!(data, 0);
    }

    #[test]
    fn test_bcd_encode_single_digit() {
        let codec = BcdCodec::new(1).unwrap();
        for digit in 0..=9u32 {
            let data = codec.encode(digit).unwrap();
            assert_eq!(data, digit);
        }
    }

    #[test]
    fn test_bcd_encode_1234() {
        let codec = BcdCodec::new(4).unwrap();
        let data = codec.encode(1234).unwrap();
        // 1234 -> nibbles: 1, 2, 3, 4
        // Packed: 0x1234
        assert_eq!(data, 0x1234);
    }

    #[test]
    fn test_bcd_encode_9999() {
        let codec = BcdCodec::new(4).unwrap();
        let data = codec.encode(9999).unwrap();
        assert_eq!(data, 0x9999);
    }

    #[test]
    fn test_bcd_encode_out_of_range() {
        let codec = BcdCodec::new(4).unwrap();
        assert_eq!(codec.encode(10000), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_bcd_encode_2_digits() {
        let codec = BcdCodec::new(2).unwrap();
        let data = codec.encode(42).unwrap();
        assert_eq!(data, 0x42);
    }

    // -----------------------------------------------------------------------
    // BCD decode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bcd_decode_zero() {
        let codec = BcdCodec::new(4).unwrap();
        assert_eq!(codec.decode(0).unwrap(), 0);
    }

    #[test]
    fn test_bcd_decode_1234() {
        let codec = BcdCodec::new(4).unwrap();
        assert_eq!(codec.decode(0x1234).unwrap(), 1234);
    }

    #[test]
    fn test_bcd_decode_9999() {
        let codec = BcdCodec::new(4).unwrap();
        assert_eq!(codec.decode(0x9999).unwrap(), 9999);
    }

    #[test]
    fn test_bcd_decode_invalid_nibble() {
        let codec = BcdCodec::new(4).unwrap();
        // 0xA is invalid BCD
        assert_eq!(codec.decode(0xA000), Err(BusError::InvalidBcd));
        assert_eq!(codec.decode(0x0F00), Err(BusError::InvalidBcd));
    }

    // -----------------------------------------------------------------------
    // BCD roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bcd_roundtrip_all_4digit() {
        let codec = BcdCodec::new(4).unwrap();
        for val in [0, 1, 9, 10, 99, 100, 999, 1000, 5555, 9999] {
            let data = codec.encode(val).unwrap();
            let decoded = codec.decode(data).unwrap();
            assert_eq!(decoded, val, "BCD roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_bcd_roundtrip_3digit() {
        let codec = BcdCodec::new(3).unwrap();
        for val in [0, 1, 50, 123, 456, 789, 999] {
            let data = codec.encode(val).unwrap();
            let decoded = codec.decode(data).unwrap();
            assert_eq!(decoded, val);
        }
    }

    // -----------------------------------------------------------------------
    // BCD word encode/decode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bcd_word_roundtrip() {
        let codec = BcdCodec::new(4).unwrap();
        let label = Label::new(0o205);
        let sdi = Sdi::One;
        let ssm = Ssm::NormalOp;

        let raw = codec.encode_word(label, sdi, 3456, ssm).unwrap();
        let (value, decoded_ssm) = codec.decode_word(raw).unwrap();

        assert_eq!(value, 3456);
        assert_eq!(decoded_ssm, Ssm::NormalOp);
    }

    #[test]
    fn test_bcd_word_failure_warning() {
        let codec = BcdCodec::new(4).unwrap();
        let raw = codec
            .encode_word(Label::new(0o100), Sdi::Zero, 0, Ssm::FailureWarning)
            .unwrap();
        let (_, ssm) = codec.decode_word(raw).unwrap();
        assert_eq!(ssm, Ssm::FailureWarning);
    }

    #[test]
    fn test_bcd_word_parity_error() {
        let codec = BcdCodec::new(4).unwrap();
        let raw = codec
            .encode_word(Label::new(0), Sdi::Zero, 0, Ssm::NormalOp)
            .unwrap();
        let corrupted = raw ^ 1;
        assert_eq!(codec.decode_word(corrupted), Err(BusError::ParityError));
    }

    #[test]
    fn test_bcd_word_out_of_range() {
        let codec = BcdCodec::new(2).unwrap();
        assert_eq!(
            codec.encode_word(Label::new(0), Sdi::Zero, 100, Ssm::NormalOp),
            Err(BusError::OutOfRange)
        );
    }
}
