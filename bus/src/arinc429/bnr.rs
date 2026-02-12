// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 BNR (Binary / Two's Complement) data format.
//!
//! BNR encodes a signed value into the 19-bit data field (bits 29:11)
//! using two's complement representation with configurable resolution.
//!
//! - Bit 29: Sign bit (0 = positive, 1 = negative)
//! - Bits 28:11: Magnitude (18 bits)
//! - Resolution: `range / 2^18` per LSB
//!
//! For example, with range = 180.0 degrees:
//! - Resolution = 180.0 / 262144 = 0.000687 degrees per LSB
//! - +90.0 degrees = 90.0 / 0.000687 = 131072 = 0x20000

use super::word::{Arinc429Word, Label, Sdi, Ssm};
use crate::BusError;

/// Number of magnitude bits in a BNR data field.
const BNR_MAGNITUDE_BITS: u32 = 18;

/// Maximum magnitude value (2^18 - 1).
const BNR_MAX_MAGNITUDE: u32 = (1 << BNR_MAGNITUDE_BITS) - 1;

/// BNR codec with configurable range and resolution.
///
/// The resolution is derived from the range: `resolution = range / 2^18`.
#[derive(Debug, Clone, Copy)]
pub struct BnrCodec {
    /// Full-scale range (positive side). The value can be -range..+range.
    range: f64,
    /// Resolution per LSB = range / 2^18.
    resolution: f64,
}

impl BnrCodec {
    /// Create a new BNR codec with the given full-scale range.
    ///
    /// The resolution is automatically computed as `range / 2^18`.
    pub fn new(range: f64) -> Self {
        let resolution = range / (1u64 << BNR_MAGNITUDE_BITS) as f64;
        BnrCodec { range, resolution }
    }

    /// Returns the full-scale range.
    pub fn range(&self) -> f64 {
        self.range
    }

    /// Returns the resolution (engineering units per LSB).
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Encode a floating-point value into a 19-bit BNR data field.
    ///
    /// Returns [`BusError::OutOfRange`] if the value exceeds the configured range.
    pub fn encode(&self, value: f64) -> Result<u32, BusError> {
        if value > self.range || value < -self.range {
            return Err(BusError::OutOfRange);
        }

        let negative = value < 0.0;
        let abs_value = if negative { -value } else { value };

        // Scale to integer magnitude
        let magnitude = (abs_value / self.resolution) as u32;
        let magnitude = if magnitude > BNR_MAX_MAGNITUDE {
            BNR_MAX_MAGNITUDE
        } else {
            magnitude
        };

        let mut data: u32 = magnitude & BNR_MAX_MAGNITUDE;
        if negative {
            // Two's complement: invert and add 1, masked to 18 bits
            data = ((!data) + 1) & BNR_MAX_MAGNITUDE;
            // Set sign bit (bit 18 of the 19-bit field)
            data |= 1 << BNR_MAGNITUDE_BITS;
        }

        Ok(data)
    }

    /// Decode a 19-bit BNR data field to a floating-point value.
    pub fn decode(&self, data: u32) -> f64 {
        let sign_bit = (data >> BNR_MAGNITUDE_BITS) & 1;
        let magnitude_field = data & BNR_MAX_MAGNITUDE;

        if sign_bit == 0 {
            // Positive
            magnitude_field as f64 * self.resolution
        } else {
            // Negative: two's complement
            let magnitude = ((!magnitude_field) + 1) & BNR_MAX_MAGNITUDE;
            -(magnitude as f64 * self.resolution)
        }
    }

    /// Encode a BNR value into a complete ARINC 429 word.
    pub fn encode_word(
        &self,
        label: Label,
        sdi: Sdi,
        value: f64,
        ssm: Ssm,
    ) -> Result<u32, BusError> {
        let data = self.encode(value)?;
        let word = Arinc429Word {
            label,
            sdi,
            data,
            ssm,
        };
        Ok(word.encode())
    }

    /// Decode a BNR value from a complete ARINC 429 word.
    ///
    /// Returns the decoded engineering-unit value and the SSM status.
    pub fn decode_word(&self, raw: u32) -> Result<(f64, Ssm), BusError> {
        let word = Arinc429Word::decode(raw)?;
        let value = self.decode(word.data);
        Ok((value, word.ssm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_bnr_codec_new() {
        let codec = BnrCodec::new(180.0);
        assert_eq!(codec.range(), 180.0);
        let expected_res = 180.0 / 262144.0;
        assert!(approx_eq(codec.resolution(), expected_res, 1e-10));
    }

    #[test]
    fn test_bnr_encode_zero() {
        let codec = BnrCodec::new(180.0);
        let data = codec.encode(0.0).unwrap();
        assert_eq!(data, 0);
    }

    #[test]
    fn test_bnr_encode_positive() {
        let codec = BnrCodec::new(180.0);
        let data = codec.encode(90.0).unwrap();
        // 90.0 / (180.0 / 262144) = 131072 = 0x20000
        // Sign bit = 0, magnitude = 131072
        assert_eq!(data & (1 << 18), 0); // positive
        let decoded = codec.decode(data);
        assert!(approx_eq(decoded, 90.0, codec.resolution()));
    }

    #[test]
    fn test_bnr_encode_negative() {
        let codec = BnrCodec::new(180.0);
        let data = codec.encode(-45.0).unwrap();
        // Sign bit should be set
        assert_ne!(data & (1 << 18), 0);
        let decoded = codec.decode(data);
        assert!(approx_eq(decoded, -45.0, codec.resolution()));
    }

    #[test]
    fn test_bnr_roundtrip_various() {
        let codec = BnrCodec::new(180.0);
        let values = [0.0, 1.0, -1.0, 45.0, -45.0, 90.0, -90.0, 179.999];
        for &val in &values {
            let data = codec.encode(val).unwrap();
            let decoded = codec.decode(data);
            assert!(
                approx_eq(decoded, val, codec.resolution()),
                "BNR roundtrip failed for {}: got {}",
                val,
                decoded,
            );
        }
    }

    #[test]
    fn test_bnr_out_of_range_positive() {
        let codec = BnrCodec::new(180.0);
        assert_eq!(codec.encode(180.1), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_bnr_out_of_range_negative() {
        let codec = BnrCodec::new(180.0);
        assert_eq!(codec.encode(-180.1), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_bnr_boundary_positive() {
        let codec = BnrCodec::new(180.0);
        // Exactly at range should succeed
        let data = codec.encode(180.0).unwrap();
        let decoded = codec.decode(data);
        assert!(approx_eq(decoded, 180.0, codec.resolution() * 2.0));
    }

    #[test]
    fn test_bnr_boundary_negative() {
        let codec = BnrCodec::new(180.0);
        let data = codec.encode(-180.0).unwrap();
        let decoded = codec.decode(data);
        assert!(approx_eq(decoded, -180.0, codec.resolution() * 2.0));
    }

    #[test]
    fn test_bnr_small_range() {
        let codec = BnrCodec::new(1.0);
        let data = codec.encode(0.5).unwrap();
        let decoded = codec.decode(data);
        assert!(approx_eq(decoded, 0.5, codec.resolution()));
    }

    #[test]
    fn test_bnr_encode_word_roundtrip() {
        let codec = BnrCodec::new(180.0);
        let label = Label::new(0o310);
        let sdi = Sdi::Zero;
        let ssm = Ssm::NormalOp;
        let value = 45.5;

        let raw = codec.encode_word(label, sdi, value, ssm).unwrap();
        let (decoded_val, decoded_ssm) = codec.decode_word(raw).unwrap();

        assert!(approx_eq(decoded_val, value, codec.resolution()));
        assert_eq!(decoded_ssm, Ssm::NormalOp);
    }

    #[test]
    fn test_bnr_decode_word_parity_error() {
        let codec = BnrCodec::new(180.0);
        let raw = codec
            .encode_word(Label::new(0), Sdi::Zero, 0.0, Ssm::NormalOp)
            .unwrap();
        let corrupted = raw ^ 1; // flip a bit
        assert_eq!(codec.decode_word(corrupted), Err(BusError::ParityError));
    }

    #[test]
    fn test_bnr_decode_data_19bit_range() {
        let codec = BnrCodec::new(180.0);
        // Max positive 19-bit BNR: sign=0, magnitude=0x3FFFF
        let data = 0x3FFFF; // all magnitude bits set, sign=0
        let decoded = codec.decode(data);
        assert!(decoded > 0.0);

        // Max negative 19-bit BNR: sign=1, magnitude=two's complement
        let data = 0x7FFFF; // sign=1, all magnitude bits set -> -resolution
        let decoded = codec.decode(data);
        assert!(decoded < 0.0);
    }
}
