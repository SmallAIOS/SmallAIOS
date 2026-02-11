// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 discrete data words.
//!
//! Discrete words use individual bits in the 19-bit data field (bits 29:11)
//! to represent independent boolean states. Each bit position maps to a
//! specific discrete parameter defined by the label.
//!
//! The SSM field indicates the overall validity:
//! - Normal Operation: all discrete bits are valid
//! - Failure Warning: data may be unreliable
//! - No Computed Data: data is not available

use crate::BusError;
use super::word::{Arinc429Word, Label, Sdi, Ssm};

/// Maximum number of discrete bits in the 19-bit data field.
pub const MAX_DISCRETE_BITS: u8 = 19;

/// Discrete word encoder/decoder.
///
/// Packs and extracts individual boolean bits from the 19-bit data field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscreteWord {
    /// The 19-bit discrete data (bit 0 = ARINC bit 11, bit 18 = ARINC bit 29).
    bits: u32,
}

impl DiscreteWord {
    /// Create a new discrete word with all bits cleared.
    pub fn new() -> Self {
        DiscreteWord { bits: 0 }
    }

    /// Create a discrete word from a raw 19-bit value.
    pub fn from_raw(bits: u32) -> Self {
        DiscreteWord {
            bits: bits & 0x7FFFF,
        }
    }

    /// Returns the raw 19-bit data value.
    pub fn raw(&self) -> u32 {
        self.bits
    }

    /// Get the state of a specific bit position (0-18).
    ///
    /// Returns [`BusError::OutOfRange`] if `bit_pos` >= 19.
    pub fn get_bit(&self, bit_pos: u8) -> Result<bool, BusError> {
        if bit_pos >= MAX_DISCRETE_BITS {
            return Err(BusError::OutOfRange);
        }
        Ok((self.bits >> bit_pos) & 1 == 1)
    }

    /// Set a specific bit position (0-18) to the given state.
    ///
    /// Returns [`BusError::OutOfRange`] if `bit_pos` >= 19.
    pub fn set_bit(&mut self, bit_pos: u8, state: bool) -> Result<(), BusError> {
        if bit_pos >= MAX_DISCRETE_BITS {
            return Err(BusError::OutOfRange);
        }
        if state {
            self.bits |= 1 << bit_pos;
        } else {
            self.bits &= !(1 << bit_pos);
        }
        Ok(())
    }

    /// Set multiple bits from a slice of (bit_pos, state) pairs.
    pub fn set_bits(&mut self, bits: &[(u8, bool)]) -> Result<(), BusError> {
        for &(pos, state) in bits {
            self.set_bit(pos, state)?;
        }
        Ok(())
    }

    /// Clear all bits to zero.
    pub fn clear(&mut self) {
        self.bits = 0;
    }

    /// Encode this discrete word into a complete ARINC 429 word.
    pub fn encode_word(&self, label: Label, sdi: Sdi, ssm: Ssm) -> u32 {
        let word = Arinc429Word {
            label,
            sdi,
            data: self.bits,
            ssm,
        };
        word.encode()
    }

    /// Decode a discrete word from a complete ARINC 429 word.
    ///
    /// Returns the discrete word and SSM status.
    pub fn decode_word(raw: u32) -> Result<(Self, Ssm), BusError> {
        let word = Arinc429Word::decode(raw)?;
        Ok((DiscreteWord::from_raw(word.data), word.ssm))
    }
}

impl Default for DiscreteWord {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_all_clear() {
        let dw = DiscreteWord::new();
        assert_eq!(dw.raw(), 0);
        for i in 0..19 {
            assert!(!dw.get_bit(i).unwrap());
        }
    }

    #[test]
    fn test_from_raw() {
        let dw = DiscreteWord::from_raw(0x55555);
        assert_eq!(dw.raw(), 0x55555);
    }

    #[test]
    fn test_from_raw_truncates_to_19_bits() {
        let dw = DiscreteWord::from_raw(0xFFFFF);
        assert_eq!(dw.raw(), 0x7FFFF);
    }

    #[test]
    fn test_set_and_get_bit() {
        let mut dw = DiscreteWord::new();
        dw.set_bit(0, true).unwrap();
        dw.set_bit(5, true).unwrap();
        dw.set_bit(18, true).unwrap();

        assert!(dw.get_bit(0).unwrap());
        assert!(!dw.get_bit(1).unwrap());
        assert!(dw.get_bit(5).unwrap());
        assert!(dw.get_bit(18).unwrap());
    }

    #[test]
    fn test_clear_bit() {
        let mut dw = DiscreteWord::from_raw(0x7FFFF);
        dw.set_bit(10, false).unwrap();
        assert!(!dw.get_bit(10).unwrap());
        assert!(dw.get_bit(9).unwrap());
        assert!(dw.get_bit(11).unwrap());
    }

    #[test]
    fn test_set_bit_out_of_range() {
        let mut dw = DiscreteWord::new();
        assert_eq!(dw.set_bit(19, true), Err(BusError::OutOfRange));
        assert_eq!(dw.set_bit(255, true), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_get_bit_out_of_range() {
        let dw = DiscreteWord::new();
        assert_eq!(dw.get_bit(19), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_set_bits_multiple() {
        let mut dw = DiscreteWord::new();
        dw.set_bits(&[(0, true), (3, true), (7, true), (15, true)])
            .unwrap();
        assert!(dw.get_bit(0).unwrap());
        assert!(!dw.get_bit(1).unwrap());
        assert!(dw.get_bit(3).unwrap());
        assert!(dw.get_bit(7).unwrap());
        assert!(dw.get_bit(15).unwrap());
    }

    #[test]
    fn test_set_bits_error() {
        let mut dw = DiscreteWord::new();
        assert_eq!(
            dw.set_bits(&[(0, true), (19, true)]),
            Err(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_clear() {
        let mut dw = DiscreteWord::from_raw(0x7FFFF);
        dw.clear();
        assert_eq!(dw.raw(), 0);
    }

    #[test]
    fn test_default() {
        let dw = DiscreteWord::default();
        assert_eq!(dw.raw(), 0);
    }

    // -----------------------------------------------------------------------
    // Word encode/decode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_discrete_word_roundtrip() {
        let mut dw = DiscreteWord::new();
        dw.set_bits(&[(0, true), (5, true), (10, true), (18, true)])
            .unwrap();

        let label = Label::new(0o270);
        let sdi = Sdi::One;
        let ssm = Ssm::NormalOp;

        let raw = dw.encode_word(label, sdi, ssm);
        let (decoded_dw, decoded_ssm) = DiscreteWord::decode_word(raw).unwrap();

        assert_eq!(decoded_dw, dw);
        assert_eq!(decoded_ssm, Ssm::NormalOp);
    }

    #[test]
    fn test_discrete_word_failure_warning() {
        let dw = DiscreteWord::new();
        let raw = dw.encode_word(Label::new(0o100), Sdi::Zero, Ssm::FailureWarning);
        let (_, ssm) = DiscreteWord::decode_word(raw).unwrap();
        assert_eq!(ssm, Ssm::FailureWarning);
    }

    #[test]
    fn test_discrete_word_parity_error() {
        let dw = DiscreteWord::new();
        let raw = dw.encode_word(Label::new(0o100), Sdi::Zero, Ssm::NormalOp);
        let corrupted = raw ^ 1;
        assert_eq!(
            DiscreteWord::decode_word(corrupted),
            Err(BusError::ParityError)
        );
    }

    #[test]
    fn test_discrete_all_bits_set_roundtrip() {
        let dw = DiscreteWord::from_raw(0x7FFFF);
        let raw = dw.encode_word(Label::new(0o377), Sdi::Three, Ssm::NormalOp);
        let (decoded_dw, _) = DiscreteWord::decode_word(raw).unwrap();
        assert_eq!(decoded_dw.raw(), 0x7FFFF);
    }
}
