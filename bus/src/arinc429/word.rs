// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 32-bit word encode/decode.
//!
//! # Bit layout (ARINC 429 1-based numbering, bit 1 = LSB)
//!
//! ```text
//! | Parity(32) | SSM(31:30) | Data(29:11) | SDI(10:9) | Label(8:1) |
//! ```
//!
//! Labels are encoded with reversed bit order per ARINC 429 convention.
//! For example, octal label 310 (binary 11_001_000) is stored as bits
//! 8:1 = 00_010_011 in the word.

use crate::BusError;

/// ARINC 429 label (8-bit, octal, bit-reversed in the word).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(u8);

impl Label {
    /// Create a label from an octal value (0-377 octal = 0-255 decimal).
    pub fn new(value: u8) -> Self {
        Label(value)
    }

    /// Create a label from an octal literal.
    ///
    /// Returns [`BusError::InvalidLabel`] if the value exceeds 0o377.
    pub fn from_octal(octal: u16) -> Result<Self, BusError> {
        if octal > 0o377 {
            return Err(BusError::InvalidLabel);
        }
        Ok(Label(octal as u8))
    }

    /// Returns the label value as a decimal u8.
    pub fn value(self) -> u8 {
        self.0
    }

    /// Reverse the bits of the label for wire encoding.
    ///
    /// ARINC 429 transmits the label with bit 1 (LSB of octal) first,
    /// which means the label appears bit-reversed in the 32-bit word.
    pub fn to_wire(self) -> u8 {
        self.0.reverse_bits()
    }

    /// Decode a label from wire (bit-reversed) representation.
    pub fn from_wire(wire: u8) -> Self {
        Label(wire.reverse_bits())
    }
}

/// Source/Destination Identifier (2 bits, bits 10:9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sdi {
    /// SDI = 00
    Zero,
    /// SDI = 01
    One,
    /// SDI = 10
    Two,
    /// SDI = 11
    Three,
}

impl Sdi {
    /// Create an SDI from a 2-bit raw value.
    pub fn from_raw(val: u8) -> Result<Self, BusError> {
        match val {
            0 => Ok(Sdi::Zero),
            1 => Ok(Sdi::One),
            2 => Ok(Sdi::Two),
            3 => Ok(Sdi::Three),
            _ => Err(BusError::InvalidHeader),
        }
    }

    /// Returns the raw 2-bit value.
    pub fn raw(self) -> u8 {
        match self {
            Sdi::Zero => 0,
            Sdi::One => 1,
            Sdi::Two => 2,
            Sdi::Three => 3,
        }
    }
}

/// Sign/Status Matrix (2 bits, bits 31:30).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ssm {
    /// Normal operation / positive value.
    NormalOp,
    /// No computed data.
    NoComputedData,
    /// Functional test.
    FunctionalTest,
    /// Failure warning.
    FailureWarning,
}

impl Ssm {
    /// Create an SSM from BNR-convention raw bits.
    ///
    /// BNR convention (most common):
    /// - 00 = Failure Warning
    /// - 01 = No Computed Data
    /// - 10 = Functional Test
    /// - 11 = Normal Operation
    pub fn from_bnr_raw(val: u8) -> Result<Self, BusError> {
        match val {
            0b00 => Ok(Ssm::FailureWarning),
            0b01 => Ok(Ssm::NoComputedData),
            0b10 => Ok(Ssm::FunctionalTest),
            0b11 => Ok(Ssm::NormalOp),
            _ => Err(BusError::InvalidHeader),
        }
    }

    /// Returns the BNR-convention raw 2-bit value.
    pub fn to_bnr_raw(self) -> u8 {
        match self {
            Ssm::FailureWarning => 0b00,
            Ssm::NoComputedData => 0b01,
            Ssm::FunctionalTest => 0b10,
            Ssm::NormalOp => 0b11,
        }
    }

    /// Create an SSM from BCD-convention raw bits.
    ///
    /// BCD convention:
    /// - 00 = Plus / North / East / Right / To / Above
    /// - 01 = No Computed Data
    /// - 10 = Functional Test
    /// - 11 = Minus / South / West / Left / From / Below
    pub fn from_bcd_raw(val: u8) -> Result<Self, BusError> {
        match val {
            0b00 => Ok(Ssm::NormalOp),
            0b01 => Ok(Ssm::NoComputedData),
            0b10 => Ok(Ssm::FunctionalTest),
            0b11 => Ok(Ssm::FailureWarning),
            _ => Err(BusError::InvalidHeader),
        }
    }

    /// Returns the BCD-convention raw 2-bit value.
    pub fn to_bcd_raw(self) -> u8 {
        match self {
            Ssm::NormalOp => 0b00,
            Ssm::NoComputedData => 0b01,
            Ssm::FunctionalTest => 0b10,
            Ssm::FailureWarning => 0b11,
        }
    }
}

/// A decoded ARINC 429 32-bit word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arinc429Word {
    /// Label (8-bit, octal).
    pub label: Label,
    /// Source/Destination Identifier (2 bits).
    pub sdi: Sdi,
    /// Data field (19 bits, bits 29:11).
    pub data: u32,
    /// Sign/Status Matrix (2 bits).
    pub ssm: Ssm,
}

impl Arinc429Word {
    /// Data field mask (19 bits).
    const DATA_MASK: u32 = 0x7FFFF; // 19 bits

    /// Encode this word into a 32-bit value with correct parity.
    ///
    /// Layout (bit 0 = LSB):
    /// - Bits 7:0   = Label (bit-reversed)
    /// - Bits 9:8   = SDI
    /// - Bits 28:10 = Data (19 bits)
    /// - Bits 30:29 = SSM (BNR convention)
    /// - Bit 31     = Parity (odd)
    pub fn encode(&self) -> u32 {
        self.encode_with_ssm_raw(self.ssm.to_bnr_raw())
    }

    /// Encode with a custom SSM raw value (for BCD which uses different convention).
    pub fn encode_with_ssm_raw(&self, ssm_raw: u8) -> u32 {
        let mut word: u32 = 0;

        // Label (bits 7:0, bit-reversed)
        word |= self.label.to_wire() as u32;

        // SDI (bits 9:8)
        word |= (self.sdi.raw() as u32) << 8;

        // Data (bits 28:10)
        word |= (self.data & Self::DATA_MASK) << 10;

        // SSM (bits 30:29)
        word |= (ssm_raw as u32 & 0x03) << 29;

        // Parity (bit 31, odd parity over bits 0-30)
        let ones = (word & 0x7FFF_FFFF).count_ones();
        if ones.is_multiple_of(2) {
            word |= 1 << 31; // set parity bit to make odd
        }

        word
    }

    /// Decode a 32-bit word, verifying odd parity.
    ///
    /// Returns [`BusError::ParityError`] if parity check fails.
    pub fn decode(word: u32) -> Result<Self, BusError> {
        // Verify odd parity over all 32 bits
        if word.count_ones().is_multiple_of(2) {
            return Err(BusError::ParityError);
        }

        let label = Label::from_wire((word & 0xFF) as u8);
        let sdi = Sdi::from_raw(((word >> 8) & 0x03) as u8)?;
        let data = (word >> 10) & Self::DATA_MASK;
        let ssm_raw = ((word >> 29) & 0x03) as u8;
        let ssm = Ssm::from_bnr_raw(ssm_raw)?;

        Ok(Arinc429Word {
            label,
            sdi,
            data,
            ssm,
        })
    }

    /// Decode a 32-bit word using BCD SSM convention, verifying odd parity.
    pub fn decode_bcd(word: u32) -> Result<Self, BusError> {
        if word.count_ones().is_multiple_of(2) {
            return Err(BusError::ParityError);
        }

        let label = Label::from_wire((word & 0xFF) as u8);
        let sdi = Sdi::from_raw(((word >> 8) & 0x03) as u8)?;
        let data = (word >> 10) & Self::DATA_MASK;
        let ssm_raw = ((word >> 29) & 0x03) as u8;
        let ssm = Ssm::from_bcd_raw(ssm_raw)?;

        Ok(Arinc429Word {
            label,
            sdi,
            data,
            ssm,
        })
    }

    /// Extract the raw 19-bit data field from a 32-bit word without
    /// full validation (no parity check).
    pub fn raw_data(word: u32) -> u32 {
        (word >> 10) & Self::DATA_MASK
    }

    /// Extract the label from a raw 32-bit word (no parity check).
    pub fn raw_label(word: u32) -> Label {
        Label::from_wire((word & 0xFF) as u8)
    }
}

/// Compute odd parity over a 32-bit value (returns true if parity is valid).
pub fn check_parity(word: u32) -> bool {
    !word.count_ones().is_multiple_of(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Label tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_new() {
        let label = Label::new(0o310);
        assert_eq!(label.value(), 0o310);
    }

    #[test]
    fn test_label_from_octal_valid() {
        let label = Label::from_octal(0o310).unwrap();
        assert_eq!(label.value(), 0o310);
    }

    #[test]
    fn test_label_from_octal_max() {
        let label = Label::from_octal(0o377).unwrap();
        assert_eq!(label.value(), 255);
    }

    #[test]
    fn test_label_from_octal_too_large() {
        assert_eq!(Label::from_octal(0o400), Err(BusError::InvalidLabel));
    }

    #[test]
    fn test_label_bit_reversal_roundtrip() {
        for val in 0..=255u8 {
            let label = Label::new(val);
            let wire = label.to_wire();
            let decoded = Label::from_wire(wire);
            assert_eq!(decoded.value(), val, "bit reversal failed for {}", val);
        }
    }

    #[test]
    fn test_label_known_reversal() {
        // Octal 310 = 0b11_001_000
        // Reversed = 0b00_010_011 = 0x13
        let label = Label::new(0o310);
        assert_eq!(label.to_wire(), 0b11_001_000u8.reverse_bits());
    }

    // -----------------------------------------------------------------------
    // SDI tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sdi_roundtrip() {
        for val in 0..=3u8 {
            let sdi = Sdi::from_raw(val).unwrap();
            assert_eq!(sdi.raw(), val);
        }
    }

    #[test]
    fn test_sdi_invalid() {
        assert_eq!(Sdi::from_raw(4), Err(BusError::InvalidHeader));
    }

    // -----------------------------------------------------------------------
    // SSM tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssm_bnr_roundtrip() {
        let ssms = [
            Ssm::FailureWarning,
            Ssm::NoComputedData,
            Ssm::FunctionalTest,
            Ssm::NormalOp,
        ];
        for ssm in ssms {
            let raw = ssm.to_bnr_raw();
            let decoded = Ssm::from_bnr_raw(raw).unwrap();
            assert_eq!(decoded, ssm);
        }
    }

    #[test]
    fn test_ssm_bcd_roundtrip() {
        let ssms = [
            Ssm::NormalOp,
            Ssm::NoComputedData,
            Ssm::FunctionalTest,
            Ssm::FailureWarning,
        ];
        for ssm in ssms {
            let raw = ssm.to_bcd_raw();
            let decoded = Ssm::from_bcd_raw(raw).unwrap();
            assert_eq!(decoded, ssm);
        }
    }

    // -----------------------------------------------------------------------
    // Word encode/decode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_has_odd_parity() {
        let word = Arinc429Word {
            label: Label::new(0o310),
            sdi: Sdi::Zero,
            data: 0x12345,
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        assert!(check_parity(encoded), "parity should be odd");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let original = Arinc429Word {
            label: Label::new(0o205),
            sdi: Sdi::One,
            data: 0x3ABCD,
            ssm: Ssm::NormalOp,
        };
        let encoded = original.encode();
        let decoded = Arinc429Word::decode(encoded).unwrap();
        assert_eq!(decoded.label, original.label);
        assert_eq!(decoded.sdi, original.sdi);
        assert_eq!(decoded.data, original.data & 0x7FFFF); // 19 bits
        assert_eq!(decoded.ssm, original.ssm);
    }

    #[test]
    fn test_encode_decode_all_ssm() {
        for ssm in [
            Ssm::FailureWarning,
            Ssm::NoComputedData,
            Ssm::FunctionalTest,
            Ssm::NormalOp,
        ] {
            let word = Arinc429Word {
                label: Label::new(0o100),
                sdi: Sdi::Zero,
                data: 0,
                ssm,
            };
            let encoded = word.encode();
            let decoded = Arinc429Word::decode(encoded).unwrap();
            assert_eq!(decoded.ssm, ssm);
        }
    }

    #[test]
    fn test_encode_decode_all_sdi() {
        for sdi in [Sdi::Zero, Sdi::One, Sdi::Two, Sdi::Three] {
            let word = Arinc429Word {
                label: Label::new(0o200),
                sdi,
                data: 0x55555,
                ssm: Ssm::NormalOp,
            };
            let encoded = word.encode();
            let decoded = Arinc429Word::decode(encoded).unwrap();
            assert_eq!(decoded.sdi, sdi);
        }
    }

    #[test]
    fn test_decode_parity_error() {
        let word = Arinc429Word {
            label: Label::new(0o310),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        // Flip a bit to break parity
        let corrupted = encoded ^ 1;
        assert_eq!(Arinc429Word::decode(corrupted), Err(BusError::ParityError));
    }

    #[test]
    fn test_encode_zero_data() {
        let word = Arinc429Word {
            label: Label::new(0),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::FailureWarning,
        };
        let encoded = word.encode();
        assert!(check_parity(encoded));
        let decoded = Arinc429Word::decode(encoded).unwrap();
        assert_eq!(decoded.data, 0);
    }

    #[test]
    fn test_encode_max_data() {
        let word = Arinc429Word {
            label: Label::new(0o377),
            sdi: Sdi::Three,
            data: 0x7FFFF, // max 19-bit value
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        assert!(check_parity(encoded));
        let decoded = Arinc429Word::decode(encoded).unwrap();
        assert_eq!(decoded.data, 0x7FFFF);
    }

    #[test]
    fn test_data_field_truncated_to_19_bits() {
        let word = Arinc429Word {
            label: Label::new(0o100),
            sdi: Sdi::Zero,
            data: 0xFFFFF, // 20 bits, should be truncated to 19
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        let decoded = Arinc429Word::decode(encoded).unwrap();
        assert_eq!(decoded.data, 0x7FFFF); // truncated to 19 bits
    }

    #[test]
    fn test_raw_data_extraction() {
        let word = Arinc429Word {
            label: Label::new(0o250),
            sdi: Sdi::Two,
            data: 0x12345,
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        assert_eq!(Arinc429Word::raw_data(encoded), 0x12345);
    }

    #[test]
    fn test_raw_label_extraction() {
        let word = Arinc429Word {
            label: Label::new(0o310),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        assert_eq!(Arinc429Word::raw_label(encoded), Label::new(0o310));
    }

    // -----------------------------------------------------------------------
    // BCD SSM convention decode
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_bcd_ssm() {
        let word = Arinc429Word {
            label: Label::new(0o100),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::NormalOp,
        };
        // Encode with BCD SSM convention
        let encoded = word.encode_with_ssm_raw(word.ssm.to_bcd_raw());
        let decoded = Arinc429Word::decode_bcd(encoded).unwrap();
        assert_eq!(decoded.ssm, Ssm::NormalOp);
    }

    // -----------------------------------------------------------------------
    // All labels roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_labels_roundtrip() {
        for val in 0..=255u8 {
            let word = Arinc429Word {
                label: Label::new(val),
                sdi: Sdi::Zero,
                data: 0,
                ssm: Ssm::NormalOp,
            };
            let encoded = word.encode();
            assert!(check_parity(encoded), "parity failed for label {}", val);
            let decoded = Arinc429Word::decode(encoded).unwrap();
            assert_eq!(
                decoded.label.value(),
                val,
                "label roundtrip failed for {}",
                val
            );
        }
    }

    // -----------------------------------------------------------------------
    // Parity function
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_parity_odd() {
        assert!(check_parity(0b001)); // 1 one -> odd
        assert!(!check_parity(0b011)); // 2 ones -> even
        assert!(check_parity(0b111)); // 3 ones -> odd
    }
}
