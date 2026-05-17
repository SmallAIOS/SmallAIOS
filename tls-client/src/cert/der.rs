// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DER (Distinguished Encoding Rules) TLV decoder.
//!
//! The X.509 wire format is BER/DER. We only ever decode DER
//! (no indefinite-length BER), and only the subset of tags
//! X.509v3 actually uses. Every length field is bounded against
//! the enclosing structure BEFORE any allocation, eliminating
//! the standard X.509 parser CVE class.
//!
//! ## What we accept
//!
//! - Standard ASN.1 universal tags: INTEGER (0x02), BIT STRING
//!   (0x03), OCTET STRING (0x04), NULL (0x05), OBJECT
//!   IDENTIFIER (0x06), UTF8String (0x0c), SEQUENCE (0x30),
//!   SET (0x31), and a handful of date types
//!   (UTCTime 0x17, GeneralizedTime 0x18) + PrintableString
//!   (0x13) + IA5String (0x16) needed for DN / SAN parsing.
//! - Context-specific tags `[0]`, `[1]`, `[2]`, `[3]`
//!   (constructed and primitive) needed for X.509 extensions
//!   and the Version field.
//! - Long-form length (up to 4 bytes, encoding values ≤
//!   2^32-1). Any length byte count > 4 is rejected.
//!
//! ## What we refuse
//!
//! - Indefinite-length encoding (length byte 0x80).
//! - Multi-byte tag forms (high tag number).
//! - Length encodings that would exceed the enclosing
//!   structure or [`MAX_FIELD_LEN`].

use crate::{Result, TlsClientError};

/// Per-field cap on length decoding. Any TLV declaring a
/// length above this is rejected as malformed.
pub const MAX_FIELD_LEN: usize = 1024 * 1024;

/// ASN.1 universal tag bytes (class = 00, primitive/constructed
/// matters but is folded into the byte value below).
pub mod tag {
    pub const INTEGER: u8 = 0x02;
    pub const BIT_STRING: u8 = 0x03;
    pub const OCTET_STRING: u8 = 0x04;
    pub const NULL: u8 = 0x05;
    pub const OID: u8 = 0x06;
    pub const UTF8_STRING: u8 = 0x0c;
    pub const PRINTABLE_STRING: u8 = 0x13;
    pub const IA5_STRING: u8 = 0x16;
    pub const UTC_TIME: u8 = 0x17;
    pub const GENERALIZED_TIME: u8 = 0x18;
    pub const SEQUENCE: u8 = 0x30;
    pub const SET: u8 = 0x31;
}

/// A decoded TLV (tag-length-value) header pointing at the
/// raw value bytes within an input slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tlv<'a> {
    pub tag: u8,
    pub value: &'a [u8],
}

impl<'a> Tlv<'a> {
    /// True iff the tag matches the expected universal tag
    /// (or context-specific tag the caller supplied).
    pub fn is(&self, tag: u8) -> bool {
        self.tag == tag
    }
}

/// Parser cursor over a DER input slice.
#[derive(Debug, Clone, Copy)]
pub struct Reader<'a> {
    input: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self { input }
    }

    pub fn is_empty(&self) -> bool {
        self.input.is_empty()
    }

    pub fn remaining(&self) -> &'a [u8] {
        self.input
    }

    /// Decode the next TLV. Returns the tag + a slice over the
    /// value bytes, and advances the cursor past the whole
    /// TLV. Refuses indefinite-length and multi-byte tag forms.
    pub fn next_tlv(&mut self) -> Result<Tlv<'a>> {
        if self.input.is_empty() {
            return Err(TlsClientError::BadCertificate);
        }
        let tag = self.input[0];
        // High tag number form (multi-byte tag): low 5 bits of
        // the first octet are all 1.
        if tag & 0x1f == 0x1f {
            return Err(TlsClientError::BadCertificate);
        }
        if self.input.len() < 2 {
            return Err(TlsClientError::BadCertificate);
        }
        let length_byte = self.input[1];
        let (length, length_size) = if length_byte & 0x80 == 0 {
            (length_byte as usize, 1)
        } else {
            // Long-form length: low 7 bits give the number of
            // length octets that follow.
            let n = (length_byte & 0x7f) as usize;
            if n == 0 {
                // Indefinite-length encoding — not allowed in DER.
                return Err(TlsClientError::BadCertificate);
            }
            if n > 4 {
                // We accept at most 4 length octets (values up
                // to 2^32-1); anything larger is malformed.
                return Err(TlsClientError::BadCertificate);
            }
            if self.input.len() < 2 + n {
                return Err(TlsClientError::BadCertificate);
            }
            let mut value: u64 = 0;
            for i in 0..n {
                value = (value << 8) | (self.input[2 + i] as u64);
            }
            // Reject obvious overflow / non-minimal long forms.
            if value < 128 {
                return Err(TlsClientError::BadCertificate);
            }
            if value > MAX_FIELD_LEN as u64 {
                return Err(TlsClientError::BadCertificate);
            }
            (value as usize, 1 + n)
        };
        let header_size = 1 + length_size;
        if length > MAX_FIELD_LEN {
            return Err(TlsClientError::BadCertificate);
        }
        if self.input.len() < header_size + length {
            return Err(TlsClientError::BadCertificate);
        }
        let value = &self.input[header_size..header_size + length];
        self.input = &self.input[header_size + length..];
        Ok(Tlv { tag, value })
    }

    /// Decode the next TLV and assert it matches `expected_tag`.
    pub fn expect_tlv(&mut self, expected_tag: u8) -> Result<&'a [u8]> {
        let tlv = self.next_tlv()?;
        if tlv.tag != expected_tag {
            return Err(TlsClientError::BadCertificate);
        }
        Ok(tlv.value)
    }

    /// Convenience: expect a SEQUENCE and return a sub-reader
    /// over its contents.
    pub fn expect_sequence(&mut self) -> Result<Reader<'a>> {
        Ok(Reader::new(self.expect_tlv(tag::SEQUENCE)?))
    }
}

/// Decode an INTEGER as a non-negative `u64`. Negative integers
/// (high bit set in the first content byte) are rejected. Empty
/// content is also rejected (DER mandates at least one byte).
pub fn read_uint(bytes: &[u8]) -> Result<u64> {
    if bytes.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    if bytes[0] & 0x80 != 0 {
        return Err(TlsClientError::BadCertificate);
    }
    if bytes.len() > 9 {
        // 9 because DER allows a leading 0x00 to mark
        // non-negativity on 8-byte values.
        return Err(TlsClientError::BadCertificate);
    }
    let mut value: u64 = 0;
    for &b in bytes {
        value = (value << 8) | (b as u64);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn short_form_length_round_trip() {
        // SEQUENCE { OCTET STRING(5) "hello" }
        let buf = [0x30, 7, 0x04, 5, b'h', b'e', b'l', b'l', b'o'];
        let mut r = Reader::new(&buf);
        let seq = r.expect_sequence().unwrap();
        let mut inner = seq;
        let os = inner.expect_tlv(tag::OCTET_STRING).unwrap();
        assert_eq!(os, b"hello");
        assert!(inner.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn long_form_length_round_trip() {
        // OCTET STRING with 200 bytes — needs 2-byte length encoding.
        let body = [0u8; 200];
        let mut buf = vec![0x04, 0x81, 200];
        buf.extend_from_slice(&body);
        let mut r = Reader::new(&buf);
        let os = r.expect_tlv(tag::OCTET_STRING).unwrap();
        assert_eq!(os.len(), 200);
    }

    #[test]
    fn indefinite_length_rejected() {
        // Length byte 0x80 = indefinite-length BER, not DER.
        let buf = [0x30, 0x80];
        let mut r = Reader::new(&buf);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn long_form_length_oversized_rejected() {
        // 5 length octets — refuse.
        let buf = [0x30, 0x85, 0, 0, 0, 0, 0];
        let mut r = Reader::new(&buf);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn long_form_non_minimal_rejected() {
        // 0x81 0x05 — long-form for a value < 128 is non-minimal.
        let buf = [0x04, 0x81, 0x05, 1, 2, 3, 4, 5];
        let mut r = Reader::new(&buf);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn length_exceeds_input_rejected() {
        let buf = [0x04, 10, 1, 2, 3]; // declares 10 bytes but only 3 follow
        let mut r = Reader::new(&buf);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn multi_byte_tag_rejected() {
        // 0x1f indicates a multi-byte tag form.
        let buf = [0x1f, 0x05];
        let mut r = Reader::new(&buf);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn empty_input_rejected() {
        let mut r = Reader::new(&[]);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn expect_tlv_mismatch_rejected() {
        let buf = [0x04, 1, 0xaa]; // OCTET STRING
        let mut r = Reader::new(&buf);
        assert_eq!(
            r.expect_tlv(tag::SEQUENCE).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn read_uint_small() {
        assert_eq!(read_uint(&[42]).unwrap(), 42);
        assert_eq!(read_uint(&[0x00, 0x80]).unwrap(), 128);
        assert_eq!(read_uint(&[0x01, 0x00]).unwrap(), 256);
    }

    #[test]
    fn read_uint_rejects_negative() {
        // High bit set on first byte ⇒ negative in two's complement.
        assert_eq!(
            read_uint(&[0x80]).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn read_uint_rejects_empty() {
        assert_eq!(read_uint(&[]).unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn read_uint_rejects_oversized() {
        // 10 bytes wouldn't fit in u64.
        let buf = [0u8; 10];
        assert_eq!(read_uint(&buf).unwrap_err(), TlsClientError::BadCertificate);
    }

    #[test]
    fn declared_length_above_cap_rejected() {
        // Encode 2 MiB declared length (above 1 MiB cap).
        let buf = [0x04, 0x84, 0x00, 0x20, 0x00, 0x00];
        let mut r = Reader::new(&buf);
        assert_eq!(r.next_tlv().unwrap_err(), TlsClientError::BadCertificate);
    }
}
