// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HPACK static-table-only encoder/decoder (RFC 7541).
//!
//! We deliberately implement **only** the subset of HPACK that
//! covers gRPC's needs:
//!
//! - Indexed header field with index in the static table (1–61).
//! - Literal header field with name index in the static table,
//!   value as a literal string (no Huffman, no indexing).
//! - Literal header field with both name and value as literal
//!   strings (no Huffman, no indexing).
//!
//! Any reference into the dynamic table — index ≥ 62 in an
//! indexed-or-literal-with-name field — is rejected with
//! [`Http2Error::HpackDynamicReference`]. This eliminates the
//! "HPACK Bomb" class of decompression-amplification CVEs at the
//! protocol-design level.
//!
//! Huffman input is also rejected (the H bit on a string-length
//! integer prefix). The encoder never emits Huffman.

use super::{Http2Error, Result};
use alloc::string::String;
use alloc::vec::Vec;

/// RFC 7541 Appendix A static table (1-indexed). Entries 1..=61.
/// `(name, value_or_empty)`.
pub const STATIC_TABLE: &[(&str, &str)] = &[
    (":authority", ""),                   // 1
    (":method", "GET"),                   // 2
    (":method", "POST"),                  // 3
    (":path", "/"),                       // 4
    (":path", "/index.html"),             // 5
    (":scheme", "http"),                  // 6
    (":scheme", "https"),                 // 7
    (":status", "200"),                   // 8
    (":status", "204"),                   // 9
    (":status", "206"),                   // 10
    (":status", "304"),                   // 11
    (":status", "400"),                   // 12
    (":status", "404"),                   // 13
    (":status", "500"),                   // 14
    ("accept-charset", ""),               // 15
    ("accept-encoding", "gzip, deflate"), // 16
    ("accept-language", ""),              // 17
    ("accept-ranges", ""),                // 18
    ("accept", ""),                       // 19
    ("access-control-allow-origin", ""),  // 20
    ("age", ""),                          // 21
    ("allow", ""),                        // 22
    ("authorization", ""),                // 23
    ("cache-control", ""),                // 24
    ("content-disposition", ""),          // 25
    ("content-encoding", ""),             // 26
    ("content-language", ""),             // 27
    ("content-length", ""),               // 28
    ("content-location", ""),             // 29
    ("content-range", ""),                // 30
    ("content-type", ""),                 // 31
    ("cookie", ""),                       // 32
    ("date", ""),                         // 33
    ("etag", ""),                         // 34
    ("expect", ""),                       // 35
    ("expires", ""),                      // 36
    ("from", ""),                         // 37
    ("host", ""),                         // 38
    ("if-match", ""),                     // 39
    ("if-modified-since", ""),            // 40
    ("if-none-match", ""),                // 41
    ("if-range", ""),                     // 42
    ("if-unmodified-since", ""),          // 43
    ("last-modified", ""),                // 44
    ("link", ""),                         // 45
    ("location", ""),                     // 46
    ("max-forwards", ""),                 // 47
    ("proxy-authenticate", ""),           // 48
    ("proxy-authorization", ""),          // 49
    ("range", ""),                        // 50
    ("referer", ""),                      // 51
    ("refresh", ""),                      // 52
    ("retry-after", ""),                  // 53
    ("server", ""),                       // 54
    ("set-cookie", ""),                   // 55
    ("strict-transport-security", ""),    // 56
    ("transfer-encoding", ""),            // 57
    ("user-agent", ""),                   // 58
    ("vary", ""),                         // 59
    ("via", ""),                          // 60
    ("www-authenticate", ""),             // 61
];

/// Cap on a single literal string length (name or value) we will
/// accept. 8 KiB is comfortably above any header gRPC requires.
const MAX_LITERAL_LEN: usize = 8 * 1024;

/// One decoded header (lower-cased name, raw value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub name: String,
    pub value: String,
}

/// Decode an integer with N-bit prefix per RFC 7541 § 5.1.
fn decode_integer(input: &[u8], prefix_bits: u8, cursor: &mut usize) -> Result<u64> {
    if *cursor >= input.len() {
        return Err(Http2Error::HpackBadEncoding);
    }
    let mask = (1u8 << prefix_bits) - 1;
    let first = input[*cursor] & mask;
    *cursor += 1;
    if first < mask {
        return Ok(first as u64);
    }
    let mut value: u64 = first as u64;
    let mut m = 0u32;
    loop {
        if *cursor >= input.len() {
            return Err(Http2Error::HpackBadEncoding);
        }
        let b = input[*cursor];
        *cursor += 1;
        value = value
            .checked_add(((b & 0x7f) as u64) << m)
            .ok_or(Http2Error::HpackBadEncoding)?;
        if b & 0x80 == 0 {
            return Ok(value);
        }
        m = m.checked_add(7).ok_or(Http2Error::HpackBadEncoding)?;
        if m >= 64 {
            return Err(Http2Error::HpackBadEncoding);
        }
    }
}

/// Encode an integer with N-bit prefix into `out`. The caller is
/// responsible for OR-ing the high bits of `out`'s last byte
/// with any flag bits before this call returns.
fn encode_integer(value: u64, prefix_bits: u8, out: &mut Vec<u8>) {
    let mask = ((1u64 << prefix_bits) - 1) as u8;
    if value < mask as u64 {
        out.push(value as u8);
        return;
    }
    out.push(mask);
    let mut v = value - mask as u64;
    while v >= 0x80 {
        out.push(((v & 0x7f) as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decode a literal string: 1-byte H+length prefix, then bytes.
/// Huffman is rejected.
fn decode_literal_string(input: &[u8], cursor: &mut usize) -> Result<String> {
    if *cursor >= input.len() {
        return Err(Http2Error::HpackBadEncoding);
    }
    let huffman = (input[*cursor] & 0x80) != 0;
    if huffman {
        return Err(Http2Error::HpackBadEncoding);
    }
    let len = decode_integer(input, 7, cursor)? as usize;
    if len > MAX_LITERAL_LEN {
        return Err(Http2Error::HpackBadEncoding);
    }
    if *cursor + len > input.len() {
        return Err(Http2Error::HpackBadEncoding);
    }
    let s = core::str::from_utf8(&input[*cursor..*cursor + len])
        .map_err(|_| Http2Error::HpackBadEncoding)?;
    *cursor += len;
    Ok(s.into())
}

/// Encode a literal string (no Huffman) into `out`.
fn encode_literal_string(s: &str, out: &mut Vec<u8>) {
    encode_integer(s.len() as u64, 7, out);
    out.extend_from_slice(s.as_bytes());
}

/// Decode a single HPACK header block into a vector of [`Header`]s.
///
/// Every reference into the dynamic table is rejected; every
/// indexing-style instruction (literal-with-incremental-indexing)
/// is converted to a literal-without-indexing on the decode side
/// (we do not maintain a dynamic table).
pub fn decode_block(input: &[u8]) -> Result<Vec<Header>> {
    let mut out = Vec::new();
    let mut cur = 0usize;
    while cur < input.len() {
        let b = input[cur];
        if b & 0x80 != 0 {
            // Indexed Header Field — RFC 7541 § 6.1.
            let idx = decode_integer(input, 7, &mut cur)? as usize;
            if idx == 0 || idx > STATIC_TABLE.len() {
                return Err(Http2Error::HpackDynamicReference);
            }
            let (n, v) = STATIC_TABLE[idx - 1];
            out.push(Header {
                name: n.into(),
                value: v.into(),
            });
        } else if b & 0xc0 == 0x40 {
            // Literal Header Field with Incremental Indexing — § 6.2.1.
            // We accept the wire form but never store in a dynamic table.
            let idx = decode_integer(input, 6, &mut cur)? as usize;
            let name = if idx == 0 {
                decode_literal_string(input, &mut cur)?
            } else if idx <= STATIC_TABLE.len() {
                STATIC_TABLE[idx - 1].0.into()
            } else {
                return Err(Http2Error::HpackDynamicReference);
            };
            let value = decode_literal_string(input, &mut cur)?;
            out.push(Header { name, value });
        } else if b & 0xe0 == 0x20 {
            // Dynamic Table Size Update — § 6.3. We do not maintain a
            // dynamic table; accept and ignore (size MUST be ≤ the value
            // we advertised in SETTINGS, which is 0).
            let new_max = decode_integer(input, 5, &mut cur)?;
            if new_max != 0 {
                return Err(Http2Error::HpackDynamicReference);
            }
        } else {
            // Either Without Indexing (0x00 prefix) or Never Indexed (0x10).
            // Both have a 4-bit prefix and behave identically for us.
            let idx = decode_integer(input, 4, &mut cur)? as usize;
            let name = if idx == 0 {
                decode_literal_string(input, &mut cur)?
            } else if idx <= STATIC_TABLE.len() {
                STATIC_TABLE[idx - 1].0.into()
            } else {
                return Err(Http2Error::HpackDynamicReference);
            };
            let value = decode_literal_string(input, &mut cur)?;
            out.push(Header { name, value });
        }
    }
    Ok(out)
}

/// Encode a list of headers into an HPACK block, using static-table
/// indices when an exact name+value match exists, otherwise emitting
/// a literal-without-indexing reference (with name index when
/// available).
pub fn encode_block(headers: &[Header]) -> Vec<u8> {
    let mut out = Vec::with_capacity(headers.len() * 32);
    for h in headers {
        if let Some(i) = static_full_match(&h.name, &h.value) {
            // Indexed Header Field: 1xxx_xxxx with 7-bit prefix.
            let mark = out.len();
            encode_integer(i as u64, 7, &mut out);
            out[mark] |= 0x80;
            continue;
        }
        let name_idx = static_name_match(&h.name);
        // Literal Header Field without Indexing — Indexed Name (4-bit prefix, top bits 0000).
        let mark = out.len();
        encode_integer(name_idx as u64, 4, &mut out);
        // Top 4 bits are already zero (literal-without-indexing pattern).
        if name_idx == 0 {
            encode_literal_string(&h.name, &mut out);
        }
        encode_literal_string(&h.value, &mut out);
        let _ = mark; // silence unused on no-debug builds.
    }
    out
}

fn static_full_match(name: &str, value: &str) -> Option<usize> {
    for (i, (n, v)) in STATIC_TABLE.iter().enumerate() {
        if *n == name && *v == value {
            return Some(i + 1);
        }
    }
    None
}

fn static_name_match(name: &str) -> usize {
    for (i, (n, _)) in STATIC_TABLE.iter().enumerate() {
        if *n == name {
            return i + 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(n: &str, v: &str) -> Header {
        Header {
            name: n.into(),
            value: v.into(),
        }
    }

    #[test]
    fn integer_roundtrip_small() {
        let mut out = Vec::new();
        encode_integer(10, 5, &mut out);
        let mut cur = 0;
        assert_eq!(decode_integer(&out, 5, &mut cur).unwrap(), 10);
    }

    #[test]
    fn integer_roundtrip_at_prefix_max() {
        // 5-bit prefix, value 31 — needs continuation.
        let mut out = Vec::new();
        encode_integer(31, 5, &mut out);
        let mut cur = 0;
        assert_eq!(decode_integer(&out, 5, &mut cur).unwrap(), 31);
    }

    #[test]
    fn integer_roundtrip_large() {
        let mut out = Vec::new();
        encode_integer(1337, 5, &mut out);
        let mut cur = 0;
        assert_eq!(decode_integer(&out, 5, &mut cur).unwrap(), 1337);
    }

    #[test]
    fn integer_overflow_rejected() {
        // 12 continuation bytes with high bit set ⇒ shift > 64 ⇒ reject.
        let bad = [
            0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        let mut cur = 0;
        assert_eq!(
            decode_integer(&bad, 5, &mut cur).unwrap_err(),
            Http2Error::HpackBadEncoding
        );
    }

    #[test]
    fn indexed_method_post() {
        // 0x83 = 1000_0011 = indexed table entry 3 (:method POST).
        let block = [0x83];
        let hdrs = decode_block(&block).unwrap();
        assert_eq!(hdrs, alloc::vec![h(":method", "POST")]);
    }

    #[test]
    fn dynamic_index_rejected() {
        // 0xbe = 1011_1110 = indexed entry 62 (first dynamic slot).
        let block = [0xbe];
        assert_eq!(
            decode_block(&block).unwrap_err(),
            Http2Error::HpackDynamicReference
        );
    }

    #[test]
    fn huffman_input_rejected() {
        // Literal-without-indexing, name index 0, then H=1 length prefix.
        let block = [0x00, 0x82, 0xaa, 0xbb];
        assert_eq!(
            decode_block(&block).unwrap_err(),
            Http2Error::HpackBadEncoding
        );
    }

    #[test]
    fn dynamic_table_size_update_nonzero_rejected() {
        // 0010_xxxx prefix; nonzero size update.
        let block = [0x3a]; // table size 26.
        assert_eq!(
            decode_block(&block).unwrap_err(),
            Http2Error::HpackDynamicReference
        );
    }

    #[test]
    fn encode_decode_roundtrip_grpc_request_headers() {
        let headers = alloc::vec![
            h(":method", "POST"),
            h(":scheme", "https"),
            h(":authority", "immudb.example.com:3322"),
            h(":path", "/immudb.schema.ImmuService/VerifiableSet"),
            h("content-type", "application/grpc+proto"),
            h("te", "trailers"),
            h("authorization", "Bearer s3cr3t"),
            h("user-agent", "smallaios-audit-export/0.2.1"),
        ];
        let block = encode_block(&headers);
        let decoded = decode_block(&block).unwrap();
        assert_eq!(decoded, headers);
    }
}
