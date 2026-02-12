// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Property-based fuzz harness for ONNX protobuf parser (Task 13.5).
//!
//! Feeds pseudo-random and malformed inputs into the protobuf decoder
//! to verify no panics occur. Uses a simple xorshift PRNG for
//! deterministic, cargo-test-compatible testing.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use smallaios_onnx_rt::protobuf::{zigzag_decode_32, zigzag_decode_64, ProtoDecoder, WireType};

// ============================================================================
// Simple xorshift64 PRNG
// ============================================================================

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.next_byte();
        }
    }

    fn gen_bytes(&mut self, max_len: usize) -> Vec<u8> {
        let len = (self.next() as usize) % (max_len + 1);
        let mut buf = vec![0u8; len];
        self.fill_bytes(&mut buf);
        buf
    }
}

// ============================================================================
// Varint fuzz
// ============================================================================

#[test]
fn fuzz_varint_random_bytes() {
    let mut rng = Xorshift64::new(0xDEAD_BEEF_0001);

    for _ in 0..50_000 {
        let data = rng.gen_bytes(16);
        let mut dec = ProtoDecoder::new(&data);
        // Must not panic
        let _ = dec.read_varint();
    }
}

#[test]
fn fuzz_varint_all_continuation_patterns() {
    // Test all combinations of continuation bits in first 3 bytes
    for b0 in 0..=255u8 {
        for b1 in 0..=255u8 {
            let data = [b0, b1, 0x00]; // 3rd byte terminates
            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_varint();
        }
    }
}

// ============================================================================
// Tag decode fuzz
// ============================================================================

#[test]
fn fuzz_tag_decode_random() {
    let mut rng = Xorshift64::new(0xCAFE_BABE_0002);

    for _ in 0..50_000 {
        let data = rng.gen_bytes(12);
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.read_tag();
    }
}

#[test]
fn fuzz_tag_valid_field_numbers() {
    // Test tags with all valid wire types and field numbers 1..100
    for field_num in 1u32..100 {
        for wt in [0u8, 1, 2, 5] {
            // Encode as varint: (field_num << 3) | wire_type
            let tag_val = (field_num << 3) | (wt as u32);
            let mut data = Vec::new();
            let mut v = tag_val;
            loop {
                if v < 0x80 {
                    data.push(v as u8);
                    break;
                }
                data.push((v as u8 & 0x7F) | 0x80);
                v >>= 7;
            }
            let mut dec = ProtoDecoder::new(&data);
            let header = dec.read_tag().unwrap();
            assert_eq!(header.field_number, field_num);
        }
    }
}

// ============================================================================
// Length-delimited fuzz
// ============================================================================

#[test]
fn fuzz_length_delimited_random() {
    let mut rng = Xorshift64::new(0x1234_5678_0003);

    for _ in 0..20_000 {
        let data = rng.gen_bytes(64);
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.read_length_delimited();
    }
}

#[test]
fn fuzz_length_delimited_huge_lengths() {
    // Test with length prefixes that exceed buffer
    let lengths: &[u64] = &[0, 1, 255, 256, 65535, 65536, u32::MAX as u64, u64::MAX];

    for &len in lengths {
        // Encode length as varint
        let mut data = Vec::new();
        let mut v = len;
        loop {
            if v < 0x80 {
                data.push(v as u8);
                break;
            }
            data.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        // Add some trailing bytes
        data.extend_from_slice(&[0xAA; 10]);
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.read_length_delimited();
    }
}

// ============================================================================
// String decode fuzz
// ============================================================================

#[test]
fn fuzz_string_decode_random() {
    let mut rng = Xorshift64::new(0xAAAA_BBBB_0004);

    for _ in 0..20_000 {
        let data = rng.gen_bytes(32);
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.read_string();
    }
}

#[test]
fn fuzz_string_valid_utf8_roundtrip() {
    let test_strings = [
        "",
        "hello",
        "SmallAIOS",
        "input_0",
        "tensor-data-float32",
        "\u{00E9}\u{00F1}\u{00FC}", // accented chars
        "\u{1F600}",                // emoji
        "a/b/c",
    ];

    for s in &test_strings {
        let bytes = s.as_bytes();
        // Encode length as varint
        let mut data = Vec::new();
        let len = bytes.len() as u64;
        let mut v = len;
        loop {
            if v < 0x80 {
                data.push(v as u8);
                break;
            }
            data.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        data.extend_from_slice(bytes);
        let mut dec = ProtoDecoder::new(&data);
        let decoded = dec.read_string().unwrap();
        assert_eq!(decoded, *s);
    }
}

// ============================================================================
// Fixed32 / Fixed64 fuzz
// ============================================================================

#[test]
fn fuzz_fixed32_random() {
    let mut rng = Xorshift64::new(0xCCCC_DDDD_0005);

    for _ in 0..20_000 {
        let data = rng.gen_bytes(8);
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.read_fixed32();
    }
}

#[test]
fn fuzz_fixed64_random() {
    let mut rng = Xorshift64::new(0xEEEE_FFFF_0006);

    for _ in 0..20_000 {
        let data = rng.gen_bytes(16);
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.read_fixed64();
    }
}

#[test]
fn fuzz_float_random() {
    let mut rng = Xorshift64::new(0x1111_2222_0007);

    for _ in 0..10_000 {
        let data = rng.gen_bytes(8);
        let mut dec = ProtoDecoder::new(&data);
        if let Ok(f) = dec.read_float() {
            // NaN and Inf are valid IEEE 754 values
            let _ = f.is_nan();
            let _ = f.is_infinite();
        }
    }
}

#[test]
fn fuzz_double_random() {
    let mut rng = Xorshift64::new(0x3333_4444_0008);

    for _ in 0..10_000 {
        let data = rng.gen_bytes(16);
        let mut dec = ProtoDecoder::new(&data);
        if let Ok(d) = dec.read_double() {
            let _ = d.is_nan();
            let _ = d.is_infinite();
        }
    }
}

// ============================================================================
// Skip field fuzz
// ============================================================================

#[test]
fn fuzz_skip_field_random() {
    let mut rng = Xorshift64::new(0x5555_6666_0009);

    let wire_types = [
        WireType::Varint,
        WireType::Fixed64,
        WireType::LengthDelimited,
        WireType::Fixed32,
    ];

    for _ in 0..20_000 {
        let data = rng.gen_bytes(32);
        let wt = wire_types[(rng.next() as usize) % wire_types.len()];
        let mut dec = ProtoDecoder::new(&data);
        let _ = dec.skip_field(wt);
    }
}

// ============================================================================
// Sequential operation fuzz — simulate parsing a protobuf message
// ============================================================================

#[test]
fn fuzz_sequential_protobuf_parsing() {
    let mut rng = Xorshift64::new(0x7777_8888_000A);

    for _ in 0..5_000 {
        let data = rng.gen_bytes(128);
        let mut dec = ProtoDecoder::new(&data);

        // Simulate: read tag, then skip or read field value
        while !dec.is_empty() {
            match dec.read_tag() {
                Ok(header) => {
                    let _ = dec.skip_field(header.wire_type);
                }
                Err(_) => break,
            }
        }
    }
}

#[test]
fn fuzz_sequential_typed_reads() {
    let mut rng = Xorshift64::new(0x9999_AAAA_000B);

    for _ in 0..5_000 {
        let data = rng.gen_bytes(64);
        let mut dec = ProtoDecoder::new(&data);

        // Try various read operations in sequence
        let _ = dec.read_varint();
        let _ = dec.read_fixed32();
        let _ = dec.read_varint();
        let _ = dec.read_fixed64();
        let _ = dec.read_length_delimited();
        let _ = dec.read_string();
        let _ = dec.read_sint32();
        let _ = dec.read_sint64();
        let _ = dec.read_float();
        let _ = dec.read_double();
    }
}

// ============================================================================
// Zigzag decode fuzz
// ============================================================================

#[test]
fn fuzz_zigzag_decode_random_32() {
    let mut rng = Xorshift64::new(0xBBBB_CCCC_000C);

    for _ in 0..100_000 {
        let val = rng.next() as u32;
        let decoded = zigzag_decode_32(val);
        // Verify the roundtrip property:
        // zigzag_encode(decoded) == val
        let re_encoded = if decoded >= 0 {
            (decoded as u32) << 1
        } else {
            (((-decoded - 1) as u32) << 1) | 1
        };
        assert_eq!(re_encoded, val);
    }
}

#[test]
fn fuzz_zigzag_decode_random_64() {
    let mut rng = Xorshift64::new(0xDDDD_EEEE_000D);

    for _ in 0..100_000 {
        let val = rng.next();
        let decoded = zigzag_decode_64(val);
        let re_encoded = if decoded >= 0 {
            (decoded as u64) << 1
        } else {
            (((-decoded - 1) as u64) << 1) | 1
        };
        assert_eq!(re_encoded, val);
    }
}

// ============================================================================
// Bit-flip fuzz — valid protobuf message with random bit flips
// ============================================================================

#[test]
fn fuzz_bitflip_protobuf_message() {
    // Build a valid protobuf-like message:
    // Field 1 (varint): value 150
    // Field 2 (string): "ABC"
    // Field 3 (fixed32): 0x3F800000
    let valid_message: Vec<u8> = vec![
        0x08, 0x96, 0x01, // field 1, varint 150
        0x12, 0x03, 0x41, 0x42, 0x43, // field 2, string "ABC"
        0x1D, 0x00, 0x00, 0x80, 0x3F, // field 3, fixed32
    ];

    let mut rng = Xorshift64::new(0xFFFF_0000_000E);

    for _ in 0..5_000 {
        let mut corrupted = valid_message.clone();

        // Flip 1-3 random bits
        let num_flips = 1 + (rng.next() as usize % 3);
        for _ in 0..num_flips {
            let byte_idx = (rng.next() as usize) % corrupted.len();
            let bit_idx = (rng.next() as usize) % 8;
            corrupted[byte_idx] ^= 1 << bit_idx;
        }

        // Parse the corrupted message — must not panic
        let mut dec = ProtoDecoder::new(&corrupted);
        while !dec.is_empty() {
            match dec.read_tag() {
                Ok(header) => {
                    let _ = dec.skip_field(header.wire_type);
                }
                Err(_) => break,
            }
        }
    }
}

// ============================================================================
// Boundary value tests
// ============================================================================

#[test]
fn fuzz_varint_boundary_values() {
    // Test varints at power-of-2 boundaries
    let values: &[u64] = &[
        0,
        1,
        0x7F,
        0x80,
        0xFF,
        0x100,
        0x3FFF,
        0x4000,
        0xFFFF,
        0x10000,
        0x1FFFFF,
        0x200000,
        0xFFFFFFF,
        0x10000000,
        u32::MAX as u64,
        u64::MAX >> 1,
        u64::MAX,
    ];

    for &val in values {
        // Encode as LEB128
        let mut encoded = Vec::new();
        let mut v = val;
        loop {
            if v < 0x80 {
                encoded.push(v as u8);
                break;
            }
            encoded.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }

        let mut dec = ProtoDecoder::new(&encoded);
        let decoded = dec.read_varint().unwrap();
        assert_eq!(decoded, val);
    }
}
