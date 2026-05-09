// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HMAC-SHA1 KAT vectors from RFC 2202 §3.
//!
//! All values are public test vectors taken verbatim from the RFC; they
//! are NOT credentials. Imported solely as oracle inputs for the
//! regression suite.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2202 §3 KAT vectors

#![allow(dead_code)]

// ─── Test Case 1 ────────────────────────────────────────────────────────────
//
// Key  = 0x0b * 20
// Data = "Hi There"

/// RFC 2202 §3 Test Case 1: 20-byte key of repeated 0x0b.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2202 KAT
pub const RFC2202_KEY_1: &[u8] = &[
    0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b,
    0x0b, 0x0b, 0x0b, 0x0b,
];

/// RFC 2202 §3 Test Case 1: data "Hi There".
pub const RFC2202_DATA_1: &[u8] = b"Hi There";

/// RFC 2202 §3 Test Case 1: expected tag.
pub const RFC2202_TAG_1_HEX: &str = "b617318655057264e28bc0b6fb378c8ef146be00";

// ─── Test Case 2 ────────────────────────────────────────────────────────────
//
// Key  = "Jefe"
// Data = "what do ya want for nothing?"

/// RFC 2202 §3 Test Case 2: ASCII key.
pub const RFC2202_KEY_2: &[u8] = b"Jefe";

/// RFC 2202 §3 Test Case 2: data.
pub const RFC2202_DATA_2: &[u8] = b"what do ya want for nothing?";

/// RFC 2202 §3 Test Case 2: expected tag.
pub const RFC2202_TAG_2_HEX: &str = "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79";

// ─── Test Case 3 ────────────────────────────────────────────────────────────
//
// Key  = 0xaa * 20, Data = 0xdd * 50  → constructed in tests via vec![…].

/// RFC 2202 §3 Test Case 3: expected tag.
pub const RFC2202_TAG_3_HEX: &str = "125d7342b9ac11cd91a39af48aa17b4f63f175d3";

// ─── Test Case 4 ────────────────────────────────────────────────────────────
//
// Key  = 0x01..0x19 (25 bytes)
// Data = 0xcd * 50

/// RFC 2202 §3 Test Case 4: 25-byte ascending key.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2202 KAT
pub const RFC2202_KEY_4: &[u8] = &[
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
];

/// RFC 2202 §3 Test Case 4: expected tag.
pub const RFC2202_TAG_4_HEX: &str = "4c9007f4026250c6bc8414f9bf50c86c2d7235da";

// ─── Test Case 5 ────────────────────────────────────────────────────────────
//
// Key  = 0x0c * 20
// Data = "Test With Truncation"

/// RFC 2202 §3 Test Case 5: data.
pub const RFC2202_DATA_5: &[u8] = b"Test With Truncation";

/// RFC 2202 §3 Test Case 5: expected (full-length) tag.
pub const RFC2202_TAG_5_HEX: &str = "4c1a03424b55e07fe7f27be1d58bb9324a9a5a04";

// ─── Test Case 6 ────────────────────────────────────────────────────────────
//
// Key  = 0xaa * 80
// Data = "Test Using Larger Than Block-Size Key - Hash Key First"

/// RFC 2202 §3 Test Case 6: data.
pub const RFC2202_DATA_6: &[u8] = b"Test Using Larger Than Block-Size Key - Hash Key First";

/// RFC 2202 §3 Test Case 6: expected tag.
pub const RFC2202_TAG_6_HEX: &str = "aa4ae5e15272d00e95705637ce8a3b55ed402112";

// ─── Test Case 7 ────────────────────────────────────────────────────────────
//
// Key  = 0xaa * 80
// Data = "Test Using Larger Than Block-Size Key and Larger Than One \
//         Block-Size Data"

/// RFC 2202 §3 Test Case 7: data.
pub const RFC2202_DATA_7: &[u8] =
    b"Test Using Larger Than Block-Size Key and Larger Than One Block-Size Data";

/// RFC 2202 §3 Test Case 7: expected tag.
pub const RFC2202_TAG_7_HEX: &str = "e8e99d0f45237d786d6bbaa7965c7808bbff1a91";
