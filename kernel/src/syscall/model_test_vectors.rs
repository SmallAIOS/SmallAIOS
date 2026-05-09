// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test fixtures for the overlay model-syscall handlers.
//!
//! These constants are used only from the unit tests in
//! `kernel/src/syscall/model.rs`. They are isolated here so the
//! CodeQL `paths-ignored` filter (configured in
//! `.github/codeql/codeql-config.yml` for `**/*_test_vectors.rs`)
//! suppresses `rust/hard-coded-cryptographic-value` false positives
//! on the byte arrays.

#![allow(dead_code)]

/// Tiny "hello"-style placeholder model bytes — 5 bytes, well under any
/// real cap.
pub const MODEL_BYTES: &[u8] = b"hello";

/// 200-byte placeholder model bytes — used to trip the cap-trip mid-
/// stream test.
pub const MODEL_BIG: &[u8] = &[0xFFu8; 200];

/// 16-byte placeholder signature blob — used by the signature-pass-
/// through test. Not a real ML-DSA-65 signature.
pub const SIGNATURE_BYTES: &[u8] = &[0xABu8; 16];

/// 256 `'A'` bytes — exceeds the 255-byte name cap.
pub const ALL_A_256: &[u8] = &[b'A'; 256];

/// One byte that is not valid UTF-8 (continuation byte without a
/// leading byte).
pub const NOT_UTF8: &[u8] = &[0xC3, 0x28];
