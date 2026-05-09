// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Byte-array test vectors for `embedded-overlay-v1` Phase 2 conformance
//! tests.
//!
//! Per `docs/codeql-test-vectors.md`: keypairs, fingerprints, and any
//! `[u8; N]` constants used as tests-only inputs live in this sibling
//! file (NOT in `overlay_write_conformance.rs`) so CodeQL's
//! `hard-coded-cryptographic-value` query treats them as test fixtures.
//!
//! Loaded via `#[path = "overlay_write_test_vectors.rs"] mod test_vectors;`
//! from the conformance test binary.

#![cfg(feature = "fs-overlay-mounts")]
#![allow(dead_code)] // Some helpers exist for tests not yet enabled.

extern crate alloc;

use alloc::vec::Vec;

/// 32-byte deterministic seed for the ML-DSA-65 trust-anchor key pair
/// the conformance tests use. Test-only — not a real signing key.
// lgtm[rust/hard-coded-cryptographic-value] — test fixture seed
pub const SIGNING_KEY_SEED: [u8; 32] = [
    0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x07,
    0x18, // lgtm[rust/hard-coded-cryptographic-value]
    0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f,
    0x90, // lgtm[rust/hard-coded-cryptographic-value]
    0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67,
    0x78, // lgtm[rust/hard-coded-cryptographic-value]
    0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef,
    0xf0, // lgtm[rust/hard-coded-cryptographic-value]
];

/// 32-byte randomness used by `ml_dsa_65_sign` for hedged signing in
/// the conformance tests. Reused across tests for determinism.
// lgtm[rust/hard-coded-cryptographic-value] — test fixture
pub const SIGNING_RANDOMNESS: [u8; 32] = [
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, // lgtm[rust/hard-coded-cryptographic-value]
    0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    0x00, // lgtm[rust/hard-coded-cryptographic-value]
    0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,
    0x80, // lgtm[rust/hard-coded-cryptographic-value]
    0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
    0x01, // lgtm[rust/hard-coded-cryptographic-value]
];

/// Synthetic "tiny model" payload used by every test that only cares
/// about the bytes-flow path. Not real ONNX, but the overlay treats
/// model files as opaque blobs so any 16-byte sequence works.
pub const TINY_MODEL_BYTES: &[u8] = b"\x00onnxhelloworld!";

/// Larger ~64 KiB synthetic model for cap-stress tests.
pub fn build_kib_payload(kib: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(kib * 1024);
    for i in 0..(kib * 1024) {
        out.push((i & 0xff) as u8);
    }
    out
}
