// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Test fixtures for the Phase 4 model verb handlers — sample model
// payloads, request IDs, signatures, peer identities. Lives in a
// `*_test_vectors.rs` file so CodeQL's byte-array / hex-constant
// scanners (`paths-ignored` rule in `.github/codeql/codeql-config.yml`)
// skip the file. Production code never references these constants;
// only `#[cfg(test)]` consumers in this module tree do.

#![cfg(test)]

use crate::surface_zenoh::session::{PeerIdentity, Token};

/// Small synthetic model payload (well below the 256 KiB JSON cap).
pub const MODEL_BYTES_SMALL: &[u8] = b"\x08\x09ONNX-FAKE-PAYLOAD-FOR-TESTS-ONLY";

/// Larger synthetic model payload — 4 KiB worth of bytes, exercises
/// the streaming chunked path with a few frames.
pub const MODEL_BYTES_LARGE: &[u8] = &[0xAB; 4096];

/// Sample ML-DSA-65 signature bytes — content irrelevant; the kernel-
/// side test backend just stores them under `<name>.sig`.
pub const SAMPLE_SIGNATURE: &[u8] = &[0x42; 64];

/// Stable `request_id` used for the streamed-add happy-path test.
pub const REQUEST_ID_OK: &str = "rid-ok-01";

/// Stable `request_id` used for the disconnect-mid-upload test.
pub const REQUEST_ID_DISCONNECT: &str = "rid-disconnect-01";

/// Primary peer-identity fixture for Phase 4 tests.
pub const PEER_AA: PeerIdentity = [0xAA; 32];

/// Secondary peer-identity fixture for cross-peer replay tests.
pub const PEER_BB: PeerIdentity = [0xBB; 32];

/// Pool of opaque token fixtures keyed by (user_id % 8). Each token
/// is 16 bytes of patterned data — *not* a credential for any real
/// account; never reaches outside test scope.
pub const TOKEN_FIXTURE_FOR_USER: &[Token] = &[
    [0x00; 16], [0x11; 16], [0x22; 16], [0x33; 16], [0x44; 16], [0x55; 16], [0x66; 16], [0x77; 16],
    // Wrap-around for `% N` indexing.
    [0x88; 16], [0x99; 16], [0xAA; 16], [0xBB; 16], [0xCC; 16], [0xDD; 16], [0xEE; 16], [0xF0; 16],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_constants_are_distinct() {
        assert_ne!(MODEL_BYTES_SMALL, MODEL_BYTES_LARGE);
    }

    #[test]
    fn request_id_constants_are_distinct() {
        assert_ne!(REQUEST_ID_OK, REQUEST_ID_DISCONNECT);
    }

    #[test]
    fn token_fixtures_have_at_least_eight_distinct_bytes() {
        let pool = TOKEN_FIXTURE_FOR_USER;
        assert!(pool.len() >= 8);
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert_ne!(pool[i], pool[j]);
            }
        }
    }

    #[test]
    fn peer_constants_are_distinct() {
        assert_ne!(PEER_AA, PEER_BB);
    }

    #[test]
    fn signature_is_64_bytes() {
        assert_eq!(SAMPLE_SIGNATURE.len(), 64);
    }
}
