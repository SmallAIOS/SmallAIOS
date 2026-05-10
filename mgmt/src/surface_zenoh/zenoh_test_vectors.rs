// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Test fixtures for the Zenoh admin surface — sample opaque tokens,
// dummy peer fingerprints, sample claim payloads. Lives in a
// `*_test_vectors.rs` file so CodeQL's byte-array / hex-constant
// scanners (`paths-ignored` rule in `.github/codeql/codeql-config.yml`)
// skip the file. Production code never references these constants;
// only `#[cfg(test)]` consumers in this module tree do.

#![cfg(test)]

use crate::surface_zenoh::session::{PeerIdentity, Token};

/// Worked-example opaque token used by the round-trip hex codec test.
/// 16 bytes of patterned data — *not* a credential for any real
/// account; it never reaches the kernel session table outside test
/// scope.
pub const SAMPLE_TOKEN_PATTERNED: Token = [
    0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
];

/// Lowercase hex form of [`SAMPLE_TOKEN_PATTERNED`].
pub const SAMPLE_TOKEN_PATTERNED_HEX: &str = "0123456789abcdeffedcba9876543210";

/// All-`0xAA` peer identity used as a primary fixture in the wrapper
/// tests. Stable so an audit-trace assertion can pin the rendered
/// `peer_identity` byte string.
pub const SAMPLE_PEER_AA: PeerIdentity = [0xAA; 32];

/// All-`0xBB` peer identity for cross-peer replay tests.
pub const SAMPLE_PEER_BB: PeerIdentity = [0xBB; 32];

/// All-`0xCC` peer identity for the per-identity cap test.
pub const SAMPLE_PEER_CC: PeerIdentity = [0xCC; 32];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_zenoh::session::{token_from_hex, token_hex};

    #[test]
    fn sample_token_round_trips_through_hex() {
        let hex = token_hex(&SAMPLE_TOKEN_PATTERNED);
        assert_eq!(
            core::str::from_utf8(&hex).unwrap(),
            SAMPLE_TOKEN_PATTERNED_HEX
        );
        let back = token_from_hex(hex.as_slice()).unwrap();
        assert_eq!(back, SAMPLE_TOKEN_PATTERNED);
    }

    #[test]
    fn sample_peer_constants_are_distinct() {
        assert_ne!(SAMPLE_PEER_AA, SAMPLE_PEER_BB);
        assert_ne!(SAMPLE_PEER_AA, SAMPLE_PEER_CC);
        assert_ne!(SAMPLE_PEER_BB, SAMPLE_PEER_CC);
    }

    #[test]
    fn sample_peer_constants_are_32_bytes() {
        assert_eq!(SAMPLE_PEER_AA.len(), 32);
        assert_eq!(SAMPLE_PEER_BB.len(), 32);
        assert_eq!(SAMPLE_PEER_CC.len(), 32);
    }
}
