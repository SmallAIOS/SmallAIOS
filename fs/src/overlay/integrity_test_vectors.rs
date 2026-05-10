// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test fixtures for the overlay Phase 5 signature-policy enforcement.
//!
//! These constants are used only from the unit tests in
//! `fs/src/overlay/integrity.rs`. They are isolated here so the
//! CodeQL `paths-ignored` filter (configured in
//! `.github/codeql/codeql-config.yml` for `**/*_test_vectors.rs`)
//! suppresses `rust/hard-coded-cryptographic-value` false positives
//! on the byte arrays.
//!
//! ## Why seeds, not keys
//!
//! ML-DSA-65 public keys are 1952 bytes and signatures are 3309 bytes
//! — too large to inline as constant slices without bloating the
//! repo. Instead, this module holds a handful of 32-byte seeds and
//! the helper [`generate_keypair`] / [`sign_digest`] derive the
//! keypair / signature deterministically at test time. ML-DSA-65
//! `ml_dsa_65_keygen` is a pure function of the seed, so the vectors
//! are reproducible across runs and platforms (and across CI / local
//! `cargo test`).
//!
//! ## Seed catalogue
//!
//! | Seed             | Use                                              |
//! |------------------|--------------------------------------------------|
//! | `SEED_PRIMARY`   | Operator-deployed model-signing key (the canonical "right" key) |
//! | `SEED_ATTACKER`  | Wrong key used to fabricate a forged signature   |
//! | `SIGN_RNG_SEED`  | RNG seed for [`ml_dsa_65_sign`]'s hedged-signing nonce |

#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;
use smallaios_security::crypto::ml_dsa::{
    ml_dsa_65_keygen, ml_dsa_65_sign, MlDsaKeyPair, MlDsaSignature,
};

/// Seed for the canonical operator-deployed model-signing keypair.
/// All "right key" tests derive from this.
pub const SEED_PRIMARY: [u8; 32] = [0x11u8; 32];

/// Seed for an attacker-controlled keypair. Used to construct
/// signatures that verify under the WRONG public key (so the strict-
/// policy test sees a signature that's well-formed but mis-signed).
pub const SEED_ATTACKER: [u8; 32] = [0x22u8; 32];

/// RNG seed for `ml_dsa_65_sign`. Hedged signing means each `sign`
/// call needs an external 32-byte randomness input.
pub const SIGN_RNG_SEED: [u8; 32] = [0x33u8; 32];

/// Tiny model bytes — same payload as the kernel's
/// `MODEL_BYTES` so the two test suites use a consistent fixture.
pub const MODEL_BYTES: &[u8] = b"hello";

/// A second model body used in the "two-key cross-verify" test —
/// different bytes mean a different SHA-3 fingerprint, so signing
/// fingerprint A and verifying against fingerprint B fails.
pub const MODEL_BYTES_OTHER: &[u8] = b"another model body";

/// Generate the canonical primary keypair from [`SEED_PRIMARY`].
/// Every test that needs the "right" trust anchor calls this.
pub fn primary_keypair() -> MlDsaKeyPair {
    ml_dsa_65_keygen(&SEED_PRIMARY).expect("keygen from SEED_PRIMARY")
}

/// Generate the attacker keypair from [`SEED_ATTACKER`]. Used to
/// forge a signature that verifies under the wrong pubkey.
pub fn attacker_keypair() -> MlDsaKeyPair {
    ml_dsa_65_keygen(&SEED_ATTACKER).expect("keygen from SEED_ATTACKER")
}

/// Sign `message` with `kp.secret_key` and return the serialised
/// signature bytes. Wraps the test-only helper in
/// `smallaios_security::crypto::ml_dsa`.
pub fn sign_message(kp: &MlDsaKeyPair, message: &[u8]) -> MlDsaSignature {
    ml_dsa_65_sign(&kp.secret_key, message, &SIGN_RNG_SEED).expect("sign")
}

/// Convenience: sign `message` and return the raw signature bytes
/// (drops the wrapper struct, useful for `verify_signed` which takes
/// raw bytes).
pub fn sign_message_bytes(kp: &MlDsaKeyPair, message: &[u8]) -> Vec<u8> {
    sign_message(kp, message).as_bytes().to_vec()
}
