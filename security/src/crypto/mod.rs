// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cryptographic primitives for SmallAIOS.
//!
//! Implements post-quantum and classical cryptography:
//! - SHA-3-256 and SHAKE256 (FIPS 202)
//! - Blake2b (RFC 7693) — backend for Argon2id's variable-length hash
//! - AES-256-GCM (FIPS 197 + SP 800-38D)
//! - CSPRNG (SHAKE256-based, seeded from RDRAND/RNDR)
//! - ML-KEM-768 (FIPS 203) — key encapsulation
//! - ML-DSA-65 (FIPS 204) — digital signatures
//! - ECDSA-P256/SHA-256 signature verification (X9.62, verify-only)
//! - Hybrid modes (classical + PQC)
//! - Constant-time utilities

pub mod aes_gcm;
/// Private big-integer arithmetic backing RSA (`rsa_pss`); not part of the
/// public crypto surface.
mod big_int;
pub mod blake2b;
pub mod chacha20;
pub mod chacha20_poly1305;
pub mod constant_time;
pub mod csprng;
pub mod ecdsa_p256;
#[cfg(test)]
mod ecdsa_p256_test_vectors;
pub mod ed25519;
pub mod field25519;
pub mod hybrid;
pub mod key_manager;
pub mod mgf1;
pub mod ml_dsa;
pub mod ml_kem;
pub mod p256;
pub mod poly1305;
pub mod rsa_pss;
#[cfg(test)]
mod rsa_pss_test_vectors;
pub mod sha3;
pub mod verify;
pub mod x25519;
