// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cryptographic primitives for SmallAIOS.
//!
//! Implements post-quantum and classical cryptography:
//! - SHA-3-256 and SHAKE256 (FIPS 202)
//! - AES-256-GCM (FIPS 197 + SP 800-38D)
//! - CSPRNG (SHAKE256-based, seeded from RDRAND/RNDR)
//! - ML-KEM-768 (FIPS 203) — key encapsulation
//! - ML-DSA-65 (FIPS 204) — digital signatures
//! - Hybrid modes (classical + PQC)
//! - Constant-time utilities

pub mod aes_gcm;
pub mod constant_time;
pub mod csprng;
pub mod hybrid;
pub mod key_manager;
pub mod ml_dsa;
pub mod ml_kem;
pub mod sha3;
pub mod verify;
