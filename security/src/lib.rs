// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Security
//!
//! Defense-in-depth security with:
//! - Capability-based access control (no ambient authority)
//! - Post-quantum cryptography (FIPS 203/204/205):
//!   - ML-KEM-768 (Kyber) for key encapsulation
//!   - ML-DSA-65 (Dilithium) for digital signatures
//!   - SLH-DSA (SPHINCS+) for hash-based signature fallback
//!   - Hybrid mode: classical + PQC combined
//! - AES-256-GCM symmetric encryption (hardware-accelerated)
//! - SHA-3 / SHAKE256 hashing
//! - CSPRNG seeded from hardware RNG (RDRAND/RNDR)
//! - TLS 1.3 with post-quantum key exchange
//! - ONNX model signature verification
//! - Hardware security feature management (NX, SMEP, SMAP, CET, PAC, BTI, MTE)
//! - Security audit logging

#![no_std]
