// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! X.509v3 parser + chain verification.
//!
//! Phase 5 of `tls-tcp-client-v1`. Implementation lands in
//! two commits:
//!
//! 1. **This commit**: DER TLV decoder (`der`) + RFC 6125
//!    hostname matcher (`hostname`). Both are fully testable
//!    without synthetic cert scaffolding and form the
//!    foundations the structure parser + chain verifier rely
//!    on.
//! 2. **Follow-on commit**: full X.509v3 structure parser
//!    (SerialNumber, signature.algorithm, issuer, subject,
//!    validity, SubjectPublicKeyInfo, SAN / BasicConstraints
//!    / KeyUsage / ExtKeyUsage extensions); chain
//!    verification against the trust store; signature
//!    verification via Ed25519 (immediate), with RSA-PSS +
//!    ECDSA-P256 deferred to their own primitive sub-adds.

pub mod der;
pub mod hostname;
