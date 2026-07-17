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
pub mod verify;
pub mod x509;

#[cfg(test)]
mod corpus_tests;
#[cfg(test)]
pub(crate) mod test_certs;

use crate::Result;
use alloc::vec::Vec;

/// The leaf certificate's SubjectPublicKeyInfo key, in the
/// algorithms the workspace can verify CertificateVerify with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeafPublicKey {
    Ed25519([u8; 32]),
    /// Uncompressed SEC1 point (`0x04 || X || Y`, 65 bytes).
    EcdsaP256(Vec<u8>),
    /// DER `RSAPublicKey ::= SEQUENCE { INTEGER n, INTEGER e }`.
    Rsa(Vec<u8>),
}

/// Chain verification seam between the handshake driver (Phase 4)
/// and the X.509 chain verifier (Phase 5).
///
/// `certs` is the server's chain exactly as received — leaf
/// first, raw DER. Implementations MUST verify the chain anchors
/// in the operator trust store, the validity windows, and that
/// the leaf's SAN matches `server_name`, returning the leaf's
/// public key for CertificateVerify checking. The production
/// implementation lands with Phase 5's structure parser; until
/// then the only impls are test doubles inside this crate.
pub trait ServerCertVerifier {
    fn verify_chain(&self, certs: &[Vec<u8>], server_name: Option<&str>) -> Result<LeafPublicKey>;
}
