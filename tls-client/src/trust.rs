// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Operator-controlled trust store (Phase 6, design.md D5).
//!
//! SmallAIOS ships **no** baked-in CA bundle. The operator points
//! `tls.trust_store_path` at a PEM bundle of the CA roots they
//! choose to trust; an empty store refuses every chain. Optional
//! pinning (`tls.trust_store_pin`) restricts acceptance to a single
//! anchor by its SHA-256 fingerprint.
//!
//! This module owns the in-memory anchor set and the PEM loader.
//! Chain construction against it lives in [`crate::cert::verify`].

use crate::cert::x509::{Certificate, SubjectPublicKey};
use alloc::vec::Vec;
use smallaios_security::sha2::sha256;

/// Errors from loading or validating a trust bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustStoreError {
    /// The bundle parsed to zero anchors (design.md D5: an empty
    /// store is a configuration error, not a silent allow-none).
    Empty,
    /// An anchor is not a CA certificate (`BasicConstraints.cA`
    /// absent or false).
    NotCa,
    /// Two anchors share the same Subject DN.
    DuplicateSubject,
    /// A PEM block or its base64 payload was malformed.
    Pem,
    /// An anchor's DER failed X.509 parsing.
    Parse,
}

/// A single trusted CA anchor.
#[derive(Debug, Clone)]
pub struct TrustAnchor {
    /// Raw Subject DN (SEQUENCE contents) for issuer matching.
    pub subject: Vec<u8>,
    /// The anchor's public key, used to verify the cert it signed.
    pub public_key: SubjectPublicKey,
    /// SHA-256 over the full anchor certificate DER (for pinning).
    pub fingerprint: [u8; 32],
}

/// An operator trust store: a set of CA anchors plus an optional
/// pin.
#[derive(Debug, Clone, Default)]
pub struct TrustStore {
    anchors: Vec<TrustAnchor>,
    pin: Option<[u8; 32]>,
}

impl TrustStore {
    /// An empty store. Refuses every chain until anchors are added.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when no anchors are loaded (every chain will be refused).
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Number of loaded anchors.
    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    /// All anchors (for the chain verifier).
    pub fn anchors(&self) -> &[TrustAnchor] {
        &self.anchors
    }

    /// Set the optional pin: a SHA-256 fingerprint over a single
    /// anchor's DER. When set, only chains anchored at that exact
    /// certificate are accepted.
    pub fn set_pin(&mut self, fingerprint: [u8; 32]) {
        self.pin = Some(fingerprint);
    }

    /// The configured pin, if any.
    pub fn pin(&self) -> Option<&[u8; 32]> {
        self.pin.as_ref()
    }

    /// Add one anchor from raw certificate DER. Rejects non-CA
    /// certificates and duplicate Subjects.
    pub fn add_anchor_der(&mut self, der: &[u8]) -> Result<(), TrustStoreError> {
        let cert = Certificate::parse(der).map_err(|_| TrustStoreError::Parse)?;
        if !cert.basic_constraints.ca {
            return Err(TrustStoreError::NotCa);
        }
        if self.anchors.iter().any(|a| a.subject == cert.subject) {
            return Err(TrustStoreError::DuplicateSubject);
        }
        self.anchors.push(TrustAnchor {
            subject: cert.subject.to_vec(),
            public_key: cert.public_key.clone(),
            fingerprint: sha256(der),
        });
        Ok(())
    }

    /// Load a PEM bundle of `BEGIN CERTIFICATE` / `END CERTIFICATE`
    /// blocks. Returns [`TrustStoreError::Empty`] if no anchors
    /// result.
    pub fn from_pem(pem: &str) -> Result<Self, TrustStoreError> {
        let mut store = TrustStore::new();
        for block in pem_certificate_blocks(pem)? {
            store.add_anchor_der(&block)?;
        }
        if store.is_empty() {
            return Err(TrustStoreError::Empty);
        }
        Ok(store)
    }
}

/// Extract every base64-decoded `CERTIFICATE` block from a PEM text.
fn pem_certificate_blocks(pem: &str) -> Result<Vec<Vec<u8>>, TrustStoreError> {
    const BEGIN: &str = "-----BEGIN CERTIFICATE-----";
    const END: &str = "-----END CERTIFICATE-----";
    let mut out = Vec::new();
    let mut rest = pem;
    while let Some(begin) = rest.find(BEGIN) {
        let after_begin = &rest[begin + BEGIN.len()..];
        let end = after_begin.find(END).ok_or(TrustStoreError::Pem)?;
        let body = &after_begin[..end];
        out.push(b64_decode(body)?);
        rest = &after_begin[end + END.len()..];
    }
    Ok(out)
}

/// Decode standard RFC 4648 base64 (with `=` padding), skipping
/// ASCII whitespace. Rejects any other character.
fn b64_decode(input: &str) -> Result<Vec<u8>, TrustStoreError> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::new();
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in input.as_bytes() {
        if c.is_ascii_whitespace() {
            continue;
        }
        if c == b'=' {
            break;
        }
        let v = val(c).ok_or(TrustStoreError::Pem)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::test_certs::{keypair, CertBuilder};
    use alloc::string::String;

    /// Re-encode DER as a PEM CERTIFICATE block (test mirror of the
    /// loader's decoder).
    fn to_pem(der: &[u8]) -> String {
        const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut b64 = String::new();
        for chunk in der.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            b64.push(ALPHA[(n >> 18 & 63) as usize] as char);
            b64.push(ALPHA[(n >> 12 & 63) as usize] as char);
            b64.push(if chunk.len() > 1 {
                ALPHA[(n >> 6 & 63) as usize] as char
            } else {
                '='
            });
            b64.push(if chunk.len() > 2 {
                ALPHA[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        let mut pem = String::from("-----BEGIN CERTIFICATE-----\n");
        pem.push_str(&b64);
        pem.push_str("\n-----END CERTIFICATE-----\n");
        pem
    }

    #[test]
    fn loads_ca_anchor_from_pem() {
        let ca = keypair(2);
        let der = CertBuilder::ca("Root CA").build_self_signed(&ca);
        let store = TrustStore::from_pem(&to_pem(&der)).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn empty_bundle_rejected() {
        assert_eq!(
            TrustStore::from_pem("no certs here").unwrap_err(),
            TrustStoreError::Empty
        );
    }

    #[test]
    fn non_ca_anchor_rejected() {
        let leaf = keypair(1);
        let ca = keypair(2);
        let der = CertBuilder::leaf("leaf.example.com", "leaf.example.com").build(&leaf, &ca);
        let mut store = TrustStore::new();
        assert_eq!(
            store.add_anchor_der(&der).unwrap_err(),
            TrustStoreError::NotCa
        );
    }

    #[test]
    fn duplicate_subject_rejected() {
        let ca = keypair(2);
        let der = CertBuilder::ca("Root CA").build_self_signed(&ca);
        let mut store = TrustStore::new();
        store.add_anchor_der(&der).unwrap();
        assert_eq!(
            store.add_anchor_der(&der).unwrap_err(),
            TrustStoreError::DuplicateSubject
        );
    }

    #[test]
    fn two_distinct_anchors_load() {
        let ca1 = keypair(2);
        let ca2 = keypair(3);
        let mut pem = to_pem(&CertBuilder::ca("Root One").build_self_signed(&ca1));
        pem.push_str(&to_pem(
            &CertBuilder::ca("Root Two").build_self_signed(&ca2),
        ));
        let store = TrustStore::from_pem(&pem).unwrap();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn malformed_base64_rejected() {
        let pem = "-----BEGIN CERTIFICATE-----\n!!!not base64!!!\n-----END CERTIFICATE-----";
        assert_eq!(TrustStore::from_pem(pem).unwrap_err(), TrustStoreError::Pem);
    }

    #[test]
    fn fingerprint_matches_sha256_of_der() {
        let ca = keypair(2);
        let der = CertBuilder::ca("Root CA").build_self_signed(&ca);
        let mut store = TrustStore::new();
        store.add_anchor_der(&der).unwrap();
        assert_eq!(store.anchors()[0].fingerprint, sha256(&der));
    }
}
