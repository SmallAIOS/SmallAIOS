// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Certificate-chain verification (tasks 5.4, 5.5, 5.6).
//!
//! [`TrustStoreVerifier`] is the production [`ServerCertVerifier`]
//! the handshake driver calls. It:
//!
//! 1. parses the leaf, requires a SubjectAltName, binds it to the
//!    operator hostname (RFC 6125, via [`super::hostname`]), and
//!    requires the serverAuth EKU;
//! 2. builds a chain leaf → intermediate(s) → a trust-store anchor,
//!    requiring `BasicConstraints.cA` + `keyUsage.keyCertSign` on
//!    every CA link;
//! 3. verifies the signature on every link;
//! 4. checks each certificate's validity window against the wall
//!    clock, with the design.md unsynced-clock sentinel.
//!
//! **Signature-algorithm coverage:** Ed25519 links are verified
//! today. ECDSA-P256 and RSA-PSS/PKCS#1 links are recognised by the
//! parser but cannot yet be verified — their `security/` primitives
//! are tracked by `security-ecdsa-p256-v1` / `security-rsa-pss-v1`.
//! A chain that needs one of them is refused with `ChainUntrusted`
//! rather than silently accepted.

use super::hostname::match_hostname;
use super::x509::{Certificate, SignatureAlgorithm, SubjectPublicKey};
use super::{LeafPublicKey, ServerCertVerifier};
use crate::trust::TrustStore;
use crate::{Result, TlsClientError};
use alloc::vec::Vec;
use smallaios_security::crypto::ed25519::{ed25519_verify, Ed25519PublicKey, Ed25519Signature};

/// Wall-clock sentinel (design.md): a "now" earlier than
/// 2026-01-01T00:00:00Z is treated as an unsynchronized clock.
const SYNCED_CLOCK_THRESHOLD: i64 = 1_767_225_600;

/// Maximum certificates considered while building a chain. Bounds
/// the search regardless of how many the peer sent.
const MAX_CHAIN_DEPTH: usize = 8;

/// Production chain verifier over an operator [`TrustStore`].
pub struct TrustStoreVerifier<'a> {
    trust: &'a TrustStore,
    /// Current wall-clock time as Unix seconds (from `kernel::clock`
    /// in production; injected in tests).
    now_unix: i64,
    /// When true, an unsynchronized clock refuses the chain instead
    /// of bypassing the validity window.
    require_synced_clock: bool,
}

impl<'a> TrustStoreVerifier<'a> {
    /// Construct with an explicit `now` (Unix seconds).
    ///
    /// The caller supplies the wall-clock time. In production the
    /// integration layer reads it from `kernel::clock()` (the
    /// `tls-client` crate stays decoupled from the kernel allocator);
    /// tests inject a fixed value.
    pub fn new(trust: &'a TrustStore, now_unix: i64, require_synced_clock: bool) -> Self {
        Self {
            trust,
            now_unix,
            require_synced_clock,
        }
    }

    /// True when the wall clock looks synchronized (≥ the sentinel).
    fn clock_synced(&self) -> bool {
        self.now_unix >= SYNCED_CLOCK_THRESHOLD
    }

    /// Enforce a certificate's validity window. Bypassed (with the
    /// design.md caveat) when the clock is unsynced unless
    /// `require_synced_clock`.
    fn check_validity(&self, cert: &Certificate<'_>) -> Result<()> {
        if !self.clock_synced() {
            // Unsynchronized clock. The audit hook
            // (`audit_export_unsynced_clock`) is wired in Phase 8;
            // here we either bypass or refuse per operator policy.
            if self.require_synced_clock {
                return Err(TlsClientError::Expired);
            }
            return Ok(());
        }
        if self.now_unix < cert.not_before || self.now_unix > cert.not_after {
            return Err(TlsClientError::Expired);
        }
        Ok(())
    }
}

impl ServerCertVerifier for TrustStoreVerifier<'_> {
    fn verify_chain(&self, certs: &[Vec<u8>], server_name: Option<&str>) -> Result<LeafPublicKey> {
        // An empty trust store refuses everything (design.md D5).
        if self.trust.is_empty() {
            return Err(TlsClientError::ChainUntrusted);
        }
        let leaf_der = certs.first().ok_or(TlsClientError::BadCertificate)?;
        let leaf = Certificate::parse(leaf_der)?;

        // ── Leaf identity policy ──────────────────────────────────
        // SAN is mandatory (design.md D4 / RFC 6125 §6.4.4).
        if !leaf.san_present || leaf.san.is_empty() {
            return Err(TlsClientError::BadCertificate);
        }
        if let Some(name) = server_name {
            match_hostname(name, &leaf.san)?;
        }
        // serverAuth EKU required on the leaf (design.md D4).
        if !leaf.ext_key_usage_present || !leaf.server_auth_eku {
            return Err(TlsClientError::BadCertificate);
        }
        self.check_validity(&leaf)?;

        // ── Chain construction ────────────────────────────────────
        // Parse the intermediates the peer offered (leaf excluded).
        let mut intermediates: Vec<Certificate<'_>> = Vec::new();
        for der in &certs[1..] {
            intermediates.push(Certificate::parse(der)?);
        }

        // Walk issuer links until we hit a trust-store anchor.
        let mut current = leaf.clone();
        let mut used = alloc::vec![false; intermediates.len()];
        for _ in 0..MAX_CHAIN_DEPTH {
            // Anchored? Find an anchor whose Subject is our issuer.
            if let Some(anchor) = self
                .trust
                .anchors()
                .iter()
                .find(|a| a.subject == current.issuer)
            {
                verify_signature(&current, &anchor.public_key)?;
                // Pin check (task 6.3): the anchoring cert must match.
                if let Some(pin) = self.trust.pin() {
                    if &anchor.fingerprint != pin {
                        return Err(TlsClientError::ChainUntrusted);
                    }
                }
                // Whole chain verified — hand back the leaf key.
                return leaf_public_key(&leaf);
            }

            // Otherwise extend via an offered intermediate whose
            // Subject is our issuer and which is a usable CA.
            let next = intermediates
                .iter()
                .enumerate()
                .find(|(i, c)| !used[*i] && c.subject == current.issuer && is_ca(c));
            match next {
                Some((idx, issuer)) => {
                    verify_signature(&current, &issuer.public_key)?;
                    self.check_validity(issuer)?;
                    used[idx] = true;
                    current = issuer.clone();
                }
                None => return Err(TlsClientError::ChainUntrusted),
            }
        }
        Err(TlsClientError::ChainUntrusted)
    }
}

/// A certificate is a usable CA link iff `BasicConstraints.cA` is
/// set and, when a `keyUsage` extension is present, it asserts
/// `keyCertSign`.
fn is_ca(cert: &Certificate<'_>) -> bool {
    if !cert.basic_constraints.ca {
        return false;
    }
    if cert.key_usage.present && !cert.key_usage.key_cert_sign {
        return false;
    }
    true
}

/// Verify `cert`'s outer signature over its `tbs_der` using the
/// issuer's public key. Only Ed25519 is supported today; other
/// algorithms are refused as `ChainUntrusted`.
fn verify_signature(cert: &Certificate<'_>, issuer_key: &SubjectPublicKey) -> Result<()> {
    match (cert.signature_algorithm, issuer_key) {
        (SignatureAlgorithm::Ed25519, SubjectPublicKey::Ed25519(pk)) => {
            if cert.signature.len() != 64 {
                return Err(TlsClientError::BadCertificate);
            }
            let mut sig = [0u8; 64];
            sig.copy_from_slice(cert.signature);
            ed25519_verify(
                &Ed25519PublicKey::from_bytes(*pk),
                cert.tbs_der,
                &Ed25519Signature::from_bytes(sig),
            )
            .map_err(|_| TlsClientError::ChainUntrusted)
        }
        // Algorithm/key mismatch, or an algorithm whose primitive is
        // not yet available (ECDSA-P256, RSA-PSS/PKCS#1).
        _ => Err(TlsClientError::ChainUntrusted),
    }
}

/// Map the leaf's SPKI to the [`LeafPublicKey`] the driver uses for
/// CertificateVerify. Non-Ed25519 leaves cannot be used yet.
fn leaf_public_key(leaf: &Certificate<'_>) -> Result<LeafPublicKey> {
    match &leaf.public_key {
        SubjectPublicKey::Ed25519(pk) => Ok(LeafPublicKey::Ed25519(*pk)),
        _ => Err(TlsClientError::ChainUntrusted),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::test_certs::{keypair, CertBuilder, San};

    /// Build a (trust store, leaf-der, [intermediate-ders]) fixture:
    /// root CA → leaf signed directly by the root.
    fn direct_chain(now_ok: bool) -> (TrustStore, Vec<Vec<u8>>) {
        let root = keypair(10);
        let leaf = keypair(11);
        let na = if now_ok {
            (2099, 1, 1, 0, 0, 0)
        } else {
            (2024, 6, 1, 0, 0, 0)
        };
        let mut lb = CertBuilder::leaf("leaf.example.com", "leaf.example.com");
        lb.issuer = "Root CA";
        lb.not_after = na;
        let leaf_der = lb.build(&leaf, &root);
        let root_der = CertBuilder::ca("Root CA").build_self_signed(&root);

        let mut store = TrustStore::new();
        store.add_anchor_der(&root_der).unwrap();
        (store, alloc::vec![leaf_der])
    }

    // 2026-06-01 — a synced "now".
    const NOW: i64 = 1_780_000_000;

    #[test]
    fn accepts_direct_chain_and_returns_leaf_key() {
        let (store, certs) = direct_chain(true);
        let v = TrustStoreVerifier::new(&store, NOW, false);
        let key = v.verify_chain(&certs, Some("leaf.example.com")).unwrap();
        assert!(matches!(key, LeafPublicKey::Ed25519(_)));
    }

    #[test]
    fn accepts_two_link_chain() {
        let root = keypair(20);
        let inter = keypair(21);
        let leaf = keypair(22);

        let mut ib = CertBuilder::ca("Intermediate CA");
        ib.issuer = "Root CA";
        let inter_der = ib.build(&inter, &root);
        let root_der = CertBuilder::ca("Root CA").build_self_signed(&root);

        let mut lb = CertBuilder::leaf("svc.example.com", "svc.example.com");
        lb.issuer = "Intermediate CA";
        let leaf_der = lb.build(&leaf, &inter);

        let mut store = TrustStore::new();
        store.add_anchor_der(&root_der).unwrap();
        let v = TrustStoreVerifier::new(&store, NOW, false);
        v.verify_chain(&alloc::vec![leaf_der, inter_der], Some("svc.example.com"))
            .unwrap();
    }

    #[test]
    fn empty_trust_store_refuses() {
        let (_full, certs) = direct_chain(true);
        let store = TrustStore::new();
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert_eq!(
            v.verify_chain(&certs, Some("leaf.example.com"))
                .unwrap_err(),
            TlsClientError::ChainUntrusted
        );
    }

    #[test]
    fn unknown_anchor_refused() {
        let (_store, certs) = direct_chain(true);
        // A trust store holding an unrelated root.
        let other = keypair(99);
        let other_der = CertBuilder::ca("Other Root").build_self_signed(&other);
        let mut store = TrustStore::new();
        store.add_anchor_der(&other_der).unwrap();
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert_eq!(
            v.verify_chain(&certs, Some("leaf.example.com"))
                .unwrap_err(),
            TlsClientError::ChainUntrusted
        );
    }

    #[test]
    fn tampered_leaf_signature_refused() {
        let (store, mut certs) = direct_chain(true);
        // Flip a byte in the leaf's signatureValue region (tail).
        let last = certs[0].len() - 1;
        certs[0][last] ^= 0x01;
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert!(v.verify_chain(&certs, Some("leaf.example.com")).is_err());
    }

    #[test]
    fn hostname_mismatch_refused() {
        let (store, certs) = direct_chain(true);
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert_eq!(
            v.verify_chain(&certs, Some("evil.example.com"))
                .unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn expired_leaf_refused_when_clock_synced() {
        let (store, certs) = direct_chain(false); // notAfter 2024-06
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert_eq!(
            v.verify_chain(&certs, Some("leaf.example.com"))
                .unwrap_err(),
            TlsClientError::Expired
        );
    }

    #[test]
    fn expired_leaf_bypassed_when_clock_unsynced() {
        let (store, certs) = direct_chain(false);
        // now < 2026 sentinel ⇒ unsynced ⇒ validity bypassed.
        let v = TrustStoreVerifier::new(&store, 1_000, false);
        v.verify_chain(&certs, Some("leaf.example.com")).unwrap();
    }

    #[test]
    fn unsynced_clock_refused_when_required() {
        let (store, certs) = direct_chain(true);
        let v = TrustStoreVerifier::new(&store, 1_000, true);
        assert_eq!(
            v.verify_chain(&certs, Some("leaf.example.com"))
                .unwrap_err(),
            TlsClientError::Expired
        );
    }

    #[test]
    fn leaf_without_san_refused() {
        let root = keypair(30);
        let leaf = keypair(31);
        let mut lb = CertBuilder::leaf("leaf.example.com", "leaf.example.com");
        lb.issuer = "Root CA";
        lb.san_extension = false;
        let leaf_der = lb.build(&leaf, &root);
        let root_der = CertBuilder::ca("Root CA").build_self_signed(&root);
        let mut store = TrustStore::new();
        store.add_anchor_der(&root_der).unwrap();
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert_eq!(
            v.verify_chain(&alloc::vec![leaf_der], Some("leaf.example.com"))
                .unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn non_ca_intermediate_refused() {
        let root = keypair(40);
        let inter = keypair(41);
        let leaf = keypair(42);
        // "Intermediate" lacks cA=true (built as a leaf-shaped cert).
        let mut ib = CertBuilder::leaf("Intermediate", "intermediate.invalid");
        ib.issuer = "Root CA";
        let inter_der = ib.build(&inter, &root);
        let root_der = CertBuilder::ca("Root CA").build_self_signed(&root);
        let mut lb = CertBuilder::leaf("svc.example.com", "svc.example.com");
        lb.issuer = "Intermediate";
        let leaf_der = lb.build(&leaf, &inter);
        let mut store = TrustStore::new();
        store.add_anchor_der(&root_der).unwrap();
        let v = TrustStoreVerifier::new(&store, NOW, false);
        assert_eq!(
            v.verify_chain(&alloc::vec![leaf_der, inter_der], Some("svc.example.com"))
                .unwrap_err(),
            TlsClientError::ChainUntrusted
        );
    }

    #[test]
    fn pin_match_accepts_mismatch_refuses() {
        let (mut store, certs) = direct_chain(true);
        let good_pin = store.anchors()[0].fingerprint;
        store.set_pin(good_pin);
        let v = TrustStoreVerifier::new(&store, NOW, false);
        v.verify_chain(&certs, Some("leaf.example.com")).unwrap();

        let (mut store2, certs2) = direct_chain(true);
        store2.set_pin([0xaa; 32]); // wrong pin
        let v2 = TrustStoreVerifier::new(&store2, NOW, false);
        assert_eq!(
            v2.verify_chain(&certs2, Some("leaf.example.com"))
                .unwrap_err(),
            TlsClientError::ChainUntrusted
        );
    }

    #[test]
    fn ip_san_leaf_accepts_matching_literal() {
        let root = keypair(50);
        let leaf = keypair(51);
        let mut lb = CertBuilder::leaf("server", "unused");
        lb.issuer = "Root CA";
        lb.sans = alloc::vec![San::Ip(alloc::vec![192, 0, 2, 1])];
        let leaf_der = lb.build(&leaf, &root);
        let root_der = CertBuilder::ca("Root CA").build_self_signed(&root);
        let mut store = TrustStore::new();
        store.add_anchor_der(&root_der).unwrap();
        let v = TrustStoreVerifier::new(&store, NOW, false);
        v.verify_chain(&alloc::vec![leaf_der], Some("192.0.2.1"))
            .unwrap();
    }
}
