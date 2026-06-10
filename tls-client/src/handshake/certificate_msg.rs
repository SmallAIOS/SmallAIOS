// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Certificate and CertificateVerify message wire layer
//! (RFC 8446 §4.4.2, §4.4.3).
//!
//! This module owns the *message* parsing and the
//! CertificateVerify signature check. Chain construction and
//! X.509 validation live behind the [`crate::cert`] verifier
//! (Phase 5 of `tls-tcp-client-v1`); the handshake driver hands
//! the raw DER blobs parsed here to that verifier.
//!
//! ```text
//! struct {
//!     opaque certificate_request_context<0..2^8-1>;
//!     CertificateEntry certificate_list<0..2^24-1>;
//! } Certificate;
//!
//! struct {
//!     opaque cert_data<1..2^24-1>;
//!     Extension extensions<0..2^16-1>;
//! } CertificateEntry;
//!
//! struct {
//!     SignatureScheme algorithm;
//!     opaque signature<0..2^16-1>;
//! } CertificateVerify;
//! ```

use super::extensions::sig_scheme;
use super::{HandshakeHeader, HandshakeType};
use crate::{Result, TlsClientError};
use alloc::vec::Vec;
use smallaios_security::crypto::ed25519::{ed25519_verify, Ed25519PublicKey, Ed25519Signature};

/// Parsed Certificate message: the server's chain, leaf first,
/// as raw DER blobs.
#[derive(Debug, Clone)]
pub struct CertificateMsg {
    pub certs: Vec<Vec<u8>>,
}

/// Parse a Certificate message starting at its handshake header.
///
/// The `certificate_request_context` MUST be empty for
/// server-initiated authentication (RFC 8446 §4.4.2) — we never
/// send post-handshake CertificateRequest, so any non-empty
/// context is refused.
pub fn parse_certificate(buf: &[u8]) -> Result<CertificateMsg> {
    let header = HandshakeHeader::parse(buf)?;
    if header.msg_type != HandshakeType::Certificate {
        return Err(TlsClientError::BadHandshake);
    }
    let body_start = HandshakeHeader::LEN;
    let body_end = body_start + header.length as usize;
    if buf.len() < body_end {
        return Err(TlsClientError::BadHandshake);
    }
    let body = &buf[body_start..body_end];

    if body.is_empty() {
        return Err(TlsClientError::BadHandshake);
    }
    let ctx_len = body[0] as usize;
    if ctx_len != 0 {
        return Err(TlsClientError::BadHandshake);
    }
    let mut cur = 1;

    if cur + 3 > body.len() {
        return Err(TlsClientError::BadHandshake);
    }
    let list_len = u32::from_be_bytes([0, body[cur], body[cur + 1], body[cur + 2]]) as usize;
    cur += 3;
    if cur + list_len != body.len() {
        return Err(TlsClientError::BadHandshake);
    }

    let mut certs = Vec::new();
    let list_end = cur + list_len;
    while cur < list_end {
        if cur + 3 > list_end {
            return Err(TlsClientError::BadHandshake);
        }
        let cert_len = u32::from_be_bytes([0, body[cur], body[cur + 1], body[cur + 2]]) as usize;
        cur += 3;
        if cert_len == 0 || cur + cert_len > list_end {
            return Err(TlsClientError::BadHandshake);
        }
        certs.push(body[cur..cur + cert_len].to_vec());
        cur += cert_len;
        // Per-entry extensions (e.g. OCSP staple) — length-check
        // and skip; v1 consumes none of them.
        if cur + 2 > list_end {
            return Err(TlsClientError::BadHandshake);
        }
        let ext_len = u16::from_be_bytes([body[cur], body[cur + 1]]) as usize;
        cur += 2;
        if cur + ext_len > list_end {
            return Err(TlsClientError::BadHandshake);
        }
        cur += ext_len;
    }
    if certs.is_empty() {
        // An empty certificate_list from the server means it has
        // no certificate — fatal for server auth.
        return Err(TlsClientError::BadCertificate);
    }
    Ok(CertificateMsg { certs })
}

/// Parsed CertificateVerify message.
#[derive(Debug, Clone)]
pub struct CertificateVerifyMsg {
    pub scheme: u16,
    pub signature: Vec<u8>,
}

/// Signature schemes the client accepts in CertificateVerify, per
/// the `tls-client-handshake` spec allow-list. Everything else —
/// in particular every SHA-1 suite (`rsa_pkcs1_sha1` 0x0201,
/// `ecdsa_sha1` 0x0203) and `dsa_*` — is refused with
/// `BadCertificate`.
pub const CERT_VERIFY_ALLOW_LIST: [u16; 5] = [
    sig_scheme::ED25519,
    sig_scheme::ECDSA_SECP256R1_SHA256,
    sig_scheme::RSA_PSS_RSAE_SHA256,
    sig_scheme::RSA_PSS_RSAE_SHA384,
    sig_scheme::RSA_PSS_RSAE_SHA512,
];

/// Parse a CertificateVerify message starting at its handshake
/// header, enforcing the signature-scheme allow-list.
pub fn parse_certificate_verify(buf: &[u8]) -> Result<CertificateVerifyMsg> {
    let header = HandshakeHeader::parse(buf)?;
    if header.msg_type != HandshakeType::CertificateVerify {
        return Err(TlsClientError::BadHandshake);
    }
    let body_start = HandshakeHeader::LEN;
    let body_end = body_start + header.length as usize;
    if buf.len() < body_end {
        return Err(TlsClientError::BadHandshake);
    }
    let body = &buf[body_start..body_end];
    if body.len() < 4 {
        return Err(TlsClientError::BadHandshake);
    }
    let scheme = u16::from_be_bytes([body[0], body[1]]);
    if !CERT_VERIFY_ALLOW_LIST.contains(&scheme) {
        return Err(TlsClientError::BadCertificate);
    }
    let sig_len = u16::from_be_bytes([body[2], body[3]]) as usize;
    if 4 + sig_len != body.len() {
        return Err(TlsClientError::BadHandshake);
    }
    Ok(CertificateVerifyMsg {
        scheme,
        signature: body[4..].to_vec(),
    })
}

/// Build the content the server signs in CertificateVerify
/// (RFC 8446 §4.4.3): 64 spaces, the context string, one zero
/// byte, then Transcript-Hash(CH..Certificate).
pub fn certificate_verify_content(transcript_hash: &[u8]) -> Vec<u8> {
    let mut content = Vec::with_capacity(64 + 33 + 1 + transcript_hash.len());
    content.extend_from_slice(&[0x20u8; 64]);
    content.extend_from_slice(b"TLS 1.3, server CertificateVerify");
    content.push(0);
    content.extend_from_slice(transcript_hash);
    content
}

/// Verify a CertificateVerify signature against the leaf's
/// public key and the transcript hash at the Certificate message
/// (task 4.7's signature step).
///
/// Only `ed25519` is verifiable today — ECDSA-P256 and RSA-PSS
/// primitives are deferred to their own `security/` sub-adds
/// (tasks 5.5 note). A chain whose leaf needs one of those
/// surfaces `BadCertificate` rather than silently passing.
pub fn verify_certificate_verify(
    msg: &CertificateVerifyMsg,
    leaf_pubkey_ed25519: &[u8; 32],
    transcript_hash: &[u8],
) -> Result<()> {
    if msg.scheme != sig_scheme::ED25519 {
        return Err(TlsClientError::BadCertificate);
    }
    if msg.signature.len() != 64 {
        return Err(TlsClientError::BadCertificate);
    }
    let content = certificate_verify_content(transcript_hash);
    let pk = Ed25519PublicKey::from_bytes(*leaf_pubkey_ed25519);
    let mut sig = [0u8; 64];
    sig.copy_from_slice(&msg.signature);
    ed25519_verify(&pk, &content, &Ed25519Signature::from_bytes(sig))
        .map_err(|_| TlsClientError::BadCertificate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_security::crypto::ed25519::{ed25519_keygen, ed25519_sign};

    pub(crate) fn build_certificate(certs: &[&[u8]]) -> Vec<u8> {
        let mut list = Vec::new();
        for c in certs {
            list.extend_from_slice(&(*c).len().to_be_bytes()[5..]); // u24
            list.extend_from_slice(c);
            list.extend_from_slice(&[0, 0]); // no per-entry extensions
        }
        let mut body = Vec::new();
        body.push(0); // empty certificate_request_context
        body.extend_from_slice(&list.len().to_be_bytes()[5..]); // u24
        body.extend_from_slice(&list);
        let mut out = Vec::new();
        HandshakeHeader {
            msg_type: HandshakeType::Certificate,
            length: body.len() as u32,
        }
        .encode(&mut out)
        .unwrap();
        out.extend_from_slice(&body);
        out
    }

    fn build_cert_verify(scheme: u16, sig: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&scheme.to_be_bytes());
        body.extend_from_slice(&(sig.len() as u16).to_be_bytes());
        body.extend_from_slice(sig);
        let mut out = Vec::new();
        HandshakeHeader {
            msg_type: HandshakeType::CertificateVerify,
            length: body.len() as u32,
        }
        .encode(&mut out)
        .unwrap();
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn certificate_round_trip() {
        let bytes = build_certificate(&[b"leaf-der-bytes", b"intermediate-der"]);
        let msg = parse_certificate(&bytes).unwrap();
        assert_eq!(msg.certs.len(), 2);
        assert_eq!(msg.certs[0], b"leaf-der-bytes");
        assert_eq!(msg.certs[1], b"intermediate-der");
    }

    #[test]
    fn certificate_empty_list_rejected() {
        assert_eq!(
            parse_certificate(&build_certificate(&[])).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn certificate_nonempty_context_rejected() {
        let mut bytes = build_certificate(&[b"leaf"]);
        // context length byte sits right after the 4-byte header;
        // claim 1 byte of context.
        bytes[4] = 1;
        assert!(parse_certificate(&bytes).is_err());
    }

    #[test]
    fn certificate_truncated_rejected() {
        let bytes = build_certificate(&[b"leaf-der-bytes"]);
        for cut in [5, 8, 12, bytes.len() - 1] {
            assert!(parse_certificate(&bytes[..cut]).is_err(), "cut={cut}");
        }
    }

    #[test]
    fn cert_verify_round_trip_ed25519() {
        let sig = [0xabu8; 64];
        let msg = parse_certificate_verify(&build_cert_verify(sig_scheme::ED25519, &sig)).unwrap();
        assert_eq!(msg.scheme, sig_scheme::ED25519);
        assert_eq!(msg.signature, sig);
    }

    #[test]
    fn cert_verify_sha1_schemes_rejected() {
        // rsa_pkcs1_sha1 (0x0201), ecdsa_sha1 (0x0203) — the spec's
        // "SHA-1 signature refused" scenario.
        for scheme in [0x0201u16, 0x0203, 0x0202 /* dsa_sha1 */] {
            assert_eq!(
                parse_certificate_verify(&build_cert_verify(scheme, &[0u8; 64])).unwrap_err(),
                TlsClientError::BadCertificate,
                "scheme {scheme:#06x}"
            );
        }
    }

    #[test]
    fn cert_verify_rsa_pss_parses() {
        // Allow-listed schemes parse fine (verification of non-Ed25519
        // schemes is a Phase 5 follow-on).
        for scheme in [
            sig_scheme::RSA_PSS_RSAE_SHA256,
            sig_scheme::RSA_PSS_RSAE_SHA384,
            sig_scheme::RSA_PSS_RSAE_SHA512,
            sig_scheme::ECDSA_SECP256R1_SHA256,
        ] {
            assert!(parse_certificate_verify(&build_cert_verify(scheme, &[0u8; 96])).is_ok());
        }
    }

    #[test]
    fn cert_verify_signature_verifies() {
        let kp = ed25519_keygen(&[7u8; 32]);
        let transcript_hash = [0x44u8; 32];
        let content = certificate_verify_content(&transcript_hash);
        let sig = ed25519_sign(&kp.secret_key, &content);
        let msg = parse_certificate_verify(&build_cert_verify(sig_scheme::ED25519, sig.as_bytes()))
            .unwrap();
        verify_certificate_verify(&msg, kp.public_key.as_bytes(), &transcript_hash).unwrap();
    }

    #[test]
    fn cert_verify_wrong_transcript_rejected() {
        let kp = ed25519_keygen(&[7u8; 32]);
        let content = certificate_verify_content(&[0x44u8; 32]);
        let sig = ed25519_sign(&kp.secret_key, &content);
        let msg = parse_certificate_verify(&build_cert_verify(sig_scheme::ED25519, sig.as_bytes()))
            .unwrap();
        assert_eq!(
            verify_certificate_verify(&msg, kp.public_key.as_bytes(), &[0x45u8; 32]).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn cert_verify_content_layout() {
        let content = certificate_verify_content(&[0xaa; 32]);
        assert_eq!(content.len(), 64 + 33 + 1 + 32);
        assert!(content[..64].iter().all(|&b| b == 0x20));
        assert_eq!(&content[64..97], b"TLS 1.3, server CertificateVerify");
        assert_eq!(content[97], 0);
        assert_eq!(&content[98..], &[0xaa; 32]);
    }
}
