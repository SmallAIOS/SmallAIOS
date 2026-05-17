// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ClientHello builder (RFC 8446 § 4.1.2).
//!
//! Wire layout:
//! ```text
//! struct {
//!     ProtocolVersion legacy_version = 0x0303;
//!     Random random;                     // 32 bytes
//!     opaque legacy_session_id<0..32>;
//!     CipherSuite cipher_suites<2..2^16-2>;
//!     opaque legacy_compression_methods<1..2^8-1> = [0];
//!     Extension extensions<8..2^16-1>;
//! } ClientHello;
//! ```
//!
//! The 4-byte handshake header is prepended by the caller.

use super::extensions::{
    self, build_key_share_client, build_server_name, build_signature_algorithms_default,
    build_supported_groups, build_supported_versions_client, emit_extension, ext_type, named_group,
    KeyShareEntry,
};
use super::{HandshakeHeader, HandshakeType};
use crate::record::CipherSuite;
use crate::{Result, TlsClientError};
use alloc::vec::Vec;

/// What goes into the client_hello key_share extension. Caller
/// owns the actual byte buffer for the public key (so we can
/// stay `#![no_std]` without baking in a CSPRNG).
pub struct ClientHelloKeyShare<'a> {
    pub group: u16,
    pub public_key: &'a [u8],
}

/// Inputs the caller provides for `build_client_hello`.
pub struct ClientHelloInput<'a> {
    /// 32-byte ClientHello.random. Caller is responsible for
    /// CSPRNG-generating this.
    pub random: [u8; 32],
    /// Optional SNI hostname; pass `None` for IP-literal
    /// endpoints (RFC 6066 § 3 says SNI MUST NOT be sent then).
    pub server_name: Option<&'a str>,
    /// Cipher-suite preference list, advertised in order.
    pub cipher_suites: &'a [CipherSuite],
    /// supported_groups list, advertised in order. PQC hybrid
    /// goes first when `require_pqc = true`.
    pub supported_groups: &'a [u16],
    /// Exactly one key_share entry today (the primary group).
    pub key_share: ClientHelloKeyShare<'a>,
}

/// Build the full ClientHello message including the 4-byte
/// handshake header (`type=ClientHello || length:u24`).
pub fn build_client_hello(input: &ClientHelloInput) -> Result<Vec<u8>> {
    // First build the body, then prepend the header.
    let mut body = Vec::new();

    // legacy_version = TLS 1.2 (RFC 8446 § 4.1.2 mandates this on
    // the wire for TLS 1.3 ClientHello).
    body.extend_from_slice(&0x0303u16.to_be_bytes());

    // 32-byte random.
    body.extend_from_slice(&input.random);

    // legacy_session_id: opaque <0..32> — TLS 1.3 sends 0 bytes.
    // (RFC 8446 § 4.1.2 allows an empty list or compat values; we
    // pick 0 bytes for the simplest interoperable form.)
    body.push(0);

    // cipher_suites: opaque <2..2^16-2>, each suite 2 bytes.
    let cs_len = (input.cipher_suites.len() * 2) as u16;
    body.extend_from_slice(&cs_len.to_be_bytes());
    for cs in input.cipher_suites {
        body.extend_from_slice(&cs.wire_value().to_be_bytes());
    }

    // legacy_compression_methods = [0]. Length-1 prefix.
    body.push(1);
    body.push(0);

    // Extensions block (outer u16-prefixed).
    let mut exts = Vec::new();
    emit_extension(
        &mut exts,
        ext_type::SUPPORTED_VERSIONS,
        &build_supported_versions_client(),
    )?;
    emit_extension(
        &mut exts,
        ext_type::SUPPORTED_GROUPS,
        &build_supported_groups(input.supported_groups)?,
    )?;
    emit_extension(
        &mut exts,
        ext_type::SIGNATURE_ALGORITHMS,
        &build_signature_algorithms_default(),
    )?;
    let ks_entry = KeyShareEntry {
        group: input.key_share.group,
        key_exchange: input.key_share.public_key,
    };
    emit_extension(
        &mut exts,
        ext_type::KEY_SHARE,
        &build_key_share_client(&[ks_entry])?,
    )?;
    if let Some(host) = input.server_name {
        emit_extension(&mut exts, ext_type::SERVER_NAME, &build_server_name(host)?)?;
    }
    if exts.len() > 0xffff {
        return Err(TlsClientError::BadHandshake);
    }
    body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
    body.extend_from_slice(&exts);

    // 4-byte handshake header prepended.
    let mut out = Vec::with_capacity(HandshakeHeader::LEN + body.len());
    let hdr = HandshakeHeader {
        msg_type: HandshakeType::ClientHello,
        length: body.len() as u32,
    };
    hdr.encode(&mut out)?;
    out.extend_from_slice(&body);
    Ok(out)
}

/// Suppress dead-code warnings on the imports until the
/// state-machine driver consumes them.
#[allow(dead_code)]
const _USED: () = {
    let _ = extensions::TLS_1_3;
    let _ = named_group::X25519_MLKEM768;
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> ([u8; 32], [u8; 32]) {
        ([0x42u8; 32], [0xcc; 32])
    }

    #[test]
    fn build_client_hello_minimal() {
        let (random, pk) = sample_input();
        let suites = [CipherSuite::Aes256GcmSha384];
        let groups = [named_group::X25519];
        let out = build_client_hello(&ClientHelloInput {
            random,
            server_name: Some("immudb.example.com"),
            cipher_suites: &suites,
            supported_groups: &groups,
            key_share: ClientHelloKeyShare {
                group: named_group::X25519,
                public_key: &pk,
            },
        })
        .unwrap();
        // Header: type=1 ClientHello, length=body_len.
        assert_eq!(out[0], 1);
        let body_len = u32::from_be_bytes([0, out[1], out[2], out[3]]) as usize;
        assert_eq!(body_len + 4, out.len());
        // legacy_version = 0x0303.
        assert_eq!(&out[4..6], &[0x03, 0x03]);
        // Random follows.
        assert_eq!(&out[6..38], &random);
        // legacy_session_id length = 0.
        assert_eq!(out[38], 0);
    }

    #[test]
    fn build_client_hello_omits_sni_when_none() {
        let (random, pk) = sample_input();
        let suites = [CipherSuite::ChaCha20Poly1305Sha256];
        let groups = [named_group::X25519];
        let out = build_client_hello(&ClientHelloInput {
            random,
            server_name: None,
            cipher_suites: &suites,
            supported_groups: &groups,
            key_share: ClientHelloKeyShare {
                group: named_group::X25519,
                public_key: &pk,
            },
        })
        .unwrap();
        // Build the same input WITH SNI for comparison.
        let with_sni = build_client_hello(&ClientHelloInput {
            random,
            server_name: Some("immudb.example.com"),
            cipher_suites: &suites,
            supported_groups: &groups,
            key_share: ClientHelloKeyShare {
                group: named_group::X25519,
                public_key: &pk,
            },
        })
        .unwrap();
        // The without-SNI ClientHello MUST be shorter and MUST NOT
        // contain the hostname bytes.
        assert!(out.len() < with_sni.len());
        assert!(!out.windows(18).any(|w| w == b"immudb.example.com"));
    }

    #[test]
    fn build_client_hello_two_suites() {
        let (random, pk) = sample_input();
        let suites = [
            CipherSuite::Aes256GcmSha384,
            CipherSuite::ChaCha20Poly1305Sha256,
        ];
        let groups = [named_group::X25519];
        let out = build_client_hello(&ClientHelloInput {
            random,
            server_name: None,
            cipher_suites: &suites,
            supported_groups: &groups,
            key_share: ClientHelloKeyShare {
                group: named_group::X25519,
                public_key: &pk,
            },
        })
        .unwrap();
        // After header(4) + version(2) + random(32) + session_id(1) =
        // offset 39, we have cipher_suites: u16 length + 4 bytes
        // (two suites of 2 bytes each).
        assert_eq!(&out[39..41], &[0x00, 0x04]);
        // First suite = AES_256_GCM_SHA384 (0x1302).
        assert_eq!(&out[41..43], &[0x13, 0x02]);
        // Second suite = CHACHA20_POLY1305_SHA256 (0x1303).
        assert_eq!(&out[43..45], &[0x13, 0x03]);
    }

    #[test]
    fn build_client_hello_pqc_supported_groups_lists_hybrid_first() {
        let (random, pk) = sample_input();
        // For a real hybrid share the public key would be 32 + 1184
        // bytes; we use 32 here just to confirm the wire shape.
        let suites = [CipherSuite::Aes256GcmSha384];
        let groups = [named_group::X25519_MLKEM768, named_group::X25519];
        let out = build_client_hello(&ClientHelloInput {
            random,
            server_name: None,
            cipher_suites: &suites,
            supported_groups: &groups,
            key_share: ClientHelloKeyShare {
                group: named_group::X25519_MLKEM768,
                public_key: &pk,
            },
        })
        .unwrap();
        // Confirm hybrid group bytes appear before classical bytes.
        let mut iter = out.windows(2);
        let hybrid_pos = iter
            .position(|w| w == [0x11, 0xec])
            .expect("hybrid group present");
        let x25519_pos = out
            .windows(2)
            .position(|w| w == [0x00, 0x1d])
            .expect("x25519 group present");
        assert!(hybrid_pos < x25519_pos);
    }
}
