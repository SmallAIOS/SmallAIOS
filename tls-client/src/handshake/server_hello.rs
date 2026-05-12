// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ServerHello parser (RFC 8446 § 4.1.3).
//!
//! Wire layout:
//! ```text
//! struct {
//!     ProtocolVersion legacy_version = 0x0303;
//!     Random random;                     // 32 bytes (last 8 bytes
//!                                        // MAY be the "downgrade
//!                                        // protection" sentinel)
//!     opaque legacy_session_id_echo<0..32>;
//!     CipherSuite cipher_suite;
//!     uint8 legacy_compression_method = 0;
//!     Extension extensions<6..2^16-1>;
//! } ServerHello;
//! ```
//!
//! Pinning is strict: the `supported_versions` extension MUST
//! select TLS 1.3 (`0x0304`); the `cipher_suite` MUST be one
//! we advertised; the `key_share` group MUST match what we
//! sent.

use super::extensions::{
    ext_type, parse_extensions_block, parse_key_share_server, parse_supported_versions_server,
    TLS_1_3,
};
use super::{HandshakeHeader, HandshakeType};
use crate::record::CipherSuite;
use crate::{Result, TlsClientError};
use alloc::vec::Vec;

/// Parsed ServerHello with the fields the state machine
/// needs for further handshake steps.
#[derive(Debug, Clone)]
pub struct ServerHello {
    /// 32 bytes of `random` from the wire. Caller checks the
    /// downgrade sentinel separately if PQC mode demands it.
    pub random: [u8; 32],
    /// Echoed session id (we typically send an empty one;
    /// servers may echo it).
    pub session_id_echo: Vec<u8>,
    /// Cipher suite the server selected. Must be one of the
    /// values the client advertised — `from_wire` enforces this.
    pub cipher_suite: CipherSuite,
    /// Server's key_share: `(group, public_key)`.
    pub key_share_group: u16,
    pub key_share_public: Vec<u8>,
    /// Selected TLS version (always 0x0304 — `parse` enforces).
    pub selected_version: u16,
}

/// Parse a ServerHello starting with the 4-byte handshake header.
pub fn parse_server_hello(buf: &[u8]) -> Result<ServerHello> {
    let header = HandshakeHeader::parse(buf)?;
    if header.msg_type != HandshakeType::ServerHello {
        return Err(TlsClientError::BadHandshake);
    }
    let body_start = HandshakeHeader::LEN;
    let body_end = body_start + header.length as usize;
    if buf.len() < body_end {
        return Err(TlsClientError::BadHandshake);
    }
    let body = &buf[body_start..body_end];

    // legacy_version field is 0x0303 (TLS 1.2 on the wire per
    // RFC 8446 § 4.1.3 — actual selected version is in the
    // supported_versions extension).
    if body.len() < 2 + 32 + 1 + 2 + 1 + 2 {
        return Err(TlsClientError::BadHandshake);
    }
    if body[0] != 0x03 || body[1] != 0x03 {
        return Err(TlsClientError::Version);
    }
    let mut cur = 2;

    let mut random = [0u8; 32];
    random.copy_from_slice(&body[cur..cur + 32]);
    cur += 32;

    let sid_len = body[cur] as usize;
    cur += 1;
    if sid_len > 32 || cur + sid_len > body.len() {
        return Err(TlsClientError::BadHandshake);
    }
    let session_id_echo = body[cur..cur + sid_len].to_vec();
    cur += sid_len;

    if cur + 2 > body.len() {
        return Err(TlsClientError::BadHandshake);
    }
    let cs_code = u16::from_be_bytes([body[cur], body[cur + 1]]);
    cur += 2;
    let cipher_suite = CipherSuite::from_wire(cs_code)?;

    if cur + 1 > body.len() {
        return Err(TlsClientError::BadHandshake);
    }
    if body[cur] != 0 {
        return Err(TlsClientError::BadHandshake);
    }
    cur += 1;

    let extensions = parse_extensions_block(&body[cur..])?;

    let mut selected_version: Option<u16> = None;
    let mut key_share_group: Option<u16> = None;
    let mut key_share_public: Option<Vec<u8>> = None;
    for ext in &extensions {
        match ext.ext_type {
            t if t == ext_type::SUPPORTED_VERSIONS => {
                if selected_version.is_some() {
                    return Err(TlsClientError::BadHandshake);
                }
                selected_version = Some(parse_supported_versions_server(ext.data)?);
            }
            t if t == ext_type::KEY_SHARE => {
                if key_share_group.is_some() {
                    return Err(TlsClientError::BadHandshake);
                }
                let (g, k) = parse_key_share_server(ext.data)?;
                key_share_group = Some(g);
                key_share_public = Some(k.to_vec());
            }
            _ => {
                // Server MAY send other extensions in
                // ServerHello (e.g., pre_shared_key for PSK
                // resumption — out of scope in v1). Ignore
                // them for now.
            }
        }
    }

    let selected_version = selected_version.ok_or(TlsClientError::Version)?;
    if selected_version != TLS_1_3 {
        return Err(TlsClientError::Version);
    }
    let key_share_group = key_share_group.ok_or(TlsClientError::BadHandshake)?;
    let key_share_public = key_share_public.ok_or(TlsClientError::BadHandshake)?;

    Ok(ServerHello {
        random,
        session_id_echo,
        cipher_suite,
        key_share_group,
        key_share_public,
        selected_version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handshake::extensions::{ext_type, named_group};

    /// Build a synthetic ServerHello for testing the parser.
    fn build_server_hello(
        cipher_suite_code: u16,
        selected_version: Option<u16>,
        key_share_group: u16,
        key_share_pk: &[u8],
    ) -> Vec<u8> {
        let mut body = Vec::new();
        // legacy_version
        body.extend_from_slice(&[0x03, 0x03]);
        // random
        body.extend_from_slice(&[0xaa; 32]);
        // session_id_echo: empty
        body.push(0);
        // cipher_suite
        body.extend_from_slice(&cipher_suite_code.to_be_bytes());
        // legacy_compression_method
        body.push(0);
        // extensions block
        let mut exts = Vec::new();
        if let Some(v) = selected_version {
            // supported_versions extension (server form: bare u16)
            exts.extend_from_slice(&ext_type::SUPPORTED_VERSIONS.to_be_bytes());
            exts.extend_from_slice(&2u16.to_be_bytes());
            exts.extend_from_slice(&v.to_be_bytes());
        }
        // key_share
        let mut ks_body = Vec::new();
        ks_body.extend_from_slice(&key_share_group.to_be_bytes());
        ks_body.extend_from_slice(&(key_share_pk.len() as u16).to_be_bytes());
        ks_body.extend_from_slice(key_share_pk);
        exts.extend_from_slice(&ext_type::KEY_SHARE.to_be_bytes());
        exts.extend_from_slice(&(ks_body.len() as u16).to_be_bytes());
        exts.extend_from_slice(&ks_body);
        // outer length
        body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
        body.extend_from_slice(&exts);

        // 4-byte handshake header
        let mut out = Vec::new();
        let hdr = HandshakeHeader {
            msg_type: HandshakeType::ServerHello,
            length: body.len() as u32,
        };
        hdr.encode(&mut out).unwrap();
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn parse_ok_minimal() {
        let pk = [0xbb; 32];
        let bytes = build_server_hello(0x1302, Some(TLS_1_3), named_group::X25519, &pk);
        let sh = parse_server_hello(&bytes).unwrap();
        assert_eq!(sh.cipher_suite, CipherSuite::Aes256GcmSha384);
        assert_eq!(sh.key_share_group, named_group::X25519);
        assert_eq!(sh.key_share_public.len(), 32);
        assert_eq!(sh.selected_version, TLS_1_3);
    }

    #[test]
    fn parse_rejects_tls12_selection() {
        let pk = [0xbb; 32];
        let bytes = build_server_hello(0x1302, Some(0x0303), named_group::X25519, &pk);
        assert_eq!(
            parse_server_hello(&bytes).unwrap_err(),
            TlsClientError::Version
        );
    }

    #[test]
    fn parse_rejects_missing_supported_versions() {
        let pk = [0xbb; 32];
        let bytes = build_server_hello(0x1302, None, named_group::X25519, &pk);
        assert_eq!(
            parse_server_hello(&bytes).unwrap_err(),
            TlsClientError::Version
        );
    }

    #[test]
    fn parse_rejects_unsupported_cipher_suite() {
        let pk = [0xbb; 32];
        let bytes = build_server_hello(0x1301, Some(TLS_1_3), named_group::X25519, &pk);
        assert_eq!(
            parse_server_hello(&bytes).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn parse_rejects_non_zero_compression_method() {
        let pk = [0xbb; 32];
        let mut bytes = build_server_hello(0x1302, Some(TLS_1_3), named_group::X25519, &pk);
        // Compression method byte sits at offset header(4) + version(2)
        // + random(32) + session_id_len(1) + session_id(0) +
        // cipher_suite(2) = 41. Flip it from 0 to 1.
        bytes[41] = 1;
        assert!(parse_server_hello(&bytes).is_err());
    }

    #[test]
    fn parse_rejects_wrong_legacy_version() {
        let pk = [0xbb; 32];
        let mut bytes = build_server_hello(0x1302, Some(TLS_1_3), named_group::X25519, &pk);
        // legacy_version sits at offset 4..6.
        bytes[4] = 0x03;
        bytes[5] = 0x02;
        assert_eq!(
            parse_server_hello(&bytes).unwrap_err(),
            TlsClientError::Version
        );
    }

    #[test]
    fn parse_rejects_truncated() {
        let pk = [0xbb; 32];
        let bytes = build_server_hello(0x1302, Some(TLS_1_3), named_group::X25519, &pk);
        let truncated = &bytes[..bytes.len() / 2];
        assert!(parse_server_hello(truncated).is_err());
    }

    #[test]
    fn parse_pqc_hybrid_key_share() {
        // Real hybrid pubkey is 32 + 1184 = 1216 bytes; use the
        // shape for the test.
        let mut pk = alloc::vec![0xcc; 1216];
        pk[0] = 0x42;
        let bytes = build_server_hello(0x1302, Some(TLS_1_3), named_group::X25519_MLKEM768, &pk);
        let sh = parse_server_hello(&bytes).unwrap();
        assert_eq!(sh.key_share_group, named_group::X25519_MLKEM768);
        assert_eq!(sh.key_share_public.len(), 1216);
    }
}
