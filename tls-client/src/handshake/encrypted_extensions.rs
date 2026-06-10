// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! EncryptedExtensions parser (RFC 8446 §4.3.1).
//!
//! ```text
//! struct {
//!     Extension extensions<0..2^16-1>;
//! } EncryptedExtensions;
//! ```
//!
//! The server uses this message for extensions that are not needed
//! to establish keys. An empty `server_name` extension here is the
//! server's acknowledgement that it used our SNI (RFC 6066 §3).
//! Extensions that RFC 8446 §4.2 pins to other messages
//! (`key_share`, `supported_versions`, `pre_shared_key`) are
//! illegal in EncryptedExtensions and refused.

use super::extensions::{ext_type, parse_extensions_block};
use super::{HandshakeHeader, HandshakeType};
use crate::{Result, TlsClientError};

/// Extension type codepoints RFC 8446 §4.2 forbids in
/// EncryptedExtensions (the ones a TLS 1.3 client could otherwise
/// see): key_share, supported_versions, pre_shared_key.
const FORBIDDEN_IN_EE: [u16; 3] = [ext_type::KEY_SHARE, ext_type::SUPPORTED_VERSIONS, 0x0029];

/// Parsed EncryptedExtensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncryptedExtensions {
    /// `true` when the server echoed an empty `server_name`
    /// extension, acknowledging the SNI we sent (task 4.5).
    pub sni_acked: bool,
}

/// Parse an EncryptedExtensions message starting at its 4-byte
/// handshake header.
pub fn parse_encrypted_extensions(buf: &[u8]) -> Result<EncryptedExtensions> {
    let header = HandshakeHeader::parse(buf)?;
    if header.msg_type != HandshakeType::EncryptedExtensions {
        return Err(TlsClientError::BadHandshake);
    }
    let body_start = HandshakeHeader::LEN;
    let body_end = body_start + header.length as usize;
    if buf.len() < body_end {
        return Err(TlsClientError::BadHandshake);
    }
    let extensions = parse_extensions_block(&buf[body_start..body_end])?;

    let mut sni_acked = false;
    for ext in &extensions {
        if FORBIDDEN_IN_EE.contains(&ext.ext_type) {
            return Err(TlsClientError::BadHandshake);
        }
        if ext.ext_type == ext_type::SERVER_NAME {
            // The ack form is an empty extension (RFC 6066 §3).
            if !ext.data.is_empty() {
                return Err(TlsClientError::BadHandshake);
            }
            sni_acked = true;
        }
        // Other extensions (max_fragment_length, ALPN,
        // supported_groups, ...) are tolerated and ignored in v1.
    }
    Ok(EncryptedExtensions { sni_acked })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn build_ee(exts: &[(u16, &[u8])]) -> Vec<u8> {
        let mut block = Vec::new();
        for (t, data) in exts {
            block.extend_from_slice(&t.to_be_bytes());
            block.extend_from_slice(&(data.len() as u16).to_be_bytes());
            block.extend_from_slice(data);
        }
        let mut body = Vec::new();
        body.extend_from_slice(&(block.len() as u16).to_be_bytes());
        body.extend_from_slice(&block);
        let mut out = Vec::new();
        HandshakeHeader {
            msg_type: HandshakeType::EncryptedExtensions,
            length: body.len() as u32,
        }
        .encode(&mut out)
        .unwrap();
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn empty_ee_parses() {
        let ee = parse_encrypted_extensions(&build_ee(&[])).unwrap();
        assert!(!ee.sni_acked);
    }

    #[test]
    fn sni_ack_detected() {
        let ee = parse_encrypted_extensions(&build_ee(&[(ext_type::SERVER_NAME, &[])])).unwrap();
        assert!(ee.sni_acked);
    }

    #[test]
    fn non_empty_server_name_rejected() {
        assert_eq!(
            parse_encrypted_extensions(&build_ee(&[(ext_type::SERVER_NAME, b"x")])).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn forbidden_extensions_rejected() {
        for t in [ext_type::KEY_SHARE, ext_type::SUPPORTED_VERSIONS, 0x0029] {
            assert_eq!(
                parse_encrypted_extensions(&build_ee(&[(t, &[0, 0])])).unwrap_err(),
                TlsClientError::BadHandshake,
                "ext_type {t:#06x} must be refused in EE"
            );
        }
    }

    #[test]
    fn unknown_extensions_tolerated() {
        // ALPN (16) and max_fragment_length (1) are fine.
        let ee = parse_encrypted_extensions(&build_ee(&[(16, b"\x00\x05\x04h2-x"), (1, b"\x01")]))
            .unwrap();
        assert!(!ee.sni_acked);
    }

    #[test]
    fn wrong_message_type_rejected() {
        let mut bytes = build_ee(&[]);
        bytes[0] = HandshakeType::Finished as u8;
        assert_eq!(
            parse_encrypted_extensions(&bytes).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn truncated_rejected() {
        let bytes = build_ee(&[(ext_type::SERVER_NAME, &[])]);
        assert!(parse_encrypted_extensions(&bytes[..bytes.len() - 1]).is_err());
    }
}
