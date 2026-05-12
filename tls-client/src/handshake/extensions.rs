// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TLS 1.3 extension encoders / parsers.
//!
//! The client emits exactly the extension set
//! `tls-client-handshake` declares:
//!
//! - `supported_versions` — pins TLS 1.3.
//! - `supported_groups` — `x25519`, optionally `X25519MLKEM768`
//!   when `require_pqc = true`.
//! - `signature_algorithms` — the CertificateVerify allow-list.
//! - `key_share` — one entry matching the primary
//!   supported_group.
//! - `server_name` — when the endpoint is a DNS name.

use crate::{Result, TlsClientError};
use alloc::vec::Vec;

/// Extension type codes from RFC 8446 § 4.2.
pub mod ext_type {
    pub const SERVER_NAME: u16 = 0x0000;
    pub const SUPPORTED_GROUPS: u16 = 0x000a;
    pub const SIGNATURE_ALGORITHMS: u16 = 0x000d;
    pub const SUPPORTED_VERSIONS: u16 = 0x002b;
    pub const KEY_SHARE: u16 = 0x0033;
}

/// Named-group codes the client emits or accepts.
pub mod named_group {
    /// `x25519` (RFC 7748).
    pub const X25519: u16 = 0x001d;
    /// `X25519MLKEM768` hybrid (draft-ietf-tls-hybrid-design).
    pub const X25519_MLKEM768: u16 = 0x11ec;
}

/// Signature scheme codes the client advertises and accepts
/// (RFC 8446 § 4.2.3).
pub mod sig_scheme {
    /// `ed25519`.
    pub const ED25519: u16 = 0x0807;
    /// `ecdsa_secp256r1_sha256`.
    pub const ECDSA_SECP256R1_SHA256: u16 = 0x0403;
    /// `rsa_pss_rsae_sha256`.
    pub const RSA_PSS_RSAE_SHA256: u16 = 0x0804;
    /// `rsa_pss_rsae_sha384`.
    pub const RSA_PSS_RSAE_SHA384: u16 = 0x0805;
    /// `rsa_pss_rsae_sha512`.
    pub const RSA_PSS_RSAE_SHA512: u16 = 0x0806;
}

/// The TLS 1.3 version code carried inside `supported_versions`.
pub const TLS_1_3: u16 = 0x0304;

/// Helper: write a length-prefixed body. `prefix_len_bytes`
/// is 1 for u8, 2 for u16, 3 for u24.
fn emit_with_length<F: FnOnce(&mut Vec<u8>) -> Result<()>>(
    out: &mut Vec<u8>,
    prefix_len_bytes: usize,
    write_body: F,
) -> Result<()> {
    let len_pos = out.len();
    // Reserve space.
    for _ in 0..prefix_len_bytes {
        out.push(0);
    }
    let body_start = out.len();
    write_body(out)?;
    let body_len = out.len() - body_start;
    let max = match prefix_len_bytes {
        1 => 0xff,
        2 => 0xffff,
        3 => 0x00ff_ffff,
        _ => return Err(TlsClientError::BadHandshake),
    };
    if body_len > max {
        return Err(TlsClientError::BadHandshake);
    }
    match prefix_len_bytes {
        1 => out[len_pos] = body_len as u8,
        2 => {
            let l = (body_len as u16).to_be_bytes();
            out[len_pos] = l[0];
            out[len_pos + 1] = l[1];
        }
        3 => {
            let l = (body_len as u32).to_be_bytes();
            out[len_pos] = l[1];
            out[len_pos + 1] = l[2];
            out[len_pos + 2] = l[3];
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Emit a single extension: `type:u16 || data:opaque<u16>`.
pub fn emit_extension(out: &mut Vec<u8>, ext_type: u16, data: &[u8]) -> Result<()> {
    out.extend_from_slice(&ext_type.to_be_bytes());
    if data.len() > 0xffff {
        return Err(TlsClientError::BadHandshake);
    }
    out.extend_from_slice(&(data.len() as u16).to_be_bytes());
    out.extend_from_slice(data);
    Ok(())
}

/// Build the `supported_versions` extension body (client form):
/// `versions:opaque<u8>` containing only TLS 1.3.
pub fn build_supported_versions_client() -> Vec<u8> {
    let mut body = Vec::new();
    // One u8-prefixed list with a single 2-byte version.
    body.push(2); // length of inner list
    body.extend_from_slice(&TLS_1_3.to_be_bytes());
    body
}

/// Parse the server's `supported_versions` extension. The
/// server form is `selected_version:u16` only (no list).
pub fn parse_supported_versions_server(data: &[u8]) -> Result<u16> {
    if data.len() != 2 {
        return Err(TlsClientError::BadHandshake);
    }
    Ok(u16::from_be_bytes([data[0], data[1]]))
}

/// Build the `supported_groups` extension body.
pub fn build_supported_groups(groups: &[u16]) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    emit_with_length(&mut body, 2, |o| {
        for g in groups {
            o.extend_from_slice(&g.to_be_bytes());
        }
        Ok(())
    })?;
    Ok(body)
}

/// Build the `signature_algorithms` extension body using the
/// allow-list declared in `tls-client-handshake` spec.
pub fn build_signature_algorithms_default() -> Vec<u8> {
    let schemes = &[
        sig_scheme::ED25519,
        sig_scheme::ECDSA_SECP256R1_SHA256,
        sig_scheme::RSA_PSS_RSAE_SHA256,
        sig_scheme::RSA_PSS_RSAE_SHA384,
        sig_scheme::RSA_PSS_RSAE_SHA512,
    ];
    let mut body = Vec::new();
    emit_with_length(&mut body, 2, |o| {
        for s in schemes {
            o.extend_from_slice(&s.to_be_bytes());
        }
        Ok(())
    })
    .expect("schemes fit in u16");
    body
}

/// One entry in the client `key_share` extension.
pub struct KeyShareEntry<'a> {
    pub group: u16,
    pub key_exchange: &'a [u8],
}

/// Build the client `key_share` extension body:
/// `client_shares:opaque<u16>` of `(group:u16 ||
/// key_exchange:opaque<u16>)` entries.
pub fn build_key_share_client(shares: &[KeyShareEntry]) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    emit_with_length(&mut body, 2, |o| {
        for share in shares {
            o.extend_from_slice(&share.group.to_be_bytes());
            if share.key_exchange.len() > 0xffff {
                return Err(TlsClientError::BadHandshake);
            }
            o.extend_from_slice(&(share.key_exchange.len() as u16).to_be_bytes());
            o.extend_from_slice(share.key_exchange);
        }
        Ok(())
    })?;
    Ok(body)
}

/// Parse the server `key_share` extension: a single
/// `(group:u16 || key_exchange:opaque<u16>)` entry.
pub fn parse_key_share_server(data: &[u8]) -> Result<(u16, &[u8])> {
    if data.len() < 4 {
        return Err(TlsClientError::BadHandshake);
    }
    let group = u16::from_be_bytes([data[0], data[1]]);
    let len = u16::from_be_bytes([data[2], data[3]]) as usize;
    if data.len() != 4 + len {
        return Err(TlsClientError::BadHandshake);
    }
    Ok((group, &data[4..]))
}

/// Build the `server_name` extension body (RFC 6066 § 3) for
/// a DNS host. `host` MUST be ASCII; non-ASCII names get the
/// punycode treatment outside this layer.
pub fn build_server_name(host: &str) -> Result<Vec<u8>> {
    if !host.is_ascii() || host.is_empty() || host.len() > 0xffff {
        return Err(TlsClientError::BadHandshake);
    }
    let mut body = Vec::new();
    emit_with_length(&mut body, 2, |o| {
        o.push(0); // name_type = host_name(0)
        o.extend_from_slice(&(host.len() as u16).to_be_bytes());
        o.extend_from_slice(host.as_bytes());
        Ok(())
    })?;
    Ok(body)
}

/// Parsed extension as `(type, body slice)`.
#[derive(Debug, Clone, Copy)]
pub struct ParsedExt<'a> {
    pub ext_type: u16,
    pub data: &'a [u8],
}

/// Parse an extensions block (`extensions:opaque<u16>`).
/// Returns a vector of `(type, slice)`. Duplicate extension
/// types are NOT rejected here — the caller checks per-type
/// semantics.
pub fn parse_extensions_block(buf: &[u8]) -> Result<Vec<ParsedExt<'_>>> {
    if buf.len() < 2 {
        return Err(TlsClientError::BadHandshake);
    }
    let total = u16::from_be_bytes([buf[0], buf[1]]) as usize;
    if buf.len() != 2 + total {
        return Err(TlsClientError::BadHandshake);
    }
    let mut cur = 2;
    let mut out = Vec::new();
    while cur < buf.len() {
        if buf.len() - cur < 4 {
            return Err(TlsClientError::BadHandshake);
        }
        let ext_type = u16::from_be_bytes([buf[cur], buf[cur + 1]]);
        let len = u16::from_be_bytes([buf[cur + 2], buf[cur + 3]]) as usize;
        cur += 4;
        if buf.len() - cur < len {
            return Err(TlsClientError::BadHandshake);
        }
        out.push(ParsedExt {
            ext_type,
            data: &buf[cur..cur + len],
        });
        cur += len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_versions_client_lists_only_tls13() {
        let body = build_supported_versions_client();
        assert_eq!(body, alloc::vec![0x02, 0x03, 0x04]);
    }

    #[test]
    fn supported_versions_server_parses_tls13() {
        let v = parse_supported_versions_server(&[0x03, 0x04]).unwrap();
        assert_eq!(v, TLS_1_3);
    }

    #[test]
    fn supported_versions_server_wrong_length_rejected() {
        assert!(parse_supported_versions_server(&[0x03]).is_err());
        assert!(parse_supported_versions_server(&[0x03, 0x04, 0x05]).is_err());
    }

    #[test]
    fn supported_groups_default_x25519() {
        let body = build_supported_groups(&[named_group::X25519]).unwrap();
        // u16 list length + 1 group = 2 + 2 = 4 bytes.
        assert_eq!(body, alloc::vec![0x00, 0x02, 0x00, 0x1d]);
    }

    #[test]
    fn supported_groups_pqc_first() {
        let body =
            build_supported_groups(&[named_group::X25519_MLKEM768, named_group::X25519]).unwrap();
        assert_eq!(body, alloc::vec![0x00, 0x04, 0x11, 0xec, 0x00, 0x1d]);
    }

    #[test]
    fn signature_algorithms_default_includes_ed25519_first() {
        let body = build_signature_algorithms_default();
        // u16 list length + 5 schemes * 2 = 2 + 10 = 12 bytes.
        assert_eq!(body.len(), 12);
        // First scheme in body[2..4] is ed25519.
        assert_eq!(&body[2..4], &[0x08, 0x07]);
    }

    #[test]
    fn key_share_client_one_entry() {
        let pk = [0xaa; 32];
        let body = build_key_share_client(&[KeyShareEntry {
            group: named_group::X25519,
            key_exchange: &pk,
        }])
        .unwrap();
        // u16 outer length + (u16 group + u16 inner length + 32 pk)
        // = 2 + 2 + 2 + 32 = 38 bytes.
        assert_eq!(body.len(), 38);
        assert_eq!(&body[..2], &[0x00, 36]);
        assert_eq!(&body[2..4], &[0x00, 0x1d]);
        assert_eq!(&body[4..6], &[0x00, 32]);
    }

    #[test]
    fn key_share_server_parse_single_entry() {
        let mut input = Vec::new();
        input.extend_from_slice(&named_group::X25519.to_be_bytes());
        input.extend_from_slice(&32u16.to_be_bytes());
        input.extend_from_slice(&[0xbb; 32]);
        let (group, key) = parse_key_share_server(&input).unwrap();
        assert_eq!(group, named_group::X25519);
        assert_eq!(key.len(), 32);
        assert_eq!(key[0], 0xbb);
    }

    #[test]
    fn server_name_dns_round_trip() {
        let body = build_server_name("immudb.example.com").unwrap();
        // outer u16 + name_type u8 + name length u16 + 18 bytes name.
        assert_eq!(body.len(), 2 + 1 + 2 + 18);
        assert_eq!(body[2], 0); // name_type = host_name(0)
        assert_eq!(&body[3..5], &[0x00, 0x12]); // length = 18
        assert_eq!(&body[5..], b"immudb.example.com");
    }

    #[test]
    fn server_name_non_ascii_rejected() {
        assert!(build_server_name("ünicode.example.com").is_err());
    }

    #[test]
    fn server_name_empty_rejected() {
        assert!(build_server_name("").is_err());
    }

    #[test]
    fn emit_extension_round_trip() {
        let mut buf = Vec::new();
        emit_extension(&mut buf, ext_type::SERVER_NAME, b"abc").unwrap();
        assert_eq!(&buf[..2], &[0x00, 0x00]);
        assert_eq!(&buf[2..4], &[0x00, 0x03]);
        assert_eq!(&buf[4..], b"abc");
    }

    #[test]
    fn parse_extensions_block_two_extensions() {
        // Outer u16 length wraps two extensions.
        let mut inner = Vec::new();
        emit_extension(&mut inner, ext_type::SUPPORTED_VERSIONS, &[0x03, 0x04]).unwrap();
        emit_extension(&mut inner, ext_type::KEY_SHARE, &[0xaa, 0xbb]).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(inner.len() as u16).to_be_bytes());
        buf.extend_from_slice(&inner);
        let parsed = parse_extensions_block(&buf).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].ext_type, ext_type::SUPPORTED_VERSIONS);
        assert_eq!(parsed[0].data, &[0x03, 0x04]);
        assert_eq!(parsed[1].ext_type, ext_type::KEY_SHARE);
        assert_eq!(parsed[1].data, &[0xaa, 0xbb]);
    }

    #[test]
    fn parse_extensions_block_truncated_rejected() {
        // Outer length claims 10 bytes; only 5 follow.
        let bad = [0x00, 0x0a, 0x00, 0x00, 0x00, 0x01, 0xaa];
        assert!(parse_extensions_block(&bad).is_err());
    }

    #[test]
    fn parse_extensions_block_empty_ok() {
        let buf = [0x00, 0x00];
        let parsed = parse_extensions_block(&buf).unwrap();
        assert!(parsed.is_empty());
    }
}
