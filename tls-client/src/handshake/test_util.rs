// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for `handshake` unit tests.
//!
//! One copy of the synthetic server-flight message builders and the
//! hex formatter, consumed by the `server_hello`,
//! `encrypted_extensions`, `certificate_msg`, `key_schedule`, and
//! `driver` test modules — so the wire-building logic under test
//! exists exactly once and the per-file test harnesses cannot drift.

#![cfg(test)]

use super::extensions::ext_type;
use super::{HandshakeHeader, HandshakeType};
use alloc::string::String;
use alloc::vec::Vec;

/// Lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

/// Prepend the 4-byte handshake header to `body`.
pub(crate) fn wrap_handshake(msg_type: HandshakeType, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HandshakeHeader::LEN + body.len());
    HandshakeHeader {
        msg_type,
        length: body.len() as u32,
    }
    .encode(&mut out)
    .unwrap();
    out.extend_from_slice(body);
    out
}

/// Build a synthetic ServerHello message (header included).
/// `selected_version = None` omits the supported_versions extension.
pub(crate) fn build_server_hello(
    cipher_suite_code: u16,
    selected_version: Option<u16>,
    key_share_group: u16,
    key_share_pk: &[u8],
    random: [u8; 32],
) -> Vec<u8> {
    build_server_hello_with_session_id(
        cipher_suite_code,
        selected_version,
        key_share_group,
        key_share_pk,
        random,
        &[],
    )
}

/// [`build_server_hello`] with a caller-chosen
/// `legacy_session_id_echo`. The client under test always sends an
/// empty `legacy_session_id`, so tests pass a non-empty echo here
/// to provoke the RFC 8446 §4.1.3 mismatch abort.
pub(crate) fn build_server_hello_with_session_id(
    cipher_suite_code: u16,
    selected_version: Option<u16>,
    key_share_group: u16,
    key_share_pk: &[u8],
    random: [u8; 32],
    session_id_echo: &[u8],
) -> Vec<u8> {
    let mut body = Vec::new();
    // legacy_version
    body.extend_from_slice(&[0x03, 0x03]);
    body.extend_from_slice(&random);
    // session_id_echo
    body.push(session_id_echo.len() as u8);
    body.extend_from_slice(session_id_echo);
    body.extend_from_slice(&cipher_suite_code.to_be_bytes());
    // legacy_compression_method
    body.push(0);
    let mut exts = Vec::new();
    if let Some(v) = selected_version {
        // supported_versions extension (server form: bare u16)
        exts.extend_from_slice(&ext_type::SUPPORTED_VERSIONS.to_be_bytes());
        exts.extend_from_slice(&2u16.to_be_bytes());
        exts.extend_from_slice(&v.to_be_bytes());
    }
    let mut ks_body = Vec::new();
    ks_body.extend_from_slice(&key_share_group.to_be_bytes());
    ks_body.extend_from_slice(&(key_share_pk.len() as u16).to_be_bytes());
    ks_body.extend_from_slice(key_share_pk);
    exts.extend_from_slice(&ext_type::KEY_SHARE.to_be_bytes());
    exts.extend_from_slice(&(ks_body.len() as u16).to_be_bytes());
    exts.extend_from_slice(&ks_body);
    body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
    body.extend_from_slice(&exts);
    wrap_handshake(HandshakeType::ServerHello, &body)
}

/// Build an EncryptedExtensions message carrying the given raw
/// `(ext_type, data)` pairs.
pub(crate) fn build_encrypted_extensions(exts: &[(u16, &[u8])]) -> Vec<u8> {
    let mut block = Vec::new();
    for (t, data) in exts {
        block.extend_from_slice(&t.to_be_bytes());
        block.extend_from_slice(&(data.len() as u16).to_be_bytes());
        block.extend_from_slice(data);
    }
    let mut body = Vec::new();
    body.extend_from_slice(&(block.len() as u16).to_be_bytes());
    body.extend_from_slice(&block);
    wrap_handshake(HandshakeType::EncryptedExtensions, &body)
}

/// Build a Certificate message from raw DER blobs (no per-entry
/// extensions, empty certificate_request_context).
pub(crate) fn build_certificate(certs: &[&[u8]]) -> Vec<u8> {
    let mut list = Vec::new();
    for c in certs {
        list.extend_from_slice(&(c.len() as u32).to_be_bytes()[1..]); // u24
        list.extend_from_slice(c);
        list.extend_from_slice(&[0, 0]); // no per-entry extensions
    }
    let mut body = Vec::new();
    body.push(0); // empty certificate_request_context
    body.extend_from_slice(&(list.len() as u32).to_be_bytes()[1..]); // u24
    body.extend_from_slice(&list);
    wrap_handshake(HandshakeType::Certificate, &body)
}

/// Build a CertificateVerify message from a scheme code and raw
/// signature bytes.
pub(crate) fn build_certificate_verify(scheme: u16, sig: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&scheme.to_be_bytes());
    body.extend_from_slice(&(sig.len() as u16).to_be_bytes());
    body.extend_from_slice(sig);
    wrap_handshake(HandshakeType::CertificateVerify, &body)
}
