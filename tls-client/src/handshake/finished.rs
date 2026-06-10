// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Finished message (RFC 8446 §4.4.4).
//!
//! ```text
//! struct {
//!     opaque verify_data[Hash.length];
//! } Finished;
//! ```
//!
//! `verify_data = HMAC(finished_key, Transcript-Hash(...))` where
//! the finished_key is expanded from the sender's handshake-traffic
//! secret. The MAC comparison is constant-time
//! (`security::crypto::constant_time::ct_eq`) — a forged Finished
//! must not leak how many bytes matched.

use super::{HandshakeHeader, HandshakeType};
use crate::{Result, TlsClientError};
use alloc::vec::Vec;
use smallaios_security::crypto::constant_time::ct_eq;

/// Parse a Finished message starting at its handshake header,
/// returning the raw verify_data.
pub fn parse_finished(buf: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let header = HandshakeHeader::parse(buf)?;
    if header.msg_type != HandshakeType::Finished {
        return Err(TlsClientError::BadHandshake);
    }
    let body_start = HandshakeHeader::LEN;
    let body_end = body_start + header.length as usize;
    if buf.len() < body_end || header.length as usize != expected_len {
        return Err(TlsClientError::BadHandshake);
    }
    Ok(buf[body_start..body_end].to_vec())
}

/// Verify a received Finished's verify_data in constant time
/// (task 4.8).
pub fn check_verify_data(received: &[u8], expected: &[u8]) -> Result<()> {
    if received.len() != expected.len() || !ct_eq(received, expected).to_bool() {
        return Err(TlsClientError::BadHandshake);
    }
    Ok(())
}

/// Build a Finished message (header + verify_data) — the client's
/// final flight (task 4.9).
pub fn build_finished(verify_data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(HandshakeHeader::LEN + verify_data.len());
    HandshakeHeader {
        msg_type: HandshakeType::Finished,
        length: verify_data.len() as u32,
    }
    .encode(&mut out)?;
    out.extend_from_slice(verify_data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finished_round_trip() {
        let vd = [0x5au8; 32];
        let msg = build_finished(&vd).unwrap();
        assert_eq!(msg[0], HandshakeType::Finished as u8);
        let parsed = parse_finished(&msg, 32).unwrap();
        assert_eq!(parsed, vd);
        check_verify_data(&parsed, &vd).unwrap();
    }

    #[test]
    fn finished_wrong_length_rejected() {
        let msg = build_finished(&[0x5au8; 32]).unwrap();
        // SHA-384 suite expects 48 bytes.
        assert_eq!(
            parse_finished(&msg, 48).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn tampered_verify_data_rejected() {
        let vd = [0x5au8; 32];
        let mut bad = vd;
        bad[31] ^= 1;
        assert_eq!(
            check_verify_data(&bad, &vd).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn length_mismatch_rejected() {
        assert!(check_verify_data(&[0u8; 32], &[0u8; 48]).is_err());
    }
}
