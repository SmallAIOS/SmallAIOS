// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TLS 1.3 handshake messages and state.
//!
//! Phase 4 of `tls-tcp-client-v1`. This module is the
//! message-layer foundation:
//!
//! - [`HandshakeType`] + [`HandshakeHeader`] — the 4-byte
//!   header every handshake message carries.
//! - [`extensions`] — encoders/parsers for the extensions
//!   the client emits or expects.
//! - [`client_hello`] — `build_client_hello` per the cipher
//!   suite + key-share + signature-algorithm allow-lists.
//! - [`server_hello`] — `parse_server_hello` with strict
//!   version-pinning.
//!
//! The handshake state machine (`Handshake::step()`) that
//! drives this module against a `TlsStreamLike` lands in a
//! follow-on commit alongside Phase 5's cert verifier and
//! Phase 7's `TcpTlsStream`.

pub mod client_hello;
pub mod extensions;
pub mod server_hello;

use crate::{Result, TlsClientError};
use alloc::vec::Vec;

/// RFC 8446 § 4 HandshakeType values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HandshakeType {
    ClientHello = 1,
    ServerHello = 2,
    NewSessionTicket = 4,
    EncryptedExtensions = 8,
    Certificate = 11,
    CertificateRequest = 13,
    CertificateVerify = 15,
    Finished = 20,
    KeyUpdate = 24,
}

impl HandshakeType {
    /// Decode the wire byte. Refuses anything not on the
    /// TLS 1.3 list (we never accept legacy
    /// `HelloRequest`, `ServerKeyExchange`, etc.).
    pub fn from_u8(b: u8) -> Result<Self> {
        match b {
            1 => Ok(Self::ClientHello),
            2 => Ok(Self::ServerHello),
            4 => Ok(Self::NewSessionTicket),
            8 => Ok(Self::EncryptedExtensions),
            11 => Ok(Self::Certificate),
            13 => Ok(Self::CertificateRequest),
            15 => Ok(Self::CertificateVerify),
            20 => Ok(Self::Finished),
            24 => Ok(Self::KeyUpdate),
            _ => Err(TlsClientError::BadHandshake),
        }
    }
}

/// 4-byte handshake message header: type + 24-bit length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandshakeHeader {
    pub msg_type: HandshakeType,
    pub length: u32,
}

impl HandshakeHeader {
    pub const LEN: usize = 4;

    /// Parse the first 4 bytes as a handshake header.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::LEN {
            return Err(TlsClientError::BadHandshake);
        }
        let msg_type = HandshakeType::from_u8(buf[0])?;
        // 24-bit big-endian length.
        let length = u32::from_be_bytes([0, buf[1], buf[2], buf[3]]);
        Ok(Self { msg_type, length })
    }

    /// Serialize a handshake header.
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<()> {
        if self.length > 0x00ff_ffff {
            return Err(TlsClientError::BadHandshake);
        }
        out.push(self.msg_type as u8);
        let l = self.length.to_be_bytes();
        out.extend_from_slice(&l[1..4]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_header_round_trip() {
        let h = HandshakeHeader {
            msg_type: HandshakeType::ClientHello,
            length: 0x123456,
        };
        let mut buf = Vec::new();
        h.encode(&mut buf).unwrap();
        let parsed = HandshakeHeader::parse(&buf).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn handshake_type_unknown_rejected() {
        assert_eq!(
            HandshakeType::from_u8(99).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn handshake_header_too_short_rejected() {
        assert_eq!(
            HandshakeHeader::parse(&[1, 0, 0]).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn handshake_header_oversized_length_rejected() {
        let h = HandshakeHeader {
            msg_type: HandshakeType::ClientHello,
            length: 0x0100_0000,
        };
        let mut buf = Vec::new();
        assert_eq!(
            h.encode(&mut buf).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }
}
