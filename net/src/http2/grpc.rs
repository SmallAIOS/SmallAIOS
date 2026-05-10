// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! gRPC length-prefixed-message framing (gRPC over HTTP/2, § 4.2).
//!
//! Wire format of one gRPC message:
//!
//! ```text
//! [ 1 byte: compressed-flag ]
//! [ 4 bytes: message-length, big-endian ]
//! [ message-length bytes:   message body ]
//! ```
//!
//! We never advertise compression (we set `grpc-encoding: identity`
//! implicitly by omitting the header), and we reject any inbound
//! message whose compressed-flag is set.
//!
//! The message-length cap is configurable per call; the default
//! mirrors immudb's `MaxRecvMsgSize` (4 MiB).

use super::{Http2Error, Result};
use alloc::vec::Vec;

/// gRPC framing prefix length.
pub const PREFIX_LEN: usize = 5;

/// Default gRPC inbound message-size cap (4 MiB).
pub const DEFAULT_MAX_MESSAGE_LEN: u32 = 4 * 1024 * 1024;

/// Encode `body` as a single gRPC length-prefixed message. The
/// prefix's compressed-flag is always zero.
pub fn encode_message(body: &[u8]) -> Vec<u8> {
    let len = body.len() as u32;
    let mut out = Vec::with_capacity(PREFIX_LEN + body.len());
    out.push(0); // compressed-flag = 0.
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(body);
    out
}

/// Try to decode the prefix of `input` as a single gRPC message.
/// Returns `Ok((body_slice, total_consumed))` on a complete frame,
/// `Ok((..0..,  0))` when the input is too short to determine the
/// length yet, and `Err` on a protocol violation.
///
/// The compressed-flag byte MUST be zero; any other value is
/// rejected with [`Http2Error::GrpcCompressed`].
pub fn try_decode_message(input: &[u8], max_len: u32) -> Result<Option<(&[u8], usize)>> {
    if input.len() < PREFIX_LEN {
        return Ok(None);
    }
    if input[0] != 0 {
        return Err(Http2Error::GrpcCompressed);
    }
    let len = u32::from_be_bytes([input[1], input[2], input[3], input[4]]);
    if len > max_len {
        return Err(Http2Error::GrpcOversized);
    }
    let total = PREFIX_LEN + len as usize;
    if input.len() < total {
        return Ok(None);
    }
    Ok(Some((&input[PREFIX_LEN..total], total)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let body = [1u8, 2, 3, 4, 5];
        let frame = encode_message(&body);
        let (decoded, consumed) = try_decode_message(&frame, DEFAULT_MAX_MESSAGE_LEN)
            .unwrap()
            .unwrap();
        assert_eq!(decoded, &body);
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn empty_message_round_trip() {
        let frame = encode_message(&[]);
        assert_eq!(frame.len(), PREFIX_LEN);
        let (decoded, consumed) = try_decode_message(&frame, DEFAULT_MAX_MESSAGE_LEN)
            .unwrap()
            .unwrap();
        assert!(decoded.is_empty());
        assert_eq!(consumed, PREFIX_LEN);
    }

    #[test]
    fn short_input_returns_none() {
        assert!(try_decode_message(&[0, 0, 0], DEFAULT_MAX_MESSAGE_LEN)
            .unwrap()
            .is_none());
    }

    #[test]
    fn partial_body_returns_none() {
        // Prefix says length 10 but only 3 body bytes follow.
        let mut frame = Vec::from([0u8, 0, 0, 0, 10]);
        frame.extend_from_slice(&[1, 2, 3]);
        assert!(try_decode_message(&frame, DEFAULT_MAX_MESSAGE_LEN)
            .unwrap()
            .is_none());
    }

    #[test]
    fn compressed_flag_rejected() {
        let mut frame = Vec::from([1u8, 0, 0, 0, 1]);
        frame.push(0xff);
        assert_eq!(
            try_decode_message(&frame, DEFAULT_MAX_MESSAGE_LEN).unwrap_err(),
            Http2Error::GrpcCompressed
        );
    }

    #[test]
    fn oversized_rejected() {
        let frame = [0u8, 0, 0x10, 0, 0]; // length 0x100000 = 1 MiB.
        assert_eq!(
            try_decode_message(&frame, 1024).unwrap_err(),
            Http2Error::GrpcOversized
        );
    }
}
