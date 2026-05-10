// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 frame layout and the minimum frame-type set needed by
//! a gRPC unary client.
//!
//! Reference: RFC 7540 § 4 (frame format) and § 6 (frame definitions).

use super::{Http2Error, Result, MAX_FRAME_PAYLOAD};
use alloc::vec::Vec;

/// Length of the fixed HTTP/2 frame header (RFC 7540 § 4.1).
pub const HEADER_LEN: usize = 9;

/// Frame types we emit or accept. Unknown types are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Data = 0x0,
    Headers = 0x1,
    // Priority = 0x2 — parsed-and-discarded inline; never instantiated.
    RstStream = 0x3,
    Settings = 0x4,
    // PushPromise = 0x5 — refused; see Http2Error::RefusedFrameType.
    Ping = 0x6,
    GoAway = 0x7,
    WindowUpdate = 0x8,
    // Continuation = 0x9 — accepted only as a follow-on to HEADERS.
    Continuation = 0x9,
}

impl FrameType {
    /// Decode a wire frame-type byte into a [`FrameType`].
    ///
    /// Returns `Err(RefusedFrameType)` for `PUSH_PROMISE` (0x5)
    /// per the design. `PRIORITY` (0x2) is decoded as `Err(Protocol)`
    /// so the caller drops the frame without state change.
    pub fn from_u8(b: u8) -> Result<Self> {
        match b {
            0x0 => Ok(Self::Data),
            0x1 => Ok(Self::Headers),
            0x2 => Err(Http2Error::Protocol),
            0x3 => Ok(Self::RstStream),
            0x4 => Ok(Self::Settings),
            0x5 => Err(Http2Error::RefusedFrameType),
            0x6 => Ok(Self::Ping),
            0x7 => Ok(Self::GoAway),
            0x8 => Ok(Self::WindowUpdate),
            0x9 => Ok(Self::Continuation),
            _ => Err(Http2Error::RefusedFrameType),
        }
    }
}

/// Common flag bits.
pub mod flags {
    pub const END_STREAM: u8 = 0x1;
    pub const ACK: u8 = 0x1;
    pub const END_HEADERS: u8 = 0x4;
    pub const PADDED: u8 = 0x8;
    pub const PRIORITY: u8 = 0x20;
}

/// Decoded fixed frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub length: u32,
    pub frame_type: FrameType,
    pub flags: u8,
    pub stream_id: u32,
}

impl FrameHeader {
    /// Parse a 9-byte frame header.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_LEN {
            return Err(Http2Error::ShortFrameHeader);
        }
        let length = u32::from_be_bytes([0, buf[0], buf[1], buf[2]]);
        if length as usize > MAX_FRAME_PAYLOAD {
            return Err(Http2Error::OversizedPayload);
        }
        let frame_type = FrameType::from_u8(buf[3])?;
        let flags = buf[4];
        // The "R" reserved bit (high bit of stream_id) MUST be ignored
        // on receipt per RFC 7540 § 4.1.
        let stream_id = u32::from_be_bytes([buf[5] & 0x7f, buf[6], buf[7], buf[8]]);
        Ok(Self {
            length,
            frame_type,
            flags,
            stream_id,
        })
    }

    /// Encode the frame header into the first 9 bytes of `out`.
    pub fn encode(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < HEADER_LEN {
            return Err(Http2Error::ShortFrameHeader);
        }
        if self.length as usize > MAX_FRAME_PAYLOAD {
            return Err(Http2Error::OversizedPayload);
        }
        let l = self.length.to_be_bytes();
        out[0] = l[1];
        out[1] = l[2];
        out[2] = l[3];
        out[3] = self.frame_type as u8;
        out[4] = self.flags;
        let s = self.stream_id.to_be_bytes();
        out[5] = s[0] & 0x7f;
        out[6] = s[1];
        out[7] = s[2];
        out[8] = s[3];
        Ok(())
    }
}

/// Build a SETTINGS frame with the given (id, value) pairs. ACK
/// SETTINGS frames have an empty payload and `flags = ACK`.
pub fn build_settings(ack: bool, pairs: &[(u16, u32)]) -> Vec<u8> {
    let payload_len = (pairs.len() * 6) as u32;
    let mut out = Vec::with_capacity(HEADER_LEN + payload_len as usize);
    out.resize(HEADER_LEN, 0);
    let hdr = FrameHeader {
        length: payload_len,
        frame_type: FrameType::Settings,
        flags: if ack { flags::ACK } else { 0 },
        stream_id: 0,
    };
    hdr.encode(&mut out[..HEADER_LEN]).expect("header fits");
    for (id, val) in pairs {
        out.extend_from_slice(&id.to_be_bytes());
        out.extend_from_slice(&val.to_be_bytes());
    }
    out
}

/// Build a WINDOW_UPDATE frame for `stream_id` (0 = connection level).
pub fn build_window_update(stream_id: u32, increment: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 4);
    out.resize(HEADER_LEN, 0);
    let hdr = FrameHeader {
        length: 4,
        frame_type: FrameType::WindowUpdate,
        flags: 0,
        stream_id,
    };
    hdr.encode(&mut out[..HEADER_LEN]).expect("header fits");
    out.extend_from_slice(&(increment & 0x7fff_ffff).to_be_bytes());
    out
}

/// Build a PING frame. Opaque data is exactly 8 bytes per RFC 7540 § 6.7.
pub fn build_ping(ack: bool, data: [u8; 8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 8);
    out.resize(HEADER_LEN, 0);
    let hdr = FrameHeader {
        length: 8,
        frame_type: FrameType::Ping,
        flags: if ack { flags::ACK } else { 0 },
        stream_id: 0,
    };
    hdr.encode(&mut out[..HEADER_LEN]).expect("header fits");
    out.extend_from_slice(&data);
    out
}

/// Build a GOAWAY frame.
pub fn build_goaway(last_stream_id: u32, error_code: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 8);
    out.resize(HEADER_LEN, 0);
    let hdr = FrameHeader {
        length: 8,
        frame_type: FrameType::GoAway,
        flags: 0,
        stream_id: 0,
    };
    hdr.encode(&mut out[..HEADER_LEN]).expect("header fits");
    out.extend_from_slice(&(last_stream_id & 0x7fff_ffff).to_be_bytes());
    out.extend_from_slice(&error_code.to_be_bytes());
    out
}

/// Build a HEADERS frame from an already-encoded HPACK block.
///
/// `end_stream` and `end_headers` map to the corresponding flags.
pub fn build_headers(stream_id: u32, block: &[u8], end_stream: bool, end_headers: bool) -> Vec<u8> {
    let mut flags_byte = 0u8;
    if end_stream {
        flags_byte |= flags::END_STREAM;
    }
    if end_headers {
        flags_byte |= flags::END_HEADERS;
    }
    let mut out = Vec::with_capacity(HEADER_LEN + block.len());
    out.resize(HEADER_LEN, 0);
    let hdr = FrameHeader {
        length: block.len() as u32,
        frame_type: FrameType::Headers,
        flags: flags_byte,
        stream_id,
    };
    hdr.encode(&mut out[..HEADER_LEN]).expect("header fits");
    out.extend_from_slice(block);
    out
}

/// Build a DATA frame.
pub fn build_data(stream_id: u32, payload: &[u8], end_stream: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.resize(HEADER_LEN, 0);
    let hdr = FrameHeader {
        length: payload.len() as u32,
        frame_type: FrameType::Data,
        flags: if end_stream { flags::END_STREAM } else { 0 },
        stream_id,
    };
    hdr.encode(&mut out[..HEADER_LEN]).expect("header fits");
    out.extend_from_slice(payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = FrameHeader {
            length: 12,
            frame_type: FrameType::Headers,
            flags: flags::END_HEADERS | flags::END_STREAM,
            stream_id: 1,
        };
        let mut buf = [0u8; HEADER_LEN];
        h.encode(&mut buf).unwrap();
        let parsed = FrameHeader::parse(&buf).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn reserved_bit_ignored_on_parse() {
        // stream_id = 1 with the reserved high bit set — must parse to 1.
        let buf = [0, 0, 0, 0x1, 0, 0x80, 0, 0, 1];
        let h = FrameHeader::parse(&buf).unwrap();
        assert_eq!(h.stream_id, 1);
    }

    #[test]
    fn push_promise_refused() {
        let buf = [0, 0, 0, 0x5, 0, 0, 0, 0, 1];
        assert_eq!(
            FrameHeader::parse(&buf).unwrap_err(),
            Http2Error::RefusedFrameType
        );
    }

    #[test]
    fn priority_decoded_as_protocol_error() {
        // PRIORITY (0x2) is parsed-and-discarded by the higher layer.
        let buf = [0, 0, 5, 0x2, 0, 0, 0, 0, 1];
        assert_eq!(FrameHeader::parse(&buf).unwrap_err(), Http2Error::Protocol);
    }

    #[test]
    fn oversized_payload_rejected() {
        let buf = [0xff, 0xff, 0xff, 0x0, 0, 0, 0, 0, 1];
        assert_eq!(
            FrameHeader::parse(&buf).unwrap_err(),
            Http2Error::OversizedPayload
        );
    }

    #[test]
    fn build_settings_ack_empty() {
        let f = build_settings(true, &[]);
        let h = FrameHeader::parse(&f).unwrap();
        assert_eq!(h.frame_type, FrameType::Settings);
        assert_eq!(h.flags, flags::ACK);
        assert_eq!(h.length, 0);
        assert_eq!(f.len(), HEADER_LEN);
    }

    #[test]
    fn build_settings_pairs() {
        let f = build_settings(false, &[(0x3, 100), (0x5, 16384)]);
        let h = FrameHeader::parse(&f).unwrap();
        assert_eq!(h.length, 12);
        assert_eq!(&f[HEADER_LEN..HEADER_LEN + 6], &[0, 3, 0, 0, 0, 100]);
    }

    #[test]
    fn build_window_update_masks_high_bit() {
        let f = build_window_update(1, u32::MAX);
        // payload bytes 9..13 — high bit must be clear.
        assert_eq!(f[HEADER_LEN] & 0x80, 0);
    }
}
