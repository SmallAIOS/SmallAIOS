// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC frame encode/decode (RFC 9000 Section 19).
//!
//! Supports PADDING, PING, ACK, CRYPTO, STREAM, NEW_CONNECTION_ID,
//! RETIRE_CONNECTION_ID, CONNECTION_CLOSE, MAX_DATA, MAX_STREAM_DATA,
//! MAX_STREAMS, and RESET_STREAM frames.

use alloc::vec::Vec;
use crate::NetError;
use super::packet::{encode_var_int, decode_var_int};

/// QUIC frame type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// PADDING (type 0x00).
    Padding,
    /// PING (type 0x01).
    Ping,
    /// ACK (type 0x02).
    Ack,
    /// CRYPTO (type 0x06).
    Crypto,
    /// NEW_TOKEN (type 0x07).
    NewToken,
    /// STREAM (type 0x08-0x0f).
    Stream,
    /// MAX_DATA (type 0x10).
    MaxData,
    /// MAX_STREAM_DATA (type 0x11).
    MaxStreamData,
    /// MAX_STREAMS bidi (type 0x12).
    MaxStreamsBidi,
    /// MAX_STREAMS uni (type 0x13).
    MaxStreamsUni,
    /// CONNECTION_CLOSE (type 0x1c).
    ConnectionClose,
    /// NEW_CONNECTION_ID (type 0x18).
    NewConnectionId,
    /// RETIRE_CONNECTION_ID (type 0x19).
    RetireConnectionId,
    /// RESET_STREAM (type 0x04).
    ResetStream,
    /// STOP_SENDING (type 0x05).
    StopSending,
    /// PATH_CHALLENGE (type 0x1a).
    PathChallenge,
    /// PATH_RESPONSE (type 0x1b).
    PathResponse,
    /// Unknown frame type.
    Unknown(u64),
}

impl FrameType {
    /// Get the type ID.
    pub fn to_id(self) -> u64 {
        match self {
            FrameType::Padding => 0x00,
            FrameType::Ping => 0x01,
            FrameType::Ack => 0x02,
            FrameType::Crypto => 0x06,
            FrameType::NewToken => 0x07,
            FrameType::Stream => 0x08,
            FrameType::MaxData => 0x10,
            FrameType::MaxStreamData => 0x11,
            FrameType::MaxStreamsBidi => 0x12,
            FrameType::MaxStreamsUni => 0x13,
            FrameType::ConnectionClose => 0x1c,
            FrameType::NewConnectionId => 0x18,
            FrameType::RetireConnectionId => 0x19,
            FrameType::ResetStream => 0x04,
            FrameType::StopSending => 0x05,
            FrameType::PathChallenge => 0x1a,
            FrameType::PathResponse => 0x1b,
            FrameType::Unknown(id) => id,
        }
    }

    /// Parse from type ID.
    pub fn from_id(id: u64) -> Self {
        match id {
            0x00 => FrameType::Padding,
            0x01 => FrameType::Ping,
            0x02 | 0x03 => FrameType::Ack,
            0x04 => FrameType::ResetStream,
            0x05 => FrameType::StopSending,
            0x06 => FrameType::Crypto,
            0x07 => FrameType::NewToken,
            0x08..=0x0f => FrameType::Stream,
            0x10 => FrameType::MaxData,
            0x11 => FrameType::MaxStreamData,
            0x12 => FrameType::MaxStreamsBidi,
            0x13 => FrameType::MaxStreamsUni,
            0x18 => FrameType::NewConnectionId,
            0x19 => FrameType::RetireConnectionId,
            0x1a => FrameType::PathChallenge,
            0x1b => FrameType::PathResponse,
            0x1c | 0x1d => FrameType::ConnectionClose,
            other => FrameType::Unknown(other),
        }
    }
}

/// Parsed QUIC frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuicFrame {
    /// PADDING frame (zero or more bytes).
    Padding(usize),
    /// PING frame (no payload).
    Ping,
    /// ACK frame.
    Ack(AckFrame),
    /// CRYPTO frame.
    Crypto(CryptoFrame),
    /// STREAM frame.
    Stream(StreamFrame),
    /// MAX_DATA frame.
    MaxData { maximum: u64 },
    /// MAX_STREAM_DATA frame.
    MaxStreamData { stream_id: u64, maximum: u64 },
    /// MAX_STREAMS (bidirectional).
    MaxStreamsBidi { maximum: u64 },
    /// MAX_STREAMS (unidirectional).
    MaxStreamsUni { maximum: u64 },
    /// CONNECTION_CLOSE frame.
    ConnectionClose(ConnectionCloseFrame),
    /// NEW_CONNECTION_ID frame.
    NewConnectionId(NewConnectionIdFrame),
    /// RETIRE_CONNECTION_ID frame.
    RetireConnectionId { sequence: u64 },
    /// RESET_STREAM frame.
    ResetStream { stream_id: u64, error_code: u64, final_size: u64 },
    /// STOP_SENDING frame.
    StopSending { stream_id: u64, error_code: u64 },
    /// PATH_CHALLENGE frame.
    PathChallenge { data: [u8; 8] },
    /// PATH_RESPONSE frame.
    PathResponse { data: [u8; 8] },
}

/// ACK frame data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AckFrame {
    /// Largest acknowledged packet number.
    pub largest_ack: u64,
    /// ACK delay (in microseconds, encoded as multiples of ack_delay_exponent).
    pub ack_delay: u64,
    /// Number of ACK range entries (after the first range).
    pub ack_range_count: u64,
    /// First ACK range (number of contiguous packets before largest_ack).
    pub first_ack_range: u64,
    /// Additional ACK ranges: (gap, range_length).
    pub ranges: Vec<(u64, u64)>,
}

/// CRYPTO frame data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CryptoFrame {
    /// Byte offset in the crypto stream.
    pub offset: u64,
    /// Crypto data.
    pub data: Vec<u8>,
}

/// STREAM frame data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFrame {
    /// Stream ID.
    pub stream_id: u64,
    /// Byte offset (if present).
    pub offset: u64,
    /// Stream data.
    pub data: Vec<u8>,
    /// FIN bit — last data on this stream.
    pub fin: bool,
}

/// CONNECTION_CLOSE frame data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionCloseFrame {
    /// Error code.
    pub error_code: u64,
    /// Frame type that triggered the error (0 if unknown).
    pub frame_type: u64,
    /// Human-readable reason phrase.
    pub reason: Vec<u8>,
}

/// NEW_CONNECTION_ID frame data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewConnectionIdFrame {
    /// Sequence number for this CID.
    pub sequence: u64,
    /// Retire prior to this sequence number.
    pub retire_prior_to: u64,
    /// Connection ID bytes.
    pub connection_id: Vec<u8>,
    /// Stateless reset token (16 bytes).
    pub stateless_reset_token: [u8; 16],
}

impl QuicFrame {
    /// Encode a QUIC frame into a buffer. Returns bytes written.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, NetError> {
        match self {
            QuicFrame::Padding(count) => {
                if buf.len() < *count {
                    return Err(NetError::BufferTooSmall);
                }
                for b in buf[..*count].iter_mut() {
                    *b = 0;
                }
                Ok(*count)
            }
            QuicFrame::Ping => {
                if buf.is_empty() {
                    return Err(NetError::BufferTooSmall);
                }
                let off = encode_var_int(0x01, buf);
                Ok(off)
            }
            QuicFrame::Ack(ack) => encode_ack(ack, buf),
            QuicFrame::Crypto(crypto) => encode_crypto(crypto, buf),
            QuicFrame::Stream(stream) => encode_stream(stream, buf),
            QuicFrame::MaxData { maximum } => {
                let mut off = encode_var_int(0x10, buf);
                off += encode_var_int(*maximum, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::MaxStreamData { stream_id, maximum } => {
                let mut off = encode_var_int(0x11, buf);
                off += encode_var_int(*stream_id, &mut buf[off..]);
                off += encode_var_int(*maximum, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::MaxStreamsBidi { maximum } => {
                let mut off = encode_var_int(0x12, buf);
                off += encode_var_int(*maximum, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::MaxStreamsUni { maximum } => {
                let mut off = encode_var_int(0x13, buf);
                off += encode_var_int(*maximum, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::ConnectionClose(cc) => encode_connection_close(cc, buf),
            QuicFrame::NewConnectionId(ncid) => encode_new_connection_id(ncid, buf),
            QuicFrame::RetireConnectionId { sequence } => {
                let mut off = encode_var_int(0x19, buf);
                off += encode_var_int(*sequence, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::ResetStream { stream_id, error_code, final_size } => {
                let mut off = encode_var_int(0x04, buf);
                off += encode_var_int(*stream_id, &mut buf[off..]);
                off += encode_var_int(*error_code, &mut buf[off..]);
                off += encode_var_int(*final_size, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::StopSending { stream_id, error_code } => {
                let mut off = encode_var_int(0x05, buf);
                off += encode_var_int(*stream_id, &mut buf[off..]);
                off += encode_var_int(*error_code, &mut buf[off..]);
                Ok(off)
            }
            QuicFrame::PathChallenge { data } => {
                if buf.len() < 9 {
                    return Err(NetError::BufferTooSmall);
                }
                let mut off = encode_var_int(0x1a, buf);
                buf[off..off + 8].copy_from_slice(data);
                off += 8;
                Ok(off)
            }
            QuicFrame::PathResponse { data } => {
                if buf.len() < 9 {
                    return Err(NetError::BufferTooSmall);
                }
                let mut off = encode_var_int(0x1b, buf);
                buf[off..off + 8].copy_from_slice(data);
                off += 8;
                Ok(off)
            }
        }
    }

    /// Decode a single QUIC frame from a buffer. Returns (frame, bytes_consumed).
    pub fn decode(data: &[u8]) -> Result<(Self, usize), NetError> {
        if data.is_empty() {
            return Err(NetError::PacketTooShort);
        }

        // Check for PADDING (type 0x00)
        if data[0] == 0x00 {
            let mut count = 0;
            while count < data.len() && data[count] == 0x00 {
                count += 1;
            }
            return Ok((QuicFrame::Padding(count), count));
        }

        let (frame_type, mut off) = decode_var_int(data)?;
        let ft = FrameType::from_id(frame_type);

        match ft {
            FrameType::Ping => Ok((QuicFrame::Ping, off)),
            FrameType::Ack => {
                let (frame, consumed) = decode_ack(&data[off..])?;
                Ok((frame, off + consumed))
            }
            FrameType::Crypto => {
                let (frame, consumed) = decode_crypto(&data[off..])?;
                Ok((frame, off + consumed))
            }
            FrameType::Stream => {
                let flags = frame_type as u8;
                let (frame, consumed) = decode_stream(flags, &data[off..])?;
                Ok((frame, off + consumed))
            }
            FrameType::MaxData => {
                let (maximum, c) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::MaxData { maximum }, off + c))
            }
            FrameType::MaxStreamData => {
                let (stream_id, c1) = decode_var_int(&data[off..])?;
                off += c1;
                let (maximum, c2) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::MaxStreamData { stream_id, maximum }, off + c2))
            }
            FrameType::MaxStreamsBidi => {
                let (maximum, c) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::MaxStreamsBidi { maximum }, off + c))
            }
            FrameType::MaxStreamsUni => {
                let (maximum, c) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::MaxStreamsUni { maximum }, off + c))
            }
            FrameType::ConnectionClose => {
                let (frame, consumed) = decode_connection_close(&data[off..])?;
                Ok((frame, off + consumed))
            }
            FrameType::NewConnectionId => {
                let (frame, consumed) = decode_new_connection_id(&data[off..])?;
                Ok((frame, off + consumed))
            }
            FrameType::RetireConnectionId => {
                let (sequence, c) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::RetireConnectionId { sequence }, off + c))
            }
            FrameType::ResetStream => {
                let (stream_id, c1) = decode_var_int(&data[off..])?;
                off += c1;
                let (error_code, c2) = decode_var_int(&data[off..])?;
                off += c2;
                let (final_size, c3) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::ResetStream { stream_id, error_code, final_size }, off + c3))
            }
            FrameType::StopSending => {
                let (stream_id, c1) = decode_var_int(&data[off..])?;
                off += c1;
                let (error_code, c2) = decode_var_int(&data[off..])?;
                Ok((QuicFrame::StopSending { stream_id, error_code }, off + c2))
            }
            FrameType::PathChallenge => {
                if data.len() < off + 8 {
                    return Err(NetError::PacketTooShort);
                }
                let mut d = [0u8; 8];
                d.copy_from_slice(&data[off..off + 8]);
                Ok((QuicFrame::PathChallenge { data: d }, off + 8))
            }
            FrameType::PathResponse => {
                if data.len() < off + 8 {
                    return Err(NetError::PacketTooShort);
                }
                let mut d = [0u8; 8];
                d.copy_from_slice(&data[off..off + 8]);
                Ok((QuicFrame::PathResponse { data: d }, off + 8))
            }
            _ => Err(NetError::InvalidProtocol),
        }
    }
}

fn encode_ack(ack: &AckFrame, buf: &mut [u8]) -> Result<usize, NetError> {
    let mut off = encode_var_int(0x02, buf);
    off += encode_var_int(ack.largest_ack, &mut buf[off..]);
    off += encode_var_int(ack.ack_delay, &mut buf[off..]);
    off += encode_var_int(ack.ack_range_count, &mut buf[off..]);
    off += encode_var_int(ack.first_ack_range, &mut buf[off..]);
    for (gap, range_len) in &ack.ranges {
        off += encode_var_int(*gap, &mut buf[off..]);
        off += encode_var_int(*range_len, &mut buf[off..]);
    }
    Ok(off)
}

fn decode_ack(data: &[u8]) -> Result<(QuicFrame, usize), NetError> {
    let mut off = 0;
    let (largest_ack, c) = decode_var_int(&data[off..])?;
    off += c;
    let (ack_delay, c) = decode_var_int(&data[off..])?;
    off += c;
    let (ack_range_count, c) = decode_var_int(&data[off..])?;
    off += c;
    let (first_ack_range, c) = decode_var_int(&data[off..])?;
    off += c;
    let mut ranges = Vec::new();
    for _ in 0..ack_range_count {
        let (gap, c1) = decode_var_int(&data[off..])?;
        off += c1;
        let (range_len, c2) = decode_var_int(&data[off..])?;
        off += c2;
        ranges.push((gap, range_len));
    }
    Ok((
        QuicFrame::Ack(AckFrame {
            largest_ack,
            ack_delay,
            ack_range_count,
            first_ack_range,
            ranges,
        }),
        off,
    ))
}

fn encode_crypto(crypto: &CryptoFrame, buf: &mut [u8]) -> Result<usize, NetError> {
    let mut off = encode_var_int(0x06, buf);
    off += encode_var_int(crypto.offset, &mut buf[off..]);
    off += encode_var_int(crypto.data.len() as u64, &mut buf[off..]);
    if buf.len() < off + crypto.data.len() {
        return Err(NetError::BufferTooSmall);
    }
    buf[off..off + crypto.data.len()].copy_from_slice(&crypto.data);
    off += crypto.data.len();
    Ok(off)
}

fn decode_crypto(data: &[u8]) -> Result<(QuicFrame, usize), NetError> {
    let mut off = 0;
    let (offset, c) = decode_var_int(&data[off..])?;
    off += c;
    let (length, c) = decode_var_int(&data[off..])?;
    off += c;
    if off + length as usize > data.len() {
        return Err(NetError::PacketTooShort);
    }
    let crypto_data = Vec::from(&data[off..off + length as usize]);
    off += length as usize;
    Ok((
        QuicFrame::Crypto(CryptoFrame {
            offset,
            data: crypto_data,
        }),
        off,
    ))
}

fn encode_stream(stream: &StreamFrame, buf: &mut [u8]) -> Result<usize, NetError> {
    // Stream frame type: 0x08 | OFF(0x04) | LEN(0x02) | FIN(0x01)
    let mut frame_type: u8 = 0x08;
    if stream.offset > 0 {
        frame_type |= 0x04;
    }
    frame_type |= 0x02; // Always include length
    if stream.fin {
        frame_type |= 0x01;
    }

    let mut off = encode_var_int(frame_type as u64, buf);
    off += encode_var_int(stream.stream_id, &mut buf[off..]);
    if stream.offset > 0 {
        off += encode_var_int(stream.offset, &mut buf[off..]);
    }
    off += encode_var_int(stream.data.len() as u64, &mut buf[off..]);
    if buf.len() < off + stream.data.len() {
        return Err(NetError::BufferTooSmall);
    }
    buf[off..off + stream.data.len()].copy_from_slice(&stream.data);
    off += stream.data.len();
    Ok(off)
}

fn decode_stream(flags: u8, data: &[u8]) -> Result<(QuicFrame, usize), NetError> {
    let has_offset = flags & 0x04 != 0;
    let has_length = flags & 0x02 != 0;
    let fin = flags & 0x01 != 0;

    let mut off = 0;
    let (stream_id, c) = decode_var_int(&data[off..])?;
    off += c;

    let offset = if has_offset {
        let (o, c) = decode_var_int(&data[off..])?;
        off += c;
        o
    } else {
        0
    };

    let stream_data = if has_length {
        let (length, c) = decode_var_int(&data[off..])?;
        off += c;
        if off + length as usize > data.len() {
            return Err(NetError::PacketTooShort);
        }
        let d = Vec::from(&data[off..off + length as usize]);
        off += length as usize;
        d
    } else {
        // No length field: consume rest of packet
        let d = Vec::from(&data[off..]);
        off = data.len();
        d
    };

    Ok((
        QuicFrame::Stream(StreamFrame {
            stream_id,
            offset,
            data: stream_data,
            fin,
        }),
        off,
    ))
}

fn encode_connection_close(cc: &ConnectionCloseFrame, buf: &mut [u8]) -> Result<usize, NetError> {
    let mut off = encode_var_int(0x1c, buf);
    off += encode_var_int(cc.error_code, &mut buf[off..]);
    off += encode_var_int(cc.frame_type, &mut buf[off..]);
    off += encode_var_int(cc.reason.len() as u64, &mut buf[off..]);
    if buf.len() < off + cc.reason.len() {
        return Err(NetError::BufferTooSmall);
    }
    buf[off..off + cc.reason.len()].copy_from_slice(&cc.reason);
    off += cc.reason.len();
    Ok(off)
}

fn decode_connection_close(data: &[u8]) -> Result<(QuicFrame, usize), NetError> {
    let mut off = 0;
    let (error_code, c) = decode_var_int(&data[off..])?;
    off += c;
    let (frame_type, c) = decode_var_int(&data[off..])?;
    off += c;
    let (reason_len, c) = decode_var_int(&data[off..])?;
    off += c;
    if off + reason_len as usize > data.len() {
        return Err(NetError::PacketTooShort);
    }
    let reason = Vec::from(&data[off..off + reason_len as usize]);
    off += reason_len as usize;
    Ok((
        QuicFrame::ConnectionClose(ConnectionCloseFrame {
            error_code,
            frame_type,
            reason,
        }),
        off,
    ))
}

fn encode_new_connection_id(ncid: &NewConnectionIdFrame, buf: &mut [u8]) -> Result<usize, NetError> {
    let mut off = encode_var_int(0x18, buf);
    off += encode_var_int(ncid.sequence, &mut buf[off..]);
    off += encode_var_int(ncid.retire_prior_to, &mut buf[off..]);
    if buf.len() < off + 1 + ncid.connection_id.len() + 16 {
        return Err(NetError::BufferTooSmall);
    }
    buf[off] = ncid.connection_id.len() as u8;
    off += 1;
    buf[off..off + ncid.connection_id.len()].copy_from_slice(&ncid.connection_id);
    off += ncid.connection_id.len();
    buf[off..off + 16].copy_from_slice(&ncid.stateless_reset_token);
    off += 16;
    Ok(off)
}

fn decode_new_connection_id(data: &[u8]) -> Result<(QuicFrame, usize), NetError> {
    let mut off = 0;
    let (sequence, c) = decode_var_int(&data[off..])?;
    off += c;
    let (retire_prior_to, c) = decode_var_int(&data[off..])?;
    off += c;
    if off >= data.len() {
        return Err(NetError::PacketTooShort);
    }
    let cid_len = data[off] as usize;
    off += 1;
    if off + cid_len + 16 > data.len() {
        return Err(NetError::PacketTooShort);
    }
    let connection_id = Vec::from(&data[off..off + cid_len]);
    off += cid_len;
    let mut token = [0u8; 16];
    token.copy_from_slice(&data[off..off + 16]);
    off += 16;
    Ok((
        QuicFrame::NewConnectionId(NewConnectionIdFrame {
            sequence,
            retire_prior_to,
            connection_id,
            stateless_reset_token: token,
        }),
        off,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_frame_type_roundtrip() {
        for id in [0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x08, 0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1a, 0x1b, 0x1c] {
            let ft = FrameType::from_id(id);
            if !matches!(ft, FrameType::Unknown(_)) {
                // Stream and Ack have range of IDs, so to_id may differ
                if !matches!(ft, FrameType::Stream | FrameType::Ack) {
                    assert_eq!(ft.to_id(), id);
                }
            }
        }
    }

    #[test]
    fn test_padding_encode_decode() {
        let frame = QuicFrame::Padding(5);
        let mut buf = [0xFFu8; 32];
        let len = frame.encode(&mut buf).unwrap();
        assert_eq!(len, 5);
        assert_eq!(&buf[..5], &[0, 0, 0, 0, 0]);
        let (decoded, consumed) = QuicFrame::decode(&buf[..5]).unwrap();
        assert_eq!(consumed, 5);
        assert_eq!(decoded, QuicFrame::Padding(5));
    }

    #[test]
    fn test_ping_encode_decode() {
        let frame = QuicFrame::Ping;
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, consumed) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        assert_eq!(decoded, QuicFrame::Ping);
    }

    #[test]
    fn test_ack_encode_decode() {
        let frame = QuicFrame::Ack(AckFrame {
            largest_ack: 100,
            ack_delay: 50,
            ack_range_count: 1,
            first_ack_range: 5,
            ranges: vec![(2, 3)],
        });
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, consumed) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        if let QuicFrame::Ack(ack) = decoded {
            assert_eq!(ack.largest_ack, 100);
            assert_eq!(ack.ack_delay, 50);
            assert_eq!(ack.first_ack_range, 5);
            assert_eq!(ack.ranges, vec![(2, 3)]);
        } else {
            panic!("Expected Ack frame");
        }
    }

    #[test]
    fn test_ack_no_ranges() {
        let frame = QuicFrame::Ack(AckFrame {
            largest_ack: 10,
            ack_delay: 0,
            ack_range_count: 0,
            first_ack_range: 10,
            ranges: vec![],
        });
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        if let QuicFrame::Ack(ack) = decoded {
            assert_eq!(ack.largest_ack, 10);
            assert!(ack.ranges.is_empty());
        } else {
            panic!("Expected Ack frame");
        }
    }

    #[test]
    fn test_crypto_encode_decode() {
        let frame = QuicFrame::Crypto(CryptoFrame {
            offset: 0,
            data: vec![0x01, 0x02, 0x03, 0x04],
        });
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, consumed) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        if let QuicFrame::Crypto(c) = decoded {
            assert_eq!(c.offset, 0);
            assert_eq!(c.data, vec![0x01, 0x02, 0x03, 0x04]);
        } else {
            panic!("Expected Crypto frame");
        }
    }

    #[test]
    fn test_crypto_with_offset() {
        let frame = QuicFrame::Crypto(CryptoFrame {
            offset: 1000,
            data: vec![0xAA; 10],
        });
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        if let QuicFrame::Crypto(c) = decoded {
            assert_eq!(c.offset, 1000);
            assert_eq!(c.data.len(), 10);
        } else {
            panic!("Expected Crypto frame");
        }
    }

    #[test]
    fn test_stream_encode_decode() {
        let frame = QuicFrame::Stream(StreamFrame {
            stream_id: 4,
            offset: 100,
            data: vec![0xDE, 0xAD],
            fin: true,
        });
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, consumed) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        if let QuicFrame::Stream(s) = decoded {
            assert_eq!(s.stream_id, 4);
            assert_eq!(s.offset, 100);
            assert_eq!(s.data, vec![0xDE, 0xAD]);
            assert!(s.fin);
        } else {
            panic!("Expected Stream frame");
        }
    }

    #[test]
    fn test_stream_no_offset() {
        let frame = QuicFrame::Stream(StreamFrame {
            stream_id: 0,
            offset: 0,
            data: vec![0x01],
            fin: false,
        });
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        if let QuicFrame::Stream(s) = decoded {
            assert_eq!(s.stream_id, 0);
            assert_eq!(s.offset, 0);
            assert!(!s.fin);
        } else {
            panic!("Expected Stream frame");
        }
    }

    #[test]
    fn test_max_data_encode_decode() {
        let frame = QuicFrame::MaxData { maximum: 1_000_000 };
        let mut buf = [0u8; 16];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, QuicFrame::MaxData { maximum: 1_000_000 });
    }

    #[test]
    fn test_max_stream_data_encode_decode() {
        let frame = QuicFrame::MaxStreamData { stream_id: 4, maximum: 500_000 };
        let mut buf = [0u8; 16];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, QuicFrame::MaxStreamData { stream_id: 4, maximum: 500_000 });
    }

    #[test]
    fn test_max_streams_bidi_encode_decode() {
        let frame = QuicFrame::MaxStreamsBidi { maximum: 100 };
        let mut buf = [0u8; 8];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, QuicFrame::MaxStreamsBidi { maximum: 100 });
    }

    #[test]
    fn test_connection_close_encode_decode() {
        let frame = QuicFrame::ConnectionClose(ConnectionCloseFrame {
            error_code: 0x0A,
            frame_type: 0x06,
            reason: Vec::from(b"test error" as &[u8]),
        });
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        if let QuicFrame::ConnectionClose(cc) = decoded {
            assert_eq!(cc.error_code, 0x0A);
            assert_eq!(cc.frame_type, 0x06);
            assert_eq!(cc.reason, b"test error");
        } else {
            panic!("Expected ConnectionClose frame");
        }
    }

    #[test]
    fn test_new_connection_id_encode_decode() {
        let frame = QuicFrame::NewConnectionId(NewConnectionIdFrame {
            sequence: 1,
            retire_prior_to: 0,
            connection_id: vec![0x01, 0x02, 0x03, 0x04],
            stateless_reset_token: [0xAA; 16],
        });
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        if let QuicFrame::NewConnectionId(ncid) = decoded {
            assert_eq!(ncid.sequence, 1);
            assert_eq!(ncid.retire_prior_to, 0);
            assert_eq!(ncid.connection_id, vec![0x01, 0x02, 0x03, 0x04]);
            assert_eq!(ncid.stateless_reset_token, [0xAA; 16]);
        } else {
            panic!("Expected NewConnectionId frame");
        }
    }

    #[test]
    fn test_retire_connection_id_encode_decode() {
        let frame = QuicFrame::RetireConnectionId { sequence: 3 };
        let mut buf = [0u8; 8];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, QuicFrame::RetireConnectionId { sequence: 3 });
    }

    #[test]
    fn test_reset_stream_encode_decode() {
        let frame = QuicFrame::ResetStream {
            stream_id: 4,
            error_code: 0x01,
            final_size: 1000,
        };
        let mut buf = [0u8; 16];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn test_stop_sending_encode_decode() {
        let frame = QuicFrame::StopSending { stream_id: 8, error_code: 0x02 };
        let mut buf = [0u8; 8];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn test_path_challenge_encode_decode() {
        let frame = QuicFrame::PathChallenge { data: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] };
        let mut buf = [0u8; 16];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn test_path_response_encode_decode() {
        let frame = QuicFrame::PathResponse { data: [0xAA; 8] };
        let mut buf = [0u8; 16];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn test_decode_empty() {
        assert_eq!(QuicFrame::decode(&[]).err(), Some(NetError::PacketTooShort));
    }

    #[test]
    fn test_stream_fin_only() {
        let frame = QuicFrame::Stream(StreamFrame {
            stream_id: 0,
            offset: 0,
            data: vec![],
            fin: true,
        });
        let mut buf = [0u8; 16];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = QuicFrame::decode(&buf[..len]).unwrap();
        if let QuicFrame::Stream(s) = decoded {
            assert!(s.fin);
            assert!(s.data.is_empty());
        } else {
            panic!("Expected Stream frame");
        }
    }
}
