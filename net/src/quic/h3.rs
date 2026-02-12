// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal HTTP/3 framing (RFC 9114).
//!
//! Implements the subset of HTTP/3 needed for the SmallAIOS management API:
//! - HTTP/3 frame encoding/decoding (DATA, HEADERS, SETTINGS)
//! - QPACK static-table-only header compression (no dynamic table)
//! - Simple request/response for GET/POST on /health, /metrics, /deploy
//!
//! This is intentionally minimal — SmallAIOS only needs to serve a handful
//! of management endpoints, not a full HTTP/3 implementation.

use super::packet::{decode_var_int, encode_var_int};
use crate::NetError;
use alloc::vec::Vec;

/// HTTP/3 frame types (RFC 9114 Section 7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3FrameType {
    /// DATA frame (type 0x00).
    Data,
    /// HEADERS frame (type 0x01).
    Headers,
    /// CANCEL_PUSH frame (type 0x03).
    CancelPush,
    /// SETTINGS frame (type 0x04).
    Settings,
    /// PUSH_PROMISE frame (type 0x05).
    PushPromise,
    /// GOAWAY frame (type 0x07).
    GoAway,
    /// MAX_PUSH_ID frame (type 0x0D).
    MaxPushId,
    /// Unknown frame type.
    Unknown(u64),
}

impl H3FrameType {
    /// Get the frame type ID.
    pub fn to_id(self) -> u64 {
        match self {
            H3FrameType::Data => 0x00,
            H3FrameType::Headers => 0x01,
            H3FrameType::CancelPush => 0x03,
            H3FrameType::Settings => 0x04,
            H3FrameType::PushPromise => 0x05,
            H3FrameType::GoAway => 0x07,
            H3FrameType::MaxPushId => 0x0D,
            H3FrameType::Unknown(id) => id,
        }
    }

    /// Parse from type ID.
    pub fn from_id(id: u64) -> Self {
        match id {
            0x00 => H3FrameType::Data,
            0x01 => H3FrameType::Headers,
            0x03 => H3FrameType::CancelPush,
            0x04 => H3FrameType::Settings,
            0x05 => H3FrameType::PushPromise,
            0x07 => H3FrameType::GoAway,
            0x0D => H3FrameType::MaxPushId,
            other => H3FrameType::Unknown(other),
        }
    }
}

/// HTTP/3 frame (generic: type + payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3Frame {
    /// Frame type.
    pub frame_type: H3FrameType,
    /// Frame payload.
    pub payload: Vec<u8>,
}

impl H3Frame {
    /// Create a DATA frame.
    pub fn data(payload: &[u8]) -> Self {
        Self {
            frame_type: H3FrameType::Data,
            payload: Vec::from(payload),
        }
    }

    /// Create a HEADERS frame with pre-encoded header block.
    pub fn headers(encoded_headers: &[u8]) -> Self {
        Self {
            frame_type: H3FrameType::Headers,
            payload: Vec::from(encoded_headers),
        }
    }

    /// Encode the frame into a buffer. Returns bytes written.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, NetError> {
        let type_id = self.frame_type.to_id();
        let mut off = encode_var_int(type_id, buf);
        off += encode_var_int(self.payload.len() as u64, &mut buf[off..]);
        if buf.len() < off + self.payload.len() {
            return Err(NetError::BufferTooSmall);
        }
        buf[off..off + self.payload.len()].copy_from_slice(&self.payload);
        off += self.payload.len();
        Ok(off)
    }

    /// Decode a frame from a buffer. Returns (frame, bytes_consumed).
    pub fn decode(data: &[u8]) -> Result<(Self, usize), NetError> {
        if data.is_empty() {
            return Err(NetError::PacketTooShort);
        }
        let (type_id, c1) = decode_var_int(data)?;
        let (length, c2) = decode_var_int(&data[c1..])?;
        let off = c1 + c2;
        let length = length as usize;
        if data.len() < off + length {
            return Err(NetError::PacketTooShort);
        }
        let payload = Vec::from(&data[off..off + length]);
        Ok((
            H3Frame {
                frame_type: H3FrameType::from_id(type_id),
                payload,
            },
            off + length,
        ))
    }
}

/// HTTP method (subset used by SmallAIOS management API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// GET request.
    Get,
    /// POST request.
    Post,
    /// HEAD request.
    Head,
}

impl HttpMethod {
    /// QPACK static table index for :method.
    pub fn static_index(self) -> u8 {
        match self {
            HttpMethod::Get => 17,  // :method GET
            HttpMethod::Post => 20, // :method POST
            HttpMethod::Head => 18, // :method HEAD (not in static table, use literal)
        }
    }

    /// Parse from method bytes.
    pub fn from_bytes(method: &[u8]) -> Option<Self> {
        match method {
            b"GET" => Some(HttpMethod::Get),
            b"POST" => Some(HttpMethod::Post),
            b"HEAD" => Some(HttpMethod::Head),
            _ => None,
        }
    }

    /// Method name as bytes.
    pub fn as_bytes(self) -> &'static [u8] {
        match self {
            HttpMethod::Get => b"GET",
            HttpMethod::Post => b"POST",
            HttpMethod::Head => b"HEAD",
        }
    }
}

/// HTTP/3 status code (subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpStatus {
    /// 200 OK.
    Ok,
    /// 204 No Content.
    NoContent,
    /// 400 Bad Request.
    BadRequest,
    /// 404 Not Found.
    NotFound,
    /// 405 Method Not Allowed.
    MethodNotAllowed,
    /// 500 Internal Server Error.
    InternalServerError,
}

impl HttpStatus {
    /// Get the numeric status code.
    pub fn code(self) -> u16 {
        match self {
            HttpStatus::Ok => 200,
            HttpStatus::NoContent => 204,
            HttpStatus::BadRequest => 400,
            HttpStatus::NotFound => 404,
            HttpStatus::MethodNotAllowed => 405,
            HttpStatus::InternalServerError => 500,
        }
    }

    /// QPACK static table index for :status (only some are indexed).
    pub fn static_index(self) -> Option<u8> {
        match self {
            HttpStatus::Ok => Some(25),        // :status 200
            HttpStatus::NoContent => Some(26), // :status 204
            HttpStatus::NotFound => Some(27),  // :status 404
            _ => None,
        }
    }

    /// Parse from status code.
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            200 => Some(HttpStatus::Ok),
            204 => Some(HttpStatus::NoContent),
            400 => Some(HttpStatus::BadRequest),
            404 => Some(HttpStatus::NotFound),
            405 => Some(HttpStatus::MethodNotAllowed),
            500 => Some(HttpStatus::InternalServerError),
            _ => None,
        }
    }
}

/// HTTP/3 SETTINGS parameters (RFC 9114 Section 7.2.4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3Settings {
    /// SETTINGS_MAX_FIELD_SECTION_SIZE (0x06).
    pub max_field_section_size: u64,
    /// SETTINGS_QPACK_MAX_TABLE_CAPACITY (0x01).
    /// SmallAIOS sets this to 0 (no dynamic table).
    pub qpack_max_table_capacity: u64,
    /// SETTINGS_QPACK_BLOCKED_STREAMS (0x07).
    /// SmallAIOS sets this to 0 (no blocked streams).
    pub qpack_blocked_streams: u64,
}

impl Default for H3Settings {
    fn default() -> Self {
        Self {
            max_field_section_size: 16384,
            qpack_max_table_capacity: 0, // Static table only
            qpack_blocked_streams: 0,
        }
    }
}

impl H3Settings {
    /// Encode settings into a SETTINGS frame payload.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, NetError> {
        let mut off = 0;

        // QPACK max table capacity (0x01)
        off += encode_var_int(0x01, &mut buf[off..]);
        off += encode_var_int(self.qpack_max_table_capacity, &mut buf[off..]);

        // Max field section size (0x06)
        off += encode_var_int(0x06, &mut buf[off..]);
        off += encode_var_int(self.max_field_section_size, &mut buf[off..]);

        // QPACK blocked streams (0x07)
        off += encode_var_int(0x07, &mut buf[off..]);
        off += encode_var_int(self.qpack_blocked_streams, &mut buf[off..]);

        Ok(off)
    }

    /// Decode settings from a SETTINGS frame payload.
    pub fn decode(data: &[u8]) -> Result<Self, NetError> {
        let mut settings = H3Settings::default();
        let mut off = 0;

        while off < data.len() {
            let (id, c1) = decode_var_int(&data[off..])?;
            off += c1;
            let (value, c2) = decode_var_int(&data[off..])?;
            off += c2;

            match id {
                0x01 => settings.qpack_max_table_capacity = value,
                0x06 => settings.max_field_section_size = value,
                0x07 => settings.qpack_blocked_streams = value,
                _ => {} // Ignore unknown settings
            }
        }

        Ok(settings)
    }
}

/// Minimal QPACK encoder (static table only, no Huffman).
///
/// Encodes HTTP field sections using only QPACK static table references
/// and literal header fields. This avoids any dynamic table state.
pub struct QpackEncoder;

impl QpackEncoder {
    /// Encode a request header block (minimal: :method, :path, :scheme).
    ///
    /// Format: Required Insert Count (0) + Delta Base (0) + field lines.
    pub fn encode_request(
        method: HttpMethod,
        path: &[u8],
        buf: &mut [u8],
    ) -> Result<usize, NetError> {
        let mut off = 0;

        // Required Insert Count = 0 (no dynamic table entries needed)
        if buf.len() < 2 {
            return Err(NetError::BufferTooSmall);
        }
        buf[off] = 0x00; // Required Insert Count (encoded as 0)
        off += 1;
        buf[off] = 0x00; // Delta Base
        off += 1;

        // :method — use static table indexed reference
        // Indexed Field Line (static): 1nnnnnnn where n = static table index
        if method == HttpMethod::Get || method == HttpMethod::Post {
            if buf.len() < off + 1 {
                return Err(NetError::BufferTooSmall);
            }
            buf[off] = 0xC0 | method.static_index();
            off += 1;
        } else {
            // Literal with name reference for other methods
            off += Self::encode_literal_name_ref(23, method.as_bytes(), &mut buf[off..])?;
        }

        // :scheme https — static index 23
        if buf.len() < off + 1 {
            return Err(NetError::BufferTooSmall);
        }
        buf[off] = 0xC0 | 23; // :scheme https
        off += 1;

        // :path — literal with name reference (static index 1 for :path /)
        off += Self::encode_literal_name_ref(1, path, &mut buf[off..])?;

        Ok(off)
    }

    /// Encode a response header block (minimal: :status).
    pub fn encode_response(
        status: HttpStatus,
        content_type: Option<&[u8]>,
        buf: &mut [u8],
    ) -> Result<usize, NetError> {
        let mut off = 0;

        if buf.len() < 2 {
            return Err(NetError::BufferTooSmall);
        }
        buf[off] = 0x00; // Required Insert Count
        off += 1;
        buf[off] = 0x00; // Delta Base
        off += 1;

        // :status — indexed if in static table
        if let Some(idx) = status.static_index() {
            if buf.len() < off + 1 {
                return Err(NetError::BufferTooSmall);
            }
            buf[off] = 0xC0 | idx;
            off += 1;
        } else {
            // Literal with name reference
            let code_str = status.code();
            let mut code_buf = [0u8; 3];
            code_buf[0] = b'0' + (code_str / 100) as u8;
            code_buf[1] = b'0' + ((code_str / 10) % 10) as u8;
            code_buf[2] = b'0' + (code_str % 10) as u8;
            off += Self::encode_literal_name_ref(24, &code_buf, &mut buf[off..])?;
        }

        // content-type (optional)
        if let Some(ct) = content_type {
            off += Self::encode_literal_name_ref(44, ct, &mut buf[off..])?;
        }

        Ok(off)
    }

    /// Encode a literal header field with name reference.
    ///
    /// Format: 01NTnnnn (N=1 for static, T=0 for no Huffman) + value length + value
    fn encode_literal_name_ref(
        static_index: u8,
        value: &[u8],
        buf: &mut [u8],
    ) -> Result<usize, NetError> {
        // Need at least: prefix(1) + value_length(1) + value
        let needed = 2 + value.len();
        if buf.len() < needed {
            return Err(NetError::BufferTooSmall);
        }

        let mut off = 0;
        // 0101nnnn = literal with name reference, static table, no Huffman
        buf[off] = 0x50 | (static_index & 0x0F);
        off += 1;

        // Value length (simplified: single byte, max 127)
        if value.len() > 127 {
            return Err(NetError::BufferTooSmall);
        }
        buf[off] = value.len() as u8;
        off += 1;

        buf[off..off + value.len()].copy_from_slice(value);
        off += value.len();

        Ok(off)
    }
}

/// SmallAIOS management API paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementPath {
    /// GET /health — liveness/readiness probe.
    Health,
    /// GET /metrics — Prometheus metrics.
    Metrics,
    /// POST /deploy — deploy ONNX model.
    Deploy,
    /// POST /undeploy — remove deployed model.
    Undeploy,
    /// GET /status — system status.
    Status,
}

impl ManagementPath {
    /// Parse from path bytes.
    pub fn from_path(path: &[u8]) -> Option<Self> {
        match path {
            b"/health" => Some(Self::Health),
            b"/metrics" => Some(Self::Metrics),
            b"/deploy" => Some(Self::Deploy),
            b"/undeploy" => Some(Self::Undeploy),
            b"/status" => Some(Self::Status),
            _ => None,
        }
    }

    /// Path as bytes.
    pub fn as_bytes(self) -> &'static [u8] {
        match self {
            Self::Health => b"/health",
            Self::Metrics => b"/metrics",
            Self::Deploy => b"/deploy",
            Self::Undeploy => b"/undeploy",
            Self::Status => b"/status",
        }
    }

    /// Expected HTTP method for this path.
    pub fn expected_method(self) -> HttpMethod {
        match self {
            Self::Health | Self::Metrics | Self::Status => HttpMethod::Get,
            Self::Deploy | Self::Undeploy => HttpMethod::Post,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ---- H3Frame ----

    #[test]
    fn test_data_frame_encode_decode() {
        let frame = H3Frame::data(b"hello h3");
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();

        let (decoded, consumed) = H3Frame::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        assert_eq!(decoded.frame_type, H3FrameType::Data);
        assert_eq!(decoded.payload, b"hello h3");
    }

    #[test]
    fn test_headers_frame_encode_decode() {
        let frame = H3Frame::headers(&[0x00, 0x00, 0xC0 | 17]); // Minimal QPACK
        let mut buf = [0u8; 32];
        let len = frame.encode(&mut buf).unwrap();

        let (decoded, consumed) = H3Frame::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        assert_eq!(decoded.frame_type, H3FrameType::Headers);
    }

    #[test]
    fn test_frame_encode_buffer_too_small() {
        let frame = H3Frame::data(&[0xAA; 100]);
        let mut buf = [0u8; 10];
        assert_eq!(frame.encode(&mut buf).err(), Some(NetError::BufferTooSmall));
    }

    #[test]
    fn test_frame_decode_empty() {
        assert_eq!(H3Frame::decode(&[]).err(), Some(NetError::PacketTooShort));
    }

    #[test]
    fn test_frame_decode_truncated() {
        // Type=0x00 (DATA), length=10, but only 5 bytes of payload
        let mut buf = [0u8; 8];
        let mut off = encode_var_int(0x00, &mut buf);
        off += encode_var_int(10, &mut buf[off..]);
        assert_eq!(
            H3Frame::decode(&buf[..off + 5]).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_frame_empty_payload() {
        let frame = H3Frame::data(&[]);
        let mut buf = [0u8; 8];
        let len = frame.encode(&mut buf).unwrap();
        let (decoded, _) = H3Frame::decode(&buf[..len]).unwrap();
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_frame_type_roundtrip() {
        for id in [0x00, 0x01, 0x03, 0x04, 0x05, 0x07, 0x0D] {
            let ft = H3FrameType::from_id(id);
            assert_eq!(ft.to_id(), id);
        }
    }

    #[test]
    fn test_frame_type_unknown() {
        let ft = H3FrameType::from_id(0xFF);
        assert_eq!(ft, H3FrameType::Unknown(0xFF));
        assert_eq!(ft.to_id(), 0xFF);
    }

    // ---- H3Settings ----

    #[test]
    fn test_settings_default() {
        let s = H3Settings::default();
        assert_eq!(s.qpack_max_table_capacity, 0);
        assert_eq!(s.qpack_blocked_streams, 0);
        assert_eq!(s.max_field_section_size, 16384);
    }

    #[test]
    fn test_settings_encode_decode() {
        let settings = H3Settings {
            max_field_section_size: 8192,
            qpack_max_table_capacity: 0,
            qpack_blocked_streams: 0,
        };
        let mut buf = [0u8; 32];
        let len = settings.encode(&mut buf).unwrap();

        let decoded = H3Settings::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.max_field_section_size, 8192);
        assert_eq!(decoded.qpack_max_table_capacity, 0);
        assert_eq!(decoded.qpack_blocked_streams, 0);
    }

    #[test]
    fn test_settings_unknown_params() {
        // Encode an unknown setting ID 0xFF = 42
        let mut buf = [0u8; 8];
        let mut off = encode_var_int(0xFF, &mut buf);
        off += encode_var_int(42, &mut buf[off..]);

        // Should parse without error (ignore unknown)
        let settings = H3Settings::decode(&buf[..off]).unwrap();
        assert_eq!(settings.max_field_section_size, 16384); // default
    }

    #[test]
    fn test_settings_frame() {
        let settings = H3Settings::default();
        let mut payload = [0u8; 32];
        let payload_len = settings.encode(&mut payload).unwrap();

        let frame = H3Frame {
            frame_type: H3FrameType::Settings,
            payload: Vec::from(&payload[..payload_len]),
        };
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();

        let (decoded, _) = H3Frame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.frame_type, H3FrameType::Settings);
        let decoded_settings = H3Settings::decode(&decoded.payload).unwrap();
        assert_eq!(decoded_settings.qpack_max_table_capacity, 0);
    }

    // ---- HttpMethod ----

    #[test]
    fn test_http_method_from_bytes() {
        assert_eq!(HttpMethod::from_bytes(b"GET"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::from_bytes(b"POST"), Some(HttpMethod::Post));
        assert_eq!(HttpMethod::from_bytes(b"HEAD"), Some(HttpMethod::Head));
        assert_eq!(HttpMethod::from_bytes(b"PUT"), None);
    }

    #[test]
    fn test_http_method_as_bytes() {
        assert_eq!(HttpMethod::Get.as_bytes(), b"GET");
        assert_eq!(HttpMethod::Post.as_bytes(), b"POST");
        assert_eq!(HttpMethod::Head.as_bytes(), b"HEAD");
    }

    // ---- HttpStatus ----

    #[test]
    fn test_http_status_codes() {
        assert_eq!(HttpStatus::Ok.code(), 200);
        assert_eq!(HttpStatus::NoContent.code(), 204);
        assert_eq!(HttpStatus::BadRequest.code(), 400);
        assert_eq!(HttpStatus::NotFound.code(), 404);
        assert_eq!(HttpStatus::MethodNotAllowed.code(), 405);
        assert_eq!(HttpStatus::InternalServerError.code(), 500);
    }

    #[test]
    fn test_http_status_from_code() {
        assert_eq!(HttpStatus::from_code(200), Some(HttpStatus::Ok));
        assert_eq!(HttpStatus::from_code(404), Some(HttpStatus::NotFound));
        assert_eq!(HttpStatus::from_code(999), None);
    }

    #[test]
    fn test_http_status_static_index() {
        assert!(HttpStatus::Ok.static_index().is_some());
        assert!(HttpStatus::NoContent.static_index().is_some());
        assert!(HttpStatus::NotFound.static_index().is_some());
        assert!(HttpStatus::BadRequest.static_index().is_none());
    }

    // ---- QPACK Encoder ----

    #[test]
    fn test_qpack_encode_get_request() {
        let mut buf = [0u8; 64];
        let len = QpackEncoder::encode_request(HttpMethod::Get, b"/health", &mut buf).unwrap();
        assert!(len > 0);
        // First two bytes are required insert count + delta base
        assert_eq!(buf[0], 0x00);
        assert_eq!(buf[1], 0x00);
    }

    #[test]
    fn test_qpack_encode_post_request() {
        let mut buf = [0u8; 64];
        let len = QpackEncoder::encode_request(HttpMethod::Post, b"/deploy", &mut buf).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn test_qpack_encode_response_200() {
        let mut buf = [0u8; 64];
        let len =
            QpackEncoder::encode_response(HttpStatus::Ok, Some(b"application/json"), &mut buf)
                .unwrap();
        assert!(len > 0);
        assert_eq!(buf[0], 0x00); // Required insert count
    }

    #[test]
    fn test_qpack_encode_response_no_content_type() {
        let mut buf = [0u8; 32];
        let len = QpackEncoder::encode_response(HttpStatus::NotFound, None, &mut buf).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn test_qpack_encode_response_non_indexed_status() {
        let mut buf = [0u8; 32];
        let len = QpackEncoder::encode_response(HttpStatus::BadRequest, None, &mut buf).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn test_qpack_encode_request_buffer_too_small() {
        let mut buf = [0u8; 1];
        assert_eq!(
            QpackEncoder::encode_request(HttpMethod::Get, b"/", &mut buf).err(),
            Some(NetError::BufferTooSmall)
        );
    }

    // ---- ManagementPath ----

    #[test]
    fn test_management_path_from_path() {
        assert_eq!(
            ManagementPath::from_path(b"/health"),
            Some(ManagementPath::Health)
        );
        assert_eq!(
            ManagementPath::from_path(b"/metrics"),
            Some(ManagementPath::Metrics)
        );
        assert_eq!(
            ManagementPath::from_path(b"/deploy"),
            Some(ManagementPath::Deploy)
        );
        assert_eq!(
            ManagementPath::from_path(b"/undeploy"),
            Some(ManagementPath::Undeploy)
        );
        assert_eq!(
            ManagementPath::from_path(b"/status"),
            Some(ManagementPath::Status)
        );
        assert_eq!(ManagementPath::from_path(b"/unknown"), None);
    }

    #[test]
    fn test_management_path_as_bytes() {
        assert_eq!(ManagementPath::Health.as_bytes(), b"/health");
        assert_eq!(ManagementPath::Deploy.as_bytes(), b"/deploy");
    }

    #[test]
    fn test_management_path_expected_method() {
        assert_eq!(ManagementPath::Health.expected_method(), HttpMethod::Get);
        assert_eq!(ManagementPath::Metrics.expected_method(), HttpMethod::Get);
        assert_eq!(ManagementPath::Status.expected_method(), HttpMethod::Get);
        assert_eq!(ManagementPath::Deploy.expected_method(), HttpMethod::Post);
        assert_eq!(ManagementPath::Undeploy.expected_method(), HttpMethod::Post);
    }

    // ---- Multiple frames ----

    #[test]
    fn test_multiple_frames_in_buffer() {
        let frame1 = H3Frame::data(b"first");
        let frame2 = H3Frame::data(b"second");

        let mut buf = [0u8; 64];
        let len1 = frame1.encode(&mut buf).unwrap();
        let len2 = frame2.encode(&mut buf[len1..]).unwrap();

        let (d1, c1) = H3Frame::decode(&buf[..len1 + len2]).unwrap();
        assert_eq!(d1.payload, b"first");
        let (d2, _) = H3Frame::decode(&buf[c1..len1 + len2]).unwrap();
        assert_eq!(d2.payload, b"second");
    }
}
