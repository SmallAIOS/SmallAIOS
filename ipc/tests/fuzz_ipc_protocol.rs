// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Property-based fuzz harness for IPC protocol parsing (Task 13.5).
//!
//! Feeds pseudo-random and malformed inputs into the IPC wire protocol
//! parsers to verify no panics occur. Uses a simple xorshift PRNG for
//! deterministic, cargo-test-compatible fuzzing.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use smallaios_ipc::inference_proto::{InferenceRequest, InferenceResponse};
use smallaios_ipc::tcp_transport::{Frame, FrameType, TcpTransport, TransportConfig};

// ============================================================================
// Simple xorshift64 PRNG for deterministic fuzz inputs
// ============================================================================

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.next_byte();
        }
    }

    fn gen_bytes(&mut self, max_len: usize) -> Vec<u8> {
        let len = (self.next() as usize) % (max_len + 1);
        let mut buf = vec![0u8; len];
        self.fill_bytes(&mut buf);
        buf
    }
}

// ============================================================================
// Frame decode fuzz
// ============================================================================

#[test]
fn fuzz_frame_decode_random_bytes() {
    let mut rng = Xorshift64::new(0xDEAD_BEEF_CAFE_1234);

    for _ in 0..10_000 {
        let data = rng.gen_bytes(64);
        // Must not panic
        let _ = Frame::decode(&data);
    }
}

#[test]
fn fuzz_frame_decode_with_valid_prefix() {
    let mut rng = Xorshift64::new(0x1234_5678);

    for _ in 0..5_000 {
        let body_len = (rng.next() % 256) as u32;
        let mut data = Vec::with_capacity(4 + body_len as usize);
        data.extend_from_slice(&body_len.to_be_bytes());
        for _ in 0..body_len {
            data.push(rng.next_byte());
        }
        let _ = Frame::decode(&data);
    }
}

#[test]
fn fuzz_frame_encode_decode_roundtrip() {
    let mut rng = Xorshift64::new(0xABCD_EF01);

    let frame_types = [
        FrameType::Init,
        FrameType::InitAck,
        FrameType::Open,
        FrameType::OpenAck,
        FrameType::Close,
        FrameType::KeepAlive,
        FrameType::Data,
        FrameType::Query,
        FrameType::Reply,
    ];

    for _ in 0..2_000 {
        let ft = frame_types[(rng.next() as usize) % frame_types.len()];
        let flags = rng.next_byte();
        let payload = rng.gen_bytes(128);
        let frame = Frame::new(ft, flags, payload.clone());

        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, ft);
        assert_eq!(decoded.flags, flags);
        assert_eq!(decoded.payload, payload);
        assert_eq!(consumed, encoded.len());
    }
}

// ============================================================================
// TCP transport state machine fuzz
// ============================================================================

#[test]
fn fuzz_tcp_transport_random_frames() {
    let mut rng = Xorshift64::new(0xFEED_FACE);

    let frame_types = [
        FrameType::Init,
        FrameType::InitAck,
        FrameType::Open,
        FrameType::OpenAck,
        FrameType::Close,
        FrameType::KeepAlive,
        FrameType::Data,
        FrameType::Query,
        FrameType::Reply,
    ];

    for _ in 0..1_000 {
        let mut transport = TcpTransport::new(TransportConfig::default());

        // Send random sequence of frames
        for _ in 0..10 {
            let ft = frame_types[(rng.next() as usize) % frame_types.len()];
            let payload = rng.gen_bytes(32);
            let frame = Frame::new(ft, 0x00, payload);
            // Must not panic (may return Err)
            let _ = transport.handle_frame(&frame);
        }
    }
}

#[test]
fn fuzz_tcp_transport_feed_random_data() {
    let mut rng = Xorshift64::new(0xCAFE_BABE);

    for _ in 0..2_000 {
        let mut transport = TcpTransport::new(TransportConfig::default());
        let data = rng.gen_bytes(128);
        transport.feed_data(&data);

        // Try decoding — must not panic
        loop {
            match transport.try_decode_frame() {
                Ok(Some(_frame)) => continue,
                Ok(None) => break,
                Err(_) => break,
            }
        }
    }
}

// ============================================================================
// Inference protocol fuzz
// ============================================================================

#[test]
fn fuzz_inference_request_decode_random() {
    let mut rng = Xorshift64::new(0x1111_2222_3333_4444);

    for _ in 0..10_000 {
        let data = rng.gen_bytes(256);
        // Must not panic
        let _ = InferenceRequest::decode(&data);
    }
}

#[test]
fn fuzz_inference_response_decode_random() {
    let mut rng = Xorshift64::new(0x5555_6666_7777_8888);

    for _ in 0..10_000 {
        let data = rng.gen_bytes(256);
        // Must not panic
        let _ = InferenceResponse::decode(&data);
    }
}

#[test]
fn fuzz_inference_request_with_valid_magic() {
    let mut rng = Xorshift64::new(0xAAAA_BBBB);

    for _ in 0..5_000 {
        let mut data = vec![0x4F, 0x4E, 0x4E, 0x58]; // "ONNX" magic
        data.push(1); // version
        // Random remainder
        let rest = rng.gen_bytes(128);
        data.extend_from_slice(&rest);
        let _ = InferenceRequest::decode(&data);
    }
}

#[test]
fn fuzz_inference_response_with_valid_magic() {
    let mut rng = Xorshift64::new(0xCCCC_DDDD);

    for _ in 0..5_000 {
        let mut data = vec![0x4F, 0x4E, 0x4E, 0x58]; // "ONNX" magic
        data.push(1); // version
        let rest = rng.gen_bytes(128);
        data.extend_from_slice(&rest);
        let _ = InferenceResponse::decode(&data);
    }
}

#[test]
fn fuzz_inference_request_encode_decode_roundtrip() {
    let mut rng = Xorshift64::new(0xEEEE_FFFF);

    let req_types = [
        smallaios_ipc::inference_proto::InferenceRequestType::LoadModel,
        smallaios_ipc::inference_proto::InferenceRequestType::UnloadModel,
        smallaios_ipc::inference_proto::InferenceRequestType::RunInference,
        smallaios_ipc::inference_proto::InferenceRequestType::GetModelInfo,
    ];

    for _ in 0..1_000 {
        let req = InferenceRequest {
            request_type: req_types[(rng.next() as usize) % req_types.len()],
            request_id: rng.next(),
            model_id: rng.next() as u32,
            inputs: Vec::new(),
            timeout_ms: rng.next() as u32,
        };
        let encoded = req.encode();
        let decoded = InferenceRequest::decode(&encoded).unwrap();
        assert_eq!(req, decoded);
    }
}

// ============================================================================
// Truncation fuzz — valid message truncated at every position
// ============================================================================

#[test]
fn fuzz_inference_request_truncation() {
    let req = InferenceRequest {
        request_type: smallaios_ipc::inference_proto::InferenceRequestType::RunInference,
        request_id: 12345,
        model_id: 42,
        inputs: vec![smallaios_ipc::inference_proto::TensorData {
            name: String::from("input"),
            data_type: smallaios_ipc::inference_proto::TensorDataType::Float32,
            shape: vec![1, 3, 224, 224],
            data: vec![0u8; 32],
        }],
        timeout_ms: 5000,
    };
    let encoded = req.encode();

    // Try decoding at every truncation point
    for len in 0..encoded.len() {
        let _ = InferenceRequest::decode(&encoded[..len]);
    }
}

#[test]
fn fuzz_inference_response_truncation() {
    let resp = InferenceResponse {
        request_id: 99,
        status: smallaios_ipc::inference_proto::ResponseStatus::Ok,
        outputs: vec![smallaios_ipc::inference_proto::TensorData {
            name: String::from("output"),
            data_type: smallaios_ipc::inference_proto::TensorDataType::Float32,
            shape: vec![1, 1000],
            data: vec![0u8; 64],
        }],
        elapsed_us: 1500,
    };
    let encoded = resp.encode();

    for len in 0..encoded.len() {
        let _ = InferenceResponse::decode(&encoded[..len]);
    }
}

#[test]
fn fuzz_frame_truncation() {
    let frame = Frame::new(FrameType::Data, 0x42, vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]);
    let encoded = frame.encode();

    for len in 0..encoded.len() {
        let _ = Frame::decode(&encoded[..len]);
    }
}
