// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the SmallAIOS ONNX runtime.
//!
//! Covers session creation and configuration, tensor construction,
//! inference I/O types, model loading with malformed data, and
//! protobuf decoding with valid and invalid inputs.

extern crate alloc;

use smallaios_onnx_rt::optimizer::OptimizationLevel;
use smallaios_onnx_rt::protobuf::{ProtoDecoder, ProtoError, WireType};
use smallaios_onnx_rt::session::{
    load_model, validate_model, InferenceInput, InferenceOutput, Session, SessionConfig,
    SessionError,
};
use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// 1. Session creation and SessionConfig
// ---------------------------------------------------------------------------

#[test]
fn session_default_config() {
    let config = SessionConfig::default();
    assert_eq!(config.optimization_level, OptimizationLevel::Basic);
    assert!(!config.enable_profiling);
    assert_eq!(config.max_batch_size, 1);
    assert_eq!(config.thread_count, 1);
}

#[test]
fn session_custom_config() {
    let config = SessionConfig {
        optimization_level: OptimizationLevel::Extended,
        enable_profiling: true,
        max_batch_size: 16,
        thread_count: 4,
        parallel: smallaios_onnx_rt::parallel::ParallelConfig::default(),
        gpu_config: None,
        gpu_residency: smallaios_onnx_rt::session::GpuResidency::default(),
    };
    let session = Session::new(config);
    assert!(!session.is_initialized());
    assert!(session.input_names().is_empty());
    assert!(session.output_names().is_empty());
    assert!(session.graph.is_none());
}

#[test]
fn session_ids_are_unique_and_monotonic() {
    let s1 = Session::new(SessionConfig::default());
    let s2 = Session::new(SessionConfig::default());
    let s3 = Session::new(SessionConfig::default());
    assert_ne!(s1.id, s2.id);
    assert_ne!(s2.id, s3.id);
    assert!(s2.id.0 > s1.id.0);
    assert!(s3.id.0 > s2.id.0);
}

#[test]
fn session_no_optimization_config() {
    let config = SessionConfig {
        optimization_level: OptimizationLevel::None,
        enable_profiling: false,
        max_batch_size: 1,
        thread_count: 1,
        parallel: smallaios_onnx_rt::parallel::ParallelConfig::default(),
        gpu_config: None,
        gpu_residency: smallaios_onnx_rt::session::GpuResidency::default(),
    };
    let session = Session::new(config);
    assert!(!session.is_initialized());
}

// ---------------------------------------------------------------------------
// 2. Tensor, TensorShape, and DataType
// ---------------------------------------------------------------------------

#[test]
fn tensor_with_float32_shape() {
    let shape = TensorShape::new(vec![1, 3, 224, 224]);
    assert_eq!(shape.ndim(), 4);
    assert_eq!(shape.total_elements(), 3 * 224 * 224);
    assert!(shape.is_valid());

    let tensor = Tensor::new(DataType::Float, shape, String::from("images"));
    assert_eq!(tensor.data_type, DataType::Float);
    assert_eq!(tensor.name, "images");
    assert!(tensor.is_empty());
    assert_eq!(tensor.byte_size(), 3 * 224 * 224 * 4);
}

#[test]
fn tensor_scalar() {
    let shape = TensorShape::scalar();
    assert!(shape.is_scalar());
    assert_eq!(shape.ndim(), 0);
    assert_eq!(shape.total_elements(), 1);

    let tensor = Tensor::new(DataType::Double, shape, String::from("loss"));
    assert_eq!(tensor.byte_size(), 8);
}

#[test]
fn tensor_with_symbolic_dims() {
    let shape = TensorShape::new(vec![-1, 3, 224, 224]);
    assert!(shape.is_valid());
    // Symbolic dims (-1) treated as 1 for element count.
    assert_eq!(shape.total_elements(), 3 * 224 * 224);
}

#[test]
fn tensor_shape_invalid_dims() {
    // Zero-sized dimension is invalid.
    assert!(!TensorShape::new(vec![0, 3]).is_valid());
    // Negative dims other than -1 are invalid.
    assert!(!TensorShape::new(vec![-2, 5]).is_valid());
}

#[test]
fn tensor_with_raw_data() {
    let shape = TensorShape::new(vec![4]);
    let mut tensor = Tensor::new(DataType::Uint8, shape, String::from("mask"));
    assert!(tensor.is_empty());

    tensor.raw_data = vec![0, 1, 1, 0];
    assert!(!tensor.is_empty());
    assert_eq!(tensor.byte_size(), 4);
}

#[test]
fn data_type_properties() {
    assert!(DataType::Float.is_float());
    assert!(!DataType::Float.is_integer());
    assert_eq!(DataType::Float.element_size(), 4);
    assert_eq!(DataType::Float.name(), "float32");

    assert!(DataType::Int64.is_integer());
    assert!(!DataType::Int64.is_float());
    assert_eq!(DataType::Int64.element_size(), 8);

    assert_eq!(DataType::String.element_size(), 0);
}

#[test]
fn data_type_from_i32_round_trip() {
    let known_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16];
    for &val in &known_types {
        assert!(DataType::from_i32(val).is_some());
    }
    // Unknown type codes return None.
    assert!(DataType::from_i32(0).is_none());
    assert!(DataType::from_i32(14).is_none());
    assert!(DataType::from_i32(99).is_none());
}

// ---------------------------------------------------------------------------
// 3. InferenceInput / InferenceOutput construction
// ---------------------------------------------------------------------------

#[test]
fn inference_input_construction() {
    let tensor = Tensor::new(
        DataType::Float,
        TensorShape::new(vec![1, 3, 224, 224]),
        String::from("input_0"),
    );
    let input = InferenceInput {
        name: String::from("input_0"),
        tensor,
    };
    assert_eq!(input.name, "input_0");
    assert_eq!(input.tensor.data_type, DataType::Float);
    assert_eq!(input.tensor.shape.ndim(), 4);
}

#[test]
fn inference_output_construction() {
    let tensor = Tensor::new(
        DataType::Float,
        TensorShape::new(vec![1, 1000]),
        String::from("probabilities"),
    );
    let output = InferenceOutput {
        name: String::from("probabilities"),
        tensor,
    };
    assert_eq!(output.name, "probabilities");
    assert_eq!(output.tensor.shape.total_elements(), 1000);
}

#[test]
fn inference_input_with_int64_tensor() {
    let tensor = Tensor::new(
        DataType::Int64,
        TensorShape::new(vec![1, 128]),
        String::from("token_ids"),
    );
    let input = InferenceInput {
        name: String::from("token_ids"),
        tensor,
    };
    assert_eq!(input.tensor.byte_size(), 128 * 8);
}

// ---------------------------------------------------------------------------
// 4. Model loading with malformed data
// ---------------------------------------------------------------------------

#[test]
fn load_model_empty_data_fails() {
    let result = load_model(&[]);
    assert!(result.is_err());
    match result {
        Err(SessionError::ModelLoadFailed(msg)) => {
            assert!(msg.contains("too small"));
        }
        other => panic!("expected ModelLoadFailed, got {:?}", other),
    }
}

#[test]
fn load_model_too_small_fails() {
    let data = [0x08, 0x01, 0x00, 0x00]; // Only 4 bytes, below minimum.
    let result = load_model(&data);
    assert!(result.is_err());
    match result {
        Err(SessionError::ModelLoadFailed(msg)) => {
            assert!(msg.contains("too small"));
        }
        other => panic!("expected ModelLoadFailed, got {:?}", other),
    }
}

#[test]
fn load_model_invalid_magic_byte_fails() {
    // First byte is not the ONNX magic (0x08).
    let data = [0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = load_model(&data);
    assert!(result.is_err());
    match result {
        Err(SessionError::ModelLoadFailed(msg)) => {
            assert!(msg.contains("magic"));
        }
        other => panic!("expected ModelLoadFailed, got {:?}", other),
    }
}

#[test]
fn load_model_all_zeros_invalid_magic() {
    let data = [0x00; 16];
    let result = load_model(&data);
    assert!(result.is_err());
    match result {
        Err(SessionError::ModelLoadFailed(msg)) => {
            assert!(msg.contains("magic"));
        }
        other => panic!("expected ModelLoadFailed, got {:?}", other),
    }
}

#[test]
fn load_model_valid_header_decodes_model() {
    // Valid magic byte (0x08) and sufficient size; decoder now parses protobuf.
    // Construct a minimal valid model: ir_version=7, opset version 17.
    let data = [
        0x08, 0x07, // field 1, varint 7 = ir_version
        0x42, 0x04, // field 8, length 4 (opset_import)
        0x0A, 0x00, // field 1 (domain), length 0
        0x10, 0x11, // field 2 (version), varint 17
    ];
    let result = load_model(&data);
    assert!(result.is_ok(), "expected Ok, got {:?}", result);
    let model = result.unwrap();
    assert_eq!(model.ir_version, 7);
    assert_eq!(model.opset_import.len(), 1);
    assert_eq!(model.opset_import[0].version, 17);
}

#[test]
fn load_model_random_garbage_fails_gracefully() {
    // Random bytes that happen to start with a non-magic byte.
    let data = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x00, 0x00];
    let result = load_model(&data);
    assert!(result.is_err());
    // Should not panic regardless of content.
}

// ---------------------------------------------------------------------------
// 4b. Model validation
// ---------------------------------------------------------------------------

#[test]
fn validate_model_without_graph_fails() {
    use smallaios_onnx_rt::onnx_types::ModelProto;
    let model = ModelProto {
        ir_version: 9,
        graph: None,
        ..ModelProto::default()
    };
    let result = validate_model(&model);
    assert!(matches!(result, Err(SessionError::InvalidModel(_))));
}

// ---------------------------------------------------------------------------
// 5. Session run without initialization
// ---------------------------------------------------------------------------

#[test]
fn session_run_without_init_fails() {
    let session = Session::new(SessionConfig::default());
    let input = InferenceInput {
        name: String::from("x"),
        tensor: Tensor::new(
            DataType::Float,
            TensorShape::new(vec![1, 3]),
            String::from("x"),
        ),
    };
    let result = session.run(&[input]);
    match result {
        Err(SessionError::ExecutionFailed(msg)) => {
            assert!(msg.contains("not initialized"));
        }
        other => panic!("expected ExecutionFailed, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 6. Protobuf decoder with valid data
// ---------------------------------------------------------------------------

#[test]
fn protobuf_decode_varint_values() {
    // Single-byte varint: 1
    let mut dec = ProtoDecoder::new(&[0x01]);
    assert_eq!(dec.read_varint().unwrap(), 1);

    // Two-byte varint: 300 = 0xAC 0x02
    let mut dec = ProtoDecoder::new(&[0xAC, 0x02]);
    assert_eq!(dec.read_varint().unwrap(), 300);

    // Single-byte varint: 0
    let mut dec = ProtoDecoder::new(&[0x00]);
    assert_eq!(dec.read_varint().unwrap(), 0);
}

#[test]
fn protobuf_decode_tag() {
    // Field 1, wire type Varint: tag = (1 << 3) | 0 = 0x08
    let mut dec = ProtoDecoder::new(&[0x08]);
    let header = dec.read_tag().unwrap();
    assert_eq!(header.field_number, 1);
    assert_eq!(header.wire_type, WireType::Varint);

    // Field 2, wire type LengthDelimited: tag = (2 << 3) | 2 = 0x12
    let mut dec = ProtoDecoder::new(&[0x12]);
    let header = dec.read_tag().unwrap();
    assert_eq!(header.field_number, 2);
    assert_eq!(header.wire_type, WireType::LengthDelimited);
}

#[test]
fn protobuf_decode_length_delimited() {
    // Length 3, then bytes [0x41, 0x42, 0x43] ("ABC")
    let data = [0x03, 0x41, 0x42, 0x43];
    let mut dec = ProtoDecoder::new(&data);
    let bytes = dec.read_length_delimited().unwrap();
    assert_eq!(bytes, b"ABC");
    assert!(dec.is_empty());
}

#[test]
fn protobuf_decode_string_valid_utf8() {
    let msg = "hello";
    let mut data = vec![msg.len() as u8];
    data.extend_from_slice(msg.as_bytes());
    let mut dec = ProtoDecoder::new(&data);
    assert_eq!(dec.read_string().unwrap(), "hello");
}

#[test]
fn protobuf_decode_fixed_values() {
    // fixed32: 0x01020304 little-endian
    let mut dec = ProtoDecoder::new(&[0x04, 0x03, 0x02, 0x01]);
    assert_eq!(dec.read_fixed32().unwrap(), 0x01020304);

    // fixed64
    let mut dec = ProtoDecoder::new(&[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    assert_eq!(dec.read_fixed64().unwrap(), 0x0102030405060708);
}

#[test]
fn protobuf_decoder_remaining_and_position() {
    let data = [0x01, 0x02, 0x03, 0x04];
    let mut dec = ProtoDecoder::new(&data);
    assert_eq!(dec.remaining(), 4);
    assert_eq!(dec.position(), 0);
    assert!(!dec.is_empty());

    dec.read_byte().unwrap();
    assert_eq!(dec.remaining(), 3);
    assert_eq!(dec.position(), 1);
}

#[test]
fn protobuf_skip_field_all_wire_types() {
    // Skip Varint
    let mut dec = ProtoDecoder::new(&[0xAC, 0x02, 0xFF]);
    dec.skip_field(WireType::Varint).unwrap();
    assert_eq!(dec.read_byte().unwrap(), 0xFF);

    // Skip Fixed32
    let mut dec = ProtoDecoder::new(&[0x01, 0x02, 0x03, 0x04, 0xFF]);
    dec.skip_field(WireType::Fixed32).unwrap();
    assert_eq!(dec.read_byte().unwrap(), 0xFF);

    // Skip Fixed64
    let mut dec = ProtoDecoder::new(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xFF]);
    dec.skip_field(WireType::Fixed64).unwrap();
    assert_eq!(dec.read_byte().unwrap(), 0xFF);

    // Skip LengthDelimited (length 2, two payload bytes)
    let mut dec = ProtoDecoder::new(&[0x02, 0xAA, 0xBB, 0xFF]);
    dec.skip_field(WireType::LengthDelimited).unwrap();
    assert_eq!(dec.read_byte().unwrap(), 0xFF);
}

// ---------------------------------------------------------------------------
// 7. Protobuf decoder with invalid data
// ---------------------------------------------------------------------------

#[test]
fn protobuf_empty_decoder_all_reads_fail() {
    let mut dec = ProtoDecoder::new(&[]);
    assert!(dec.is_empty());
    assert_eq!(dec.read_varint().unwrap_err(), ProtoError::UnexpectedEof);
    assert_eq!(dec.read_tag().unwrap_err(), ProtoError::UnexpectedEof);
    assert_eq!(dec.read_fixed32().unwrap_err(), ProtoError::UnexpectedEof);
    assert_eq!(dec.read_fixed64().unwrap_err(), ProtoError::UnexpectedEof);
    assert_eq!(
        dec.read_length_delimited().unwrap_err(),
        ProtoError::UnexpectedEof
    );
    assert_eq!(dec.read_byte().unwrap_err(), ProtoError::UnexpectedEof);
}

#[test]
fn protobuf_invalid_varint_too_many_continuation_bytes() {
    let data = [0xFF; 11];
    let mut dec = ProtoDecoder::new(&data);
    assert_eq!(dec.read_varint().unwrap_err(), ProtoError::InvalidVarint);
}

#[test]
fn protobuf_length_delimited_exceeds_buffer() {
    // Claims length 5 but only 2 payload bytes follow.
    let data = [0x05, 0xAA, 0xBB];
    let mut dec = ProtoDecoder::new(&data);
    assert_eq!(
        dec.read_length_delimited().unwrap_err(),
        ProtoError::BufferTooSmall
    );
}

#[test]
fn protobuf_string_invalid_utf8() {
    let data = [0x02, 0xFF, 0xFE];
    let mut dec = ProtoDecoder::new(&data);
    assert_eq!(dec.read_string().unwrap_err(), ProtoError::InvalidUtf8);
}

#[test]
fn protobuf_tag_field_zero_rejected() {
    // Tag 0x00 => field_number=0, which is invalid.
    let mut dec = ProtoDecoder::new(&[0x00]);
    assert_eq!(dec.read_tag().unwrap_err(), ProtoError::InvalidFieldNumber);
}

#[test]
fn protobuf_tag_invalid_wire_type_rejected() {
    // Wire types 3 and 4 (deprecated groups) are now recognized.
    // Only 6 and 7 remain invalid.
    for invalid_wt in [6u8, 7] {
        let tag_byte = (1 << 3) | invalid_wt;
        let data = [tag_byte];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_tag().unwrap_err(), ProtoError::InvalidWireType);
    }
    for valid_wt in [3u8, 4] {
        let tag_byte = (1 << 3) | valid_wt;
        let data = [tag_byte];
        let mut dec = ProtoDecoder::new(&data);
        assert!(dec.read_tag().is_ok());
    }
}

#[test]
fn protobuf_wire_type_from_u8_valid_and_invalid() {
    assert_eq!(WireType::from_u8(0), Some(WireType::Varint));
    assert_eq!(WireType::from_u8(1), Some(WireType::Fixed64));
    assert_eq!(WireType::from_u8(2), Some(WireType::LengthDelimited));
    assert_eq!(WireType::from_u8(3), Some(WireType::StartGroup));
    assert_eq!(WireType::from_u8(4), Some(WireType::EndGroup));
    assert_eq!(WireType::from_u8(5), Some(WireType::Fixed32));
    assert!(WireType::from_u8(6).is_none());
    assert!(WireType::from_u8(7).is_none());
}

#[test]
fn protobuf_truncated_fixed32_fails() {
    let mut dec = ProtoDecoder::new(&[0x01, 0x02]);
    assert!(dec.read_fixed32().is_err());
}

#[test]
fn protobuf_truncated_fixed64_fails() {
    let mut dec = ProtoDecoder::new(&[0x01, 0x02, 0x03, 0x04]);
    assert!(dec.read_fixed64().is_err());
}

// ---------------------------------------------------------------------------
// 8. SessionError Display coverage
// ---------------------------------------------------------------------------

#[test]
fn session_error_display_variants() {
    assert_eq!(
        format!("{}", SessionError::ModelLoadFailed(String::from("corrupt"))),
        "model load failed: corrupt"
    );
    assert_eq!(
        format!("{}", SessionError::InvalidModel(String::from("no graph"))),
        "invalid model: no graph"
    );
    assert_eq!(
        format!("{}", SessionError::UnsupportedOpset(99)),
        "unsupported opset version: 99"
    );
    assert_eq!(
        format!("{}", SessionError::NotImplemented),
        "session operation not implemented"
    );
}
