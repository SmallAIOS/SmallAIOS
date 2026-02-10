// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Binary inference request/response protocol.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::IpcError;

/// Magic number identifying inference protocol packets ("ONNX").
pub const INFERENCE_MAGIC: u32 = 0x4F4E4E58;

/// Current protocol version.
pub const PROTOCOL_VERSION: u8 = 1;

/// Type of inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InferenceRequestType {
    /// Load a model into memory.
    LoadModel = 1,
    /// Unload a model from memory.
    UnloadModel = 2,
    /// Execute inference on a loaded model.
    RunInference = 3,
    /// Query metadata about a loaded model.
    GetModelInfo = 4,
}

impl InferenceRequestType {
    /// Convert from a raw u8 value.
    fn from_u8(val: u8) -> Result<Self, IpcError> {
        match val {
            1 => Ok(InferenceRequestType::LoadModel),
            2 => Ok(InferenceRequestType::UnloadModel),
            3 => Ok(InferenceRequestType::RunInference),
            4 => Ok(InferenceRequestType::GetModelInfo),
            _ => Err(IpcError::InvalidProtocol),
        }
    }
}

/// Data type of tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDataType {
    /// 32-bit IEEE 754 floating point.
    Float32 = 1,
    /// 16-bit IEEE 754 floating point.
    Float16 = 2,
    /// Signed 8-bit integer.
    Int8 = 3,
    /// Signed 32-bit integer.
    Int32 = 4,
    /// Signed 64-bit integer.
    Int64 = 5,
    /// Unsigned 8-bit integer.
    Uint8 = 6,
}

impl TensorDataType {
    /// Convert from a raw u8 value.
    fn from_u8(val: u8) -> Result<Self, IpcError> {
        match val {
            1 => Ok(TensorDataType::Float32),
            2 => Ok(TensorDataType::Float16),
            3 => Ok(TensorDataType::Int8),
            4 => Ok(TensorDataType::Int32),
            5 => Ok(TensorDataType::Int64),
            6 => Ok(TensorDataType::Uint8),
            _ => Err(IpcError::InvalidProtocol),
        }
    }
}

/// A named tensor with shape and raw data.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    /// Tensor name (e.g., "input_0").
    pub name: String,
    /// Element data type.
    pub data_type: TensorDataType,
    /// Shape dimensions (e.g., [1, 3, 224, 224]).
    pub shape: Vec<u32>,
    /// Raw tensor data bytes (layout depends on data_type).
    pub data: Vec<u8>,
}

impl TensorData {
    /// Encode this tensor into the wire format.
    ///
    /// Format: name_len(u16) + name + data_type(u8) + num_dims(u16) +
    ///         dims(4*n) + data_len(u32) + data
    fn encode(&self, buf: &mut Vec<u8>) {
        let name_bytes = self.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(self.data_type as u8);
        buf.extend_from_slice(&(self.shape.len() as u16).to_be_bytes());
        for &dim in &self.shape {
            buf.extend_from_slice(&dim.to_be_bytes());
        }
        buf.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        buf.extend_from_slice(&self.data);
    }

    /// Decode a tensor from the wire format starting at the given offset.
    ///
    /// Returns the decoded tensor and the number of bytes consumed.
    fn decode(data: &[u8], offset: usize) -> Result<(Self, usize), IpcError> {
        let mut pos = offset;

        // name_len (u16)
        if pos + 2 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let name_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        // name
        if pos + name_len > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let name = core::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| IpcError::InvalidProtocol)?;
        let name = String::from(name);
        pos += name_len;

        // data_type (u8)
        if pos + 1 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let data_type = TensorDataType::from_u8(data[pos])?;
        pos += 1;

        // num_dims (u16)
        if pos + 2 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let num_dims = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        // dims (4 * num_dims)
        if pos + 4 * num_dims > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            let dim = u32::from_be_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]);
            shape.push(dim);
            pos += 4;
        }

        // data_len (u32)
        if pos + 4 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let data_len = u32::from_be_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ]) as usize;
        pos += 4;

        // data
        if pos + data_len > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let tensor_data = data[pos..pos + data_len].to_vec();
        pos += data_len;

        Ok((
            TensorData {
                name,
                data_type,
                shape,
                data: tensor_data,
            },
            pos - offset,
        ))
    }
}

/// An inference request message.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRequest {
    /// Type of request.
    pub request_type: InferenceRequestType,
    /// Unique request identifier for correlation.
    pub request_id: u64,
    /// Target model identifier.
    pub model_id: u32,
    /// Input tensors (empty for non-inference requests).
    pub inputs: Vec<TensorData>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u32,
}

impl InferenceRequest {
    /// Encode this request into the binary wire format.
    ///
    /// Wire format (all multi-byte fields big-endian):
    /// ```text
    /// magic(4) + version(1) + request_type(1) + request_id(8) +
    /// model_id(4) + timeout_ms(4) + num_inputs(u16) +
    /// [for each input: tensor encoding]
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&INFERENCE_MAGIC.to_be_bytes());
        buf.push(PROTOCOL_VERSION);
        buf.push(self.request_type as u8);
        buf.extend_from_slice(&self.request_id.to_be_bytes());
        buf.extend_from_slice(&self.model_id.to_be_bytes());
        buf.extend_from_slice(&self.timeout_ms.to_be_bytes());

        // Inputs
        buf.extend_from_slice(&(self.inputs.len() as u16).to_be_bytes());
        for input in &self.inputs {
            input.encode(&mut buf);
        }

        buf
    }

    /// Decode an inference request from binary wire format.
    ///
    /// # Errors
    ///
    /// - [`IpcError::PacketTooShort`] if data is too short for required fields.
    /// - [`IpcError::InvalidProtocol`] if magic or version mismatch.
    pub fn decode(data: &[u8]) -> Result<InferenceRequest, IpcError> {
        // Minimum header: magic(4) + version(1) + type(1) + id(8) +
        //                 model_id(4) + timeout(4) + num_inputs(2) = 24
        if data.len() < 24 {
            return Err(IpcError::PacketTooShort);
        }

        // Magic
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if magic != INFERENCE_MAGIC {
            return Err(IpcError::InvalidProtocol);
        }

        // Version
        let version = data[4];
        if version != PROTOCOL_VERSION {
            return Err(IpcError::InvalidProtocol);
        }

        // Request type
        let request_type = InferenceRequestType::from_u8(data[5])?;

        // Request ID
        let request_id = u64::from_be_bytes([
            data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13],
        ]);

        // Model ID
        let model_id = u32::from_be_bytes([data[14], data[15], data[16], data[17]]);

        // Timeout
        let timeout_ms = u32::from_be_bytes([data[18], data[19], data[20], data[21]]);

        // Number of inputs
        let num_inputs = u16::from_be_bytes([data[22], data[23]]) as usize;

        // Decode inputs
        let mut inputs = Vec::with_capacity(num_inputs);
        let mut offset = 24;
        for _ in 0..num_inputs {
            let (tensor, consumed) = TensorData::decode(data, offset)?;
            inputs.push(tensor);
            offset += consumed;
        }

        Ok(InferenceRequest {
            request_type,
            request_id,
            model_id,
            inputs,
            timeout_ms,
        })
    }
}

/// Status codes for inference responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ResponseStatus {
    /// Request completed successfully.
    Ok = 0,
    /// The specified model was not found.
    ModelNotFound = 1,
    /// One or more input tensors were invalid.
    InvalidInput = 2,
    /// An error occurred during inference execution.
    InferenceError = 3,
    /// The request timed out.
    Timeout = 4,
    /// An unexpected internal error occurred.
    InternalError = 5,
}

impl ResponseStatus {
    /// Convert from a raw u8 value.
    fn from_u8(val: u8) -> Result<Self, IpcError> {
        match val {
            0 => Ok(ResponseStatus::Ok),
            1 => Ok(ResponseStatus::ModelNotFound),
            2 => Ok(ResponseStatus::InvalidInput),
            3 => Ok(ResponseStatus::InferenceError),
            4 => Ok(ResponseStatus::Timeout),
            5 => Ok(ResponseStatus::InternalError),
            _ => Err(IpcError::InvalidProtocol),
        }
    }
}

/// An inference response message.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResponse {
    /// Correlation identifier matching the original request.
    pub request_id: u64,
    /// Result status code.
    pub status: ResponseStatus,
    /// Output tensors (empty on error).
    pub outputs: Vec<TensorData>,
    /// Wall-clock inference time in microseconds.
    pub elapsed_us: u64,
}

impl InferenceResponse {
    /// Encode this response into the binary wire format.
    ///
    /// Wire format (all multi-byte fields big-endian):
    /// ```text
    /// magic(4) + version(1) + request_id(8) + status(1) +
    /// elapsed_us(8) + num_outputs(u16) + [for each output: tensor encoding]
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&INFERENCE_MAGIC.to_be_bytes());
        buf.push(PROTOCOL_VERSION);
        buf.extend_from_slice(&self.request_id.to_be_bytes());
        buf.push(self.status as u8);
        buf.extend_from_slice(&self.elapsed_us.to_be_bytes());

        // Outputs
        buf.extend_from_slice(&(self.outputs.len() as u16).to_be_bytes());
        for output in &self.outputs {
            output.encode(&mut buf);
        }

        buf
    }

    /// Decode an inference response from binary wire format.
    ///
    /// # Errors
    ///
    /// - [`IpcError::PacketTooShort`] if data is too short for required fields.
    /// - [`IpcError::InvalidProtocol`] if magic or version mismatch.
    pub fn decode(data: &[u8]) -> Result<InferenceResponse, IpcError> {
        // Minimum header: magic(4) + version(1) + request_id(8) +
        //                 status(1) + elapsed_us(8) + num_outputs(2) = 24
        if data.len() < 24 {
            return Err(IpcError::PacketTooShort);
        }

        // Magic
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if magic != INFERENCE_MAGIC {
            return Err(IpcError::InvalidProtocol);
        }

        // Version
        let version = data[4];
        if version != PROTOCOL_VERSION {
            return Err(IpcError::InvalidProtocol);
        }

        // Request ID
        let request_id = u64::from_be_bytes([
            data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12],
        ]);

        // Status
        let status = ResponseStatus::from_u8(data[13])?;

        // Elapsed microseconds
        let elapsed_us = u64::from_be_bytes([
            data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21],
        ]);

        // Number of outputs
        let num_outputs = u16::from_be_bytes([data[22], data[23]]) as usize;

        // Decode outputs
        let mut outputs = Vec::with_capacity(num_outputs);
        let mut offset = 24;
        for _ in 0..num_outputs {
            let (tensor, consumed) = TensorData::decode(data, offset)?;
            outputs.push(tensor);
            offset += consumed;
        }

        Ok(InferenceResponse {
            request_id,
            status,
            outputs,
            elapsed_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    // === Request encode/decode roundtrip ===

    #[test]
    fn request_roundtrip_no_inputs() {
        let req = InferenceRequest {
            request_type: InferenceRequestType::LoadModel,
            request_id: 12345,
            model_id: 42,
            inputs: Vec::new(),
            timeout_ms: 5000,
        };
        let encoded = req.encode();
        let decoded = InferenceRequest::decode(&encoded).unwrap();
        assert_eq!(req, decoded);
    }

    #[test]
    fn request_roundtrip_with_inputs() {
        let req = InferenceRequest {
            request_type: InferenceRequestType::RunInference,
            request_id: 0xDEADBEEF,
            model_id: 7,
            inputs: vec![
                TensorData {
                    name: String::from("input_0"),
                    data_type: TensorDataType::Float32,
                    shape: vec![1, 3, 224, 224],
                    data: vec![0u8; 16], // stub data
                },
                TensorData {
                    name: String::from("mask"),
                    data_type: TensorDataType::Int8,
                    shape: vec![1, 224, 224],
                    data: vec![1u8; 8],
                },
            ],
            timeout_ms: 10000,
        };
        let encoded = req.encode();
        let decoded = InferenceRequest::decode(&encoded).unwrap();
        assert_eq!(req, decoded);
    }

    // === Response encode/decode roundtrip ===

    #[test]
    fn response_roundtrip_no_outputs() {
        let resp = InferenceResponse {
            request_id: 12345,
            status: ResponseStatus::Ok,
            outputs: Vec::new(),
            elapsed_us: 42000,
        };
        let encoded = resp.encode();
        let decoded = InferenceResponse::decode(&encoded).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn response_roundtrip_with_outputs() {
        let resp = InferenceResponse {
            request_id: 99,
            status: ResponseStatus::Ok,
            outputs: vec![TensorData {
                name: String::from("output_0"),
                data_type: TensorDataType::Float32,
                shape: vec![1, 1000],
                data: vec![0xAB; 4000],
            }],
            elapsed_us: 1500,
        };
        let encoded = resp.encode();
        let decoded = InferenceResponse::decode(&encoded).unwrap();
        assert_eq!(resp, decoded);
    }

    // === Error cases ===

    #[test]
    fn decode_request_invalid_magic() {
        let mut data = InferenceRequest {
            request_type: InferenceRequestType::LoadModel,
            request_id: 1,
            model_id: 1,
            inputs: Vec::new(),
            timeout_ms: 100,
        }
        .encode();
        // Corrupt magic bytes
        data[0] = 0xFF;
        assert_eq!(
            InferenceRequest::decode(&data),
            Err(IpcError::InvalidProtocol)
        );
    }

    #[test]
    fn decode_request_invalid_version() {
        let mut data = InferenceRequest {
            request_type: InferenceRequestType::LoadModel,
            request_id: 1,
            model_id: 1,
            inputs: Vec::new(),
            timeout_ms: 100,
        }
        .encode();
        // Corrupt version byte
        data[4] = 0xFF;
        assert_eq!(
            InferenceRequest::decode(&data),
            Err(IpcError::InvalidProtocol)
        );
    }

    #[test]
    fn decode_request_packet_too_short() {
        let data = [0u8; 10]; // Way too short
        assert_eq!(
            InferenceRequest::decode(&data),
            Err(IpcError::PacketTooShort)
        );
    }

    #[test]
    fn decode_response_invalid_magic() {
        let mut data = InferenceResponse {
            request_id: 1,
            status: ResponseStatus::Ok,
            outputs: Vec::new(),
            elapsed_us: 0,
        }
        .encode();
        data[0] = 0x00;
        assert_eq!(
            InferenceResponse::decode(&data),
            Err(IpcError::InvalidProtocol)
        );
    }

    #[test]
    fn decode_response_packet_too_short() {
        let data = [0u8; 5];
        assert_eq!(
            InferenceResponse::decode(&data),
            Err(IpcError::PacketTooShort)
        );
    }

    // === Request types ===

    #[test]
    fn all_request_types_roundtrip() {
        let types = [
            InferenceRequestType::LoadModel,
            InferenceRequestType::UnloadModel,
            InferenceRequestType::RunInference,
            InferenceRequestType::GetModelInfo,
        ];
        for req_type in &types {
            let req = InferenceRequest {
                request_type: *req_type,
                request_id: 1,
                model_id: 1,
                inputs: Vec::new(),
                timeout_ms: 100,
            };
            let encoded = req.encode();
            let decoded = InferenceRequest::decode(&encoded).unwrap();
            assert_eq!(decoded.request_type, *req_type);
        }
    }

    // === Tensor data encoding ===

    #[test]
    fn tensor_all_data_types() {
        let types = [
            TensorDataType::Float32,
            TensorDataType::Float16,
            TensorDataType::Int8,
            TensorDataType::Int32,
            TensorDataType::Int64,
            TensorDataType::Uint8,
        ];
        for dt in &types {
            let tensor = TensorData {
                name: String::from("t"),
                data_type: *dt,
                shape: vec![2, 3],
                data: vec![0u8; 4],
            };
            let mut buf = Vec::new();
            tensor.encode(&mut buf);
            let (decoded, _consumed) = TensorData::decode(&buf, 0).unwrap();
            assert_eq!(decoded.data_type, *dt);
            assert_eq!(decoded.name, "t");
            assert_eq!(decoded.shape, vec![2, 3]);
            assert_eq!(decoded.data.len(), 4);
        }
    }

    #[test]
    fn tensor_empty_data() {
        let tensor = TensorData {
            name: String::from("empty"),
            data_type: TensorDataType::Float32,
            shape: Vec::new(),
            data: Vec::new(),
        };
        let mut buf = Vec::new();
        tensor.encode(&mut buf);
        let (decoded, _) = TensorData::decode(&buf, 0).unwrap();
        assert_eq!(decoded, tensor);
    }

    // === Response status ===

    #[test]
    fn response_status_values_roundtrip() {
        let statuses = [
            ResponseStatus::Ok,
            ResponseStatus::ModelNotFound,
            ResponseStatus::InvalidInput,
            ResponseStatus::InferenceError,
            ResponseStatus::Timeout,
            ResponseStatus::InternalError,
        ];
        for status in &statuses {
            let resp = InferenceResponse {
                request_id: 42,
                status: *status,
                outputs: Vec::new(),
                elapsed_us: 100,
            };
            let encoded = resp.encode();
            let decoded = InferenceResponse::decode(&encoded).unwrap();
            assert_eq!(decoded.status, *status);
        }
    }

    // === Magic constant ===

    #[test]
    fn magic_is_onnx_ascii() {
        let bytes = INFERENCE_MAGIC.to_be_bytes();
        assert_eq!(&bytes, b"ONNX");
    }

    // === Multi-tensor request ===

    #[test]
    fn request_multiple_tensors_preserves_order() {
        let req = InferenceRequest {
            request_type: InferenceRequestType::RunInference,
            request_id: 1,
            model_id: 1,
            inputs: vec![
                TensorData {
                    name: String::from("first"),
                    data_type: TensorDataType::Float32,
                    shape: vec![1],
                    data: vec![0x01],
                },
                TensorData {
                    name: String::from("second"),
                    data_type: TensorDataType::Int8,
                    shape: vec![2],
                    data: vec![0x02, 0x03],
                },
                TensorData {
                    name: String::from("third"),
                    data_type: TensorDataType::Uint8,
                    shape: vec![3, 4],
                    data: vec![0x04; 12],
                },
            ],
            timeout_ms: 500,
        };
        let encoded = req.encode();
        let decoded = InferenceRequest::decode(&encoded).unwrap();
        assert_eq!(decoded.inputs.len(), 3);
        assert_eq!(decoded.inputs[0].name, "first");
        assert_eq!(decoded.inputs[1].name, "second");
        assert_eq!(decoded.inputs[2].name, "third");
    }

    #[test]
    fn decode_request_truncated_tensor() {
        let req = InferenceRequest {
            request_type: InferenceRequestType::RunInference,
            request_id: 1,
            model_id: 1,
            inputs: vec![TensorData {
                name: String::from("input"),
                data_type: TensorDataType::Float32,
                shape: vec![1, 3],
                data: vec![0u8; 12],
            }],
            timeout_ms: 100,
        };
        let encoded = req.encode();
        // Truncate the data portion
        let truncated = &encoded[..encoded.len() - 5];
        assert_eq!(
            InferenceRequest::decode(truncated),
            Err(IpcError::PacketTooShort)
        );
    }
}
