// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB inference gadget — presents SmallAIOS as a USB inference appliance.
//!
//! A host PC connects via USB and submits ONNX inference requests over bulk
//! endpoints. Results are returned on a bulk IN endpoint. Requests are bridged
//! to Zenoh IPC for unified inference pipeline.

use crate::UsbError;

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// Maximum model name length in bytes.
pub const MAX_MODEL_NAME_LEN: usize = 256;

/// Maximum input tensor size (256 MiB).
pub const MAX_TENSOR_SIZE: usize = 256 * 1024 * 1024;

/// Maximum number of concurrent outstanding requests.
pub const MAX_OUTSTANDING_REQUESTS: usize = 4;

/// Inference gadget USB endpoint addresses.
pub const EP_BULK_OUT: u8 = 0x02; // Host → SmallAIOS (requests)
pub const EP_BULK_IN: u8 = 0x81;  // SmallAIOS → Host (responses)

/// Inference gadget interface class (vendor-specific).
pub const INTERFACE_CLASS: u8 = 0xFF;
pub const INTERFACE_SUBCLASS: u8 = 0x01;
pub const INTERFACE_PROTOCOL: u8 = 0x01;

// ---------------------------------------------------------------------------
// Request/Response status codes
// ---------------------------------------------------------------------------

/// Status codes for inference responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum InferenceStatus {
    /// Inference completed successfully.
    Ok = 0x0000,
    /// Request was malformed.
    InvalidRequest = 0x0001,
    /// Requested model not found/loaded.
    ModelNotFound = 0x0002,
    /// Inference runtime error.
    InferenceError = 0x0003,
    /// Too many concurrent requests.
    Busy = 0x0004,
}

impl InferenceStatus {
    /// Convert from raw u16 value.
    pub fn from_u16(v: u16) -> Self {
        match v {
            0x0000 => InferenceStatus::Ok,
            0x0001 => InferenceStatus::InvalidRequest,
            0x0002 => InferenceStatus::ModelNotFound,
            0x0003 => InferenceStatus::InferenceError,
            0x0004 => InferenceStatus::Busy,
            _ => InferenceStatus::InferenceError,
        }
    }
}

// ---------------------------------------------------------------------------
// Request parser
// ---------------------------------------------------------------------------

/// Parsed inference request header.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request identifier (echoed in the response).
    pub request_id: u32,
    /// Model name (UTF-8).
    pub model_name: [u8; MAX_MODEL_NAME_LEN],
    /// Valid bytes in `model_name`.
    pub model_name_len: usize,
    /// Size of the input tensor data in bytes.
    pub tensor_size: u32,
}

impl InferenceRequest {
    /// Parse a request header from raw bytes.
    ///
    /// Format: [4: request_id][2: model_name_len][N: model_name][4: tensor_size]
    ///
    /// Returns the parsed request and the byte offset where tensor data begins.
    pub fn parse(data: &[u8]) -> Result<(Self, usize), UsbError> {
        // Minimum: 4 + 2 + 0 + 4 = 10 bytes
        if data.len() < 10 {
            return Err(UsbError::BufferTooSmall);
        }

        let request_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let model_name_len = u16::from_le_bytes([data[4], data[5]]) as usize;

        if model_name_len > MAX_MODEL_NAME_LEN {
            return Err(UsbError::InvalidEndpoint); // Reuse error for "invalid request"
        }

        let header_end = 6 + model_name_len;
        if data.len() < header_end + 4 {
            return Err(UsbError::BufferTooSmall);
        }

        let mut model_name = [0u8; MAX_MODEL_NAME_LEN];
        model_name[..model_name_len].copy_from_slice(&data[6..6 + model_name_len]);

        let tensor_size_offset = header_end;
        let tensor_size = u32::from_le_bytes([
            data[tensor_size_offset],
            data[tensor_size_offset + 1],
            data[tensor_size_offset + 2],
            data[tensor_size_offset + 3],
        ]);

        if tensor_size as usize > MAX_TENSOR_SIZE {
            return Err(UsbError::InvalidEndpoint);
        }

        let tensor_data_offset = tensor_size_offset + 4;

        Ok((
            Self {
                request_id,
                model_name,
                model_name_len,
                tensor_size,
            },
            tensor_data_offset,
        ))
    }

    /// Get the model name as a string slice.
    pub fn model_name_str(&self) -> &str {
        core::str::from_utf8(&self.model_name[..self.model_name_len]).unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// Response formatter
// ---------------------------------------------------------------------------

/// Format an inference response header.
///
/// Format: [4: request_id][2: status][4: result_size]
///
/// Returns the 10-byte response header.
pub fn format_response_header(
    request_id: u32,
    status: InferenceStatus,
    result_size: u32,
) -> [u8; 10] {
    let mut buf = [0u8; 10];
    buf[0..4].copy_from_slice(&request_id.to_le_bytes());
    buf[4..6].copy_from_slice(&(status as u16).to_le_bytes());
    buf[6..10].copy_from_slice(&result_size.to_le_bytes());
    buf
}

// ---------------------------------------------------------------------------
// Request tracker
// ---------------------------------------------------------------------------

/// Tracks outstanding inference requests.
pub struct RequestTracker {
    /// Active request IDs (0 = slot free).
    active: [u32; MAX_OUTSTANDING_REQUESTS],
    /// Number of active requests.
    count: usize,
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            active: [0; MAX_OUTSTANDING_REQUESTS],
            count: 0,
        }
    }

    /// Track a new request. Returns error if at capacity.
    pub fn add(&mut self, request_id: u32) -> Result<(), InferenceStatus> {
        if self.count >= MAX_OUTSTANDING_REQUESTS {
            return Err(InferenceStatus::Busy);
        }
        for slot in self.active.iter_mut() {
            if *slot == 0 {
                *slot = request_id;
                self.count += 1;
                return Ok(());
            }
        }
        Err(InferenceStatus::Busy)
    }

    /// Remove a completed request.
    pub fn remove(&mut self, request_id: u32) -> bool {
        for slot in self.active.iter_mut() {
            if *slot == request_id {
                *slot = 0;
                self.count -= 1;
                return true;
            }
        }
        false
    }

    /// Check if a request ID is tracked.
    pub fn is_active(&self, request_id: u32) -> bool {
        self.active.contains(&request_id) && request_id != 0
    }

    /// Return the number of active requests.
    pub fn active_count(&self) -> usize {
        self.count
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inference_request() {
        let mut data = [0u8; 32];
        // request_id = 42
        data[0..4].copy_from_slice(&42u32.to_le_bytes());
        // model_name_len = 5
        data[4..6].copy_from_slice(&5u16.to_le_bytes());
        // model_name = "model"
        data[6..11].copy_from_slice(b"model");
        // tensor_size = 1024
        data[11..15].copy_from_slice(&1024u32.to_le_bytes());

        let (req, tensor_offset) = InferenceRequest::parse(&data).unwrap();
        assert_eq!(req.request_id, 42);
        assert_eq!(req.model_name_str(), "model");
        assert_eq!(req.tensor_size, 1024);
        assert_eq!(tensor_offset, 15); // 6 + 5 + 4
    }

    #[test]
    fn test_parse_request_too_short() {
        let data = [0u8; 5];
        assert_eq!(InferenceRequest::parse(&data), Err(UsbError::BufferTooSmall));
    }

    #[test]
    fn test_parse_request_model_name_too_long() {
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&1u32.to_le_bytes());
        data[4..6].copy_from_slice(&(MAX_MODEL_NAME_LEN as u16 + 1).to_le_bytes());
        assert!(InferenceRequest::parse(&data).is_err());
    }

    #[test]
    fn test_format_response_header() {
        let header = format_response_header(42, InferenceStatus::Ok, 256);
        assert_eq!(u32::from_le_bytes([header[0], header[1], header[2], header[3]]), 42);
        assert_eq!(u16::from_le_bytes([header[4], header[5]]), 0x0000);
        assert_eq!(u32::from_le_bytes([header[6], header[7], header[8], header[9]]), 256);
    }

    #[test]
    fn test_format_response_error() {
        let header = format_response_header(7, InferenceStatus::ModelNotFound, 0);
        assert_eq!(u16::from_le_bytes([header[4], header[5]]), 0x0002);
        assert_eq!(u32::from_le_bytes([header[6], header[7], header[8], header[9]]), 0);
    }

    #[test]
    fn test_request_tracker() {
        let mut tracker = RequestTracker::new();
        assert_eq!(tracker.active_count(), 0);

        tracker.add(1).unwrap();
        tracker.add(2).unwrap();
        assert_eq!(tracker.active_count(), 2);
        assert!(tracker.is_active(1));
        assert!(tracker.is_active(2));
        assert!(!tracker.is_active(3));

        assert!(tracker.remove(1));
        assert_eq!(tracker.active_count(), 1);
        assert!(!tracker.is_active(1));
    }

    #[test]
    fn test_request_tracker_capacity() {
        let mut tracker = RequestTracker::new();
        for i in 1..=MAX_OUTSTANDING_REQUESTS as u32 {
            tracker.add(i).unwrap();
        }
        assert_eq!(
            tracker.add(100),
            Err(InferenceStatus::Busy)
        );
    }

    #[test]
    fn test_request_tracker_remove_nonexistent() {
        let mut tracker = RequestTracker::new();
        assert!(!tracker.remove(42));
    }

    #[test]
    fn test_inference_status_from_u16() {
        assert_eq!(InferenceStatus::from_u16(0), InferenceStatus::Ok);
        assert_eq!(InferenceStatus::from_u16(1), InferenceStatus::InvalidRequest);
        assert_eq!(InferenceStatus::from_u16(2), InferenceStatus::ModelNotFound);
        assert_eq!(InferenceStatus::from_u16(3), InferenceStatus::InferenceError);
        assert_eq!(InferenceStatus::from_u16(4), InferenceStatus::Busy);
        assert_eq!(InferenceStatus::from_u16(99), InferenceStatus::InferenceError);
    }
}
