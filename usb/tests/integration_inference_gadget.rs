// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test (Task 4.10): Host submits inference request over USB ->
//! ONNX result returned.
//!
//! Tests the inference gadget protocol: request parsing, validation, response
//! formatting, request tracking, and gadget descriptor composition — all
//! exercised through the public API without real USB hardware.

#![cfg(all(feature = "gadget", feature = "inference-gadget"))]

extern crate alloc;

use smallaios_kernel::hal::UsbSetupPacket;
use smallaios_usb::gadget::{
    compose_config_descriptor, compose_device_descriptor, handle_standard_request,
    GadgetDeviceConfig, GadgetFunctionDescriptors, MAX_CONFIG_DESC_LEN,
};
use smallaios_usb::inference_gadget::{
    format_response_header, InferenceRequest, InferenceStatus, RequestTracker, EP_BULK_IN,
    EP_BULK_OUT, INTERFACE_CLASS, INTERFACE_PROTOCOL, INTERFACE_SUBCLASS, MAX_MODEL_NAME_LEN,
    MAX_OUTSTANDING_REQUESTS, MAX_TENSOR_SIZE,
};
use smallaios_usb::{descriptor_type, request, UsbError};

// ---------------------------------------------------------------------------
// Helpers: Build a mock inference request in wire format.
// ---------------------------------------------------------------------------

/// Builds a binary inference request with the given parameters.
fn build_request(
    request_id: u32,
    model: &str,
    tensor_size: u32,
    tensor_data: &[u8],
) -> alloc::vec::Vec<u8> {
    let model_bytes = model.as_bytes();
    let model_len = model_bytes.len() as u16;
    let header_len = 4 + 2 + model_bytes.len() + 4;
    let total_len = header_len + tensor_data.len();
    let mut buf = alloc::vec![0u8; total_len];

    buf[0..4].copy_from_slice(&request_id.to_le_bytes());
    buf[4..6].copy_from_slice(&model_len.to_le_bytes());
    buf[6..6 + model_bytes.len()].copy_from_slice(model_bytes);
    let ts_offset = 6 + model_bytes.len();
    buf[ts_offset..ts_offset + 4].copy_from_slice(&tensor_size.to_le_bytes());
    if !tensor_data.is_empty() {
        buf[ts_offset + 4..].copy_from_slice(tensor_data);
    }
    buf
}

// ---------------------------------------------------------------------------
// 1. Request parsing and validation
// ---------------------------------------------------------------------------

#[test]
fn inference_request_parse_valid() {
    let request_data = build_request(42, "resnet50", 1024, &[0u8; 0]);
    let (req, tensor_offset) = InferenceRequest::parse(&request_data).unwrap();

    assert_eq!(req.request_id, 42);
    assert_eq!(req.model_name_str(), "resnet50");
    assert_eq!(req.model_name_len, 8);
    assert_eq!(req.tensor_size, 1024);
    assert_eq!(tensor_offset, 6 + 8 + 4); // 4(id) + 2(name_len) + 8(name) + 4(tensor_size)
}

#[test]
fn inference_request_parse_empty_model_name() {
    let request_data = build_request(1, "", 0, &[]);
    let (req, tensor_offset) = InferenceRequest::parse(&request_data).unwrap();

    assert_eq!(req.request_id, 1);
    assert_eq!(req.model_name_str(), "");
    assert_eq!(req.model_name_len, 0);
    assert_eq!(req.tensor_size, 0);
    assert_eq!(tensor_offset, 10); // minimum header: 4 + 2 + 0 + 4
}

#[test]
fn inference_request_parse_max_model_name_length() {
    // Model name at exactly MAX_MODEL_NAME_LEN should succeed.
    let long_name = "a".repeat(MAX_MODEL_NAME_LEN);
    let request_data = build_request(99, &long_name, 512, &[]);
    let (req, _) = InferenceRequest::parse(&request_data).unwrap();
    assert_eq!(req.model_name_len, MAX_MODEL_NAME_LEN);
}

#[test]
fn inference_request_parse_too_short_buffer() {
    let short_buf = [0u8; 5];
    assert!(matches!(
        InferenceRequest::parse(&short_buf),
        Err(UsbError::BufferTooSmall)
    ));
}

#[test]
fn inference_request_parse_model_name_exceeds_max() {
    let mut data = [0u8; 16];
    data[0..4].copy_from_slice(&1u32.to_le_bytes());
    // name_len = MAX_MODEL_NAME_LEN + 1
    data[4..6].copy_from_slice(&((MAX_MODEL_NAME_LEN as u16) + 1).to_le_bytes());
    assert!(InferenceRequest::parse(&data).is_err());
}

#[test]
fn inference_request_parse_tensor_exceeds_max() {
    let excess_size = (MAX_TENSOR_SIZE + 1) as u32;
    let request_data = build_request(1, "m", excess_size, &[]);
    assert!(InferenceRequest::parse(&request_data).is_err());
}

// ---------------------------------------------------------------------------
// 2. Response formatting
// ---------------------------------------------------------------------------

#[test]
fn inference_response_header_success() {
    let header = format_response_header(42, InferenceStatus::Ok, 256);

    let resp_id = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let status = u16::from_le_bytes([header[4], header[5]]);
    let result_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]);

    assert_eq!(resp_id, 42);
    assert_eq!(status, InferenceStatus::Ok as u16);
    assert_eq!(result_size, 256);
}

#[test]
fn inference_response_header_error_statuses() {
    let test_cases = [
        (InferenceStatus::InvalidRequest, 0x0001u16),
        (InferenceStatus::ModelNotFound, 0x0002),
        (InferenceStatus::InferenceError, 0x0003),
        (InferenceStatus::Busy, 0x0004),
    ];

    for (status, expected_code) in test_cases {
        let header = format_response_header(1, status, 0);
        let code = u16::from_le_bytes([header[4], header[5]]);
        assert_eq!(
            code, expected_code,
            "status {:?} should encode as {:#06x}",
            status, expected_code
        );
    }
}

#[test]
fn inference_status_roundtrip() {
    for code in 0u16..=5 {
        let status = InferenceStatus::from_u16(code);
        // Verify all known codes parse correctly.
        match code {
            0 => assert_eq!(status, InferenceStatus::Ok),
            1 => assert_eq!(status, InferenceStatus::InvalidRequest),
            2 => assert_eq!(status, InferenceStatus::ModelNotFound),
            3 => assert_eq!(status, InferenceStatus::InferenceError),
            4 => assert_eq!(status, InferenceStatus::Busy),
            _ => assert_eq!(status, InferenceStatus::InferenceError),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Request tracker (concurrent request management)
// ---------------------------------------------------------------------------

#[test]
fn inference_tracker_add_and_complete() {
    let mut tracker = RequestTracker::new();

    // Submit an inference request.
    tracker.add(42).unwrap();
    assert_eq!(tracker.active_count(), 1);
    assert!(tracker.is_active(42));

    // Complete it.
    assert!(tracker.remove(42));
    assert_eq!(tracker.active_count(), 0);
    assert!(!tracker.is_active(42));
}

#[test]
fn inference_tracker_concurrent_requests() {
    let mut tracker = RequestTracker::new();

    // Submit MAX_OUTSTANDING_REQUESTS concurrent requests.
    for i in 1..=MAX_OUTSTANDING_REQUESTS as u32 {
        tracker.add(i).unwrap();
    }
    assert_eq!(tracker.active_count(), MAX_OUTSTANDING_REQUESTS);

    // Next submission should fail with Busy.
    assert_eq!(tracker.add(100), Err(InferenceStatus::Busy));

    // Complete one and verify we can submit again.
    tracker.remove(1);
    assert_eq!(tracker.active_count(), MAX_OUTSTANDING_REQUESTS - 1);
    tracker.add(100).unwrap();
    assert_eq!(tracker.active_count(), MAX_OUTSTANDING_REQUESTS);
}

#[test]
fn inference_tracker_remove_nonexistent() {
    let mut tracker = RequestTracker::new();
    assert!(!tracker.remove(999));
}

#[test]
fn inference_tracker_zero_id_not_active() {
    let tracker = RequestTracker::new();
    assert!(
        !tracker.is_active(0),
        "request ID 0 should never be considered active"
    );
}

// ---------------------------------------------------------------------------
// 4. Full request -> response pipeline (simulated)
// ---------------------------------------------------------------------------

#[test]
fn inference_end_to_end_request_response() {
    let mut tracker = RequestTracker::new();

    // Step 1: Host sends inference request via bulk OUT endpoint.
    let tensor_data = [1.0f32, 2.0, 3.0, 4.0];
    let tensor_bytes: alloc::vec::Vec<u8> =
        tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let request_data = build_request(
        1,
        "signal_classifier",
        tensor_bytes.len() as u32,
        &tensor_bytes,
    );

    // Step 2: Parse the request.
    let (req, tensor_offset) = InferenceRequest::parse(&request_data).unwrap();
    assert_eq!(req.request_id, 1);
    assert_eq!(req.model_name_str(), "signal_classifier");
    assert_eq!(req.tensor_size as usize, tensor_bytes.len());

    // Verify tensor data is accessible after the header.
    let received_tensor = &request_data[tensor_offset..tensor_offset + req.tensor_size as usize];
    assert_eq!(received_tensor, &tensor_bytes[..]);

    // Step 3: Track the request.
    tracker.add(req.request_id).unwrap();
    assert!(tracker.is_active(req.request_id));

    // Step 4: Simulate inference result (4 f32 classification probabilities).
    let result_data = [0.1f32, 0.7, 0.15, 0.05];
    let result_bytes: alloc::vec::Vec<u8> =
        result_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Step 5: Format response header.
    let response_header = format_response_header(
        req.request_id,
        InferenceStatus::Ok,
        result_bytes.len() as u32,
    );

    // Verify response header.
    let resp_id = u32::from_le_bytes([
        response_header[0],
        response_header[1],
        response_header[2],
        response_header[3],
    ]);
    let resp_status = u16::from_le_bytes([response_header[4], response_header[5]]);
    let resp_size = u32::from_le_bytes([
        response_header[6],
        response_header[7],
        response_header[8],
        response_header[9],
    ]);

    assert_eq!(resp_id, 1);
    assert_eq!(resp_status, 0x0000); // OK
    assert_eq!(resp_size as usize, result_bytes.len());

    // Step 6: Complete the request.
    assert!(tracker.remove(req.request_id));
    assert_eq!(tracker.active_count(), 0);
}

#[test]
fn inference_request_model_not_found_response() {
    let request_data = build_request(7, "nonexistent_model", 0, &[]);
    let (req, _) = InferenceRequest::parse(&request_data).unwrap();

    // Simulate model-not-found result.
    let response_header = format_response_header(req.request_id, InferenceStatus::ModelNotFound, 0);
    let resp_status = u16::from_le_bytes([response_header[4], response_header[5]]);
    assert_eq!(resp_status, InferenceStatus::ModelNotFound as u16);
}

// ---------------------------------------------------------------------------
// 5. Gadget descriptor composition for inference function
// ---------------------------------------------------------------------------

#[test]
fn inference_gadget_descriptor_composition() {
    // Configure gadget device.
    let config = GadgetDeviceConfig {
        vendor_id: 0x1209,
        product_id: 0x0001,
        device_class: 0xFF,
        device_sub_class: 0x01,
        device_protocol: 0x01,
        bcd_device: 0x0100,
    };

    // Build device descriptor.
    let device_desc = compose_device_descriptor(&config);
    assert_eq!(device_desc[0], 18);
    assert_eq!(device_desc[1], descriptor_type::DEVICE);
    assert_eq!(device_desc[4], INTERFACE_CLASS); // vendor-specific class

    // Build inference function's interface and endpoint descriptors.
    let mut iface = [0u8; 9];
    iface[0] = 9;
    iface[1] = descriptor_type::INTERFACE;
    iface[2] = 0; // interface number
    iface[4] = 2; // num endpoints
    iface[5] = INTERFACE_CLASS;
    iface[6] = INTERFACE_SUBCLASS;
    iface[7] = INTERFACE_PROTOCOL;

    let mut ep_in = [0u8; 7];
    ep_in[0] = 7;
    ep_in[1] = descriptor_type::ENDPOINT;
    ep_in[2] = EP_BULK_IN; // 0x81
    ep_in[3] = 0x02; // Bulk
    ep_in[4] = 0x00;
    ep_in[5] = 0x02; // 512 bytes

    let mut ep_out = [0u8; 7];
    ep_out[0] = 7;
    ep_out[1] = descriptor_type::ENDPOINT;
    ep_out[2] = EP_BULK_OUT; // 0x02
    ep_out[3] = 0x02; // Bulk
    ep_out[4] = 0x00;
    ep_out[5] = 0x02; // 512 bytes

    let func = GadgetFunctionDescriptors {
        interfaces: [iface, [0u8; 9]],
        num_interfaces: 1,
        endpoints: [ep_in, ep_out, [0u8; 7], [0u8; 7]],
        num_endpoints: 2,
        interface_class: INTERFACE_CLASS,
        interface_sub_class: INTERFACE_SUBCLASS,
        interface_protocol: INTERFACE_PROTOCOL,
    };

    // Compose config descriptor.
    let mut config_desc = [0u8; MAX_CONFIG_DESC_LEN];
    let config_len = compose_config_descriptor(&[func], &mut config_desc);
    assert_eq!(config_len, 9 + 9 + 7 + 7); // config + interface + 2 endpoints
    assert_eq!(config_desc[0], 9);
    assert_eq!(config_desc[1], descriptor_type::CONFIGURATION);
    assert_eq!(config_desc[4], 1); // 1 interface
}

// ---------------------------------------------------------------------------
// 6. EP0 standard requests with inference gadget descriptors
// ---------------------------------------------------------------------------

#[test]
fn inference_gadget_handle_get_device_descriptor() {
    let config = GadgetDeviceConfig::default();
    let device_desc = compose_device_descriptor(&config);
    let config_desc = [0u8; MAX_CONFIG_DESC_LEN];
    let mut response = [0u8; 64];

    let setup = UsbSetupPacket::new(0x80, request::GET_DESCRIPTOR, 0x0100, 0, 18);
    let len =
        handle_standard_request(&setup, &device_desc, &config_desc, 9, &mut response).unwrap();
    assert_eq!(len, 18);
    assert_eq!(response[0], 18);
    assert_eq!(response[1], descriptor_type::DEVICE);
}

#[test]
fn inference_gadget_handle_set_configuration() {
    let config = GadgetDeviceConfig::default();
    let device_desc = compose_device_descriptor(&config);
    let config_desc = [0u8; MAX_CONFIG_DESC_LEN];
    let mut response = [0u8; 64];

    let setup = UsbSetupPacket::new(0x00, request::SET_CONFIGURATION, 1, 0, 0);
    let len =
        handle_standard_request(&setup, &device_desc, &config_desc, 9, &mut response).unwrap();
    assert_eq!(len, 0);
}

// ---------------------------------------------------------------------------
// 7. Protocol constants verification
// ---------------------------------------------------------------------------

#[test]
fn inference_gadget_endpoint_addresses() {
    assert_eq!(EP_BULK_OUT, 0x02);
    assert_eq!(EP_BULK_IN, 0x81);
    // Verify directions.
    assert_eq!(
        EP_BULK_OUT & 0x80,
        0,
        "OUT endpoint should have direction bit clear"
    );
    assert_ne!(
        EP_BULK_IN & 0x80,
        0,
        "IN endpoint should have direction bit set"
    );
}

#[test]
fn inference_gadget_interface_class_is_vendor_specific() {
    assert_eq!(INTERFACE_CLASS, 0xFF);
}
