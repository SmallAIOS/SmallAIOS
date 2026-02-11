// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Mock Zenoh client integration test (Task 9.12).
//!
//! Simulates an external client connecting via the TCP transport,
//! completing the Init/Open handshake, subscribing to inference
//! results via pub/sub, issuing a query via request/reply, and
//! receiving messages through the full IPC pipeline.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use smallaios_ipc::endpoints::{EndpointKind, EndpointRegistry};
use smallaios_ipc::inference_proto::{
    InferenceRequest, InferenceRequestType, InferenceResponse, ResponseStatus, TensorData,
    TensorDataType,
};
use smallaios_ipc::key_expr::KeyExpr;
use smallaios_ipc::pubsub::{Publisher, PublisherId, Subscriber};
use smallaios_ipc::reqrep::{QueryRouter, Reply};
use smallaios_ipc::router::Router;
use smallaios_ipc::shm::ShmRingBuffer;
use smallaios_ipc::tcp_transport::{
    ConnectionState, Frame, FrameType, TcpTransport, TransportConfig,
};

// ============================================================================
// Helper: simulate a full TCP handshake between client and server
// ============================================================================

fn complete_handshake() -> (TcpTransport, TcpTransport) {
    let mut client = TcpTransport::new(TransportConfig::default());
    let mut server = TcpTransport::new(TransportConfig::default());

    // Client initiates
    let init = client.initiate();
    assert_eq!(init.frame_type, FrameType::Init);

    // Server receives Init, replies with InitAck
    let init_ack = server.handle_frame(&init).unwrap().unwrap();
    assert_eq!(init_ack.frame_type, FrameType::InitAck);
    assert_eq!(*server.state(), ConnectionState::InitReceived);

    // Client receives InitAck, sends Open
    let open = client.handle_frame(&init_ack).unwrap().unwrap();
    assert_eq!(open.frame_type, FrameType::Open);
    assert_eq!(*client.state(), ConnectionState::Open);

    // Server receives Open, replies with OpenAck
    let open_ack = server.handle_frame(&open).unwrap().unwrap();
    assert_eq!(open_ack.frame_type, FrameType::OpenAck);
    assert_eq!(*server.state(), ConnectionState::Open);

    (client, server)
}

// ============================================================================
// Test 1: Full handshake + data exchange via TCP transport (loopback)
// ============================================================================

#[test]
fn mock_client_handshake_and_data_exchange() {
    let (mut client, mut server) = complete_handshake();

    // Client sends a Data frame to the server
    let data_frame = Frame::new(FrameType::Data, 0x00, vec![0xCA, 0xFE, 0xBA, 0xBE]);
    let encoded = data_frame.encode();

    // Server feeds the raw bytes and decodes
    server.feed_data(&encoded);
    let decoded = server.try_decode_frame().unwrap().unwrap();
    assert_eq!(decoded.frame_type, FrameType::Data);
    assert_eq!(decoded.payload, vec![0xCA, 0xFE, 0xBA, 0xBE]);

    // Server processes the data frame (accepted silently in Open state)
    let response = server.handle_frame(&decoded).unwrap();
    assert!(response.is_none());

    // Clean close
    let close = client.close();
    let _ = server.handle_frame(&close).unwrap();
    assert_eq!(*server.state(), ConnectionState::Closed);
}

// ============================================================================
// Test 2: Client subscribes, publisher publishes, subscriber receives
// ============================================================================

#[test]
fn mock_client_subscribe_and_receive_inference_result() {
    // Set up router
    let mut router = Router::new();

    // Client subscribes to inference results
    let sub_id = router
        .subscribe(KeyExpr::new("sai/inference/result/**").unwrap())
        .unwrap();

    // Create a subscriber for the client
    let mut subscriber = Subscriber::new(sub_id, KeyExpr::new("sai/inference/result/**").unwrap());

    // Server-side publisher publishes inference results
    let publisher = Publisher::new(
        PublisherId(1),
        KeyExpr::new("sai/inference/result/model-1").unwrap(),
    );

    // Simulate inference output as a serialized tensor
    let output_data: Vec<u8> = vec![0x3F, 0x80, 0x00, 0x00]; // 1.0f32 in BE
    let msg = publisher.create_message(output_data.clone(), 42);

    // Router matches the published key against subscriptions
    let matches = router.match_key(&msg.key);
    assert!(matches.contains(&sub_id));

    // Deliver to subscriber
    subscriber.deliver(msg.clone()).unwrap();

    // Client receives the message
    let received = subscriber.receive().unwrap();
    assert_eq!(received.key.as_str(), "sai/inference/result/model-1");
    assert_eq!(received.payload, output_data);
    assert_eq!(received.timestamp, 42);
    assert_eq!(received.publisher, PublisherId(1));
}

// ============================================================================
// Test 3: Client issues inference request via request/reply
// ============================================================================

#[test]
fn mock_client_inference_request_reply() {
    let mut qrouter = QueryRouter::new();

    // Server registers a queryable for inference
    let qid = qrouter
        .register(KeyExpr::new("sai/inference").unwrap())
        .unwrap();

    // Client submits an inference query
    let input_tensor = TensorData {
        name: String::from("input_0"),
        data_type: TensorDataType::Float32,
        shape: vec![1, 3, 224, 224],
        data: vec![0u8; 16],
    };

    let request = InferenceRequest {
        request_type: InferenceRequestType::RunInference,
        request_id: 1001,
        model_id: 42,
        inputs: vec![input_tensor],
        timeout_ms: 5000,
    };

    // Encode and submit as query payload
    let request_bytes = request.encode();
    let query_id = qrouter
        .submit_query(
            &KeyExpr::new("sai/inference").unwrap(),
            request_bytes.clone(),
            100,
        )
        .unwrap();

    // Server-side: dequeue the query from the queryable
    let queryable = qrouter.get_queryable_mut(qid).unwrap();
    let query = queryable.next_query().unwrap();
    assert_eq!(query.id, query_id);

    // Server decodes the inference request
    let decoded_req = InferenceRequest::decode(&query.payload).unwrap();
    assert_eq!(decoded_req.request_type, InferenceRequestType::RunInference);
    assert_eq!(decoded_req.model_id, 42);
    assert_eq!(decoded_req.inputs.len(), 1);
    assert_eq!(decoded_req.inputs[0].name, "input_0");

    // Server produces an inference response
    let response = InferenceResponse {
        request_id: decoded_req.request_id,
        status: ResponseStatus::Ok,
        outputs: vec![TensorData {
            name: String::from("output_0"),
            data_type: TensorDataType::Float32,
            shape: vec![1, 1000],
            data: vec![0u8; 4000],
        }],
        elapsed_us: 1500,
    };

    // Encode response and create Reply
    let response_bytes = response.encode();
    let reply = Reply {
        query_id: query.id,
        key_expr: KeyExpr::new("sai/inference").unwrap(),
        payload: response_bytes.clone(),
        timestamp: 200,
    };

    // Client decodes the reply
    let decoded_resp = InferenceResponse::decode(&reply.payload).unwrap();
    assert_eq!(decoded_resp.status, ResponseStatus::Ok);
    assert_eq!(decoded_resp.request_id, 1001);
    assert_eq!(decoded_resp.outputs.len(), 1);
    assert_eq!(decoded_resp.outputs[0].name, "output_0");
    assert_eq!(decoded_resp.elapsed_us, 1500);
}

// ============================================================================
// Test 4: End-to-end via SHM ring buffer (loopback transport)
// ============================================================================

#[test]
fn mock_client_shm_loopback_inference() {
    let ring = ShmRingBuffer::new();

    // Client sends an inference request via SHM
    let request = InferenceRequest {
        request_type: InferenceRequestType::RunInference,
        request_id: 2002,
        model_id: 7,
        inputs: vec![TensorData {
            name: String::from("image"),
            data_type: TensorDataType::Uint8,
            shape: vec![1, 224, 224, 3],
            data: vec![128u8; 64],
        }],
        timeout_ms: 3000,
    };
    let encoded = request.encode();
    ring.push(&encoded).unwrap();

    // Server receives from SHM
    let mut buf = vec![0u8; 4096];
    let n = ring.pop(&mut buf).unwrap();
    let decoded = InferenceRequest::decode(&buf[..n]).unwrap();
    assert_eq!(decoded.request_id, 2002);
    assert_eq!(decoded.model_id, 7);

    // Server sends response back via SHM
    let response = InferenceResponse {
        request_id: 2002,
        status: ResponseStatus::Ok,
        outputs: vec![TensorData {
            name: String::from("classes"),
            data_type: TensorDataType::Float32,
            shape: vec![1, 10],
            data: vec![0u8; 40],
        }],
        elapsed_us: 500,
    };
    let resp_bytes = response.encode();
    ring.push(&resp_bytes).unwrap();

    // Client receives response
    let n = ring.pop(&mut buf).unwrap();
    let client_resp = InferenceResponse::decode(&buf[..n]).unwrap();
    assert_eq!(client_resp.status, ResponseStatus::Ok);
    assert_eq!(client_resp.request_id, 2002);
    assert_eq!(client_resp.outputs[0].name, "classes");
}

// ============================================================================
// Test 5: Full pipeline — connect, register endpoints, subscribe, query
// ============================================================================

#[test]
fn mock_client_full_pipeline() {
    // 1. TCP handshake
    let (_client, _server) = complete_handshake();

    // 2. Register default endpoints
    let mut registry = EndpointRegistry::new();
    registry.register_defaults().unwrap();
    assert_eq!(registry.len(), 5);

    // Verify inference endpoint exists
    let inf_ep = registry.find_by_kind(&EndpointKind::Inference).unwrap();
    assert_eq!(inf_ep.key_expr.as_str(), "sai/inference");

    // 3. Set up pub/sub router with wildcard subscription
    let mut router = Router::new();
    let health_sub = router
        .subscribe(KeyExpr::new("sai/health").unwrap())
        .unwrap();
    let inference_sub = router
        .subscribe(KeyExpr::new("sai/inference/**").unwrap())
        .unwrap();
    let metrics_sub = router
        .subscribe(KeyExpr::new("sai/metrics").unwrap())
        .unwrap();

    // 4. Publish health status
    let health_pub = Publisher::new(PublisherId(100), KeyExpr::new("sai/health").unwrap());
    let health_msg = health_pub.create_message(b"healthy".to_vec(), 1);
    let health_matches = router.match_key(&health_msg.key);
    assert!(health_matches.contains(&health_sub));
    assert!(!health_matches.contains(&inference_sub));

    // 5. Publish inference result
    let inference_pub = Publisher::new(
        PublisherId(200),
        KeyExpr::new("sai/inference/result").unwrap(),
    );
    let inf_msg = inference_pub.create_message(vec![0xDE, 0xAD], 2);
    let inf_matches = router.match_key(&inf_msg.key);
    assert!(inf_matches.contains(&inference_sub));
    assert!(!inf_matches.contains(&health_sub));

    // 6. Set up request/reply for inference
    let mut qrouter = QueryRouter::new();
    let _ = qrouter
        .register(KeyExpr::new("sai/inference").unwrap())
        .unwrap();

    // Submit query and verify routing
    let qid = qrouter
        .submit_query(
            &KeyExpr::new("sai/inference").unwrap(),
            vec![0x01, 0x02],
            10,
        )
        .unwrap();
    assert_eq!(qid.0, 1);

    // 7. Verify metrics subscription routing
    let metrics_pub = Publisher::new(PublisherId(300), KeyExpr::new("sai/metrics").unwrap());
    let metrics_msg = metrics_pub.create_message(b"cpu=75".to_vec(), 3);
    let metrics_matches = router.match_key(&metrics_msg.key);
    assert!(metrics_matches.contains(&metrics_sub));
}

// ============================================================================
// Test 6: Multiple clients subscribing to different topics
// ============================================================================

#[test]
fn mock_multiple_clients_different_topics() {
    let mut router = Router::new();

    // Client A subscribes to sensor data
    let client_a_sub = router
        .subscribe(KeyExpr::new("sensor/**").unwrap())
        .unwrap();

    // Client B subscribes to inference results
    let client_b_sub = router
        .subscribe(KeyExpr::new("sai/inference/**").unwrap())
        .unwrap();

    // Client C subscribes to everything
    let client_c_sub = router.subscribe(KeyExpr::new("**").unwrap()).unwrap();

    // Create subscribers
    let mut sub_a = Subscriber::new(client_a_sub, KeyExpr::new("sensor/**").unwrap());
    let mut sub_b = Subscriber::new(client_b_sub, KeyExpr::new("sai/inference/**").unwrap());
    let mut sub_c = Subscriber::new(client_c_sub, KeyExpr::new("**").unwrap());

    // Publish sensor data
    let sensor_pub = Publisher::new(PublisherId(1), KeyExpr::new("sensor/temp/room1").unwrap());
    let sensor_msg = sensor_pub.create_message(vec![22], 100);

    let matches = router.match_key(&sensor_msg.key);
    assert!(matches.contains(&client_a_sub)); // sensor/**
    assert!(!matches.contains(&client_b_sub)); // not inference
    assert!(matches.contains(&client_c_sub)); // **

    // Deliver to matching subscribers
    for id in &matches {
        if *id == client_a_sub {
            sub_a.deliver(sensor_msg.clone()).unwrap();
        }
        if *id == client_c_sub {
            sub_c.deliver(sensor_msg.clone()).unwrap();
        }
    }

    // Publish inference result
    let inf_pub = Publisher::new(
        PublisherId(2),
        KeyExpr::new("sai/inference/result").unwrap(),
    );
    let inf_msg = inf_pub.create_message(vec![99], 200);

    let matches = router.match_key(&inf_msg.key);
    assert!(!matches.contains(&client_a_sub)); // not sensor
    assert!(matches.contains(&client_b_sub)); // inference/**
    assert!(matches.contains(&client_c_sub)); // **

    for id in &matches {
        if *id == client_b_sub {
            sub_b.deliver(inf_msg.clone()).unwrap();
        }
        if *id == client_c_sub {
            sub_c.deliver(inf_msg.clone()).unwrap();
        }
    }

    // Verify each client got the right messages
    assert_eq!(sub_a.pending(), 1);
    assert_eq!(sub_a.receive().unwrap().payload, vec![22]);

    assert_eq!(sub_b.pending(), 1);
    assert_eq!(sub_b.receive().unwrap().payload, vec![99]);

    assert_eq!(sub_c.pending(), 2);
    assert_eq!(sub_c.receive().unwrap().payload, vec![22]);
    assert_eq!(sub_c.receive().unwrap().payload, vec![99]);
}

// ============================================================================
// Test 7: Wire-level frame exchange — simulate TCP stream bytes
// ============================================================================

#[test]
fn mock_client_wire_level_frame_exchange() {
    let mut client = TcpTransport::new(TransportConfig::default());
    let mut server = TcpTransport::new(TransportConfig::default());

    // Client initiates, encode Init frame to bytes
    let init_frame = client.initiate();
    let init_bytes = init_frame.encode();

    // Server receives raw bytes (simulating TCP stream)
    server.feed_data(&init_bytes);
    let decoded_init = server.try_decode_frame().unwrap().unwrap();
    assert_eq!(decoded_init.frame_type, FrameType::Init);

    // Server processes and produces InitAck
    let init_ack = server.handle_frame(&decoded_init).unwrap().unwrap();
    let init_ack_bytes = init_ack.encode();

    // Client receives InitAck bytes
    client.feed_data(&init_ack_bytes);
    let decoded_init_ack = client.try_decode_frame().unwrap().unwrap();
    let open = client.handle_frame(&decoded_init_ack).unwrap().unwrap();
    let open_bytes = open.encode();

    // Server receives Open bytes
    server.feed_data(&open_bytes);
    let decoded_open = server.try_decode_frame().unwrap().unwrap();
    let _open_ack = server.handle_frame(&decoded_open).unwrap().unwrap();

    // Both are now Open
    assert_eq!(*client.state(), ConnectionState::Open);
    assert_eq!(*server.state(), ConnectionState::Open);

    // Exchange data frames at the wire level
    let data_payload = b"inference_result:{confidence:0.95}";
    let data_frame = Frame::new(FrameType::Data, 0x00, data_payload.to_vec());
    let data_bytes = data_frame.encode();

    // Feed data in two parts (partial TCP segments)
    let mid = data_bytes.len() / 2;
    server.feed_data(&data_bytes[..mid]);
    assert!(server.try_decode_frame().unwrap().is_none()); // not enough yet

    server.feed_data(&data_bytes[mid..]);
    let decoded_data = server.try_decode_frame().unwrap().unwrap();
    assert_eq!(decoded_data.payload, data_payload.to_vec());

    // KeepAlive exchange
    let ka_frame = Frame::new(FrameType::KeepAlive, 0x00, Vec::new());
    let ka_bytes = ka_frame.encode();
    server.feed_data(&ka_bytes);
    let decoded_ka = server.try_decode_frame().unwrap().unwrap();
    let ka_response = server.handle_frame(&decoded_ka).unwrap().unwrap();
    assert_eq!(ka_response.frame_type, FrameType::KeepAlive);
}
