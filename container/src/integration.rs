// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate integration verification tests.
//!
//! These tests verify that the container subsystem correctly integrates with
//! all dependent crates: kernel safety modules, ONNX runtime, IPC messaging,
//! POSIX compatibility, security, network stack, and (optionally) NVIDIA GPU.
//!
//! The 26 tests are organized into 9 categories:
//! 1. Boot-to-Ready Integration (3)
//! 2. Network Stack Integration (3)
//! 3. ONNX Runtime Integration (3)
//! 4. IPC Messaging Integration (4)
//! 5. Security Integration (3)
//! 6. POSIX Compatibility Integration (3)
//! 7. GPU Provider Integration (2)
//! 8. Safety Verification Integration (3)
//! 9. Container Lifecycle Integration (2)

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    // =========================================================================
    // 1. Boot-to-Ready Integration (3 tests)
    // =========================================================================

    /// Verify the full boot sequence runs through all 8 phases to Ready.
    #[test]
    fn test_full_boot_sequence() {
        use crate::boot::{BootPhase, BootSequence};
        use crate::config::ContainerConfig;

        let config = ContainerConfig::default();
        let mut seq = BootSequence::new(config);

        // Should start at ConfigLoaded, not yet ready.
        assert_eq!(*seq.phase(), BootPhase::ConfigLoaded);
        assert!(!seq.is_ready());

        // Run all phases to completion.
        let statuses = seq.run_all().unwrap();

        // 8 transitions: ConfigLoaded -> ... -> Ready (8 advances).
        assert_eq!(statuses.len(), 8);
        assert!(seq.is_ready());
        assert_eq!(*seq.phase(), BootPhase::Ready);

        // All 8 intermediate phases should be recorded.
        let completed = seq.completed_phases();
        assert_eq!(completed.len(), 8);
    }

    /// Verify boot integrates with the health checker subsystem.
    #[test]
    fn test_boot_with_health_checker() {
        use crate::boot::BootSequence;
        use crate::config::ContainerConfig;
        use crate::health::{HealthChecker, HealthStatus};

        let config = ContainerConfig::default();
        let mut seq = BootSequence::new(config);
        seq.run_all().unwrap();
        assert!(seq.is_ready());

        // After boot, the health checker should start healthy.
        let mut hc = HealthChecker::new();
        hc.register_component("memory").unwrap();
        hc.register_component("scheduler").unwrap();
        hc.register_component("network").unwrap();

        // Update all components to Healthy.
        hc.update_component("memory", HealthStatus::Healthy, "ok")
            .unwrap();
        hc.update_component("scheduler", HealthStatus::Healthy, "ok")
            .unwrap();
        hc.update_component("network", HealthStatus::Healthy, "ok")
            .unwrap();

        assert_eq!(hc.check_all(), HealthStatus::Healthy);
        assert_eq!(hc.component_count(), 3);

        // Set readiness (mirrors post-boot readiness).
        hc.set_ready(true);
        assert!(hc.is_ready());

        // Health JSON should contain "healthy".
        let json = hc.to_health_json();
        assert!(json.contains("\"status\":\"healthy\""));
    }

    /// Verify boot integrates with the metrics subsystem.
    #[test]
    fn test_boot_with_metrics() {
        use crate::boot::BootSequence;
        use crate::config::ContainerConfig;
        use crate::metrics::MetricsRegistry;

        let config = ContainerConfig::default();
        let mut seq = BootSequence::new(config);
        seq.run_all().unwrap();
        assert!(seq.is_ready());

        // Register default metrics.
        let mut metrics = MetricsRegistry::new();
        metrics.register_defaults().unwrap();
        assert_eq!(metrics.metric_count(), 6);

        // Simulate updating metrics after boot (use actual default metric names).
        let _ = metrics.increment("smallaios_inference_total", 10.0);
        let _ = metrics.set("smallaios_models_loaded", 2.0);

        assert_eq!(metrics.get("smallaios_inference_total"), Some(10.0));
        assert_eq!(metrics.get("smallaios_models_loaded"), Some(2.0));

        // Prometheus exposition format should include registered metrics.
        let prom = metrics.to_prometheus_format();
        assert!(prom.contains("smallaios_inference_total"));
        assert!(prom.contains("smallaios_models_loaded"));
    }

    // =========================================================================
    // 2. Network Stack Integration (3 tests)
    // =========================================================================

    /// Verify Ethernet framing integrates with IPv4 parsing.
    #[test]
    fn test_ethernet_to_ip_stack() {
        use smallaios_net::ethernet::{EthernetFrame, MacAddress, ETHERTYPE_IPV4};
        use smallaios_net::ipv4::{Ipv4Addr, Ipv4Header};

        // Build a minimal IPv4 header as the Ethernet payload.
        let src_ip = Ipv4Addr::new(192, 168, 1, 10);
        let dst_ip = Ipv4Addr::new(192, 168, 1, 1);
        let ipv4_hdr = Ipv4Header {
            version: 4,
            ihl: 5,
            dscp: 0,
            total_length: 20,
            identification: 0x1234,
            flags: 0,
            fragment_offset: 0,
            ttl: 64,
            protocol: 1, // ICMP
            checksum: 0,
            src_addr: src_ip,
            dst_addr: dst_ip,
        };
        let ip_bytes = ipv4_hdr.serialize();

        // Build an Ethernet frame containing the IPv4 header.
        let src_mac = MacAddress::new(0x02, 0x00, 0x00, 0x00, 0x00, 0x01);
        let dst_mac = MacAddress::new(0x02, 0x00, 0x00, 0x00, 0x00, 0x02);
        let frame = EthernetFrame {
            dst_mac,
            src_mac,
            ether_type: ETHERTYPE_IPV4,
            payload: Vec::from(&ip_bytes[..]),
        };
        let serialized = frame.serialize();

        // Parse the Ethernet frame back.
        let parsed_frame = EthernetFrame::parse(&serialized).unwrap();
        assert_eq!(parsed_frame.ether_type, ETHERTYPE_IPV4);
        assert_eq!(parsed_frame.src_mac, src_mac);
        assert_eq!(parsed_frame.dst_mac, dst_mac);

        // Parse the IPv4 header from the Ethernet payload.
        let (parsed_ip, _payload) = Ipv4Header::parse(&parsed_frame.payload).unwrap();
        assert_eq!(parsed_ip.src_addr, src_ip);
        assert_eq!(parsed_ip.dst_addr, dst_ip);
        assert_eq!(parsed_ip.ttl, 64);
        assert_eq!(parsed_ip.protocol, 1);
    }

    /// Verify ARP request/reply construction and table integration.
    #[test]
    fn test_arp_request_response() {
        use smallaios_net::arp::{create_reply, create_request, ArpOperation, ArpTable};
        use smallaios_net::ethernet::MacAddress;

        let sender_mac = MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01);
        let sender_ip: [u8; 4] = [10, 0, 0, 1];
        let target_ip: [u8; 4] = [10, 0, 0, 2];

        // Create an ARP request.
        let request = create_request(sender_mac, sender_ip, target_ip);
        assert_eq!(request.operation, ArpOperation::Request);
        assert_eq!(request.sender_ip, sender_ip);
        assert_eq!(request.target_ip, target_ip);

        // Serialize and re-parse the request.
        let req_bytes = request.serialize();
        let parsed_req = smallaios_net::arp::ArpPacket::parse(&req_bytes).unwrap();
        assert_eq!(parsed_req.operation, ArpOperation::Request);

        // Simulate a reply.
        let target_mac = MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x02);
        let reply = create_reply(target_mac, target_ip, sender_mac, sender_ip);
        assert_eq!(reply.operation, ArpOperation::Reply);

        // Insert into ARP table.
        let mut arp_table = ArpTable::new();
        let _ = arp_table.insert(target_ip, target_mac, 1000);
        assert_eq!(arp_table.len(), 1);
        let resolved = arp_table.lookup(&target_ip);
        assert!(resolved.is_some());
    }

    /// Verify ICMP echo request/reply construction and checksum validity.
    #[test]
    fn test_icmp_echo_roundtrip() {
        use smallaios_net::icmp::{create_echo_reply, create_echo_request, IcmpType};

        // Create an echo request.
        let request = create_echo_request(0x1234, 1, &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(request.msg_type, IcmpType::EchoRequest as u8);

        // Serialize and re-parse.
        let req_bytes = request.serialize();
        let parsed = smallaios_net::icmp::IcmpPacket::parse(&req_bytes).unwrap();
        assert_eq!(parsed.msg_type, IcmpType::EchoRequest as u8);
        assert_eq!(parsed.code, 0);

        // Create an echo reply.
        let reply = create_echo_reply(0x1234, 1, &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(reply.msg_type, IcmpType::EchoReply as u8);

        // Verify the reply serializes correctly.
        let reply_bytes = reply.serialize();
        let parsed_reply = smallaios_net::icmp::IcmpPacket::parse(&reply_bytes).unwrap();
        assert_eq!(parsed_reply.msg_type, IcmpType::EchoReply as u8);
    }

    // =========================================================================
    // 3. ONNX Runtime Integration (3 tests)
    // =========================================================================

    /// Verify protobuf decoding integrates with tensor construction.
    #[test]
    fn test_protobuf_to_tensor() {
        use smallaios_onnx_rt::protobuf::{ProtoDecoder, WireType};
        use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};

        // Encode a varint manually: field 1, wire type 0 (varint), value 1 (DataType::Float).
        let data: &[u8] = &[0x08, 0x01]; // tag: field=1, wiretype=0; value=1
        let mut decoder = ProtoDecoder::new(data);

        // Read tag.
        let tag = decoder.read_tag().unwrap();
        assert_eq!(tag.field_number, 1);
        assert_eq!(tag.wire_type, WireType::Varint);

        // Read value.
        let val = decoder.read_varint().unwrap();
        assert_eq!(val, 1);
        assert!(decoder.is_empty());

        // Use the decoded value to determine data type.
        let dtype = DataType::from_i32(val as i32).unwrap();
        assert_eq!(dtype, DataType::Float);
        assert!(dtype.is_float());
        assert_eq!(dtype.element_size(), 4);

        // Construct a tensor with the decoded type.
        let shape = TensorShape::new(vec![1, 3, 224, 224]);
        assert_eq!(shape.ndim(), 4);
        assert_eq!(shape.total_elements(), 3 * 224 * 224);
        assert!(shape.is_valid());

        let tensor = Tensor::new(dtype, shape, String::from("input"));
        assert_eq!(tensor.data_type, DataType::Float);
        assert_eq!(tensor.name, "input");
    }

    /// Verify the execution graph builder and optimizer work together.
    #[test]
    fn test_graph_build_and_optimize() {
        use smallaios_onnx_rt::graph::build_execution_graph;
        use smallaios_onnx_rt::onnx_types::{GraphProto, NodeProto, ValueInfoProto};
        use smallaios_onnx_rt::optimizer::{optimize, OptimizationLevel, OptimizerConfig};

        // Build a simple graph: input -> Relu -> output.
        let mut graph_proto = GraphProto {
            name: String::from("test_graph"),
            ..GraphProto::default()
        };
        graph_proto.input.push(ValueInfoProto {
            name: String::from("X"),
            elem_type: 1, // Float
            shape: vec![1, 10],
        });
        graph_proto.output.push(ValueInfoProto {
            name: String::from("Y"),
            elem_type: 1,
            shape: vec![1, 10],
        });
        graph_proto.node.push(NodeProto {
            input: vec![String::from("X")],
            output: vec![String::from("Y")],
            name: String::from("relu_0"),
            op_type: String::from("Relu"),
            domain: String::new(),
            attribute: Vec::new(),
        });

        // Build execution graph.
        let mut exec_graph = build_execution_graph(&graph_proto).unwrap();
        assert_eq!(exec_graph.node_count(), 1);
        assert_eq!(exec_graph.execution_order().len(), 1);

        // Apply optimizer (stubs, but verifies integration).
        let config = OptimizerConfig::default();
        assert_eq!(config.level, OptimizationLevel::Extended);

        let result = optimize(&mut exec_graph, &config);
        // Extended applies DCE + ConstantFolding + Fusion (3 passes).
        assert_eq!(result.passes_applied.len(), 3);
        // Stubs return 0 for both counters.
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(result.nodes_fused, 0);
    }

    /// Verify the operator registry covers Tier 1 (29) plus Tier 2 (36) ops.
    #[test]
    fn test_operator_registry_covers_tier1() {
        use smallaios_onnx_rt::operators::{OpKind, OperatorRegistry};

        let registry = OperatorRegistry::new();
        assert_eq!(registry.supported_count(), 65);

        // Verify critical operators are present.
        let critical_ops = [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "MatMul",
            "Relu",
            "Sigmoid",
            "Tanh",
            "Softmax",
            "Conv",
            "MaxPool",
            "Reshape",
            "Transpose",
            "Concat",
            "Gemm",
            "BatchNormalization",
            "LayerNormalization",
            "Flatten",
        ];
        for op_name in &critical_ops {
            assert!(
                registry.is_supported(op_name),
                "Operator {} should be supported",
                op_name
            );
        }

        // Verify an unsupported operator returns false.
        assert!(!registry.is_supported("CustomOp"));
        assert!(!registry.is_supported(""));

        // Verify OpKind round-trip.
        let relu = OpKind::parse_str("Relu").unwrap();
        assert_eq!(relu.name(), "Relu");
    }

    // =========================================================================
    // 4. IPC Messaging Integration (4 tests)
    // =========================================================================

    /// Verify pub/sub message flow: publisher -> subscriber.
    #[test]
    fn test_pubsub_full_flow() {
        use smallaios_ipc::key_expr::KeyExpr;
        use smallaios_ipc::pubsub::{Publisher, PublisherId, Subscriber};
        use smallaios_ipc::router::SubscriptionId;

        let key = KeyExpr::new("sensor/temperature").unwrap();

        // Create publisher and subscriber.
        let publisher = Publisher::new(PublisherId(1), key.clone());
        let mut subscriber = Subscriber::new(SubscriptionId(100), key.clone());

        // Publish a message.
        let msg = publisher.create_message(vec![0x42, 0x43], 1000);
        assert_eq!(msg.payload, vec![0x42, 0x43]);
        assert_eq!(msg.timestamp, 1000);

        // Deliver to subscriber.
        subscriber.deliver(msg).unwrap();
        assert_eq!(subscriber.pending(), 1);

        // Receive the message.
        let received = subscriber.receive().unwrap();
        assert_eq!(received.payload, vec![0x42, 0x43]);
        assert_eq!(received.timestamp, 1000);
        assert_eq!(subscriber.pending(), 0);

        // No more messages.
        assert!(subscriber.receive().is_none());
    }

    /// Verify request/reply flow: router -> queryable -> response.
    #[test]
    fn test_request_reply_flow() {
        use smallaios_ipc::key_expr::KeyExpr;
        use smallaios_ipc::reqrep::QueryRouter;

        let mut router = QueryRouter::new();

        // Register a queryable endpoint.
        let key = KeyExpr::new("model/inference").unwrap();
        let qid = router.register(key.clone()).unwrap();
        assert_eq!(router.queryable_count(), 1);

        // Submit a query.
        let query_key = KeyExpr::new("model/inference").unwrap();
        let query_id = router.submit_query(&query_key, vec![1, 2, 3], 500).unwrap();

        // The queryable should have received the query.
        let queryable = router.get_queryable_mut(qid).unwrap();
        assert_eq!(queryable.pending_count(), 1);

        let query = queryable.next_query().unwrap();
        assert_eq!(query.id, query_id);
        assert_eq!(query.payload, vec![1, 2, 3]);
        assert_eq!(query.timestamp, 500);
        assert_eq!(queryable.pending_count(), 0);
    }

    /// Verify shared memory ring buffer throughput: push/pop multiple messages.
    #[test]
    fn test_shm_ring_buffer_throughput() {
        use smallaios_ipc::shm::ShmRingBuffer;

        let ring = ShmRingBuffer::new();
        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.len(), 0);

        // Push several messages.
        let messages: Vec<Vec<u8>> = (0..10u8).map(|i| vec![i; 64]).collect();
        for msg in &messages {
            ring.push(msg).unwrap();
        }
        assert_eq!(ring.len(), 10);
        assert!(!ring.is_empty());

        // Pop and verify order (FIFO).
        for (i, expected_msg) in messages.iter().enumerate() {
            let mut buf = [0u8; 128];
            let n = ring.pop(&mut buf).unwrap();
            assert_eq!(n, 64);
            assert_eq!(&buf[..n], &expected_msg[..], "Message {} mismatch", i);
        }

        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
    }

    /// Verify the HTTP health endpoint handler.
    #[test]
    fn test_http_health_endpoint() {
        use smallaios_ipc::http::{handle_request, HttpRequest, HttpStatus};

        // Parse a GET /health request.
        let raw = b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let request = HttpRequest::parse(raw).unwrap();
        let response = handle_request(&request);
        assert_eq!(response.status, HttpStatus::Ok);

        // Check body contains health status.
        let body_str = core::str::from_utf8(&response.body).unwrap();
        assert!(body_str.contains("healthy"));

        // Test /ready endpoint.
        let raw_ready = b"GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let req_ready = HttpRequest::parse(raw_ready).unwrap();
        let resp_ready = handle_request(&req_ready);
        assert_eq!(resp_ready.status, HttpStatus::Ok);

        // Test /live endpoint.
        let raw_live = b"GET /live HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let req_live = HttpRequest::parse(raw_live).unwrap();
        let resp_live = handle_request(&req_live);
        assert_eq!(resp_live.status, HttpStatus::Ok);

        // Test unknown path returns 404.
        let raw_unknown = b"GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let req_unknown = HttpRequest::parse(raw_unknown).unwrap();
        let resp_unknown = handle_request(&req_unknown);
        assert_eq!(resp_unknown.status, HttpStatus::NotFound);
    }

    // =========================================================================
    // 5. Security Integration (3 tests)
    // =========================================================================

    /// Verify capability creation and permission checking.
    #[test]
    fn test_capability_create_and_check() {
        use alloc::boxed::Box;
        use smallaios_security::capability::{CapRegistry, Permissions, ResourceRef, ResourceType};

        // CapRegistry contains a [Option<Capability>; 4096] array (~230 KB).
        // Box it to avoid blowing the stack.
        let mut registry = Box::new(CapRegistry::new());
        assert_eq!(registry.total_count(), 0);

        // Create a capability for task 1 on a tensor buffer.
        let resource = ResourceRef::new(ResourceType::TensorBuffer, 42);
        let perms = Permissions::READ.union(Permissions::WRITE);
        let cap_id = registry.create(1, resource, perms, 0).unwrap();
        assert_eq!(registry.total_count(), 1);
        assert_eq!(registry.count_for_task(1), 1);

        // Check that task 1 has READ access.
        let found = registry.check(1, &resource, Permissions::READ, 0);
        assert!(found.is_ok());
        assert_eq!(found.unwrap(), cap_id);

        // Check that task 1 has WRITE access.
        let found_w = registry.check(1, &resource, Permissions::WRITE, 0);
        assert!(found_w.is_ok());

        // Check that task 2 does NOT have access.
        let denied = registry.check(2, &resource, Permissions::READ, 0);
        assert!(denied.is_err());

        // Lookup the capability.
        let cap = registry.lookup(cap_id, 0).unwrap();
        assert_eq!(cap.owner, 1);
        assert!(cap.permissions.contains(Permissions::READ));
        assert!(cap.permissions.contains(Permissions::WRITE));
        assert!(!cap.permissions.contains(Permissions::EXECUTE));

        // Revoke and verify it is gone.
        registry.revoke(cap_id).unwrap();
        assert_eq!(registry.total_count(), 0);
    }

    /// Verify SHA-3-256 produces deterministic and correct hashes.
    #[test]
    fn test_sha3_hash_deterministic() {
        use smallaios_security::crypto::sha3::sha3_256;

        let data = b"SmallAIOS integration test payload";

        // Hash the same data twice.
        let digest1 = sha3_256(data);
        let digest2 = sha3_256(data);

        // Must be identical.
        assert_eq!(digest1, digest2);
        assert_eq!(digest1.as_bytes().len(), 32);

        // Different input produces different hash.
        let digest3 = sha3_256(b"different data");
        assert_ne!(digest1, digest3);

        // NIST test vector: SHA-3-256("") should produce known result.
        let empty_digest = sha3_256(b"");
        let expected_first_byte = 0xa7u8;
        assert_eq!(
            empty_digest.as_bytes()[0],
            expected_first_byte,
            "SHA-3-256 of empty string should start with 0xa7"
        );
    }

    /// Verify CSPRNG stub behavior (seed sets seeded flag, generate returns NotImplemented).
    #[test]
    fn test_csprng_produces_different_values() {
        use smallaios_security::crypto::csprng::{Csprng, CsprngError, CsprngSeed};

        let mut rng = Csprng::new();
        assert!(!rng.is_seeded());
        assert_eq!(rng.bytes_generated(), 0);

        // Generate without seeding should fail.
        let mut buf = [0u8; 32];
        assert_eq!(rng.generate(&mut buf), Err(CsprngError::NotSeeded));

        // Seed the CSPRNG.
        let seed = CsprngSeed::from_bytes([0x42; 32]);
        rng.seed(&seed).expect("seed should succeed");

        assert!(rng.is_seeded());

        // Generate random bytes.
        rng.generate(&mut buf).expect("generate should succeed");

        // Output should not be all zeros from a non-zero seed.
        assert!(
            buf.iter().any(|&b| b != 0),
            "output should not be all zeros"
        );

        // Verify reseed interval tracking.
        assert!(!rng.needs_reseed());
        assert_eq!(rng.bytes_generated(), 32);
    }

    // =========================================================================
    // 6. POSIX Compatibility Integration (3 tests)
    // =========================================================================

    /// Verify file descriptor table lifecycle: alloc, get, close, dup, count.
    #[test]
    fn test_fd_table_lifecycle() {
        use smallaios_posix::fd::{FdEntry, FdKind, FdTable, OpenFlags};

        // New table starts with stdin/stdout/stderr (3 entries).
        let mut table = FdTable::new();
        assert_eq!(table.count(), 3);
        assert!(table.is_open(0)); // stdin
        assert!(table.is_open(1)); // stdout
        assert!(table.is_open(2)); // stderr

        // Allocate a new file descriptor.
        let entry = FdEntry::new(FdKind::File, OpenFlags::new(OpenFlags::O_RDWR), 100);
        let fd = table.alloc(entry).unwrap();
        assert_eq!(fd, 3); // Lowest free fd.
        assert_eq!(table.count(), 4);

        // Get the entry back.
        let got = table.get(fd).unwrap();
        assert_eq!(got.kind, FdKind::File);
        assert_eq!(got.vnode_id, 100);

        // Close the fd (ref_count 1 -> 0, slot is freed).
        table.close(fd).unwrap();
        assert!(!table.is_open(fd));
        assert_eq!(table.count(), 3);

        // Allocate again — should reuse slot 3 (lowest free).
        let entry2 = FdEntry::new(FdKind::Socket, OpenFlags::new(OpenFlags::O_RDWR), 200);
        let fd2 = table.alloc(entry2).unwrap();
        assert_eq!(fd2, 3); // Reuses slot 3.
        assert!(table.is_open(fd2));
        assert_eq!(table.count(), 4);

        // Dup the fd. dup() bumps ref_count to 2 on both slots.
        let dup_fd = table.dup(fd2).unwrap();
        assert_eq!(dup_fd, 4); // Next free slot.
        assert!(table.is_open(dup_fd));
        assert_eq!(table.count(), 5);

        // Close original — ref_count goes 2 -> 1, entry stays.
        table.close(fd2).unwrap();
        assert!(table.is_open(fd2)); // Still open (ref_count = 1).
        assert_eq!(table.count(), 5);

        // Close again — ref_count goes 1 -> 0, slot freed.
        table.close(fd2).unwrap();
        assert!(!table.is_open(fd2));
        assert_eq!(table.count(), 4);

        // Close dup twice as well.
        table.close(dup_fd).unwrap();
        table.close(dup_fd).unwrap();
        assert!(!table.is_open(dup_fd));
        assert_eq!(table.count(), 3); // Back to just stdio.
    }

    /// Verify the default VFS tree has the expected structure.
    #[test]
    fn test_vfs_default_tree() {
        use smallaios_posix::vfs::build_default_vfs;

        let tree = build_default_vfs();
        assert_eq!(tree.node_count(), 9);

        // Verify root lookup.
        let root = tree.lookup("/").unwrap();
        assert_eq!(root.name, "");

        // Verify /dev directory.
        let dev = tree.lookup("/dev").unwrap();
        assert_eq!(dev.name, "dev");

        // Verify /dev/null character device.
        let null = tree.lookup("/dev/null").unwrap();
        assert_eq!(null.name, "null");

        // Verify /dev/urandom character device.
        let urandom = tree.lookup("/dev/urandom").unwrap();
        assert_eq!(urandom.name, "urandom");

        // Verify /models directory.
        let models = tree.lookup("/models").unwrap();
        assert_eq!(models.name, "models");

        // Verify /config directory.
        let config = tree.lookup("/config").unwrap();
        assert_eq!(config.name, "config");

        // Verify /proc/self/maps file.
        let maps = tree.lookup("/proc/self/maps").unwrap();
        assert_eq!(maps.name, "maps");

        // Non-existent path should fail.
        assert!(tree.lookup("/nonexistent").is_err());
    }

    /// Verify epoll instance creation and event registration.
    #[test]
    fn test_epoll_instance_operations() {
        use smallaios_posix::epoll::{epoll_create1, epoll_ctl, EpollEvent, EpollFlags, EpollOp};

        // Create an epoll instance.
        let mut inst = epoll_create1(0).unwrap();
        assert_eq!(inst.count(), 0);

        // Register three file descriptors.
        let ev_in = EpollEvent::new(EpollFlags::EPOLLIN, 10);
        epoll_ctl(&mut inst, EpollOp::Add, 3, &ev_in).unwrap();

        let ev_out = EpollEvent::new(EpollFlags::EPOLLOUT, 20);
        epoll_ctl(&mut inst, EpollOp::Add, 4, &ev_out).unwrap();

        let ev_both = EpollEvent::new(EpollFlags::EPOLLIN | EpollFlags::EPOLLOUT, 30);
        epoll_ctl(&mut inst, EpollOp::Add, 5, &ev_both).unwrap();

        assert_eq!(inst.count(), 3);

        // Modify an entry.
        let ev_mod = EpollEvent::new(EpollFlags::EPOLLERR, 40);
        epoll_ctl(&mut inst, EpollOp::Mod, 3, &ev_mod).unwrap();
        assert_eq!(inst.count(), 3); // Count unchanged.

        // Delete an entry.
        let dummy = EpollEvent::new(0, 0);
        epoll_ctl(&mut inst, EpollOp::Del, 4, &dummy).unwrap();
        assert_eq!(inst.count(), 2);

        // Delete non-existent returns error.
        assert!(epoll_ctl(&mut inst, EpollOp::Del, 99, &dummy).is_err());
    }

    // =========================================================================
    // 7. GPU Provider Integration (2 tests)
    // =========================================================================

    /// Verify GPU identification feeds into the CUDA provider.
    #[cfg(feature = "nvidia_gpu")]
    #[test]
    fn test_gpu_identify_and_provider() {
        use smallaios_arch_nvidia::cuda_provider::{CudaProvider, ProviderStatus};
        use smallaios_arch_nvidia::gpu_id::{identify_gpu, GpuArchitecture};

        // Identify an A100 GPU (device ID 0x20B0).
        let gpu_info = identify_gpu(0x20B0).unwrap();
        assert_eq!(gpu_info.architecture, GpuArchitecture::Ampere);
        assert_eq!(gpu_info.name, "NVIDIA A100-SXM4-40GB");
        assert_eq!(gpu_info.sm_count, 108);
        assert_eq!(gpu_info.vram_size_mb, 40960);

        // Create a CUDA provider from the identified GPU.
        let vram_bytes = (gpu_info.vram_size_mb as u64) * 1024 * 1024;
        let provider = CudaProvider::new(gpu_info, vram_bytes);
        assert_eq!(*provider.status(), ProviderStatus::Ready);

        // Also test H100 identification.
        let h100 = identify_gpu(0x2330).unwrap();
        assert_eq!(h100.architecture, GpuArchitecture::Hopper);
        assert_eq!(h100.name, "NVIDIA H100-SXM5-80GB");

        // Unknown device ID should return an error.
        assert!(identify_gpu(0x0001).is_err());
    }

    /// Verify PTX kernel registry covers all kernel types and precisions.
    #[cfg(feature = "nvidia_gpu")]
    #[test]
    fn test_ptx_registry_compatibility() {
        use smallaios_arch_nvidia::gpu_id::ComputeCapability;
        use smallaios_arch_nvidia::ptx::{DataPrecision, PtxKernelType, PtxRegistry};

        let mut registry = PtxRegistry::new();
        registry.register_defaults();
        assert_eq!(registry.kernel_count(), 11);

        // Find a GEMM kernel for Ampere (CC 8.0) with F32 precision.
        let cc_ampere = ComputeCapability::new(8, 0);
        let gemm = registry.find_kernel(PtxKernelType::Gemm, DataPrecision::F32, &cc_ampere);
        assert!(gemm.is_some(), "Should find F32 GEMM kernel for Ampere");

        // Find a Softmax kernel with F16 precision.
        let softmax = registry.find_kernel(PtxKernelType::Softmax, DataPrecision::F16, &cc_ampere);
        assert!(
            softmax.is_some(),
            "Should find F16 Softmax kernel for Ampere"
        );

        // Get all compatible kernels for Ampere.
        let compatible = registry.compatible_kernels(&cc_ampere);
        assert!(
            !compatible.is_empty(),
            "Should have compatible kernels for Ampere"
        );

        // Verify Hopper (CC 9.0) compatibility.
        let cc_hopper = ComputeCapability::new(9, 0);
        let hopper_kernels = registry.compatible_kernels(&cc_hopper);
        assert!(
            !hopper_kernels.is_empty(),
            "Should have compatible kernels for Hopper"
        );
    }

    // =========================================================================
    // 8. Safety Verification Integration (3 tests)
    // =========================================================================

    /// Verify a full traceability chain: requirement -> spec -> impl -> test.
    #[test]
    fn test_traceability_full_chain() {
        use smallaios_kernel::safety::traceability::{
            ArtifactType, DalLevel, TraceMatrix, TraceStatus,
        };

        let mut matrix = TraceMatrix::new();

        // Add a requirement.
        matrix
            .add_artifact(
                "REQ-001",
                ArtifactType::Requirement,
                "Buddy allocator shall manage memory",
                DalLevel::A,
            )
            .unwrap();

        // Add a specification.
        matrix
            .add_artifact(
                "SPEC-001",
                ArtifactType::Specification,
                "Buddy allocator design",
                DalLevel::A,
            )
            .unwrap();

        // Add an implementation.
        matrix
            .add_artifact(
                "IMPL-001",
                ArtifactType::Implementation,
                "buddy_alloc.rs",
                DalLevel::A,
            )
            .unwrap();

        // Add a test.
        matrix
            .add_artifact(
                "TEST-001",
                ArtifactType::Test,
                "test_buddy_alloc",
                DalLevel::A,
            )
            .unwrap();

        assert_eq!(matrix.artifact_count(), 4);

        // Before linking: requirement is Untraced.
        assert_eq!(matrix.trace_status("REQ-001"), TraceStatus::Untraced);

        // Link requirement -> specification.
        matrix.add_link("REQ-001", "SPEC-001").unwrap();
        assert_eq!(matrix.trace_status("REQ-001"), TraceStatus::Partial);

        // Link requirement -> implementation.
        matrix.add_link("REQ-001", "IMPL-001").unwrap();
        assert_eq!(matrix.trace_status("REQ-001"), TraceStatus::Partial);

        // Link requirement -> test (completes the chain).
        matrix.add_link("REQ-001", "TEST-001").unwrap();
        assert_eq!(matrix.trace_status("REQ-001"), TraceStatus::Traced);

        // Test linked to requirement is Traced.
        assert_eq!(matrix.trace_status("TEST-001"), TraceStatus::Traced);

        // Coverage report should show 100% (1 out of 1 requirement traced).
        let report = matrix.coverage_report();
        assert_eq!(report.total_reqs, 1);
        assert_eq!(report.traced, 1);
        assert_eq!(report.partial, 0);
        assert_eq!(report.untraced, 0);
        assert_eq!(report.orphan_tests, 0);
        assert_eq!(report.coverage_pct, 100);
    }

    /// Verify coding standard compliance: advisory violations do not affect compliance.
    #[test]
    fn test_coding_standard_compliance() {
        use smallaios_kernel::safety::coding_standard::CodingStandard;

        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        assert_eq!(cs.rule_count(), 10);
        assert_eq!(cs.required_rule_count(), 7);

        // No violations: should be compliant.
        assert!(cs.is_compliant());
        assert_eq!(cs.violation_count(), 0);

        // Add an advisory violation (MR-3.2 is Advisory).
        cs.add_violation("MR-3.2", "graph.rs", 42, "cognitive complexity > 25");
        assert_eq!(cs.violation_count(), 1);

        // Advisory violations do not affect compliance.
        assert!(cs.is_compliant());

        // Add a Required violation (MR-1.1 is Required).
        cs.add_violation("MR-1.1", "main.rs", 10, "unreachable code");
        assert_eq!(cs.violation_count(), 2);

        // Required violation makes it non-compliant.
        assert!(!cs.is_compliant());

        // Verify violations for a specific rule.
        let v = cs.violations_for_rule("MR-1.1");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].file, "main.rs");
    }

    /// Verify the verification registry with default targets and status updates.
    #[test]
    fn test_verification_registry_defaults() {
        use smallaios_kernel::safety::verification::{
            VerificationRegistry, VerificationStatus, VerificationTool,
        };

        let mut reg = VerificationRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.target_count(), 6);

        // Default targets start as NotStarted.
        assert!(!reg.all_passed());

        // Verify all 6 default targets exist.
        assert!(reg.get_target("buddy_allocator").is_some());
        assert!(reg.get_target("scheduler_fairness").is_some());
        assert!(reg.get_target("tcp_state_machine").is_some());
        assert!(reg.get_target("pubsub_routing").is_some());
        assert!(reg.get_target("capability_nonforgery").is_some());
        assert!(reg.get_target("syscall_fuzz").is_some());

        // Verify tools.
        assert_eq!(
            reg.get_target("buddy_allocator").unwrap().tool,
            VerificationTool::TlaPlus
        );
        assert_eq!(
            reg.get_target("tcp_state_machine").unwrap().tool,
            VerificationTool::Spin
        );
        assert_eq!(
            reg.get_target("capability_nonforgery").unwrap().tool,
            VerificationTool::Lean4
        );
        assert_eq!(
            reg.get_target("syscall_fuzz").unwrap().tool,
            VerificationTool::Fuzz
        );

        // Update some targets to Passed.
        reg.update_status("buddy_allocator", VerificationStatus::Passed, 15, 15)
            .unwrap();
        reg.update_status("scheduler_fairness", VerificationStatus::Passed, 8, 8)
            .unwrap();
        reg.update_status("tcp_state_machine", VerificationStatus::Passed, 22, 22)
            .unwrap();
        reg.update_status("pubsub_routing", VerificationStatus::Passed, 10, 10)
            .unwrap();
        reg.update_status("capability_nonforgery", VerificationStatus::Passed, 5, 5)
            .unwrap();
        reg.update_status("syscall_fuzz", VerificationStatus::Passed, 1000, 1000)
            .unwrap();

        // Now all should be passed.
        assert!(reg.all_passed());
        assert_eq!(reg.passed_count(), 6);

        // Summary should reflect all passed.
        let summary = reg.summary();
        assert_eq!(summary.total, 6);
        assert_eq!(summary.passed, 6);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.in_progress, 0);
        assert_eq!(summary.not_started, 0);
    }

    // =========================================================================
    // 9. Container Lifecycle Integration (2 tests)
    // =========================================================================

    /// Verify the full container lifecycle from boot through graceful shutdown.
    #[test]
    fn test_full_lifecycle_boot_to_shutdown() {
        use crate::boot::BootSequence;
        use crate::config::ContainerConfig;
        use crate::shutdown::{ShutdownManager, ShutdownPhase, ShutdownReason};

        // Phase 1: Boot.
        let config = ContainerConfig::default();
        let mut boot = BootSequence::new(config);
        let statuses = boot.run_all().unwrap();
        assert_eq!(statuses.len(), 8);
        assert!(boot.is_ready());

        // Phase 2: Register shutdown hooks.
        let mut shutdown = ShutdownManager::new();
        assert_eq!(*shutdown.phase(), ShutdownPhase::Running);
        assert!(!shutdown.is_shutting_down());

        shutdown.register_hook("flush_logs", 10).unwrap();
        shutdown.register_hook("save_checkpoint", 50).unwrap();
        shutdown.register_hook("close_connections", 100).unwrap();
        assert_eq!(shutdown.hook_count(), 3);

        // Phase 3: Execute hooks (priority order: highest first).
        let executed = shutdown.execute_hooks();
        assert_eq!(executed.len(), 3);
        assert_eq!(executed[0], "close_connections"); // priority 100
        assert_eq!(executed[1], "save_checkpoint"); // priority 50
        assert_eq!(executed[2], "flush_logs"); // priority 10

        // Phase 4: Full shutdown.
        let result = shutdown.run_full_shutdown(ShutdownReason::Sigterm);
        assert_eq!(result, Ok(ShutdownPhase::Terminated));
        assert!(shutdown.is_shutting_down());
        assert_eq!(*shutdown.phase(), ShutdownPhase::Terminated);

        // Cannot initiate again.
        let err = shutdown.initiate(ShutdownReason::Sigint);
        assert!(err.is_err());
    }

    /// Verify OCI image builder with multi-architecture manifests.
    #[test]
    fn test_oci_image_multi_arch() {
        use crate::image::{Architecture, ImageBuilder, ImageManifest};

        let mut builder = ImageBuilder::new("smallaios/inference-server", "v1.0.0");

        // Empty builder should fail validation.
        assert!(builder.validate().is_err());

        // Add AMD64 manifest with layers.
        let mut amd64 = ImageManifest::new(Architecture::Amd64);
        amd64.add_layer(
            "sha256:amd64_kernel",
            5_000_000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        amd64.add_layer(
            "sha256:amd64_runtime",
            10_000_000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        amd64.add_annotation("org.opencontainers.image.version", "1.0.0");
        assert_eq!(amd64.layer_count(), 2);
        assert_eq!(amd64.total_size_bytes(), 15_000_000);
        assert!(amd64.is_under_size_limit(20)); // Under 20 MB.

        // Add ARM64 manifest.
        let mut arm64 = ImageManifest::new(Architecture::Arm64);
        arm64.add_layer(
            "sha256:arm64_kernel",
            4_000_000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        arm64.add_layer(
            "sha256:arm64_runtime",
            8_000_000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        assert_eq!(arm64.layer_count(), 2);

        // Add RISC-V manifest.
        let mut riscv = ImageManifest::new(Architecture::RiscV64);
        riscv.add_layer(
            "sha256:riscv_kernel",
            3_000_000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );

        builder.add_manifest(amd64);
        builder.add_manifest(arm64);
        builder.add_manifest(riscv);

        // Should now be multi-arch.
        assert!(builder.is_multi_arch());
        assert_eq!(builder.manifest_count(), 3);

        // Total size across all architectures.
        assert_eq!(builder.total_size(), 15_000_000 + 12_000_000 + 3_000_000);

        // Validation should pass.
        assert!(builder.validate().is_ok());

        // Verify JSON output of the AMD64 manifest.
        let json = builder.manifests[0].to_json();
        assert!(json.contains("\"schemaVersion\":2"));
        assert!(json.contains("\"architecture\":\"amd64\""));
        assert!(json.contains("\"os\":\"smallaios\""));
        assert!(json.contains("\"digest\":\"sha256:amd64_kernel\""));
        assert!(json.contains("\"annotations\":{"));
        assert!(json.contains("\"org.opencontainers.image.version\":\"1.0.0\""));
    }
}
