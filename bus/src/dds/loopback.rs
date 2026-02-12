// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS pub/sub loopback integration test.
//!
//! Tests end-to-end DDS data flow within SmallAIOS: DataWriter publishes
//! CDR-serialized samples to a topic, DataReader receives them via matching,
//! and RTPS message framing validates the wire format.

#[cfg(test)]
mod tests {
    use crate::dds::adapter::DdsZenohAdapter;
    use crate::dds::best_effort::{BestEffortReader, BestEffortWriter};
    use crate::dds::cdr::{CdrDeserializer, CdrSerializer, Endianness};
    use crate::dds::discovery::{DiscoveredEndpoint, EndpointKind, SedpDatabase};
    use crate::dds::endpoint::{check_match, DataReader, DataWriter, MatchStatus};
    use crate::dds::qos::*;
    use crate::dds::rtps::{RtpsHeader, RtpsMessage, Submessage, SubmessageKind};
    use crate::dds::topic::Topic;
    use crate::dds::types::*;
    use crate::transport::ZenohTransport;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    fn make_guid(prefix: u8, entity: u8) -> Guid {
        Guid::new(
            GuidPrefix::new([prefix; 12]),
            EntityId::user([0, 0, entity], 0x02),
        )
    }

    fn make_topic(name: &str) -> Topic {
        Topic::new(0, name, "SensorData").unwrap()
    }

    // -----------------------------------------------------------------------
    // Loopback: Writer -> CDR -> Reader
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_writer_to_reader() {
        let topic = make_topic("SensorTopic");
        let qos = QosPolicy {
            history: HistoryQos {
                kind: HistoryKind::KeepAll,
                depth: 0,
            },
            ..Default::default()
        };

        let mut writer = DataWriter::new(make_guid(1, 1), topic.clone(), qos);
        let mut reader = DataReader::new(make_guid(2, 1), topic, qos);

        // Verify they match
        assert_eq!(check_match(&writer, &reader), MatchStatus::Matched);

        // Write 3 samples through the writer
        let samples = [
            vec![0x01, 0x02, 0x03, 0x04],
            vec![0x11, 0x12, 0x13, 0x14],
            vec![0x21, 0x22, 0x23, 0x24],
        ];

        for sample in &samples {
            let seq = writer.write(sample).unwrap();
            reader.receive(sample, seq);
        }

        // Verify all samples received
        assert_eq!(reader.sample_count(), 3);
        let received = reader.take();
        assert_eq!(received[0], samples[0]);
        assert_eq!(received[1], samples[1]);
        assert_eq!(received[2], samples[2]);
    }

    // -----------------------------------------------------------------------
    // Loopback: CDR serialize -> deserialize
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_cdr_roundtrip() {
        // Simulate a SensorData struct: timestamp(u64) + value(f32) + name(string)
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        let timestamp: u64 = 1707500000;
        let value: f32 = 23.5;
        let name = "temperature_1";

        ser.serialize_u64(timestamp);
        ser.serialize_f32(value);
        ser.serialize_string(name);

        let encoded = ser.into_bytes();

        // Deserialize
        let mut de = CdrDeserializer::new(&encoded, Endianness::LittleEndian);
        let ts = de.deserialize_u64().unwrap();
        let val = de.deserialize_f32().unwrap();
        let n = de.deserialize_string().unwrap();

        assert_eq!(ts, timestamp);
        assert_eq!(val, value);
        assert_eq!(n, name);
    }

    // -----------------------------------------------------------------------
    // Loopback: Writer -> CDR serialize -> RTPS frame -> CDR deserialize -> Reader
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_full_pipeline() {
        let topic = make_topic("FullPipeline");
        let qos = QosPolicy::default();

        let mut writer = DataWriter::new(make_guid(1, 1), topic.clone(), qos);
        let mut reader = DataReader::new(make_guid(2, 1), topic, qos);

        // Step 1: CDR serialize payload
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u32(42);
        ser.serialize_f32(3.14);
        let cdr_payload = ser.into_bytes();

        // Step 2: Writer publishes
        let seq = writer.write(&cdr_payload).unwrap();

        // Step 3: Create RTPS DATA submessage
        let header = RtpsHeader::new(GuidPrefix::new([1; 12]));
        let mut msg = RtpsMessage::new(header);
        msg.add_submessage(Submessage::new(SubmessageKind::Data, 0x05, &cdr_payload));

        // Step 4: Encode RTPS message
        let mut wire_buf = [0u8; 1024];
        let wire_len = msg.encode(&mut wire_buf).unwrap();
        assert!(wire_len > 0);

        // Step 5: Decode RTPS message
        let decoded_msg = RtpsMessage::decode(&wire_buf[..wire_len]).unwrap();
        assert_eq!(decoded_msg.submessages.len(), 1);
        let submsg = &decoded_msg.submessages[0];
        assert_eq!(submsg.kind, SubmessageKind::Data);

        // Step 6: Reader receives
        reader.receive(&submsg.payload, seq);

        // Step 7: CDR deserialize
        let samples = reader.take();
        assert_eq!(samples.len(), 1);
        let mut de = CdrDeserializer::new(&samples[0], Endianness::LittleEndian);
        assert_eq!(de.deserialize_u32().unwrap(), 42);
        assert_eq!(de.deserialize_f32().unwrap(), 3.14);
    }

    // -----------------------------------------------------------------------
    // Loopback: Best-effort writer/reader
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_best_effort() {
        let writer_guid = make_guid(1, 1);
        let reader_guid = make_guid(2, 1);

        let mut writer = BestEffortWriter::new(writer_guid);
        let mut reader = BestEffortReader::new(reader_guid, 32);

        // Write samples
        for i in 0..5u8 {
            let data = [i, i + 10, i + 20];
            let sn = writer.write(&data);
            reader.receive(sn, &data);
        }

        // Read all samples
        let samples = reader.take();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], vec![0, 10, 20]);
        assert_eq!(samples[4], vec![4, 14, 24]);
    }

    // -----------------------------------------------------------------------
    // Loopback: Discovery + match
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_discovery_match() {
        let p1_prefix = GuidPrefix::new([1; 12]);
        let p2_prefix = GuidPrefix::new([2; 12]);

        let mut sedp = SedpDatabase::new(0);

        // Register writer endpoint on participant 1
        let writer_ep = DiscoveredEndpoint {
            guid: make_guid(1, 1),
            participant_prefix: p1_prefix,
            kind: EndpointKind::Writer,
            topic_name: String::from("SensorTopic"),
            type_name: String::from("SensorData"),
            qos: QosPolicy::default(),
        };
        sedp.add_endpoint(writer_ep).unwrap();

        // Register reader endpoint on participant 2
        let reader_ep = DiscoveredEndpoint {
            guid: make_guid(2, 1),
            participant_prefix: p2_prefix,
            kind: EndpointKind::Reader,
            topic_name: String::from("SensorTopic"),
            type_name: String::from("SensorData"),
            qos: QosPolicy::default(),
        };
        sedp.add_endpoint(reader_ep).unwrap();

        // Match endpoints
        let matches = sedp.find_matches();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, make_guid(1, 1)); // writer
        assert_eq!(matches[0].1, make_guid(2, 1)); // reader
    }

    // -----------------------------------------------------------------------
    // Loopback: Zenoh adapter
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_zenoh_adapter() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();

        // Transmit a sample through the Zenoh adapter
        let data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        adapter.transmit("dds/0/SensorTopic", &data).unwrap();

        // Inject a received sample
        adapter.inject_rx("SensorTopic", &data);

        // Receive it
        let mut buf = [0u8; 512];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert!(sample.key_expr.contains("SensorTopic"));
        assert_eq!(sample.payload, &data);
    }

    // -----------------------------------------------------------------------
    // Loopback: Multi-topic multiplexing
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_multi_topic() {
        let topics = ["Altitude", "Speed", "Heading"];
        let mut writers: Vec<DataWriter> = Vec::new();
        let mut readers: Vec<DataReader> = Vec::new();

        for (i, &name) in topics.iter().enumerate() {
            let topic = make_topic(name);
            let qos = QosPolicy::default();
            writers.push(DataWriter::new(make_guid(1, i as u8), topic.clone(), qos));
            readers.push(DataReader::new(make_guid(2, i as u8), topic, qos));
        }

        // Verify each writer/reader pair matches
        for i in 0..3 {
            assert_eq!(check_match(&writers[i], &readers[i]), MatchStatus::Matched);
        }

        // Cross-topic should not match
        assert_eq!(
            check_match(&writers[0], &readers[1]),
            MatchStatus::NotMatched
        );

        // Publish to each topic
        for (i, writer) in writers.iter_mut().enumerate() {
            let data = vec![i as u8; 4];
            let seq = writer.write(&data).unwrap();
            readers[i].receive(&data, seq);
        }

        // Verify isolation: each reader got only its own topic's data
        for (i, reader) in readers.iter_mut().enumerate() {
            let samples = reader.take();
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0], vec![i as u8; 4]);
        }
    }

    // -----------------------------------------------------------------------
    // Loopback: Transient-local durability for late joiner
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_late_joiner() {
        let topic = make_topic("DurableTopic");
        let qos = QosPolicy {
            durability: DurabilityKind::TransientLocal,
            history: HistoryQos {
                kind: HistoryKind::KeepLast,
                depth: 5,
            },
            ..Default::default()
        };

        let mut writer = DataWriter::new(make_guid(1, 1), topic.clone(), qos);

        // Write 10 samples before reader joins
        for i in 0..10u8 {
            writer.write(&[i]).unwrap();
        }

        // Late-joining reader: gets retained samples
        let mut reader = DataReader::new(make_guid(2, 1), topic, qos);

        // Transfer retained samples to reader (simulating discovery)
        for sample in writer.retained_samples() {
            reader.receive(sample, 0);
        }

        // Reader should have the last 5 samples (depth=5)
        assert_eq!(reader.sample_count(), 5);
        let samples = reader.take();
        assert_eq!(samples[0], vec![5]);
        assert_eq!(samples[4], vec![9]);
    }

    // -----------------------------------------------------------------------
    // Loopback: QoS enforcement
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_qos_history_enforcement() {
        let qos = QosPolicy {
            history: HistoryQos {
                kind: HistoryKind::KeepLast,
                depth: 3,
            },
            ..Default::default()
        };

        let mut reader = DataReader::new(make_guid(2, 1), make_topic("Q"), qos);

        // Write 10 samples
        for i in 0..10u8 {
            reader.receive(&[i], i as u64 + 1);
        }

        // Should only keep last 3
        assert_eq!(reader.sample_count(), 3);
        let samples = reader.read();
        assert_eq!(samples[0], vec![7]);
        assert_eq!(samples[1], vec![8]);
        assert_eq!(samples[2], vec![9]);
    }
}
