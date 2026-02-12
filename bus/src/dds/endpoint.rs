// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS DataWriter and DataReader endpoint abstractions.
//!
//! DataWriters publish samples; DataReaders subscribe to samples.
//! Matching is based on Topic name, type consistency, and QoS compatibility.

use crate::dds::qos::QosPolicy;
use crate::dds::topic::Topic;
use crate::dds::types::Guid;
use crate::BusError;
use alloc::vec::Vec;

/// Sequence number for RTPS samples.
pub type SequenceNumber = u64;

/// Match status between a writer and reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStatus {
    /// Endpoints are matched and exchanging data.
    Matched,
    /// Type mismatch — same topic name, different type.
    TypeMismatch,
    /// QoS incompatible.
    QosIncompatible,
    /// Not matched (different topic or domain).
    NotMatched,
}

/// DDS DataWriter.
#[derive(Debug)]
pub struct DataWriter {
    /// Writer GUID.
    guid: Guid,
    /// Topic this writer publishes to.
    topic: Topic,
    /// QoS policy.
    qos: QosPolicy,
    /// Next sequence number.
    next_seq: SequenceNumber,
    /// Number of matched readers.
    matched_count: u32,
    /// Samples retained for durability (transient-local).
    retained_samples: Vec<Vec<u8>>,
}

impl DataWriter {
    /// Create a new DataWriter.
    pub fn new(guid: Guid, topic: Topic, qos: QosPolicy) -> Self {
        Self {
            guid,
            topic,
            qos,
            next_seq: 1,
            matched_count: 0,
            retained_samples: Vec::new(),
        }
    }

    /// Returns the writer GUID.
    pub fn guid(&self) -> &Guid {
        &self.guid
    }

    /// Returns the topic.
    pub fn topic(&self) -> &Topic {
        &self.topic
    }

    /// Returns the QoS policy.
    pub fn qos(&self) -> &QosPolicy {
        &self.qos
    }

    /// Returns the number of matched readers.
    pub fn matched_count(&self) -> u32 {
        self.matched_count
    }

    /// Write a sample. Returns the assigned sequence number.
    pub fn write(&mut self, data: &[u8]) -> Result<SequenceNumber, BusError> {
        let seq = self.next_seq;
        self.next_seq += 1;

        // Retain for transient-local durability
        if self.qos.durability == crate::dds::qos::DurabilityKind::TransientLocal {
            self.retained_samples.push(Vec::from(data));
            // Enforce history depth
            if self.qos.history.kind == crate::dds::qos::HistoryKind::KeepLast {
                let depth = self.qos.history.depth as usize;
                if depth > 0 && self.retained_samples.len() > depth {
                    let excess = self.retained_samples.len() - depth;
                    self.retained_samples.drain(..excess);
                }
            }
        }

        Ok(seq)
    }

    /// Get retained samples (for late-joining readers).
    pub fn retained_samples(&self) -> &[Vec<u8>] {
        &self.retained_samples
    }

    /// Record a reader match.
    pub fn add_match(&mut self) {
        self.matched_count += 1;
    }

    /// Record a reader unmatch.
    pub fn remove_match(&mut self) {
        self.matched_count = self.matched_count.saturating_sub(1);
    }
}

/// DDS DataReader.
#[derive(Debug)]
pub struct DataReader {
    /// Reader GUID.
    guid: Guid,
    /// Topic this reader subscribes to.
    topic: Topic,
    /// QoS policy.
    qos: QosPolicy,
    /// Number of matched writers.
    matched_count: u32,
    /// Received sample buffer.
    samples: Vec<Vec<u8>>,
    /// Last received sequence number.
    last_seq: SequenceNumber,
}

impl DataReader {
    /// Create a new DataReader.
    pub fn new(guid: Guid, topic: Topic, qos: QosPolicy) -> Self {
        Self {
            guid,
            topic,
            qos,
            matched_count: 0,
            samples: Vec::new(),
            last_seq: 0,
        }
    }

    /// Returns the reader GUID.
    pub fn guid(&self) -> &Guid {
        &self.guid
    }

    /// Returns the topic.
    pub fn topic(&self) -> &Topic {
        &self.topic
    }

    /// Returns the QoS policy.
    pub fn qos(&self) -> &QosPolicy {
        &self.qos
    }

    /// Returns the number of matched writers.
    pub fn matched_count(&self) -> u32 {
        self.matched_count
    }

    /// Receive a sample.
    pub fn receive(&mut self, data: &[u8], seq: SequenceNumber) {
        self.last_seq = seq;
        self.samples.push(Vec::from(data));
        // Enforce history depth
        if self.qos.history.kind == crate::dds::qos::HistoryKind::KeepLast {
            let depth = self.qos.history.depth as usize;
            if depth > 0 && self.samples.len() > depth {
                let excess = self.samples.len() - depth;
                self.samples.drain(..excess);
            }
        }
    }

    /// Take all available samples (removes them from the buffer).
    pub fn take(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.samples)
    }

    /// Read available samples (does not remove them).
    pub fn read(&self) -> &[Vec<u8>] {
        &self.samples
    }

    /// Returns the number of available samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Record a writer match.
    pub fn add_match(&mut self) {
        self.matched_count += 1;
    }

    /// Record a writer unmatch.
    pub fn remove_match(&mut self) {
        self.matched_count = self.matched_count.saturating_sub(1);
    }
}

/// Check match compatibility between a writer and reader.
pub fn check_match(writer: &DataWriter, reader: &DataReader) -> MatchStatus {
    // Different domain or topic name
    if writer.topic().domain_id() != reader.topic().domain_id()
        || writer.topic().name() != reader.topic().name()
    {
        return MatchStatus::NotMatched;
    }

    // Type mismatch
    if writer.topic().type_name() != reader.topic().type_name() {
        return MatchStatus::TypeMismatch;
    }

    // QoS compatibility
    if !QosPolicy::is_compatible(writer.qos(), reader.qos()) {
        return MatchStatus::QosIncompatible;
    }

    MatchStatus::Matched
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dds::qos::*;
    use crate::dds::types::*;

    fn test_guid(id: u8) -> Guid {
        Guid::new(GuidPrefix::new([id; 12]), EntityId::user([0, 0, id], 0x02))
    }

    fn test_topic(name: &str, type_name: &str) -> Topic {
        Topic::new(0, name, type_name).unwrap()
    }

    #[test]
    fn test_data_writer_write() {
        let mut writer = DataWriter::new(
            test_guid(1),
            test_topic("Sensor", "SensorMsg"),
            QosPolicy::default(),
        );
        let seq = writer.write(&[0xAA, 0xBB]).unwrap();
        assert_eq!(seq, 1);
        let seq2 = writer.write(&[0xCC]).unwrap();
        assert_eq!(seq2, 2);
    }

    #[test]
    fn test_data_writer_transient_local_retention() {
        let qos = QosPolicy {
            durability: DurabilityKind::TransientLocal,
            history: HistoryQos {
                kind: HistoryKind::KeepLast,
                depth: 3,
            },
            ..Default::default()
        };
        let mut writer = DataWriter::new(test_guid(1), test_topic("S", "M"), qos);
        for i in 0..5u8 {
            writer.write(&[i]).unwrap();
        }
        // Should only retain last 3
        assert_eq!(writer.retained_samples().len(), 3);
        assert_eq!(writer.retained_samples()[0], &[2]);
        assert_eq!(writer.retained_samples()[2], &[4]);
    }

    #[test]
    fn test_data_reader_receive_and_take() {
        let qos = QosPolicy {
            history: HistoryQos {
                kind: HistoryKind::KeepAll,
                depth: 0,
            },
            ..Default::default()
        };
        let mut reader = DataReader::new(test_guid(2), test_topic("Sensor", "SensorMsg"), qos);
        reader.receive(&[0x01], 1);
        reader.receive(&[0x02], 2);
        assert_eq!(reader.sample_count(), 2);
        let samples = reader.take();
        assert_eq!(samples.len(), 2);
        assert_eq!(reader.sample_count(), 0);
    }

    #[test]
    fn test_data_reader_history_depth() {
        let qos = QosPolicy {
            history: HistoryQos {
                kind: HistoryKind::KeepLast,
                depth: 2,
            },
            ..Default::default()
        };
        let mut reader = DataReader::new(test_guid(2), test_topic("S", "M"), qos);
        reader.receive(&[0x01], 1);
        reader.receive(&[0x02], 2);
        reader.receive(&[0x03], 3);
        assert_eq!(reader.sample_count(), 2);
        assert_eq!(reader.read()[0], &[0x02]);
    }

    #[test]
    fn test_check_match_success() {
        let writer = DataWriter::new(
            test_guid(1),
            test_topic("Sensor", "SensorMsg"),
            QosPolicy::default(),
        );
        let reader = DataReader::new(
            test_guid(2),
            test_topic("Sensor", "SensorMsg"),
            QosPolicy::default(),
        );
        assert_eq!(check_match(&writer, &reader), MatchStatus::Matched);
    }

    #[test]
    fn test_check_match_type_mismatch() {
        let writer = DataWriter::new(
            test_guid(1),
            test_topic("Sensor", "SensorMsg"),
            QosPolicy::default(),
        );
        let reader = DataReader::new(
            test_guid(2),
            test_topic("Sensor", "OtherMsg"),
            QosPolicy::default(),
        );
        assert_eq!(check_match(&writer, &reader), MatchStatus::TypeMismatch);
    }

    #[test]
    fn test_check_match_qos_incompatible() {
        let w_qos = QosPolicy {
            reliability: ReliabilityKind::BestEffort,
            ..Default::default()
        };
        let r_qos = QosPolicy {
            reliability: ReliabilityKind::Reliable,
            ..Default::default()
        };
        let writer = DataWriter::new(test_guid(1), test_topic("S", "M"), w_qos);
        let reader = DataReader::new(test_guid(2), test_topic("S", "M"), r_qos);
        assert_eq!(check_match(&writer, &reader), MatchStatus::QosIncompatible);
    }

    #[test]
    fn test_check_match_different_topic() {
        let writer = DataWriter::new(
            test_guid(1),
            test_topic("TopicA", "M"),
            QosPolicy::default(),
        );
        let reader = DataReader::new(
            test_guid(2),
            test_topic("TopicB", "M"),
            QosPolicy::default(),
        );
        assert_eq!(check_match(&writer, &reader), MatchStatus::NotMatched);
    }

    #[test]
    fn test_match_count_tracking() {
        let mut writer = DataWriter::new(test_guid(1), test_topic("S", "M"), QosPolicy::default());
        assert_eq!(writer.matched_count(), 0);
        writer.add_match();
        assert_eq!(writer.matched_count(), 1);
        writer.remove_match();
        assert_eq!(writer.matched_count(), 0);
        // Should not underflow
        writer.remove_match();
        assert_eq!(writer.matched_count(), 0);
    }
}
