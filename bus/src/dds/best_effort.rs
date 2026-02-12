// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RTPS best-effort delivery with sequence number tracking.
//!
//! Best-effort delivery does not use heartbeat/acknack retransmission.
//! Instead, the reader tracks the highest received sequence number and
//! detects gaps (lost samples). Out-of-order samples below the highest
//! received sequence number are dropped.

use crate::dds::endpoint::SequenceNumber;
use crate::dds::types::Guid;
use alloc::vec::Vec;

/// Best-effort writer state.
///
/// Tracks the next sequence number to assign. Does not retain samples
/// for retransmission since best-effort delivery offers no guarantees.
#[derive(Debug)]
pub struct BestEffortWriter {
    /// Writer GUID.
    writer_guid: Guid,
    /// Next sequence number to assign.
    next_sn: SequenceNumber,
    /// Number of samples written.
    samples_written: u64,
}

impl BestEffortWriter {
    /// Create a new best-effort writer.
    pub fn new(writer_guid: Guid) -> Self {
        Self {
            writer_guid,
            next_sn: 1,
            samples_written: 0,
        }
    }

    /// Returns the writer GUID.
    pub fn writer_guid(&self) -> &Guid {
        &self.writer_guid
    }

    /// Returns the next sequence number that will be assigned.
    pub fn next_sn(&self) -> SequenceNumber {
        self.next_sn
    }

    /// Returns the total number of samples written.
    pub fn samples_written(&self) -> u64 {
        self.samples_written
    }

    /// Write a sample, returning the assigned sequence number.
    ///
    /// The caller is responsible for transmitting the sample with this
    /// sequence number. No retransmission will occur.
    pub fn write(&mut self, _data: &[u8]) -> SequenceNumber {
        let sn = self.next_sn;
        self.next_sn += 1;
        self.samples_written += 1;
        sn
    }
}

/// Result of receiving a best-effort sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BestEffortReceiveResult {
    /// Sample accepted (new, in-order or fills a gap above highest_received).
    Accepted,
    /// Sample was a duplicate (already received).
    Duplicate,
    /// Sample was too old (sequence number below the acceptance window).
    TooOld,
}

/// Best-effort reader state.
///
/// Tracks the highest received sequence number. Out-of-order samples
/// with sequence numbers at or below `highest_received` are dropped.
/// Gaps are tracked as lost samples for statistics.
#[derive(Debug)]
pub struct BestEffortReader {
    /// Reader GUID.
    reader_guid: Guid,
    /// Highest sequence number received.
    highest_received: SequenceNumber,
    /// Number of samples accepted.
    samples_accepted: u64,
    /// Number of gaps detected (lost samples).
    gaps_detected: u64,
    /// Number of duplicate/old samples rejected.
    samples_rejected: u64,
    /// Sample buffer for delivered data.
    samples: Vec<Vec<u8>>,
    /// Maximum sample buffer size.
    max_samples: usize,
}

impl BestEffortReader {
    /// Create a new best-effort reader.
    pub fn new(reader_guid: Guid, max_samples: usize) -> Self {
        Self {
            reader_guid,
            highest_received: 0,
            samples_accepted: 0,
            gaps_detected: 0,
            samples_rejected: 0,
            samples: Vec::new(),
            max_samples,
        }
    }

    /// Returns the reader GUID.
    pub fn reader_guid(&self) -> &Guid {
        &self.reader_guid
    }

    /// Returns the highest received sequence number.
    pub fn highest_received(&self) -> SequenceNumber {
        self.highest_received
    }

    /// Returns the number of accepted samples.
    pub fn samples_accepted(&self) -> u64 {
        self.samples_accepted
    }

    /// Returns the number of detected gaps (lost samples).
    pub fn gaps_detected(&self) -> u64 {
        self.gaps_detected
    }

    /// Returns the number of rejected samples.
    pub fn samples_rejected(&self) -> u64 {
        self.samples_rejected
    }

    /// Receive a sample with the given sequence number.
    ///
    /// Best-effort semantics: accept only if the sequence number is
    /// strictly greater than the highest previously received. Track
    /// any gap as lost samples.
    pub fn receive(&mut self, sn: SequenceNumber, data: &[u8]) -> BestEffortReceiveResult {
        if sn == 0 {
            self.samples_rejected += 1;
            return BestEffortReceiveResult::TooOld;
        }

        if sn <= self.highest_received {
            self.samples_rejected += 1;
            if sn == self.highest_received {
                return BestEffortReceiveResult::Duplicate;
            }
            return BestEffortReceiveResult::TooOld;
        }

        // Detect gap: if sn > highest_received + 1, samples were lost
        if self.highest_received > 0 && sn > self.highest_received + 1 {
            let lost = sn - self.highest_received - 1;
            self.gaps_detected += lost;
        }

        self.highest_received = sn;
        self.samples_accepted += 1;

        // Store the sample
        self.samples.push(Vec::from(data));
        if self.samples.len() > self.max_samples {
            self.samples.remove(0);
        }

        BestEffortReceiveResult::Accepted
    }

    /// Take all buffered samples (removes them).
    pub fn take(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.samples)
    }

    /// Read buffered samples (does not remove them).
    pub fn read(&self) -> &[Vec<u8>] {
        &self.samples
    }

    /// Returns the number of buffered samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dds::types::*;

    fn test_guid(id: u8) -> Guid {
        Guid::new(GuidPrefix::new([id; 12]), EntityId::user([0, 0, id], 0x02))
    }

    // === Writer tests ===

    #[test]
    fn test_writer_new() {
        let w = BestEffortWriter::new(test_guid(1));
        assert_eq!(w.next_sn(), 1);
        assert_eq!(w.samples_written(), 0);
    }

    #[test]
    fn test_writer_write_sequence() {
        let mut w = BestEffortWriter::new(test_guid(1));
        assert_eq!(w.write(&[0xAA]), 1);
        assert_eq!(w.write(&[0xBB]), 2);
        assert_eq!(w.write(&[0xCC]), 3);
        assert_eq!(w.next_sn(), 4);
        assert_eq!(w.samples_written(), 3);
    }

    #[test]
    fn test_writer_guid() {
        let guid = test_guid(5);
        let w = BestEffortWriter::new(guid);
        assert_eq!(*w.writer_guid(), guid);
    }

    // === Reader tests ===

    #[test]
    fn test_reader_new() {
        let r = BestEffortReader::new(test_guid(2), 100);
        assert_eq!(r.highest_received(), 0);
        assert_eq!(r.samples_accepted(), 0);
        assert_eq!(r.gaps_detected(), 0);
        assert_eq!(r.samples_rejected(), 0);
        assert_eq!(r.sample_count(), 0);
    }

    #[test]
    fn test_reader_receive_sequential() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        assert_eq!(r.receive(1, &[0x01]), BestEffortReceiveResult::Accepted);
        assert_eq!(r.receive(2, &[0x02]), BestEffortReceiveResult::Accepted);
        assert_eq!(r.receive(3, &[0x03]), BestEffortReceiveResult::Accepted);
        assert_eq!(r.highest_received(), 3);
        assert_eq!(r.samples_accepted(), 3);
        assert_eq!(r.gaps_detected(), 0);
    }

    #[test]
    fn test_reader_receive_with_gap() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        r.receive(1, &[0x01]);
        // Skip 2, receive 3 — gap of 1
        assert_eq!(r.receive(3, &[0x03]), BestEffortReceiveResult::Accepted);
        assert_eq!(r.gaps_detected(), 1);
        assert_eq!(r.highest_received(), 3);
    }

    #[test]
    fn test_reader_receive_large_gap() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        r.receive(1, &[0x01]);
        // Skip 2..10, receive 11 — gap of 9
        assert_eq!(r.receive(11, &[0x0B]), BestEffortReceiveResult::Accepted);
        assert_eq!(r.gaps_detected(), 9);
    }

    #[test]
    fn test_reader_receive_duplicate() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        r.receive(1, &[0x01]);
        assert_eq!(r.receive(1, &[0x01]), BestEffortReceiveResult::Duplicate);
        assert_eq!(r.samples_rejected(), 1);
        assert_eq!(r.samples_accepted(), 1);
    }

    #[test]
    fn test_reader_receive_too_old() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        r.receive(1, &[0x01]);
        r.receive(5, &[0x05]);
        // Receive SN 3 after SN 5 — too old
        assert_eq!(r.receive(3, &[0x03]), BestEffortReceiveResult::TooOld);
        assert_eq!(r.samples_rejected(), 1);
    }

    #[test]
    fn test_reader_receive_sn_zero() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        assert_eq!(r.receive(0, &[0x00]), BestEffortReceiveResult::TooOld);
    }

    #[test]
    fn test_reader_take_samples() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        r.receive(1, &[0x01]);
        r.receive(2, &[0x02]);
        let samples = r.take();
        assert_eq!(samples.len(), 2);
        assert_eq!(r.sample_count(), 0);
    }

    #[test]
    fn test_reader_read_samples() {
        let mut r = BestEffortReader::new(test_guid(2), 100);
        r.receive(1, &[0xAA]);
        r.receive(2, &[0xBB]);
        let samples = r.read();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], &[0xAA]);
        assert_eq!(samples[1], &[0xBB]);
    }

    #[test]
    fn test_reader_max_samples_eviction() {
        let mut r = BestEffortReader::new(test_guid(2), 3);
        for i in 1..=5u64 {
            r.receive(i, &[(i & 0xFF) as u8]);
        }
        assert_eq!(r.sample_count(), 3);
        // Should have samples 3, 4, 5
        assert_eq!(r.read()[0], &[3]);
        assert_eq!(r.read()[2], &[5]);
    }

    // === Writer + Reader integration ===

    #[test]
    fn test_writer_reader_integration() {
        let mut w = BestEffortWriter::new(test_guid(1));
        let mut r = BestEffortReader::new(test_guid(2), 100);

        let data = [0xAA, 0xBB];
        let sn1 = w.write(&data);
        assert_eq!(r.receive(sn1, &data), BestEffortReceiveResult::Accepted);

        let sn2 = w.write(&[0xCC]);
        // Simulate loss of sn2
        let _ = sn2;

        let sn3 = w.write(&[0xDD]);
        assert_eq!(r.receive(sn3, &[0xDD]), BestEffortReceiveResult::Accepted);

        assert_eq!(r.samples_accepted(), 2);
        assert_eq!(r.gaps_detected(), 1); // sn2 was lost
    }

    #[test]
    fn test_reader_guid() {
        let guid = test_guid(7);
        let r = BestEffortReader::new(guid, 10);
        assert_eq!(*r.reader_guid(), guid);
    }
}
