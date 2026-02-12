// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RTPS reliable delivery protocol.
//!
//! Implements writer-side heartbeat generation and reader-side acknack
//! response for guaranteed sample delivery with retransmission.

use crate::dds::endpoint::SequenceNumber;
use crate::dds::types::Guid;
use crate::BusError;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// Maximum number of samples retained for retransmission.
const MAX_WRITER_CACHE: usize = 256;

/// Writer heartbeat state — tracks which samples are available and
/// which readers have acknowledged them.
#[derive(Debug)]
pub struct WriterHeartbeat {
    /// Writer GUID.
    writer_guid: Guid,
    /// First available sequence number.
    first_sn: SequenceNumber,
    /// Last available sequence number.
    last_sn: SequenceNumber,
    /// Heartbeat count (incremented each time a heartbeat is sent).
    heartbeat_count: u32,
    /// Retained samples for retransmission: (seq_num, data).
    cache: Vec<(SequenceNumber, Vec<u8>)>,
}

impl WriterHeartbeat {
    /// Create a new writer heartbeat tracker.
    pub fn new(writer_guid: Guid) -> Self {
        Self {
            writer_guid,
            first_sn: 1,
            last_sn: 0,
            heartbeat_count: 0,
            cache: Vec::new(),
        }
    }

    /// Returns the writer GUID.
    pub fn writer_guid(&self) -> &Guid {
        &self.writer_guid
    }

    /// Returns the first available sequence number.
    pub fn first_sn(&self) -> SequenceNumber {
        self.first_sn
    }

    /// Returns the last available sequence number.
    pub fn last_sn(&self) -> SequenceNumber {
        self.last_sn
    }

    /// Returns the current heartbeat count.
    pub fn heartbeat_count(&self) -> u32 {
        self.heartbeat_count
    }

    /// Record a new sample in the writer cache.
    pub fn add_sample(&mut self, seq: SequenceNumber, data: &[u8]) {
        self.last_sn = seq;
        self.cache.push((seq, Vec::from(data)));
        // Evict oldest if over capacity
        if self.cache.len() > MAX_WRITER_CACHE {
            self.cache.remove(0);
            if let Some((first, _)) = self.cache.first() {
                self.first_sn = *first;
            }
        }
    }

    /// Generate a heartbeat. Returns (first_sn, last_sn, count).
    pub fn generate_heartbeat(&mut self) -> (SequenceNumber, SequenceNumber, u32) {
        self.heartbeat_count = self.heartbeat_count.wrapping_add(1);
        (self.first_sn, self.last_sn, self.heartbeat_count)
    }

    /// Encode a heartbeat submessage payload.
    ///
    /// Layout (28 bytes):
    /// - writer entity_id (4 bytes)
    /// - reader entity_id (4 bytes, 0 = all readers)
    /// - first_sn (8 bytes, little-endian)
    /// - last_sn (8 bytes, little-endian)
    /// - count (4 bytes, little-endian)
    pub fn encode_heartbeat(&mut self, buf: &mut [u8]) -> Result<usize, BusError> {
        const HB_LEN: usize = 28;
        if buf.len() < HB_LEN {
            return Err(BusError::BufferTooSmall);
        }
        let (first, last, count) = self.generate_heartbeat();
        // Writer entity ID
        buf[0..4].copy_from_slice(&self.writer_guid.entity_id.0);
        // Reader entity ID (0 = all)
        buf[4..8].copy_from_slice(&[0; 4]);
        // first_sn
        buf[8..16].copy_from_slice(&first.to_le_bytes());
        // last_sn
        buf[16..24].copy_from_slice(&last.to_le_bytes());
        // count
        buf[24..28].copy_from_slice(&count.to_le_bytes());
        Ok(HB_LEN)
    }

    /// Decode a heartbeat submessage payload.
    pub fn decode_heartbeat(data: &[u8]) -> Result<HeartbeatData, BusError> {
        if data.len() < 28 {
            return Err(BusError::FrameTooShort);
        }
        let mut writer_eid = [0u8; 4];
        writer_eid.copy_from_slice(&data[0..4]);
        let mut reader_eid = [0u8; 4];
        reader_eid.copy_from_slice(&data[4..8]);
        let first_sn = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let last_sn = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let count = u32::from_le_bytes(data[24..28].try_into().unwrap());
        Ok(HeartbeatData {
            writer_entity_id: writer_eid,
            reader_entity_id: reader_eid,
            first_sn,
            last_sn,
            count,
        })
    }

    /// Get a sample by sequence number for retransmission.
    pub fn get_sample(&self, seq: SequenceNumber) -> Option<&[u8]> {
        self.cache
            .iter()
            .find(|(s, _)| *s == seq)
            .map(|(_, d)| d.as_slice())
    }

    /// Remove acknowledged samples up to (but not including) the given sequence number.
    pub fn acknowledge_up_to(&mut self, ack_sn: SequenceNumber) {
        self.cache.retain(|(s, _)| *s >= ack_sn);
        if let Some((first, _)) = self.cache.first() {
            self.first_sn = *first;
        } else {
            self.first_sn = ack_sn;
        }
    }

    /// Returns the number of cached samples.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

/// Decoded heartbeat data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeartbeatData {
    /// Writer entity ID.
    pub writer_entity_id: [u8; 4],
    /// Reader entity ID (0 = all readers).
    pub reader_entity_id: [u8; 4],
    /// First available sequence number.
    pub first_sn: SequenceNumber,
    /// Last available sequence number.
    pub last_sn: SequenceNumber,
    /// Heartbeat count.
    pub count: u32,
}

/// Reader acknack state — tracks which samples have been received and
/// which are missing for requesting retransmission.
#[derive(Debug)]
pub struct ReaderAckNack {
    /// Reader GUID.
    reader_guid: Guid,
    /// Set of missing sequence numbers.
    missing: BTreeSet<SequenceNumber>,
    /// Highest contiguously received sequence number.
    ack_sn: SequenceNumber,
    /// AckNack count (incremented each time an acknack is sent).
    acknack_count: u32,
    /// Last heartbeat count received (to detect duplicate heartbeats).
    last_heartbeat_count: u32,
}

impl ReaderAckNack {
    /// Create a new reader acknack tracker.
    pub fn new(reader_guid: Guid) -> Self {
        Self {
            reader_guid,
            missing: BTreeSet::new(),
            ack_sn: 0,
            acknack_count: 0,
            last_heartbeat_count: 0,
        }
    }

    /// Returns the reader GUID.
    pub fn reader_guid(&self) -> &Guid {
        &self.reader_guid
    }

    /// Returns the acknowledgement sequence number.
    pub fn ack_sn(&self) -> SequenceNumber {
        self.ack_sn
    }

    /// Returns the missing sequence numbers.
    pub fn missing(&self) -> &BTreeSet<SequenceNumber> {
        &self.missing
    }

    /// Returns the current acknack count.
    pub fn acknack_count(&self) -> u32 {
        self.acknack_count
    }

    /// Record receipt of a sample. Returns true if the sample was new.
    pub fn receive_sample(&mut self, seq: SequenceNumber) -> bool {
        if seq <= self.ack_sn && !self.missing.contains(&seq) {
            return false; // Duplicate
        }
        self.missing.remove(&seq);
        // If this extends the contiguous range, advance ack_sn
        if seq == self.ack_sn + 1 {
            self.ack_sn = seq;
            // Advance past any previously received samples
            while self.missing.is_empty() || !self.missing.contains(&(self.ack_sn + 1)) {
                if self.ack_sn + 1 > self.ack_sn {
                    // Check if we have the next one already
                    break;
                }
            }
        } else if seq > self.ack_sn + 1 {
            // Gap detected — mark missing range
            for s in (self.ack_sn + 1)..seq {
                self.missing.insert(s);
            }
            self.ack_sn = seq;
        }
        true
    }

    /// Process a received heartbeat. Returns true if an acknack should be sent.
    pub fn process_heartbeat(&mut self, hb: &HeartbeatData) -> bool {
        if hb.count <= self.last_heartbeat_count {
            return false; // Duplicate heartbeat
        }
        self.last_heartbeat_count = hb.count;

        // If the writer has samples beyond what we've seen, mark them missing
        if hb.last_sn > self.ack_sn {
            for s in (self.ack_sn + 1)..=hb.last_sn {
                self.missing.insert(s);
            }
        }

        // Remove any "missing" entries that the writer no longer has
        self.missing.retain(|s| *s >= hb.first_sn);

        !self.missing.is_empty()
    }

    /// Encode an acknack submessage payload.
    ///
    /// Layout:
    /// - reader entity_id (4 bytes)
    /// - writer entity_id (4 bytes)
    /// - ack_sn (8 bytes, little-endian) — everything up to this is acknowledged
    /// - num_missing (4 bytes, little-endian)
    /// - missing SNs (8 bytes each, little-endian)
    /// - count (4 bytes, little-endian)
    pub fn encode_acknack(
        &mut self,
        writer_eid: &[u8; 4],
        buf: &mut [u8],
    ) -> Result<usize, BusError> {
        let missing_count = self.missing.len();
        let total = 4 + 4 + 8 + 4 + (missing_count * 8) + 4;
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }
        self.acknack_count = self.acknack_count.wrapping_add(1);

        let mut off = 0;
        // Reader entity ID
        buf[off..off + 4].copy_from_slice(&self.reader_guid.entity_id.0);
        off += 4;
        // Writer entity ID
        buf[off..off + 4].copy_from_slice(writer_eid);
        off += 4;
        // ack_sn
        buf[off..off + 8].copy_from_slice(&self.ack_sn.to_le_bytes());
        off += 8;
        // num_missing
        buf[off..off + 4].copy_from_slice(&(missing_count as u32).to_le_bytes());
        off += 4;
        // Missing SNs
        for &sn in &self.missing {
            buf[off..off + 8].copy_from_slice(&sn.to_le_bytes());
            off += 8;
        }
        // Count
        buf[off..off + 4].copy_from_slice(&self.acknack_count.to_le_bytes());
        off += 4;
        Ok(off)
    }

    /// Decode an acknack submessage payload.
    pub fn decode_acknack(data: &[u8]) -> Result<AckNackData, BusError> {
        if data.len() < 20 {
            return Err(BusError::FrameTooShort);
        }
        let mut reader_eid = [0u8; 4];
        reader_eid.copy_from_slice(&data[0..4]);
        let mut writer_eid = [0u8; 4];
        writer_eid.copy_from_slice(&data[4..8]);
        let ack_sn = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let num_missing = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let expected = 20 + num_missing * 8 + 4;
        if data.len() < expected {
            return Err(BusError::FrameTooShort);
        }
        let mut missing = Vec::new();
        for i in 0..num_missing {
            let off = 20 + i * 8;
            let sn = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            missing.push(sn);
        }
        let count_off = 20 + num_missing * 8;
        let count = u32::from_le_bytes(data[count_off..count_off + 4].try_into().unwrap());
        Ok(AckNackData {
            reader_entity_id: reader_eid,
            writer_entity_id: writer_eid,
            ack_sn,
            missing,
            count,
        })
    }
}

/// Decoded acknack data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AckNackData {
    /// Reader entity ID.
    pub reader_entity_id: [u8; 4],
    /// Writer entity ID.
    pub writer_entity_id: [u8; 4],
    /// Acknowledgement sequence number.
    pub ack_sn: SequenceNumber,
    /// Missing sequence numbers (NACK set).
    pub missing: Vec<SequenceNumber>,
    /// AckNack count.
    pub count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dds::types::*;
    use alloc::vec;

    fn test_guid(id: u8) -> Guid {
        Guid::new(GuidPrefix::new([id; 12]), EntityId::user([0, 0, id], 0x02))
    }

    #[test]
    fn test_writer_heartbeat_new() {
        let wh = WriterHeartbeat::new(test_guid(1));
        assert_eq!(wh.first_sn(), 1);
        assert_eq!(wh.last_sn(), 0);
        assert_eq!(wh.heartbeat_count(), 0);
        assert_eq!(wh.cache_len(), 0);
    }

    #[test]
    fn test_writer_add_sample() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        wh.add_sample(1, &[0xAA]);
        wh.add_sample(2, &[0xBB]);
        assert_eq!(wh.first_sn(), 1);
        assert_eq!(wh.last_sn(), 2);
        assert_eq!(wh.cache_len(), 2);
    }

    #[test]
    fn test_writer_generate_heartbeat() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        wh.add_sample(1, &[0xAA]);
        wh.add_sample(2, &[0xBB]);
        let (first, last, count) = wh.generate_heartbeat();
        assert_eq!(first, 1);
        assert_eq!(last, 2);
        assert_eq!(count, 1);
        let (_, _, count2) = wh.generate_heartbeat();
        assert_eq!(count2, 2);
    }

    #[test]
    fn test_writer_get_sample() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        wh.add_sample(1, &[0xAA]);
        wh.add_sample(2, &[0xBB]);
        assert_eq!(wh.get_sample(1), Some(&[0xAA][..]));
        assert_eq!(wh.get_sample(2), Some(&[0xBB][..]));
        assert_eq!(wh.get_sample(3), None);
    }

    #[test]
    fn test_writer_acknowledge_up_to() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        wh.add_sample(1, &[0xAA]);
        wh.add_sample(2, &[0xBB]);
        wh.add_sample(3, &[0xCC]);
        wh.acknowledge_up_to(3);
        assert_eq!(wh.cache_len(), 1);
        assert_eq!(wh.first_sn(), 3);
        assert_eq!(wh.get_sample(1), None);
        assert_eq!(wh.get_sample(3), Some(&[0xCC][..]));
    }

    #[test]
    fn test_writer_cache_eviction() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        for i in 1..=(MAX_WRITER_CACHE as u64 + 10) {
            wh.add_sample(i, &[(i & 0xFF) as u8]);
        }
        assert_eq!(wh.cache_len(), MAX_WRITER_CACHE);
        assert!(wh.first_sn() > 1);
    }

    #[test]
    fn test_heartbeat_encode_decode() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        wh.add_sample(1, &[0xAA]);
        wh.add_sample(5, &[0xBB]);
        let mut buf = [0u8; 64];
        let len = wh.encode_heartbeat(&mut buf).unwrap();
        assert_eq!(len, 28);
        let hb = WriterHeartbeat::decode_heartbeat(&buf[..len]).unwrap();
        assert_eq!(hb.first_sn, 1);
        assert_eq!(hb.last_sn, 5);
        assert_eq!(hb.count, 1);
    }

    #[test]
    fn test_heartbeat_encode_buffer_too_small() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        let mut buf = [0u8; 10];
        assert_eq!(
            wh.encode_heartbeat(&mut buf).err(),
            Some(BusError::BufferTooSmall)
        );
    }

    #[test]
    fn test_heartbeat_decode_too_short() {
        assert_eq!(
            WriterHeartbeat::decode_heartbeat(&[0u8; 10]).err(),
            Some(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_reader_acknack_new() {
        let ra = ReaderAckNack::new(test_guid(2));
        assert_eq!(ra.ack_sn(), 0);
        assert!(ra.missing().is_empty());
        assert_eq!(ra.acknack_count(), 0);
    }

    #[test]
    fn test_reader_receive_sequential() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        assert!(ra.receive_sample(1));
        assert_eq!(ra.ack_sn(), 1);
        assert!(ra.missing().is_empty());
        assert!(ra.receive_sample(2));
        assert_eq!(ra.ack_sn(), 2);
    }

    #[test]
    fn test_reader_receive_gap() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        ra.receive_sample(1);
        // Skip 2, receive 3
        ra.receive_sample(3);
        assert_eq!(ra.ack_sn(), 3);
        assert!(ra.missing().contains(&2));
    }

    #[test]
    fn test_reader_receive_duplicate() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        assert!(ra.receive_sample(1));
        assert!(!ra.receive_sample(1)); // duplicate
    }

    #[test]
    fn test_reader_process_heartbeat() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        ra.receive_sample(1);
        let hb = HeartbeatData {
            writer_entity_id: [0; 4],
            reader_entity_id: [0; 4],
            first_sn: 1,
            last_sn: 3,
            count: 1,
        };
        let need_nack = ra.process_heartbeat(&hb);
        assert!(need_nack);
        assert!(ra.missing().contains(&2));
        assert!(ra.missing().contains(&3));
    }

    #[test]
    fn test_reader_process_duplicate_heartbeat() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        let hb = HeartbeatData {
            writer_entity_id: [0; 4],
            reader_entity_id: [0; 4],
            first_sn: 1,
            last_sn: 1,
            count: 1,
        };
        ra.process_heartbeat(&hb);
        // Same count again — should be ignored
        let need_nack = ra.process_heartbeat(&hb);
        assert!(!need_nack);
    }

    #[test]
    fn test_acknack_encode_decode() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        ra.receive_sample(1);
        ra.receive_sample(3); // Gap at 2
        let writer_eid = [0, 0, 1, 0x02];
        let mut buf = [0u8; 128];
        let len = ra.encode_acknack(&writer_eid, &mut buf).unwrap();
        let ack = ReaderAckNack::decode_acknack(&buf[..len]).unwrap();
        assert_eq!(ack.ack_sn, 3);
        assert_eq!(ack.missing, vec![2]);
        assert_eq!(ack.count, 1);
        assert_eq!(ack.writer_entity_id, writer_eid);
    }

    #[test]
    fn test_acknack_encode_no_missing() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        ra.receive_sample(1);
        ra.receive_sample(2);
        let writer_eid = [0, 0, 1, 0x02];
        let mut buf = [0u8; 128];
        let len = ra.encode_acknack(&writer_eid, &mut buf).unwrap();
        let ack = ReaderAckNack::decode_acknack(&buf[..len]).unwrap();
        assert!(ack.missing.is_empty());
        assert_eq!(ack.ack_sn, 2);
    }

    #[test]
    fn test_acknack_encode_buffer_too_small() {
        let mut ra = ReaderAckNack::new(test_guid(2));
        let writer_eid = [0; 4];
        let mut buf = [0u8; 5];
        assert_eq!(
            ra.encode_acknack(&writer_eid, &mut buf).err(),
            Some(BusError::BufferTooSmall)
        );
    }

    #[test]
    fn test_acknack_decode_too_short() {
        assert_eq!(
            ReaderAckNack::decode_acknack(&[0u8; 10]).err(),
            Some(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_full_reliable_exchange() {
        let mut wh = WriterHeartbeat::new(test_guid(1));
        let mut ra = ReaderAckNack::new(test_guid(2));

        // Writer sends samples 1, 2, 3
        wh.add_sample(1, &[0x01]);
        wh.add_sample(2, &[0x02]);
        wh.add_sample(3, &[0x03]);

        // Reader receives 1 and 3 (2 lost)
        ra.receive_sample(1);
        ra.receive_sample(3);

        // Writer sends heartbeat
        let mut hb_buf = [0u8; 64];
        let hb_len = wh.encode_heartbeat(&mut hb_buf).unwrap();
        let hb = WriterHeartbeat::decode_heartbeat(&hb_buf[..hb_len]).unwrap();

        // Reader processes heartbeat — should request retransmission
        let need_nack = ra.process_heartbeat(&hb);
        assert!(need_nack);
        assert!(ra.missing().contains(&2));

        // Writer retransmits sample 2
        let data = wh.get_sample(2).unwrap();
        assert_eq!(data, &[0x02]);

        // Reader receives retransmitted sample
        ra.receive_sample(2);
        assert!(!ra.missing().contains(&2));
    }
}
