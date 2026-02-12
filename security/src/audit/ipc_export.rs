// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Signed audit batch export via Zenoh IPC.
//!
//! Serializes signed audit batches for transport on
//! key expression `smallaios/v1/audit`.

use super::batch_signing::SignedAuditBatch;
use crate::crypto::sha3::SHA3_256_DIGEST_LEN;

/// Zenoh key expression for audit batch export.
pub const AUDIT_KEY_EXPR: &str = "smallaios/v1/audit";

/// Maximum serialized batch header size.
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | version |
/// | 1 | 8 | sequence (u64 BE) |
/// | 9 | 32 | batch_hash |
/// | 41 | 32 | previous_hash |
/// | 73 | 2 | entry_count (u16 BE) |
/// | 75 | 1 | is_signed flag |
/// | 76 | 16 | signer_key_id |
pub const BATCH_HEADER_SIZE: usize = 92;

/// Current wire format version.
const WIRE_VERSION: u8 = 1;

/// Serialize a signed audit batch header for Zenoh transport.
///
/// Returns the number of bytes written, or None if the buffer is too small.
pub fn serialize_batch_header(batch: &SignedAuditBatch, buf: &mut [u8]) -> Option<usize> {
    if buf.len() < BATCH_HEADER_SIZE {
        return None;
    }

    buf[0] = WIRE_VERSION;
    buf[1..9].copy_from_slice(&batch.batch.sequence.to_be_bytes());
    buf[9..41].copy_from_slice(&batch.batch.batch_hash);
    buf[41..73].copy_from_slice(&batch.batch.previous_hash);

    let entry_count = batch.batch.entries.len().min(u16::MAX as usize) as u16;
    buf[73..75].copy_from_slice(&entry_count.to_be_bytes());
    buf[75] = if batch.is_signed { 1 } else { 0 };
    buf[76..92].copy_from_slice(&batch.signer_key_id);

    Some(BATCH_HEADER_SIZE)
}

/// Deserialize a batch header from wire format.
///
/// Returns (sequence, batch_hash, previous_hash, entry_count, is_signed, signer_key_id).
pub fn deserialize_batch_header(buf: &[u8]) -> Option<BatchHeaderInfo> {
    if buf.len() < BATCH_HEADER_SIZE {
        return None;
    }
    if buf[0] != WIRE_VERSION {
        return None;
    }

    let sequence = u64::from_be_bytes(buf[1..9].try_into().ok()?);
    let mut batch_hash = [0u8; SHA3_256_DIGEST_LEN];
    batch_hash.copy_from_slice(&buf[9..41]);
    let mut previous_hash = [0u8; SHA3_256_DIGEST_LEN];
    previous_hash.copy_from_slice(&buf[41..73]);
    let entry_count = u16::from_be_bytes(buf[73..75].try_into().ok()?);
    let is_signed = buf[75] != 0;
    let mut signer_key_id = [0u8; 16];
    signer_key_id.copy_from_slice(&buf[76..92]);

    Some(BatchHeaderInfo {
        sequence,
        batch_hash,
        previous_hash,
        entry_count,
        is_signed,
        signer_key_id,
    })
}

/// Parsed batch header information.
#[derive(Debug, Clone)]
pub struct BatchHeaderInfo {
    pub sequence: u64,
    pub batch_hash: [u8; SHA3_256_DIGEST_LEN],
    pub previous_hash: [u8; SHA3_256_DIGEST_LEN],
    pub entry_count: u16,
    pub is_signed: bool,
    pub signer_key_id: [u8; 16],
}

/// Statistics for audit IPC export.
pub struct AuditExportStats {
    /// Total batches exported.
    pub batches_exported: u64,
    /// Total bytes exported.
    pub bytes_exported: u64,
    /// Total export failures.
    pub export_failures: u64,
}

impl AuditExportStats {
    pub fn new() -> Self {
        Self {
            batches_exported: 0,
            bytes_exported: 0,
            export_failures: 0,
        }
    }

    /// Record a successful export.
    pub fn record_success(&mut self, bytes: u64) {
        self.batches_exported += 1;
        self.bytes_exported += bytes;
    }

    /// Record an export failure.
    pub fn record_failure(&mut self) {
        self.export_failures += 1;
    }
}

impl Default for AuditExportStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::batch_signing::SignedAuditBatch;
    use super::super::entry::{AuditEntry, AuditResult, Operation};
    use super::super::integrity::IntegrityChain;
    use super::super::taxonomy::AuditEventType;
    use super::*;
    use alloc::vec;

    fn make_entry(ts: u64) -> AuditEntry {
        AuditEntry {
            timestamp_ns: ts,
            event_type: AuditEventType::capability_grant(),
            task_id: 1,
            resource_type: 0,
            resource_instance: 0,
            operation: Operation::new("test"),
            result: AuditResult::Success,
            capability_id: 0,
        }
    }

    fn make_signed_batch() -> SignedAuditBatch {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(100), make_entry(200)]);
        let mut signed = SignedAuditBatch::new(batch);
        let sig = [0xBBu8; super::super::batch_signing::ML_DSA_65_SIG_SIZE];
        signed.set_signature(&sig, [0x01u8; 16]);
        signed
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let signed = make_signed_batch();
        let mut buf = [0u8; BATCH_HEADER_SIZE];
        let written = serialize_batch_header(&signed, &mut buf).unwrap();
        assert_eq!(written, BATCH_HEADER_SIZE);

        let info = deserialize_batch_header(&buf).unwrap();
        assert_eq!(info.sequence, signed.batch.sequence);
        assert_eq!(info.batch_hash, signed.batch.batch_hash);
        assert_eq!(info.previous_hash, signed.batch.previous_hash);
        assert_eq!(info.entry_count, 2);
        assert!(info.is_signed);
        assert_eq!(info.signer_key_id, [0x01u8; 16]);
    }

    #[test]
    fn serialize_buffer_too_small() {
        let signed = make_signed_batch();
        let mut buf = [0u8; 10];
        assert!(serialize_batch_header(&signed, &mut buf).is_none());
    }

    #[test]
    fn deserialize_buffer_too_small() {
        let buf = [0u8; 10];
        assert!(deserialize_batch_header(&buf).is_none());
    }

    #[test]
    fn deserialize_wrong_version() {
        let mut buf = [0u8; BATCH_HEADER_SIZE];
        buf[0] = 99; // wrong version
        assert!(deserialize_batch_header(&buf).is_none());
    }

    #[test]
    fn unsigned_batch_roundtrip() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(100)]);
        let unsigned = SignedAuditBatch::new(batch);

        let mut buf = [0u8; BATCH_HEADER_SIZE];
        serialize_batch_header(&unsigned, &mut buf).unwrap();

        let info = deserialize_batch_header(&buf).unwrap();
        assert!(!info.is_signed);
        assert_eq!(info.entry_count, 1);
    }

    #[test]
    fn export_stats() {
        let mut stats = AuditExportStats::new();
        stats.record_success(92);
        stats.record_success(92);
        stats.record_failure();
        assert_eq!(stats.batches_exported, 2);
        assert_eq!(stats.bytes_exported, 184);
        assert_eq!(stats.export_failures, 1);
    }

    #[test]
    fn key_expression() {
        assert_eq!(AUDIT_KEY_EXPR, "smallaios/v1/audit");
    }
}
