// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Audit log entry structure.
//!
//! Each entry contains: nanosecond timestamp, event type, task ID, resource
//! reference, operation, result, and capability ID per the tamper-evident
//! audit spec.

use super::taxonomy::AuditEventType;

/// Maximum length of the operation description string stored inline.
const OP_MAX_LEN: usize = 32;

/// Result of an audited operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AuditResult {
    /// Operation succeeded.
    Success = 0,
    /// Operation failed.
    Failure = 1,
}

impl AuditResult {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Success),
            1 => Some(Self::Failure),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
        }
    }
}

/// Short inline operation description (e.g., "grant", "allocate", "tls-handshake").
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Operation {
    buf: [u8; OP_MAX_LEN],
    len: u8,
}

impl Operation {
    /// Create an operation from a string slice. Truncates to 32 bytes.
    pub fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        let len = bytes.len().min(OP_MAX_LEN);
        let mut buf = [0u8; OP_MAX_LEN];
        buf[..len].copy_from_slice(&bytes[..len]);
        Self {
            buf,
            len: len as u8,
        }
    }

    /// Return the operation as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len as usize]
    }

    /// Return the operation as a string slice.
    pub fn as_str(&self) -> &str {
        // Safety: we only store valid UTF-8 from from_str
        core::str::from_utf8(&self.buf[..self.len as usize]).unwrap_or("")
    }
}

impl core::fmt::Debug for Operation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Operation(\"{}\")", self.as_str())
    }
}

/// A single audit log entry.
///
/// Fixed-size structure suitable for ring buffer storage. Contains all
/// fields required by the tamper-evident audit spec.
#[derive(Debug, Clone, Copy)]
pub struct AuditEntry {
    /// Nanosecond-precision timestamp (monotonic clock).
    pub timestamp_ns: u64,
    /// Classified event type (category + subcategory).
    pub event_type: AuditEventType,
    /// Task that triggered the event (0 = kernel/no task).
    pub task_id: u64,
    /// Resource type being accessed (0 = not applicable).
    pub resource_type: u8,
    /// Resource instance ID (0 = not applicable).
    pub resource_instance: u64,
    /// Short description of the operation performed.
    pub operation: Operation,
    /// Whether the operation succeeded or failed.
    pub result: AuditResult,
    /// Capability ID used for the operation (0 = not applicable).
    pub capability_id: u64,
}

impl AuditEntry {
    /// Serialize the entry to a deterministic byte representation.
    ///
    /// Layout (big-endian, fixed-size):
    /// - [0..8]   timestamp_ns (u64 BE)
    /// - [8..10]  event_type (category, subcategory)
    /// - [10..18] task_id (u64 BE)
    /// - [18]     resource_type (u8)
    /// - [19..27] resource_instance (u64 BE)
    /// - [27]     operation length (u8)
    /// - [28..60] operation bytes (padded to 32)
    /// - [60]     result (u8)
    /// - [61..69] capability_id (u64 BE)
    ///
    /// Total: 69 bytes per entry.
    pub const SERIALIZED_SIZE: usize = 69;

    pub fn serialize(&self, buf: &mut [u8; Self::SERIALIZED_SIZE]) {
        buf[0..8].copy_from_slice(&self.timestamp_ns.to_be_bytes());
        let et_bytes = self.event_type.to_bytes();
        buf[8] = et_bytes[0];
        buf[9] = et_bytes[1];
        buf[10..18].copy_from_slice(&self.task_id.to_be_bytes());
        buf[18] = self.resource_type;
        buf[19..27].copy_from_slice(&self.resource_instance.to_be_bytes());
        let op_bytes = self.operation.as_bytes();
        buf[27] = op_bytes.len() as u8;
        buf[28..28 + op_bytes.len()].copy_from_slice(op_bytes);
        // Zero-pad remaining operation bytes
        for b in &mut buf[28 + op_bytes.len()..60] {
            *b = 0;
        }
        buf[60] = self.result as u8;
        buf[61..69].copy_from_slice(&self.capability_id.to_be_bytes());
    }

    /// Deserialize an entry from bytes.
    pub fn deserialize(buf: &[u8; Self::SERIALIZED_SIZE]) -> Option<Self> {
        let timestamp_ns = u64::from_be_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let event_type = AuditEventType::from_bytes([buf[8], buf[9]])?;
        let task_id = u64::from_be_bytes([
            buf[10], buf[11], buf[12], buf[13], buf[14], buf[15], buf[16], buf[17],
        ]);
        let resource_type = buf[18];
        let resource_instance = u64::from_be_bytes([
            buf[19], buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26],
        ]);
        let op_len = buf[27] as usize;
        let op_len = op_len.min(OP_MAX_LEN);
        let op_str = core::str::from_utf8(&buf[28..28 + op_len]).ok()?;
        let operation = Operation::new(op_str);
        let result = AuditResult::from_u8(buf[60])?;
        let capability_id = u64::from_be_bytes([
            buf[61], buf[62], buf[63], buf[64], buf[65], buf[66], buf[67], buf[68],
        ]);

        Some(Self {
            timestamp_ns,
            event_type,
            task_id,
            resource_type,
            resource_instance,
            operation,
            result,
            capability_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::taxonomy::AuditEventType;

    fn make_entry() -> AuditEntry {
        AuditEntry {
            timestamp_ns: 1_000_000_000,
            event_type: AuditEventType::capability_grant(),
            task_id: 42,
            resource_type: 1,
            resource_instance: 100,
            operation: Operation::new("grant"),
            result: AuditResult::Success,
            capability_id: 7,
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let entry = make_entry();
        let mut buf = [0u8; AuditEntry::SERIALIZED_SIZE];
        entry.serialize(&mut buf);
        let decoded = AuditEntry::deserialize(&buf).unwrap();
        assert_eq!(decoded.timestamp_ns, entry.timestamp_ns);
        assert_eq!(decoded.event_type, entry.event_type);
        assert_eq!(decoded.task_id, entry.task_id);
        assert_eq!(decoded.resource_type, entry.resource_type);
        assert_eq!(decoded.resource_instance, entry.resource_instance);
        assert_eq!(decoded.operation, entry.operation);
        assert_eq!(decoded.result, entry.result);
        assert_eq!(decoded.capability_id, entry.capability_id);
    }

    #[test]
    fn serialize_deterministic() {
        let entry = make_entry();
        let mut buf1 = [0u8; AuditEntry::SERIALIZED_SIZE];
        let mut buf2 = [0u8; AuditEntry::SERIALIZED_SIZE];
        entry.serialize(&mut buf1);
        entry.serialize(&mut buf2);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn operation_from_str_and_back() {
        let op = Operation::new("tls-handshake");
        assert_eq!(op.as_str(), "tls-handshake");
    }

    #[test]
    fn operation_truncates_long_string() {
        let long = "a]very-long-operation-name-that-exceeds-thirty-two-bytes";
        let op = Operation::new(long);
        assert_eq!(op.as_bytes().len(), OP_MAX_LEN);
    }

    #[test]
    fn operation_empty() {
        let op = Operation::new("");
        assert_eq!(op.as_str(), "");
        assert_eq!(op.as_bytes().len(), 0);
    }

    #[test]
    fn audit_result_roundtrip() {
        assert_eq!(AuditResult::from_u8(0), Some(AuditResult::Success));
        assert_eq!(AuditResult::from_u8(1), Some(AuditResult::Failure));
        assert_eq!(AuditResult::from_u8(2), None);
    }

    #[test]
    fn audit_result_as_str() {
        assert_eq!(AuditResult::Success.as_str(), "success");
        assert_eq!(AuditResult::Failure.as_str(), "failure");
    }

    #[test]
    fn entry_all_event_types() {
        let event_types = [
            AuditEventType::capability_grant(),
            AuditEventType::capability_revoke(),
            AuditEventType::capability_deny(),
            AuditEventType::auth_success(),
            AuditEventType::auth_failure(),
            AuditEventType::resource_allocation(),
            AuditEventType::resource_exhaustion(),
            AuditEventType::inference_request(),
            AuditEventType::inference_completion(),
            AuditEventType::inference_timeout(),
            AuditEventType::system_boot(),
            AuditEventType::system_shutdown(),
            AuditEventType::system_watchdog(),
        ];

        for et in &event_types {
            let entry = AuditEntry {
                timestamp_ns: 999,
                event_type: *et,
                task_id: 1,
                resource_type: 0,
                resource_instance: 0,
                operation: Operation::new("test"),
                result: AuditResult::Success,
                capability_id: 0,
            };
            let mut buf = [0u8; AuditEntry::SERIALIZED_SIZE];
            entry.serialize(&mut buf);
            let decoded = AuditEntry::deserialize(&buf).unwrap();
            assert_eq!(decoded.event_type, *et);
        }
    }

    #[test]
    fn entry_max_values() {
        let entry = AuditEntry {
            timestamp_ns: u64::MAX,
            event_type: AuditEventType::system_watchdog(),
            task_id: u64::MAX,
            resource_type: 255,
            resource_instance: u64::MAX,
            operation: Operation::new("x".repeat(64).as_str()),
            result: AuditResult::Failure,
            capability_id: u64::MAX,
        };
        let mut buf = [0u8; AuditEntry::SERIALIZED_SIZE];
        entry.serialize(&mut buf);
        let decoded = AuditEntry::deserialize(&buf).unwrap();
        assert_eq!(decoded.timestamp_ns, u64::MAX);
        assert_eq!(decoded.task_id, u64::MAX);
        assert_eq!(decoded.capability_id, u64::MAX);
    }

    #[test]
    fn operation_debug_format() {
        let op = Operation::new("grant");
        let debug = alloc::format!("{:?}", op);
        assert!(debug.contains("grant"));
    }
}
