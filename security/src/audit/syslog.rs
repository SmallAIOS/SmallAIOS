// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RFC 5424 syslog serialization for audit log entries.
//!
//! Produces structured data in RFC 5424 format for external log collection.
//! Uses the `smallaios-audit@0` SD-ID with typed parameters.

use core::fmt::{self, Write};

use super::entry::AuditEntry;

/// Syslog severity levels (RFC 5424 Section 6.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Severity {
    Emergency = 0,
    Alert = 1,
    Critical = 2,
    Error = 3,
    Warning = 4,
    Notice = 5,
    Informational = 6,
    Debug = 7,
}

/// Syslog facility: local use 4 (value=20) for audit subsystem.
const FACILITY_LOCAL4: u8 = 20;

/// Compute PRI value: facility * 8 + severity.
fn pri(severity: Severity) -> u8 {
    FACILITY_LOCAL4 * 8 + severity as u8
}

/// Map an audit entry to a syslog severity.
fn entry_severity(entry: &AuditEntry) -> Severity {
    use super::taxonomy::EventSubcategory;
    match entry.event_type.subcategory {
        EventSubcategory::Exhaustion | EventSubcategory::Timeout => Severity::Warning,
        EventSubcategory::TlsFailure | EventSubcategory::Deny => Severity::Notice,
        EventSubcategory::Boot | EventSubcategory::Shutdown => Severity::Informational,
        _ => match entry.result {
            super::entry::AuditResult::Failure => Severity::Warning,
            super::entry::AuditResult::Success => Severity::Informational,
        },
    }
}

/// Format a u64 nanosecond timestamp as RFC 3339 with nanosecond precision.
///
/// Outputs: `YYYY-MM-DDThh:mm:ss.nnnnnnnnnZ`
///
/// Uses a simplified epoch-based calculation (no leap seconds).
fn format_timestamp(ns: u64, w: &mut impl Write) -> fmt::Result {
    let total_secs = ns / 1_000_000_000;
    let nano_part = ns % 1_000_000_000;

    // Days since epoch
    let days = total_secs / 86400;
    let day_secs = total_secs % 86400;
    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;

    // Civil date from day count (algorithm from Howard Hinnant)
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    write!(
        w,
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
        y, m, d, hours, minutes, seconds, nano_part
    )
}

/// Serialize an audit entry to RFC 5424 syslog format.
///
/// Format:
/// `<PRI>1 TIMESTAMP HOSTNAME APP-NAME PROCID MSGID [SD-ID params] MSG`
///
/// Example:
/// `<166>1 2026-01-15T10:30:00.000000000Z smallaios audit 42 - [smallaios-audit@0 cat="capability" sub="grant" task="42" res="1:100" op="grant" result="success" cap="7"] capability grant`
pub fn format_syslog(entry: &AuditEntry, buf: &mut impl Write) -> fmt::Result {
    let severity = entry_severity(entry);
    let pri_val = pri(severity);

    // PRI and version
    write!(buf, "<{}>1 ", pri_val)?;

    // Timestamp
    format_timestamp(entry.timestamp_ns, buf)?;

    // HOSTNAME APP-NAME PROCID MSGID
    write!(buf, " smallaios audit {} - ", entry.task_id)?;

    // Structured data
    write!(
        buf,
        "[smallaios-audit@0 cat=\"{}\" sub=\"{}\" task=\"{}\" res=\"{}:{}\" op=\"{}\" result=\"{}\" cap=\"{}\"]",
        entry.event_type.category.as_str(),
        entry.event_type.subcategory.as_str(),
        entry.task_id,
        entry.resource_type,
        entry.resource_instance,
        entry.operation.as_str(),
        entry.result.as_str(),
        entry.capability_id,
    )?;

    // MSG: short human-readable summary
    write!(
        buf,
        " {} {}",
        entry.event_type.category.as_str(),
        entry.event_type.subcategory.as_str(),
    )
}

/// Maximum syslog message size (generous upper bound).
pub const MAX_SYSLOG_LEN: usize = 512;

/// Serialize to a fixed-size byte buffer. Returns the number of bytes written.
pub fn format_syslog_to_buf(entry: &AuditEntry, buf: &mut [u8]) -> usize {
    let mut writer = BufWriter::new(buf);
    let _ = format_syslog(entry, &mut writer);
    writer.pos
}

/// Adapter to write `fmt::Write` into a byte slice.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BufWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let remaining = self.buf.len() - self.pos;
        let to_write = bytes.len().min(remaining);
        self.buf[self.pos..self.pos + to_write].copy_from_slice(&bytes[..to_write]);
        self.pos += to_write;
        if to_write < bytes.len() {
            Err(fmt::Error)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditResult, Operation};
    use crate::audit::taxonomy::AuditEventType;
    use alloc::string::String;

    fn make_entry() -> AuditEntry {
        AuditEntry {
            timestamp_ns: 1_000_000_000, // 1 second after epoch
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
    fn syslog_format_basic() {
        let entry = make_entry();
        let mut out = String::new();
        format_syslog(&entry, &mut out).unwrap();

        // Check PRI (facility=20, severity=6=Informational: 20*8+6=166)
        assert!(out.starts_with("<166>1 "));
        // Check structured data
        assert!(out.contains("[smallaios-audit@0"));
        assert!(out.contains("cat=\"capability\""));
        assert!(out.contains("sub=\"grant\""));
        assert!(out.contains("task=\"42\""));
        assert!(out.contains("res=\"1:100\""));
        assert!(out.contains("op=\"grant\""));
        assert!(out.contains("result=\"success\""));
        assert!(out.contains("cap=\"7\""));
        // Check MSG
        assert!(out.contains("capability grant"));
    }

    #[test]
    fn syslog_format_failure_event() {
        let entry = AuditEntry {
            timestamp_ns: 0,
            event_type: AuditEventType::capability_deny(),
            task_id: 10,
            resource_type: 3,
            resource_instance: 5,
            operation: Operation::new("deny"),
            result: AuditResult::Failure,
            capability_id: 0,
        };
        let mut out = String::new();
        format_syslog(&entry, &mut out).unwrap();

        // Deny maps to Notice (severity=5, PRI=20*8+5=165)
        assert!(out.starts_with("<165>1 "));
        assert!(out.contains("sub=\"deny\""));
        assert!(out.contains("result=\"failure\""));
    }

    #[test]
    fn syslog_format_exhaustion() {
        let entry = AuditEntry {
            timestamp_ns: 2_000_000_000,
            event_type: AuditEventType::resource_exhaustion(),
            task_id: 5,
            resource_type: 0,
            resource_instance: 0,
            operation: Operation::new("allocate"),
            result: AuditResult::Failure,
            capability_id: 0,
        };
        let mut out = String::new();
        format_syslog(&entry, &mut out).unwrap();

        // Exhaustion maps to Warning (severity=4, PRI=20*8+4=164)
        assert!(out.starts_with("<164>1 "));
        assert!(out.contains("cat=\"resource\""));
        assert!(out.contains("sub=\"exhaustion\""));
    }

    #[test]
    fn syslog_format_system_boot() {
        let entry = AuditEntry {
            timestamp_ns: 500_000_000,
            event_type: AuditEventType::system_boot(),
            task_id: 0,
            resource_type: 0,
            resource_instance: 0,
            operation: Operation::new("boot"),
            result: AuditResult::Success,
            capability_id: 0,
        };
        let mut out = String::new();
        format_syslog(&entry, &mut out).unwrap();

        // Boot maps to Informational (severity=6, PRI=166)
        assert!(out.starts_with("<166>1 "));
        assert!(out.contains("system boot"));
    }

    #[test]
    fn syslog_format_to_buf() {
        let entry = make_entry();
        let mut buf = [0u8; MAX_SYSLOG_LEN];
        let len = format_syslog_to_buf(&entry, &mut buf);
        assert!(len > 0);
        let msg = core::str::from_utf8(&buf[..len]).unwrap();
        assert!(msg.starts_with("<166>1 "));
    }

    #[test]
    fn syslog_format_to_small_buf() {
        let entry = make_entry();
        let mut buf = [0u8; 10]; // too small
        let len = format_syslog_to_buf(&entry, &mut buf);
        // Should write what fits
        assert_eq!(len, 10);
    }

    #[test]
    fn timestamp_epoch() {
        let mut out = String::new();
        format_timestamp(0, &mut out).unwrap();
        assert_eq!(out, "1970-01-01T00:00:00.000000000Z");
    }

    #[test]
    fn timestamp_one_second() {
        let mut out = String::new();
        format_timestamp(1_000_000_000, &mut out).unwrap();
        assert_eq!(out, "1970-01-01T00:00:01.000000000Z");
    }

    #[test]
    fn timestamp_with_nanos() {
        let mut out = String::new();
        format_timestamp(1_500_000_123, &mut out).unwrap();
        assert_eq!(out, "1970-01-01T00:00:01.500000123Z");
    }

    #[test]
    fn timestamp_2026() {
        // 2026-01-01T00:00:00Z = 1767225600 seconds
        let ns = 1_767_225_600u64 * 1_000_000_000;
        let mut out = String::new();
        format_timestamp(ns, &mut out).unwrap();
        assert_eq!(out, "2026-01-01T00:00:00.000000000Z");
    }

    #[test]
    fn pri_values() {
        // facility=20(local4), severity=Informational(6): 20*8+6=166
        assert_eq!(pri(Severity::Informational), 166);
        // severity=Warning(4): 20*8+4=164
        assert_eq!(pri(Severity::Warning), 164);
        // severity=Emergency(0): 20*8+0=160
        assert_eq!(pri(Severity::Emergency), 160);
    }

    #[test]
    fn all_event_types_have_valid_severity() {
        let entries = [
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
        for et in &entries {
            let entry = AuditEntry {
                timestamp_ns: 0,
                event_type: *et,
                task_id: 0,
                resource_type: 0,
                resource_instance: 0,
                operation: Operation::new("test"),
                result: AuditResult::Success,
                capability_id: 0,
            };
            let severity = entry_severity(&entry);
            let p = pri(severity);
            assert!(
                (160..=167).contains(&p),
                "PRI {} out of range for {:?}",
                p,
                et
            );
        }
    }
}
