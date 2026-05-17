// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-binding of the local SHA-3-256 audit chain head and
//! the remote immudb signed state.
//!
//! Phase 9 of `verifiable-audit-log-v1`:
//!
//! - `audit_export-fingerprint-cross-binding` capability:
//!   extend `smallaios/metrics/audit_fingerprint` with a new
//!   `immudb_state` sub-object that is `null` until the
//!   exporter has succeeded at least once.
//! - The base payload keeps the three v1 fields (`ts`,
//!   `hex_fingerprint`, `record_count`) bit-identical so a
//!   v1 consumer continues to parse correctly per the
//!   additive-schema rule.
//! - Console-monitor `top` header line: `IMMUDB ok tx=NNN
//!   last Ns` / `IMMUDB HALT <reason>`.
//!
//! Actual Zenoh publishing lives in `mgmt-zenoh-telemetry`;
//! this module is the renderer that the container-side glue
//! invokes when it knows it must publish.

use crate::immudb::state::PersistedState;
use crate::pipeline::HaltReason;
use crate::verbs::halt_reason_str;
use alloc::format;
use alloc::string::String;

/// Optional `immudb_state` sub-object for the
/// `smallaios/metrics/audit_fingerprint` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImmudbStatePayload {
    pub db: String,
    pub tx_id: u64,
    pub tx_hash_hex: String,
    pub signature_hex: String,
    pub observed_ts_ns: u64,
}

impl ImmudbStatePayload {
    pub fn from_persisted(state: &PersistedState, observed_ts_ns: u64) -> Self {
        Self {
            db: state.db.clone(),
            tx_id: state.tx_id,
            tx_hash_hex: hex_lower(&state.tx_hash),
            signature_hex: hex_lower(&state.signature),
            observed_ts_ns,
        }
    }
}

/// Render the full `audit_fingerprint` payload, including
/// the new `immudb_state` field per the
/// `mgmt-zenoh-telemetry` spec delta. `immudb_state` is
/// `null` when the exporter has never succeeded.
///
/// On-the-wire key names stay v1-compatible:
/// `ts`, `hex_fingerprint`, `record_count`, `immudb_state`.
pub fn render_payload(
    ts_ns: u64,
    hex_fingerprint: &str,
    record_count: u64,
    immudb_state: Option<&ImmudbStatePayload>,
) -> String {
    let mut s = String::with_capacity(256);
    s.push_str("{\"ts\":");
    push_u64(&mut s, ts_ns);
    s.push_str(",\"hex_fingerprint\":\"");
    s.push_str(&json_escape(hex_fingerprint));
    s.push_str("\",\"record_count\":");
    push_u64(&mut s, record_count);
    s.push_str(",\"immudb_state\":");
    match immudb_state {
        None => s.push_str("null"),
        Some(st) => {
            s.push_str("{\"db\":\"");
            s.push_str(&json_escape(&st.db));
            s.push_str("\",\"tx_id\":");
            push_u64(&mut s, st.tx_id);
            s.push_str(",\"tx_hash_hex\":\"");
            s.push_str(&json_escape(&st.tx_hash_hex));
            s.push_str("\",\"signature_hex\":\"");
            s.push_str(&json_escape(&st.signature_hex));
            s.push_str("\",\"observed_ts_ns\":");
            push_u64(&mut s, st.observed_ts_ns);
            s.push('}');
        }
    }
    s.push('}');
    s
}

/// Console-monitor `top` header status line. Pure-function
/// helper invoked by `console-monitor-v1`'s renderer when
/// that change archives. Returns one of:
///
/// - `IMMUDB ok tx=<txId> last <N>s` — healthy export within
///   the last 60 s.
/// - `IMMUDB stale tx=<txId> last <N>s` — exporter enabled
///   but no successful export in the last 60 s.
/// - `IMMUDB HALT <reason>` — pipeline halted, alert color.
/// - `IMMUDB off` — `[exporter] enabled = false` (default).
pub fn monitor_status_line(
    enabled: bool,
    halt: Option<&HaltReason>,
    last_state: Option<&PersistedState>,
    seconds_since_success: Option<u64>,
) -> String {
    if !enabled {
        return String::from("IMMUDB off");
    }
    if let Some(reason) = halt {
        return format!("IMMUDB HALT {}", halt_reason_str(reason));
    }
    match (last_state, seconds_since_success) {
        (Some(st), Some(secs)) if secs <= 60 => {
            format!("IMMUDB ok tx={} last {}s", st.tx_id, secs)
        }
        (Some(st), Some(secs)) => {
            format!("IMMUDB stale tx={} last {}s", st.tx_id, secs)
        }
        _ => String::from("IMMUDB pending"),
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

fn push_u64(s: &mut String, v: u64) {
    s.push_str(&format!("{v}"));
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> PersistedState {
        PersistedState {
            db: "smallaios_audit".into(),
            tx_id: 1234,
            tx_hash: alloc::vec![0xab, 0xcd],
            public_key: alloc::vec![0xee; 32],
            signature: alloc::vec![0xff, 0x00, 0x11],
        }
    }

    /// Minimal v1 parser to confirm `ts`, `hex_fingerprint`, and
    /// `record_count` are present in the documented positions.
    /// Not a real JSON parser — just a substring sanity check.
    fn v1_consumer_can_read(payload: &str) -> Option<(u64, String, u64)> {
        let ts = extract_after(payload, "\"ts\":")?;
        let ts: u64 = ts.split([',', '}']).next()?.trim().parse().ok()?;
        let hex_start = payload.find("\"hex_fingerprint\":\"")? + 19;
        let hex_end = payload[hex_start..].find('"')? + hex_start;
        let hex = &payload[hex_start..hex_end];
        let rc = extract_after(payload, "\"record_count\":")?;
        let rc: u64 = rc.split([',', '}']).next()?.trim().parse().ok()?;
        Some((ts, hex.into(), rc))
    }

    fn extract_after<'a>(s: &'a str, needle: &str) -> Option<&'a str> {
        s.find(needle).map(|i| &s[i + needle.len()..])
    }

    #[test]
    fn payload_without_immudb_state_is_null_field() {
        let p = render_payload(1_700_000_000, "abc123", 42, None);
        assert!(p.contains("\"immudb_state\":null"));
    }

    #[test]
    fn payload_with_immudb_state_renders_subobject() {
        let st = ImmudbStatePayload::from_persisted(&sample_state(), 1_700_000_000);
        let p = render_payload(1_700_000_000, "abc123", 42, Some(&st));
        assert!(p.contains("\"immudb_state\":{"));
        assert!(p.contains("\"db\":\"smallaios_audit\""));
        assert!(p.contains("\"tx_id\":1234"));
        assert!(p.contains("\"tx_hash_hex\":\"abcd\""));
        assert!(p.contains("\"signature_hex\":\"ff0011\""));
        assert!(p.contains("\"observed_ts_ns\":1700000000"));
    }

    #[test]
    fn v1_consumer_still_parses_with_immudb_state() {
        let st = ImmudbStatePayload::from_persisted(&sample_state(), 1_700_000_000);
        let p = render_payload(42, "deadbeef", 7, Some(&st));
        let (ts, hex, rc) = v1_consumer_can_read(&p).unwrap();
        assert_eq!(ts, 42);
        assert_eq!(hex, "deadbeef");
        assert_eq!(rc, 7);
    }

    #[test]
    fn v1_consumer_still_parses_without_immudb_state() {
        let p = render_payload(99, "cafef00d", 13, None);
        let (ts, hex, rc) = v1_consumer_can_read(&p).unwrap();
        assert_eq!(ts, 99);
        assert_eq!(hex, "cafef00d");
        assert_eq!(rc, 13);
    }

    #[test]
    fn from_persisted_lowercases_hex() {
        let mut st = sample_state();
        st.tx_hash = alloc::vec![0xAB, 0xCD, 0xEF];
        let p = ImmudbStatePayload::from_persisted(&st, 0);
        assert_eq!(p.tx_hash_hex, "abcdef");
    }

    #[test]
    fn monitor_line_off_when_disabled() {
        assert_eq!(monitor_status_line(false, None, None, None), "IMMUDB off");
    }

    #[test]
    fn monitor_line_pending_before_first_success() {
        assert_eq!(
            monitor_status_line(true, None, None, None),
            "IMMUDB pending"
        );
    }

    #[test]
    fn monitor_line_ok_when_recent() {
        let st = sample_state();
        let line = monitor_status_line(true, None, Some(&st), Some(4));
        assert_eq!(line, "IMMUDB ok tx=1234 last 4s");
    }

    #[test]
    fn monitor_line_stale_beyond_60s() {
        let st = sample_state();
        let line = monitor_status_line(true, None, Some(&st), Some(120));
        assert_eq!(line, "IMMUDB stale tx=1234 last 120s");
    }

    #[test]
    fn monitor_line_halt_uses_reason_string() {
        let line = monitor_status_line(true, Some(&HaltReason::ProofFailure), None, None);
        assert_eq!(line, "IMMUDB HALT proof_failure");
    }

    #[test]
    fn monitor_line_halt_rollback() {
        let line = monitor_status_line(true, Some(&HaltReason::RollbackSuspected), None, None);
        assert_eq!(line, "IMMUDB HALT rollback_suspected");
    }

    #[test]
    fn payload_escapes_quote_and_backslash() {
        let mut st = ImmudbStatePayload::from_persisted(&sample_state(), 0);
        st.db = String::from("w\"e\\ird");
        let p = render_payload(0, "x", 0, Some(&st));
        assert!(p.contains("\"db\":\"w\\\"e\\\\ird\""));
    }
}
