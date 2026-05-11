// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Audit-record verbs emitted by the `audit-export` pipeline.
//!
//! `mgmt::audit::AuditAction` already supports
//! `Custom(String)` for caller-supplied verbs; this module is
//! the authoritative source of the **exact** verb names this
//! crate emits, plus helpers that build the typed `extra`
//! sidecar each verb carries (per the
//! `mgmt-audit-log/exporter-emitted audit record types`
//! spec delta in this change).
//!
//! Phase 8 of `verifiable-audit-log-v1` — the container-side
//! glue invokes [`build_record`] for each [`SelfAudit`] event
//! returned by the pipeline and feeds the resulting struct
//! through `mgmt::audit::Ring::append`. That call
//! automatically:
//!
//! - participates in the SHA-3-256 hash chain unchanged
//!   (8.2);
//! - is not subject to the denial rate limit since none of
//!   these verbs are `Deny(_)` (8.3);
//! - records the entry locally even when the
//!   `[record_filter] exclude_actions` filter prevents the
//!   record from being shipped off-box (8.4).

use crate::pipeline::{HaltReason, SelfAudit};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// `audit_export_attempt` — one record per export attempt
/// (success or per-minute coalesced failure).
pub const AUDIT_EXPORT_ATTEMPT: &str = "audit_export_attempt";

/// `audit_export_overflow` — at most once per minute,
/// summarizes dropped record count.
pub const AUDIT_EXPORT_OVERFLOW: &str = "audit_export_overflow";

/// `audit_export_proof_failure` — verifier rejected a server
/// response.
pub const AUDIT_EXPORT_PROOF_FAILURE: &str = "audit_export_proof_failure";

/// `audit_export_rollback_suspected` — server `txId`
/// regressed below the locally-stored value.
pub const AUDIT_EXPORT_ROLLBACK_SUSPECTED: &str = "audit_export_rollback_suspected";

/// `audit_export_decode_failure` — protobuf decoder rejected
/// a server response.
pub const AUDIT_EXPORT_DECODE_FAILURE: &str = "audit_export_decode_failure";

/// `audit_export_state_init` — cold start, initial
/// `currentState` recorded as the seed.
pub const AUDIT_EXPORT_STATE_INIT: &str = "audit_export_state_init";

/// `immudb_state` — emitted on every successful
/// `VerifiableSet` with the new `tx_id`, `tx_hash`, and the
/// signed-state signature bytes.
pub const IMMUDB_STATE: &str = "immudb_state";

/// All seven verbs, in stable order. The default
/// `exclude_actions` list in `PipelineConfig::default()`
/// carries the same set.
pub const ALL_VERBS: &[&str] = &[
    AUDIT_EXPORT_ATTEMPT,
    AUDIT_EXPORT_OVERFLOW,
    AUDIT_EXPORT_PROOF_FAILURE,
    AUDIT_EXPORT_ROLLBACK_SUSPECTED,
    AUDIT_EXPORT_DECODE_FAILURE,
    AUDIT_EXPORT_STATE_INIT,
    IMMUDB_STATE,
];

/// One audit-ring record this crate produces, in a layout
/// container/ can drop straight into
/// `mgmt::audit::Ring::append` after building the surrounding
/// `AuditRecord` (timestamp, who, surface) on its side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportAuditRecord {
    /// Canonical verb (one of [`ALL_VERBS`]).
    pub action: String,
    /// Result code: 0 for success, negative POSIX errno for
    /// failure (per `mgmt-audit-log` convention).
    pub code: i32,
    /// Typed sidecar fields rendered into the record's
    /// `extra` JSON. Keys never collide with the standard
    /// `mgmt::audit::AuditRecord` field names.
    pub extra: Vec<(String, String)>,
}

impl ExportAuditRecord {
    fn new(action: &str, code: i32) -> Self {
        Self {
            action: action.into(),
            code,
            extra: Vec::new(),
        }
    }

    fn with(mut self, key: &str, value: impl Into<String>) -> Self {
        self.extra.push((key.into(), value.into()));
        self
    }

    /// Render the `extra` sidecar as a JSON object string.
    /// Container/ may inline this into its own
    /// `AuditRecord::extra` field.
    pub fn extra_to_json(&self) -> String {
        let mut s = String::from("{");
        for (i, (k, v)) in self.extra.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            s.push_str(&escape_json(k));
            s.push_str("\":\"");
            s.push_str(&escape_json(v));
            s.push('"');
        }
        s.push('}');
        s
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Render the human-readable string for a [`HaltReason`].
pub fn halt_reason_str(reason: &HaltReason) -> &'static str {
    match reason {
        HaltReason::Unauthenticated => "unauthenticated",
        HaltReason::ProofFailure => "proof_failure",
        HaltReason::RollbackSuspected => "rollback_suspected",
        HaltReason::DecodeFailure => "decode_failure",
    }
}

/// Map a halt reason to the verb to emit.
pub fn halt_reason_verb(reason: &HaltReason) -> &'static str {
    match reason {
        HaltReason::Unauthenticated => AUDIT_EXPORT_ATTEMPT,
        HaltReason::ProofFailure => AUDIT_EXPORT_PROOF_FAILURE,
        HaltReason::RollbackSuspected => AUDIT_EXPORT_ROLLBACK_SUSPECTED,
        HaltReason::DecodeFailure => AUDIT_EXPORT_DECODE_FAILURE,
    }
}

/// Build the canonical `ExportAuditRecord` for a pipeline
/// [`SelfAudit`] event. Container/ takes the result, wraps it
/// in `mgmt::audit::AuditRecord`, and appends to the ring.
pub fn build_record(event: SelfAudit) -> ExportAuditRecord {
    match event {
        SelfAudit::AttemptOk => ExportAuditRecord::new(AUDIT_EXPORT_ATTEMPT, 0),
        SelfAudit::AttemptRetry { code } => ExportAuditRecord::new(AUDIT_EXPORT_ATTEMPT, code),
        SelfAudit::Overflow { dropped } => {
            ExportAuditRecord::new(AUDIT_EXPORT_OVERFLOW, 0).with("dropped", format!("{dropped}"))
        }
        SelfAudit::Halted(reason) => {
            let r = ExportAuditRecord::new(halt_reason_verb(&reason), -1);
            r.with("failure_reason", halt_reason_str(&reason))
        }
    }
}

/// Build the `immudb_state` record. Caller supplies the
/// fields directly because this record is emitted from the
/// gRPC client path (Phase 4), not from a pipeline
/// `SelfAudit` event.
pub fn build_immudb_state(tx_id: u64, tx_hash_hex: &str, signature_hex: &str) -> ExportAuditRecord {
    ExportAuditRecord::new(IMMUDB_STATE, 0)
        .with("tx_id", format!("{tx_id}"))
        .with("tx_hash_hex", tx_hash_hex)
        .with("signature_hex", signature_hex)
}

/// Build the `audit_export_state_init` record emitted on cold
/// start when no `last_state.bin` is present.
pub fn build_state_init(initial_tx_id: u64) -> ExportAuditRecord {
    ExportAuditRecord::new(AUDIT_EXPORT_STATE_INIT, 0)
        .with("initial_tx_id", format!("{initial_tx_id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_verbs_listed() {
        // Spec mandates exactly 7 verbs.
        assert_eq!(ALL_VERBS.len(), 7);
        // All verb names follow [A-Za-z0-9_]+.
        for v in ALL_VERBS {
            assert!(v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'));
        }
    }

    #[test]
    fn attempt_ok_renders_correctly() {
        let r = build_record(SelfAudit::AttemptOk);
        assert_eq!(r.action, AUDIT_EXPORT_ATTEMPT);
        assert_eq!(r.code, 0);
        assert!(r.extra.is_empty());
    }

    #[test]
    fn attempt_retry_carries_code() {
        let r = build_record(SelfAudit::AttemptRetry { code: 14 });
        assert_eq!(r.action, AUDIT_EXPORT_ATTEMPT);
        assert_eq!(r.code, 14);
    }

    #[test]
    fn overflow_carries_dropped_count() {
        let r = build_record(SelfAudit::Overflow { dropped: 42 });
        assert_eq!(r.action, AUDIT_EXPORT_OVERFLOW);
        assert_eq!(r.code, 0);
        let extras: Vec<_> = r
            .extra
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(extras, alloc::vec![("dropped", "42")]);
    }

    #[test]
    fn halted_proof_failure_uses_proof_verb() {
        let r = build_record(SelfAudit::Halted(HaltReason::ProofFailure));
        assert_eq!(r.action, AUDIT_EXPORT_PROOF_FAILURE);
        assert_eq!(r.code, -1);
        assert!(r
            .extra
            .iter()
            .any(|(k, v)| k == "failure_reason" && v == "proof_failure"));
    }

    #[test]
    fn halted_rollback_uses_rollback_verb() {
        let r = build_record(SelfAudit::Halted(HaltReason::RollbackSuspected));
        assert_eq!(r.action, AUDIT_EXPORT_ROLLBACK_SUSPECTED);
    }

    #[test]
    fn halted_decode_uses_decode_verb() {
        let r = build_record(SelfAudit::Halted(HaltReason::DecodeFailure));
        assert_eq!(r.action, AUDIT_EXPORT_DECODE_FAILURE);
    }

    #[test]
    fn halted_unauth_uses_attempt_verb_with_reason() {
        let r = build_record(SelfAudit::Halted(HaltReason::Unauthenticated));
        // Unauth halts use the generic attempt verb (per spec scenario
        // "unauthenticated stops retries").
        assert_eq!(r.action, AUDIT_EXPORT_ATTEMPT);
        assert!(r
            .extra
            .iter()
            .any(|(k, v)| k == "failure_reason" && v == "unauthenticated"));
    }

    #[test]
    fn immudb_state_carries_signature_fields() {
        let r = build_immudb_state(123, "deadbeef", "cafef00d");
        assert_eq!(r.action, IMMUDB_STATE);
        assert_eq!(r.code, 0);
        let keys: Vec<&str> = r.extra.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"tx_id"));
        assert!(keys.contains(&"tx_hash_hex"));
        assert!(keys.contains(&"signature_hex"));
    }

    #[test]
    fn state_init_carries_initial_tx_id() {
        let r = build_state_init(99);
        assert_eq!(r.action, AUDIT_EXPORT_STATE_INIT);
        assert!(r
            .extra
            .iter()
            .any(|(k, v)| k == "initial_tx_id" && v == "99"));
    }

    #[test]
    fn extra_renders_as_valid_json_object() {
        let r = build_immudb_state(1, "aa", "bb");
        let s = r.extra_to_json();
        assert!(s.starts_with('{'));
        assert!(s.ends_with('}'));
        assert!(s.contains("\"tx_id\":\"1\""));
        assert!(s.contains("\"tx_hash_hex\":\"aa\""));
    }

    #[test]
    fn extra_json_escapes_quote_and_backslash() {
        let r =
            ExportAuditRecord::new(AUDIT_EXPORT_OVERFLOW, 0).with("note", "he said \"hi\" \\back");
        let s = r.extra_to_json();
        assert!(s.contains("he said \\\"hi\\\" \\\\back"));
    }

    #[test]
    fn extra_json_empty_when_no_fields() {
        let r = ExportAuditRecord::new(AUDIT_EXPORT_ATTEMPT, 0);
        assert_eq!(r.extra_to_json(), "{}");
    }

    #[test]
    fn extra_json_escapes_control_bytes() {
        let r = ExportAuditRecord::new(AUDIT_EXPORT_OVERFLOW, 0).with("note", "a\x01b");
        let s = r.extra_to_json();
        assert!(s.contains("\\u0001"));
    }

    #[test]
    fn no_verb_clashes_with_deny_burst_or_chain_committed() {
        // The seven verbs MUST NOT collide with the built-in mgmt
        // verbs that have special rate-limit / flush semantics.
        let reserved = [
            "auth_login",
            "auth_logout",
            "auto_logout",
            "auth_create_user",
            "auth_change_password",
            "firstboot_complete",
            "audit_chain_committed",
            "checkpoint",
            "config_write",
            "model_load",
            "DENY_BURST",
        ];
        for v in ALL_VERBS {
            assert!(!reserved.contains(v), "verb '{v}' clashes with reserved");
            // Also: none of our verbs start with "DENY:" (avoiding
            // the denial rate-limit class entirely, per task 8.3).
            assert!(!v.starts_with("DENY"));
        }
    }
}
