// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHA-3-256 hash chain — the always-on tamper-evident backbone of
//! the audit log.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-audit-log/spec.md`
//! Requirement "SHA-3-256 hash chain".
//!
//! The chain head is the SHA-3-256 digest of the most recently
//! committed record's serialized form (every field except `hash`).
//! Each record's `prev_hash` is the previous record's `hash`. The
//! verifier replays [`serialize_for_hash`] over each JSONL line,
//! checks the digest, and walks the chain.
//!
//! ## Why a custom serializer rather than the JSONL form?
//!
//! - The on-disk JSONL is tied to BTreeMap iteration order — fine in
//!   practice, but a chain that depends on serialization order risks
//!   silent drift across encoder versions.
//! - The hash domain is a fixed-shape canonical byte sequence:
//!   length-prefixed `who` / `surface` / `action` / `before` / `after`
//!   strings + scalar `ts`, `code` + 32-byte `prev_hash`. The shape
//!   is stable across compiler / `BTreeMap` iteration changes.

use alloc::vec::Vec;

#[cfg(test)]
use alloc::string::ToString;

use smallaios_security::crypto::sha3::sha3_256;

use super::record::{AuditAction, AuditRecord};

/// Type alias for a SHA-3-256 chain-head digest — 32 bytes.
pub type ChainHead = [u8; 32];

/// All-zeros chain head — the spec's `prev_hash` for the first record.
pub const EMPTY_CHAIN_HEAD: ChainHead = [0u8; 32];

/// Serialize a record into the canonical byte sequence the chain
/// hashes over. Order: `ts` (u64 LE) | `who` (length-prefixed UTF-8)
/// | `surface` (length-prefixed) | `action` (length-prefixed) |
/// `before` (length-prefixed) | `after` (length-prefixed) | `code`
/// (i32 LE) | `prev_hash` (32 raw bytes).
///
/// The signature field is **not** part of the hash domain (the
/// signature is computed *after* the chain head, over the chain head
/// itself, see [`super::checkpoint`]).
pub fn serialize_for_hash(record: &AuditRecord) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    buf.extend_from_slice(&record.ts_ns.to_le_bytes());
    push_str(&mut buf, record.who.as_str());
    push_str(&mut buf, record.surface.as_str());
    let rendered_action = record.action.render();
    push_str(&mut buf, rendered_action.as_str());
    // Action discriminator extras. Without these, `Deny("a")` and
    // `Deny("b")` would both render `DENY:a`/`DENY:b` already (so the
    // hash differs), but `DenyBurst { count }` collapses to
    // `DENY_BURST` regardless of `count`. Hash the count too.
    match &record.action {
        AuditAction::DenyBurst { count } => {
            buf.extend_from_slice(&count.to_le_bytes());
        }
        AuditAction::ConfigWrite { path } => {
            push_str(&mut buf, path.as_str());
        }
        AuditAction::ModelLoad { name } => {
            push_str(&mut buf, name.as_str());
        }
        _ => {}
    }
    let before = record.before.render();
    push_str(&mut buf, before.as_str());
    let after = record.after.render();
    push_str(&mut buf, after.as_str());
    buf.extend_from_slice(&record.code.to_le_bytes());
    buf.extend_from_slice(&record.prev_hash);
    buf
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    let len = s.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Compute the SHA-3-256 chain hash over a record. The caller SHALL
/// have populated `record.prev_hash` and every other field except
/// `hash` before invoking.
pub fn compute_hash(record: &AuditRecord) -> ChainHead {
    let serialized = serialize_for_hash(record);
    *sha3_256(&serialized).as_bytes()
}

/// Walk a slice of records and confirm each `prev_hash` chains to the
/// previous record's `hash` and each `hash` matches `compute_hash`.
/// Returns the index of the first invalid record on failure.
pub fn verify_chain(records: &[AuditRecord]) -> Result<(), usize> {
    let mut prev = EMPTY_CHAIN_HEAD;
    for (i, r) in records.iter().enumerate() {
        if r.prev_hash != prev {
            return Err(i);
        }
        let computed = compute_hash(r);
        if computed != r.hash {
            return Err(i);
        }
        prev = r.hash;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::audit_test_vectors as fixtures;
    use super::super::record::{AuditAction, AuditRecord, JsonValue};
    use super::*;

    #[test]
    fn empty_chain_head_is_all_zeros() {
        assert_eq!(EMPTY_CHAIN_HEAD, [0u8; 32]);
    }

    #[test]
    fn serialize_includes_ts_who_surface() {
        let r = fixtures::sample_login_record();
        let bytes = serialize_for_hash(&r);
        assert!(bytes.len() > 32);
        // First 8 bytes are little-endian timestamp.
        let mut ts_bytes = [0u8; 8];
        ts_bytes.copy_from_slice(&bytes[..8]);
        assert_eq!(u64::from_le_bytes(ts_bytes), r.ts_ns);
    }

    #[test]
    fn compute_hash_is_deterministic() {
        let r = fixtures::sample_login_record();
        let a = compute_hash(&r);
        let b = compute_hash(&r);
        assert_eq!(a, b);
    }

    #[test]
    fn compute_hash_differs_when_payload_changes() {
        let mut r = fixtures::sample_login_record();
        let h0 = compute_hash(&r);
        r.code = -1;
        let h1 = compute_hash(&r);
        assert_ne!(h0, h1);
    }

    #[test]
    fn compute_hash_differs_when_prev_hash_changes() {
        let mut r = fixtures::sample_login_record();
        let h0 = compute_hash(&r);
        r.prev_hash[0] = 0xff;
        let h1 = compute_hash(&r);
        assert_ne!(h0, h1);
    }

    #[test]
    fn compute_hash_differs_for_deny_burst_count() {
        let r1 = fixtures::sample_burst_record(10);
        let r2 = fixtures::sample_burst_record(11);
        assert_ne!(compute_hash(&r1), compute_hash(&r2));
    }

    #[test]
    fn compute_hash_differs_for_config_write_path() {
        let mut r1 = fixtures::sample_config_write_record();
        let mut r2 = fixtures::sample_config_write_record();
        r2.action = AuditAction::ConfigWrite {
            path: "metrics.memory_interval_ms".to_string(),
        };
        // Without per-action extras the records would otherwise hash
        // identically; this test pins the `path` extra.
        r1.action = AuditAction::ConfigWrite {
            path: "metrics.cpu_interval_ms".to_string(),
        };
        assert_ne!(compute_hash(&r1), compute_hash(&r2));
    }

    #[test]
    fn verify_chain_walks_three_records() {
        let mut prev = EMPTY_CHAIN_HEAD;
        let mut chain = alloc::vec::Vec::new();
        for i in 0..3 {
            let mut r = fixtures::sample_login_record();
            r.ts_ns = 1000 + i as u64;
            r.prev_hash = prev;
            r.hash = compute_hash(&r);
            prev = r.hash;
            chain.push(r);
        }
        verify_chain(&chain).expect("clean chain");
    }

    #[test]
    fn verify_chain_detects_tampered_payload() {
        let mut chain: alloc::vec::Vec<AuditRecord> = alloc::vec::Vec::new();
        let mut prev = EMPTY_CHAIN_HEAD;
        for i in 0..3 {
            let mut r = fixtures::sample_login_record();
            r.ts_ns = 1000 + i as u64;
            r.prev_hash = prev;
            r.hash = compute_hash(&r);
            prev = r.hash;
            chain.push(r);
        }
        // Tamper with the middle record's payload (without
        // re-hashing).
        chain[1].after = JsonValue::Int(999);
        let err = verify_chain(&chain).unwrap_err();
        assert_eq!(err, 1);
    }

    #[test]
    fn verify_chain_detects_broken_prev_hash() {
        let mut chain: alloc::vec::Vec<AuditRecord> = alloc::vec::Vec::new();
        let mut prev = EMPTY_CHAIN_HEAD;
        for i in 0..3 {
            let mut r = fixtures::sample_login_record();
            r.ts_ns = 1000 + i as u64;
            r.prev_hash = prev;
            r.hash = compute_hash(&r);
            prev = r.hash;
            chain.push(r);
        }
        chain[2].prev_hash = [0xaau8; 32];
        let err = verify_chain(&chain).unwrap_err();
        assert_eq!(err, 2);
    }

    #[test]
    fn verify_chain_empty_is_ok() {
        let chain: alloc::vec::Vec<AuditRecord> = alloc::vec::Vec::new();
        verify_chain(&chain).unwrap();
    }
}
