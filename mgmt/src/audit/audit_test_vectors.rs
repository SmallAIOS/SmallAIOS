// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test vectors (sample [`AuditRecord`]s, fixture chain heads, mock
//! ML-DSA-65 keys) for the audit subsystem.
//!
//! Lives in a `*_test_vectors.rs` sibling so codeql-config's
//! `paths-ignore` rule excludes it from analysis — fixed-byte hashes
//! and signing-key seeds here would otherwise trip CodeQL's
//! "hard-coded cryptographic value" heuristics. The fixtures here are
//! deterministically derived: chain heads come from real SHA-3-256
//! invocations on synthetic records; ML-DSA seeds use a fixed
//! all-`0xa5` byte pattern that is **not** used in production.

#![cfg(test)]
#![allow(dead_code)] // shared fixture file — not every test uses every fixture

use alloc::string::ToString;

use super::chain::{compute_hash, EMPTY_CHAIN_HEAD};
use super::record::{AuditAction, AuditRecord, AuditSurface, JsonValue};

/// Minimal `auth_login` record — used by chain + record + ring tests.
pub fn sample_login_record() -> AuditRecord {
    let mut r = AuditRecord {
        ts_ns: 1_700_000_000_000_000_000,
        who: "root@tty".to_string(),
        surface: AuditSurface::Tty,
        action: AuditAction::AuthLogin,
        before: JsonValue::Null,
        after: JsonValue::Null,
        code: 0,
        prev_hash: EMPTY_CHAIN_HEAD,
        hash: EMPTY_CHAIN_HEAD,
        signature: alloc::vec::Vec::new(),
    };
    r.hash = compute_hash(&r);
    r
}

/// `DENY_BURST` record with the spec's coalesced count.
pub fn sample_burst_record(count: u32) -> AuditRecord {
    let mut r = AuditRecord {
        ts_ns: 1_700_000_000_500_000_000,
        who: "op:alice@zenoh".to_string(),
        surface: AuditSurface::Zenoh,
        action: AuditAction::DenyBurst { count },
        before: JsonValue::Null,
        after: JsonValue::Null,
        code: -1,
        prev_hash: EMPTY_CHAIN_HEAD,
        hash: EMPTY_CHAIN_HEAD,
        signature: alloc::vec::Vec::new(),
    };
    r.hash = compute_hash(&r);
    r
}

/// `config_write` record — `metrics.cpu_interval_ms` 1000 → 500.
pub fn sample_config_write_record() -> AuditRecord {
    let mut r = AuditRecord {
        ts_ns: 1_700_000_001_000_000_000,
        who: "root@toml".to_string(),
        surface: AuditSurface::Toml,
        action: AuditAction::ConfigWrite {
            path: "metrics.cpu_interval_ms".to_string(),
        },
        before: JsonValue::Int(1000),
        after: JsonValue::Int(500),
        code: 0,
        prev_hash: EMPTY_CHAIN_HEAD,
        hash: EMPTY_CHAIN_HEAD,
        signature: alloc::vec::Vec::new(),
    };
    r.hash = compute_hash(&r);
    r
}

/// Single-syscall `Deny` record — `system_power`.
pub fn sample_deny_record(syscall: &str) -> AuditRecord {
    let mut r = AuditRecord {
        ts_ns: 1_700_000_002_000_000_000,
        who: "op:alice@zenoh".to_string(),
        surface: AuditSurface::Zenoh,
        action: AuditAction::Deny(syscall.to_string()),
        before: JsonValue::Null,
        after: JsonValue::Null,
        code: -1,
        prev_hash: EMPTY_CHAIN_HEAD,
        hash: EMPTY_CHAIN_HEAD,
        signature: alloc::vec::Vec::new(),
    };
    r.hash = compute_hash(&r);
    r
}

/// Checkpoint record carrying a synthetic "signature" (3 309 bytes of
/// `0xab`). Real signed checkpoints carry an ML-DSA-65 signature; this
/// stub is for the JSONL projection test.
pub fn sample_checkpoint_record() -> AuditRecord {
    let mut r = AuditRecord {
        ts_ns: 1_700_000_003_000_000_000,
        who: "kernel".to_string(),
        surface: AuditSurface::Kernel,
        action: AuditAction::Checkpoint,
        before: JsonValue::Null,
        after: JsonValue::Null,
        code: 0,
        prev_hash: EMPTY_CHAIN_HEAD,
        hash: EMPTY_CHAIN_HEAD,
        // lgtm[rust/hard-coded-cryptographic-value]
        signature: alloc::vec![0xabu8; 64],
    };
    r.hash = compute_hash(&r);
    r
}

/// Fixed seed for the test-fixture ML-DSA-65 keypair. **Not** used in
/// production — the kernel-held key is generated fresh on first boot.
// lgtm[rust/hard-coded-cryptographic-value]
pub const FIXTURE_ML_DSA_SEED: [u8; 32] = [0xa5u8; 32];
