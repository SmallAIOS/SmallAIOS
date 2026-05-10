// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-kernel audit ring — SHA-3-256 hash chain, hybrid rotation,
//! rate-limited denial audit, optional ML-DSA-65 signed checkpoints.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-audit-log/spec.md`.
//!
//! Phase 10 of `management-login-v1` — the FINAL phase. Every spec
//! requirement listed there has a concrete implementation in this
//! module:
//!
//! - [`record`] — typed [`AuditRecord`] with the spec-mandated fields
//!   and a JSONL projection.
//! - [`chain`] — SHA-3-256 chain head + the
//!   [`chain::serialize_for_hash`] helper that the recorder hashes.
//! - [`ring`] — fixed 16 MiB in-memory [`Ring`] (4 KiB × 4096 records)
//!   with append-only semantics.
//! - [`flusher`] — JSONL flush adapter. Production wiring writes to
//!   `/data/audit/log.jsonl`; tests use [`flusher::MemoryFlusher`].
//! - [`rotation`] — hybrid (size OR age) rotation with the
//!   `audit.max_total_disk_bytes` failsafe.
//! - [`denial`] — token-bucket rate limiter (10/s/user) coalescing
//!   excess denials into a single `DENY_BURST` record.
//! - [`checkpoint`] — opt-in ML-DSA-65 signed checkpoints behind
//!   `audit.signed_checkpoints.enabled`.
//!
//! ## Architectural shape
//!
//! `Ring` owns:
//!
//! 1. The in-memory `[Option<AuditRecord>; AUDIT_RING_CAPACITY]` ring,
//! 2. The current chain head (`[u8; 32]`) + monotonic record count,
//! 3. A pluggable [`flusher::Flusher`] implementation,
//! 4. A pluggable [`rotation::Storage`] implementation,
//! 5. A pluggable [`checkpoint::Signer`] implementation.
//!
//! All three pluggables are traits — production wiring (kernel
//! filesystem + kernel-held ML-DSA key) substitutes a concrete impl;
//! the unit tests substitute in-memory fixtures.
//!
//! ## Layer
//!
//! `mgmt::audit` sits at Layer 1, peer to `auth::*` and `mgmt::config`.
//! It depends only on `security::crypto::sha3` and (when checkpoints
//! are enabled) `security::crypto::ml_dsa` from Layer 0. It never
//! reaches into Layer 2 / Layer 3 crates.

#![allow(clippy::too_many_arguments)]

extern crate alloc;

pub mod chain;
pub mod checkpoint;
pub mod denial;
pub mod flusher;
pub mod record;
pub mod ring;
pub mod rotation;

#[cfg(test)]
mod audit_test_vectors;

pub use chain::{serialize_for_hash, ChainHead, EMPTY_CHAIN_HEAD};
pub use checkpoint::{CheckpointSigner, CheckpointVerifier, NoopSigner};
pub use denial::{DenialRateLimiter, DENIAL_RATE_PER_SECOND};
pub use flusher::{Flusher, FlusherError, MemoryFlusher};
pub use record::{
    fingerprint_hex, AuditAction, AuditRecord, AuditSurface, JsonValue, AUDIT_FIELD_AFTER,
    AUDIT_FIELD_BEFORE, AUDIT_FIELD_HASH,
};
pub use ring::{
    Ring, RingError, AUDIT_RING_BYTES, AUDIT_RING_CAPACITY, AUDIT_RING_RECORD_BUDGET_BYTES,
};
pub use rotation::{ArchiveEntry, RotationError, RotationOutcome, RotationPolicy, Storage};
