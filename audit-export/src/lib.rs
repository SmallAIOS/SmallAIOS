// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![forbid(unsafe_code)]

//! Verifiable audit-log exporter for SmallAIOS.
//!
//! This crate ships SmallAIOS audit records to an external
//! [immudb](https://github.com/codenotary/immudb) instance for
//! tamper-evident off-box archival, **without** running the
//! immudb server binary inside SmallAIOS. The on-box footprint
//! is a clean-room `#![no_std]` immudb client speaking gRPC
//! over HTTP/2 + TLS 1.3.
//!
//! ## Layer
//!
//! `audit-export` sits at **Layer 1**: it depends on `security`
//! (SHA-256, Ed25519), `net` (HTTP/2 + gRPC framing), and `mgmt`
//! (ConfigSurface, audit-ring tap), all of which are Layer 0
//! or Layer 1. It is consumed by `container` only.
//!
//! ## Scope
//!
//! See `openspec/changes/verifiable-audit-log-v1/` for the full
//! design. The capability set covers:
//!
//! - **`audit-export-immudb-client`**: clean-room gRPC client.
//! - **`audit-export-pipeline`**: audit-ring tap, batcher, ring buffer.
//! - **`audit-export-config-surface`**: TOML, keyfile, role gate.
//! - **`audit-export-fingerprint-cross-binding`**: dual fingerprint publication.
//!
//! ## Status
//!
//! Implementation is sequenced as `verifiable-audit-log-v1`
//! Phases 1–12 in `tasks.md`. **Phase 1 — Vendor + scaffolding
//! — is the current phase**: this `lib.rs` plus the `Cargo.toml`,
//! the directory tree, and the `vendor/IMMUDB_SCHEMA_SHA`
//! pin file. Phases 2 onward populate the modules referenced
//! below.

// Module skeleton — populated in subsequent phases. Each module
// is the home for one capability or one design-document section.

extern crate alloc;

// pub mod pipeline; // Phase 6
// pub mod config;   // Phase 7
// pub mod cli;      // Phase 7
pub mod immudb;

/// Crate-level version string, derived from the workspace.
///
/// Surfaces as the `User-Agent` header value on outgoing
/// gRPC requests once Phase 4 wires the client.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
