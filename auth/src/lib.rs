// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Identity, authentication, and role-based access control for SmallAIOS.
//!
//! This crate is the source of truth for **who is on the box** in
//! SmallAIOS. It owns the shadow-style password file, the Argon2id
//! verifier, the three-role taxonomy (`Root`, `Operator`, `Viewer`),
//! the per-role idle-timeout policy, the in-kernel session table API,
//! and the bearer-token lifecycle for remote (Zenoh) admin sessions.
//!
//! ## Layer
//!
//! `auth` sits at **Layer 1**: it depends on `security` and `kernel`
//! (both Layer 0) and is consumed by `mgmt`, `ipc`, `peripheral` (UART
//! console-login), and `container` (passwd CLI, Zenoh admin).
//!
//! ## Status
//!
//! Implementation is sequenced as `management-login-v1` Phases 1–10:
//!
//! 1. Argon2id KDF (in `security`) + KAT vectors. **Landed.**
//! 2. `auth/` crate scaffold (this file) + shadow parser + role enum
//!    + tier auto-selection + atomic-write trait. **This phase.**
//! 3. Kernel auth syscalls + session table.
//! 4. Console-login (TTY first-boot, login, lockout, idle sweep).
//! 5. `mgmt/` crate scaffold + `Config` + `ConfigSurface` trait.
//! 6. TOML loader + universal-exposure CI walker.
//! 7. Zenoh admin keyspace + bearer-token wrapper.
//! 8. Zenoh telemetry keyspace.
//! 9. TOTP (RFC 6238) + `totp_setup` syscall.
//! 10. Audit chain + signed checkpoints + denial audit.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

// ─── Phase 2 modules ─────────────────────────────────────────────────────────

/// Three-role taxonomy: `Root`, `Operator`, `Viewer`.
pub mod role;

/// Shadow file format, parser, and atomic-rewrite helper.
pub mod shadow;

/// Atomic shadow-rewrite trait (`stage → fsync → rename`) plus an
/// in-memory test impl and a `KernelAtomicWriter` stub awaiting
/// `embedded-filesystem-v1` Phase 7 (F2FS RW).
pub mod atomic_write;

/// Argon2id per-tier parameter selection from available RAM.
pub mod tier;

// ─── Future-phase scaffolding (kept as empty stubs so downstream crates
//      can name them once the bodies land) ────────────────────────────────────

/// Kernel session table API surface.
///
/// Filled by `management-login-v1` Phase 3 (the table itself lives in
/// the kernel; this module re-exports the user-facing types).
pub mod session {}

/// Bearer-token lifecycle for remote (Zenoh) admin sessions.
///
/// Filled by `management-login-v1` Phase 7.
pub mod token {}

/// Console-login flow (TTY first-boot, login, lockout, idle sweep).
///
/// Filled by `management-login-v1` Phase 4.
pub mod console {}

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use role::{Role, RoleError};
pub use shadow::{
    parse_file, parse_record, serialize_file, serialize_record, verify_file_permissions,
    ShadowEntry, ShadowError, FLAG_MUST_CHANGE_PASSWORD,
};
pub use tier::select_tier;
