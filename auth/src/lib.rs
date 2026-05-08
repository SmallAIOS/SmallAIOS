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
//! This is a Wave 0 **skeleton**. Implementation is scheduled as
//! `management-login-v1` Phases 1–10:
//!
//! 1. Argon2id KDF (in `security`) + KAT vectors.
//! 2. `auth/` crate scaffold (this file) + shadow parser + role enum.
//! 3. Kernel auth syscalls + session table.
//! 4. Console-login (TTY first-boot, login, lockout, idle sweep).
//! 5. `mgmt/` crate scaffold + `Config` + `ConfigSurface` trait.
//! 6. TOML loader + universal-exposure CI walker.
//! 7. Zenoh admin keyspace + bearer-token wrapper.
//! 8. Zenoh telemetry keyspace.
//! 9. TOTP (RFC 6238) + `totp_setup` syscall.
//! 10. Audit chain + signed checkpoints + denial audit.
//!
//! Each phase ships as a separate PR per the agent-team plan.

#![no_std]
#![forbid(unsafe_code)]

// ─── Module skeleton ─────────────────────────────────────────────────────────
//
// These `pub mod` declarations are intentionally empty stubs in Wave 0.
// The `management-login-v1` implementation agent fills them per phase.
// Empty modules keep the public-API surface stable so downstream crates
// (`mgmt`, `ipc`, `container`) can name `auth::role::Role` etc. before
// the bodies land.

/// Three-role taxonomy: `Root`, `Operator`, `Viewer`.
///
/// Filled by `management-login-v1` Phase 2.
pub mod role {}

/// Shadow file format, parser, and atomic-rewrite helper.
///
/// Filled by `management-login-v1` Phase 2.
pub mod shadow {}

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
