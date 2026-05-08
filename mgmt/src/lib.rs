// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Typed configuration model, `ConfigSurface` trait, and audit ring
//! for SmallAIOS.
//!
//! `mgmt` is the **single source of truth** for every operator-tunable
//! configuration field in SmallAIOS. Every reachable management
//! surface (TOML loader, TTY console, Zenoh admin, future UDS-on-CAN,
//! etc.) is a thin (de)serializer over the same typed `Config` struct.
//! There is no surface-specific configuration knob held outside
//! `Config`.
//!
//! `mgmt` also owns the in-kernel audit ring with its SHA-3-256 hash
//! chain and optional ML-DSA-65 signed checkpoints (per
//! `management-login-v1` Q24).
//!
//! ## Layer
//!
//! `mgmt` sits at **Layer 1**, peer to `auth`, `ipc`, `net`, `posix`,
//! `onnx-rt`, and `usb`. It depends on `auth` (Layer 1) and `kernel`
//! (Layer 0) and is consumed by `container` (Layer 3).
//!
//! ## Status
//!
//! This is a Wave 0 **skeleton**. Implementation is scheduled as
//! `management-login-v1` Phases 5–10 plus `embedded-filesystem-v1`
//! and `embedded-overlay-v1` config-field additions.

#![no_std]
#![forbid(unsafe_code)]

// ─── Module skeleton ─────────────────────────────────────────────────────────
//
// Empty `pub mod` placeholders in Wave 0. Each `management-login-v1`
// phase replaces one with its real implementation.

/// Typed `Config` source-of-truth Rust type, validators, and the
/// `#[reload("live"|"boot")]` attribute system.
///
/// Filled by `management-login-v1` Phase 5.
pub mod config {}

/// `ConfigSurface` trait: every management surface (TOML, TTY, Zenoh,
/// future UDS) implements this trait over the same `Config`.
///
/// Filled by `management-login-v1` Phase 5.
pub mod surface {}

/// TOML loader/saver — a `ConfigSurface` impl over `/data/*.toml`.
///
/// Filled by `management-login-v1` Phase 6.
pub mod loader_toml {}

/// Zenoh admin/telemetry — a `ConfigSurface` impl over the existing
/// PQC-backed Zenoh transport at `smallaios/admin/**` and
/// `smallaios/metrics/**`.
///
/// Filled by `management-login-v1` Phases 7 + 8.
pub mod surface_zenoh {}

/// In-kernel audit ring with SHA-3-256 hash chain and optional
/// ML-DSA-65 signed checkpoints.
///
/// Filled by `management-login-v1` Phase 10.
pub mod audit {}
