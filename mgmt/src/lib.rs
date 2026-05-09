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
//! `management-login-v1` Q24) — that submodule is filled by Phase 10.
//!
//! ## Layer
//!
//! `mgmt` sits at **Layer 1**, peer to `auth`, `ipc`, `net`, `posix`,
//! `onnx-rt`, and `usb`. It depends on `auth` (Layer 1) and `kernel`
//! (Layer 0) and is consumed by `container` (Layer 3).
//!
//! ## Phase 5 status
//!
//! Phase 5 lands the typed `Config` struct, the `ConfigSurface` trait,
//! the validation pipeline, and the broadcast notify channel. The
//! TOML loader (`loader_toml`), Zenoh admin surface (`surface_zenoh`),
//! and audit ring (`audit`) are subsequent phases.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

/// Typed `Config` source-of-truth Rust type, the path registry that
/// drives validators and the universal-exposure invariant, and the
/// `#[reload("live"|"boot")]` field-kind annotation system.
pub mod config;

/// `ConfigSurface` trait + typed `ConfigPath`/`Value`/`Error`. Every
/// management surface (TOML, TTY, Zenoh, future UDS) implements this
/// trait over the same `Config`.
pub mod surface;

/// Per-field + cross-field validation pipeline. Validation runs at
/// step 2 of the apply lifecycle; failure leaves the on-disk file and
/// in-memory `Config` unchanged.
pub mod validate;

/// Cooperative `#![no_std]` MPMC broadcast channel for `Config` change
/// notifications. Subscribers pull on their own schedule.
pub mod notify;

/// TOML loader/saver — a `ConfigSurface` impl over `/data/*.toml`.
/// Filled by `management-login-v1` Phase 6.
pub mod loader_toml {}

/// Zenoh admin/telemetry — a `ConfigSurface` impl over the existing
/// PQC-backed Zenoh transport at `smallaios/admin/**` and
/// `smallaios/metrics/**`. Filled by `management-login-v1` Phases 7 + 8.
pub mod surface_zenoh {}

/// In-kernel audit ring with SHA-3-256 hash chain and optional
/// ML-DSA-65 signed checkpoints. Filled by `management-login-v1`
/// Phase 10.
pub mod audit {}

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use config::Config;
pub use surface::{ConfigPath, ConfigSurface, Error, Value};
