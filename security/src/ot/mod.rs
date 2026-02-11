// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! OT/ICS security hardening subsystem.
//!
//! Provides:
//! - WCET instrumentation and measurement for critical kernel paths
//! - Fail-safe state definitions and enforcement
//! - OT-specific anomaly detection (output range, timing, NaN/Inf)
//! - Safe shutdown procedure with bounded time

pub mod anomaly;
pub mod failsafe;
pub mod shutdown;
pub mod wcet;
pub mod wcet_bounds;
