// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tamper-evident audit logging subsystem.
//!
//! Provides structured audit logging with:
//! - Typed event taxonomy (capability, authentication, resource, inference, system)
//! - Lock-free ring buffer for non-blocking event collection
//! - RFC 5424 syslog serialization
//! - SHA-3-256 hash chain for tamper evidence

pub mod accumulator;
pub mod batch_signing;
pub mod entry;
pub mod integrity;
pub mod ipc_export;
#[cfg(test)]
mod mcdc_tests;
pub mod retention;
pub mod ring_buffer;
pub mod syslog;
pub mod taxonomy;

pub use accumulator::BatchAccumulator;
pub use entry::{AuditEntry, AuditResult, Operation};
pub use integrity::IntegrityChain;
pub use ring_buffer::AuditRingBuffer;
pub use taxonomy::{AuditEventType, EventCategory, EventSubcategory};
