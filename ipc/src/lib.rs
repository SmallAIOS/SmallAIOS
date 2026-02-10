// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS IPC — Zenoh-Inspired Messaging
//!
//! Key-expression-based pub/sub and request/reply messaging:
//! - Hierarchical key expressions with wildcard matching
//! - Publisher/Subscriber for fire-and-forget data distribution
//! - Queryable for request/reply patterns
//! - Shared memory transport (intra-kernel, zero-copy)
//! - TCP transport (external clients, Zenoh wire protocol compatible)
//! - TLS 1.3 with post-quantum key exchange (ML-KEM-768 hybrid)
//! - Built-in endpoints: health, metrics, inference, logs

#![no_std]
#![allow(dead_code)]

extern crate alloc;

pub mod endpoints;
pub mod http;
pub mod inference_proto;
pub mod key_expr;
pub mod pubsub;
pub mod reqrep;
pub mod router;
pub mod shm;
pub mod tcp_transport;
pub mod tls;

/// IPC error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcError {
    /// Key expression is syntactically invalid.
    InvalidKeyExpression,
    /// Subscription or routing table is full.
    TableFull,
    /// Requested resource was not found.
    NotFound,
    /// Message buffer is at capacity.
    BufferFull,
    /// Invalid state for this operation.
    InvalidState,
    /// Message exceeds maximum allowed size.
    MessageTooLarge,
    /// Packet too short to decode.
    PacketTooShort,
    /// Destination buffer too small.
    BufferTooSmall,
    /// Invalid or malformed header.
    InvalidHeader,
    /// Protocol magic or version mismatch.
    InvalidProtocol,
}
