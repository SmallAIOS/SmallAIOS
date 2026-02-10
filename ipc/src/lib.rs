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
