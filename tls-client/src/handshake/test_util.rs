// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for `handshake` unit tests.
//!
//! The synthetic server-flight message builders live in
//! [`crate::harness`] (one copy, also exported under the
//! `test-harness` feature for downstream integration tests); this
//! module re-exports them under their historical test-local paths
//! and keeps the test-only hex formatter.

#![cfg(test)]

pub(crate) use crate::harness::{
    build_certificate, build_certificate_verify, build_encrypted_extensions, build_server_hello,
    build_server_hello_with_session_id, wrap_handshake,
};
use alloc::string::String;

/// Lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}
