// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![forbid(unsafe_code)]

//! TLS 1.3 over TCP client for SmallAIOS.
//!
//! This crate implements `verifiable-audit-log-v1`'s deferred
//! `TlsStreamLike` impl, plus everything any future HTTPS-
//! speaking client needs. See `openspec/changes/tls-tcp-client-v1/`
//! for the full design.
//!
//! ## Layer
//!
//! `tls-client` sits at **Layer 1**: it depends on `security`
//! (SHA-256/384, Ed25519, AEAD, HKDF) and `kernel` (`clock()`
//! for cert-validity-window checks). It is consumed by
//! `container` when the `audit-export` feature is enabled.
//!
//! ## Capabilities
//!
//! - `tls-client-record-layer`: RFC 8446 § 5 record framing
//!   with strict size caps + AEAD wrap/unwrap.
//! - `tls-client-handshake`: ClientHello / ServerHello /
//!   EncryptedExtensions / Certificate / CertificateVerify /
//!   Finished state machine.
//! - `tls-client-cert-chain`: minimal X.509v3 parser +
//!   trust-store-anchored chain verification + RFC 6125
//!   hostname matcher.
//! - `tls-client-trust-store`: operator-controlled PEM
//!   bundle loader with optional per-CA SHA-256 pinning.
//!
//! ## Status
//!
//! Phase 1 of `tls-tcp-client-v1` — this scaffold ships the
//! crate boundary and the typed error enum. Phases 3-8
//! populate the modules referenced below.

extern crate alloc;

pub mod record;
// pub mod handshake;  // Phase 4
// pub mod cert;       // Phase 5
// pub mod trust;      // Phase 6
// pub mod std_io;     // Phase 7

/// Errors surfaced by the TLS client.
///
/// Each variant maps to a deterministic
/// `container::audit_export_runtime::transport::TransportError`
/// classification when the audit-export pipeline drives this
/// client: TcpConnect / Io / BadHandshake → `Retry`; every
/// other variant → `HardFail`. The audit pipeline never
/// retries a chain-untrusted or signature-failure error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlsClientError {
    /// TCP-layer connect failed.
    TcpConnect,
    /// Read/write returned an I/O error.
    Io,
    /// Record header malformed or oversized.
    BadRecord,
    /// Peer advertised TLS < 1.3 anywhere in the handshake.
    Version,
    /// Handshake message malformed or out of order.
    BadHandshake,
    /// Hybrid / classical key-exchange math failed.
    KeyExchange,
    /// Certificate DER parse failed, SHA-1 signature, no SAN, etc.
    BadCertificate,
    /// No chain to a trust-store anchor.
    ChainUntrusted,
    /// Leaf or intermediate `notAfter` in the past.
    Expired,
    /// SAN does not include the operator's hostname.
    NameMismatch,
    /// `require_pqc = true` and peer rejected hybrid.
    PqcDowngrade,
    /// AEAD decryption / authentication failed.
    Aead,
}

/// Result alias for TLS-client functions.
pub type Result<T> = core::result::Result<T, TlsClientError>;

/// Crate-level version string, derived from the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
