// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room immudb gRPC client subset.
//!
//! This module ships the wire-level pieces of the
//! `audit-export-immudb-client` capability:
//!
//! - [`wire`] — protobuf wire-format primitives (varint,
//!   length-delimited, fixed-width). Robust against malformed
//!   input: every decoder rejects, never panics, never
//!   allocates beyond the input length.
//! - [`schema`] — hand-translated subset of immudb's
//!   `pkg/api/schema/schema.proto`. Encode messages we send
//!   (`KeyValue`, `SetRequest`, `VerifiableSetRequest`); decode
//!   messages we receive (`ImmutableState`, `Signature`, `Tx`,
//!   `TxHeader`, `TxEntry`, `LinearProof`, `InclusionProof`,
//!   `DualProof`, `VerifiableTx`).
//!
//! Phases 4 (gRPC client) and 5 (verifier) build on top of
//! these.

pub mod client;
pub mod schema;
pub mod state;
pub mod transport;
pub mod wire;

/// Errors surfaced by the immudb wire-format codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    /// Input ended in the middle of a varint or length-delimited body.
    Truncated,
    /// A varint exceeded the maximum 10-byte u64 representation.
    VarintOverflow,
    /// A length-delimited field declared a length larger than the
    /// remaining input or larger than the cap configured by the
    /// caller.
    OversizedField,
    /// Wire-type bits in a field key were not one of the four
    /// values we accept (0, 1, 2, 5).
    UnknownWireType,
    /// A `string` field contained bytes that were not valid UTF-8.
    InvalidUtf8,
    /// A required field of a message was absent on decode.
    MissingRequiredField,
    /// A field's wire type did not match the expected type for its tag.
    WireTypeMismatch,
}

/// Result alias for immudb wire-format operations.
pub type Result<T> = core::result::Result<T, WireError>;

/// Default cap on a single length-delimited field's body. Immudb
/// transactions can carry large `eH` / `prevAlh` digests but
/// nothing single-field on this code path is larger than ~64 KiB.
/// The cap is enforced on every length-delimited decode.
pub const DEFAULT_FIELD_CAP: usize = 64 * 1024;
