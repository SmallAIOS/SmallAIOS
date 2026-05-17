// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Persistent immudb-state storage trait + on-disk format.
//!
//! The exporter remembers the most-recent signed state from the
//! immudb server so a cold-start can demand a consistency proof
//! from that state forward (D5 in design.md). The state file
//! lives at `/data/audit_export/last_state.bin` per the
//! `audit-export-config-surface` spec and is written via
//! stage-and-rename (D6 + Kani harness in task 7.7).
//!
//! `audit-export` is `#![no_std]` so the actual file I/O lives
//! in `container/`. This module provides:
//!
//! - [`PersistedState`] — the in-memory record of what we last
//!   shipped, plus serialize/deserialize for the disk format.
//! - [`StateStore`] — the trait the integration layer
//!   implements with real filesystem calls.

use alloc::string::String;
use alloc::vec::Vec;

/// Magic header bytes for the on-disk format. Lets the loader
/// reject a file that is not ours (e.g. a stale leftover from a
/// previous, incompatible binary).
pub const MAGIC: [u8; 8] = *b"SAIOSAEX";

/// Format version. Bump when the layout changes.
pub const FORMAT_VERSION: u8 = 1;

/// Errors the codec may surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateCodecError {
    Truncated,
    BadMagic,
    UnsupportedVersion,
    OversizedField,
    InvalidUtf8,
}

/// Errors the [`StateStore`] trait surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateStoreError {
    /// No `last_state.bin` was present (cold start).
    NotFound,
    /// Decode of the stored file failed.
    Corrupt,
    /// Filesystem I/O error.
    Io,
}

/// What we cache on disk after each successful `verifiableSet`.
///
/// Stored separately from the immudb wire response so that this
/// crate need not store every field of `ImmutableState`. Only
/// the fields the verifier needs to demand a consistency proof
/// from the next session are persisted.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PersistedState {
    pub db: String,
    pub tx_id: u64,
    pub tx_hash: Vec<u8>,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Per-field cap when decoding. Matches `DEFAULT_FIELD_CAP` in
/// the protobuf wire layer so a corrupt file cannot trigger an
/// allocation spike.
const MAX_FIELD_LEN: usize = 64 * 1024;

impl PersistedState {
    /// Serialize to the canonical on-disk byte form.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(FORMAT_VERSION);
        out.extend_from_slice(&[0u8; 7]); // reserved
        out.extend_from_slice(&self.tx_id.to_le_bytes());
        encode_blob(self.tx_hash.as_slice(), &mut out);
        encode_blob(self.public_key.as_slice(), &mut out);
        encode_blob(self.signature.as_slice(), &mut out);
        encode_blob(self.db.as_bytes(), &mut out);
        out
    }

    /// Decode from the canonical on-disk byte form.
    pub fn decode(input: &[u8]) -> Result<Self, StateCodecError> {
        if input.len() < MAGIC.len() + 1 + 7 + 8 {
            return Err(StateCodecError::Truncated);
        }
        if input[..MAGIC.len()] != MAGIC {
            return Err(StateCodecError::BadMagic);
        }
        if input[MAGIC.len()] != FORMAT_VERSION {
            return Err(StateCodecError::UnsupportedVersion);
        }
        let mut cur = MAGIC.len() + 1 + 7;
        let mut tx_id_bytes = [0u8; 8];
        tx_id_bytes.copy_from_slice(&input[cur..cur + 8]);
        cur += 8;
        let tx_id = u64::from_le_bytes(tx_id_bytes);
        let tx_hash = decode_blob(input, &mut cur)?;
        let public_key = decode_blob(input, &mut cur)?;
        let signature = decode_blob(input, &mut cur)?;
        let db_bytes = decode_blob(input, &mut cur)?;
        let db = core::str::from_utf8(&db_bytes)
            .map_err(|_| StateCodecError::InvalidUtf8)?
            .into();
        Ok(Self {
            db,
            tx_id,
            tx_hash,
            public_key,
            signature,
        })
    }
}

fn encode_blob(bytes: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn decode_blob(input: &[u8], cur: &mut usize) -> Result<Vec<u8>, StateCodecError> {
    if *cur + 4 > input.len() {
        return Err(StateCodecError::Truncated);
    }
    let len = u32::from_le_bytes([
        input[*cur],
        input[*cur + 1],
        input[*cur + 2],
        input[*cur + 3],
    ]) as usize;
    *cur += 4;
    if len > MAX_FIELD_LEN {
        return Err(StateCodecError::OversizedField);
    }
    if *cur + len > input.len() {
        return Err(StateCodecError::Truncated);
    }
    let v = input[*cur..*cur + len].to_vec();
    *cur += len;
    Ok(v)
}

/// What the `container/` layer implements to give us durable
/// state across reboots. Implementations MUST honor the
/// stage-and-rename atomicity rule.
pub trait StateStore {
    /// Load the persisted state, returning `Ok(None)` on cold
    /// start (no file) and `Err(Corrupt)` on a malformed file.
    fn load(&self) -> Result<Option<PersistedState>, StateStoreError>;

    /// Persist a new state. Implementations MUST stage to a
    /// `.tmp` sibling, fsync the body, then atomically rename.
    /// A crash mid-write MUST leave the previous valid state
    /// in place (or absent, for a cold-start).
    fn save(&mut self, state: &PersistedState) -> Result<(), StateStoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PersistedState {
        PersistedState {
            db: "smallaios_audit".into(),
            tx_id: 42,
            tx_hash: alloc::vec![0xaa; 32],
            public_key: alloc::vec![0xbb; 32],
            signature: alloc::vec![0xcc; 64],
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let s = sample();
        let bytes = s.encode();
        let back = PersistedState::decode(&bytes).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn magic_check_rejects_unrelated_file() {
        let mut bytes = sample().encode();
        bytes[0] = b'X';
        assert_eq!(
            PersistedState::decode(&bytes).unwrap_err(),
            StateCodecError::BadMagic
        );
    }

    #[test]
    fn version_check_rejects_future_format() {
        let mut bytes = sample().encode();
        bytes[MAGIC.len()] = 99;
        assert_eq!(
            PersistedState::decode(&bytes).unwrap_err(),
            StateCodecError::UnsupportedVersion
        );
    }

    #[test]
    fn truncated_input_rejected() {
        let bytes = sample().encode();
        for cut in 1..bytes.len() {
            let r = PersistedState::decode(&bytes[..cut]);
            assert!(r.is_err(), "decode should fail at truncation {cut}");
        }
    }

    #[test]
    fn oversized_field_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.extend_from_slice(&[0u8; 7]);
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tx_id
                                                      // tx_hash with declared length 1 GiB
        bytes.extend_from_slice(&(1u32 << 30).to_le_bytes());
        assert_eq!(
            PersistedState::decode(&bytes).unwrap_err(),
            StateCodecError::OversizedField
        );
    }

    #[test]
    fn empty_state_round_trips() {
        let s = PersistedState::default();
        let bytes = s.encode();
        let back = PersistedState::decode(&bytes).unwrap();
        assert_eq!(back, s);
    }
}
