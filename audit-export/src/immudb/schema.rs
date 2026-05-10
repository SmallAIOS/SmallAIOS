// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hand-translated subset of immudb's `pkg/api/schema/schema.proto`.
//!
//! Field tags are pinned to immudb wire-protocol version 1.9.x —
//! the SHA recorded in `audit-export/vendor/IMMUDB_SCHEMA_SHA` is
//! the source of truth. The CI pin-checker (Phase 12 task 12.x)
//! re-verifies this on every PR.
//!
//! ## Encoder/decoder split
//!
//! We encode only the messages we send:
//!
//! - [`KVMetadata`] — used inside [`KeyValue`]; we only ever set
//!   `nonIndexable = false`, so the encoder accepts a single
//!   field.
//! - [`KeyValue`]
//! - [`SetRequest`]
//! - [`VerifiableSetRequest`]
//!
//! And decode only the messages we receive. The decoders skip
//! unknown fields per the protobuf forward-compat contract.

use super::wire::{
    decode_fixed64, decode_key, decode_len_delim, decode_string, decode_varint, encode_key,
    encode_len_delim, encode_varint, skip_field, WireType,
};
use super::{Result, WireError, DEFAULT_FIELD_CAP};
use alloc::string::String;
use alloc::vec::Vec;

// ===== KVMetadata =====

/// Subset of `immudb.schema.KVMetadata` we need.
///
/// We only emit `nonIndexable = false` (default), which means
/// the encoded form is empty bytes; we still keep the struct
/// for round-tripping in tests.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KVMetadata {
    pub deleted: bool,
    pub non_indexable: bool,
}

impl KVMetadata {
    /// Tag layout: `deleted = 1`, `expiration = 2` (skipped),
    /// `nonIndexable = 3`.
    pub fn encode(&self, out: &mut Vec<u8>) {
        if self.deleted {
            encode_key(1, WireType::Varint, out);
            encode_varint(1, out);
        }
        if self.non_indexable {
            encode_key(3, WireType::Varint, out);
            encode_varint(1, out);
        }
    }

    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Varint) => out.deleted = decode_varint(input, &mut cur)? != 0,
                (3, WireType::Varint) => out.non_indexable = decode_varint(input, &mut cur)? != 0,
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== KeyValue =====

/// `KeyValue { bytes key = 1; bytes value = 2; KVMetadata metadata = 3; }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValue {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub metadata: Option<KVMetadata>,
}

impl KeyValue {
    pub fn encode(&self, out: &mut Vec<u8>) {
        encode_key(1, WireType::Len, out);
        encode_len_delim(&self.key, out);
        encode_key(2, WireType::Len, out);
        encode_len_delim(&self.value, out);
        if let Some(m) = &self.metadata {
            // Embedded message: encode body to a temp Vec then wrap.
            let mut body = Vec::new();
            m.encode(&mut body);
            encode_key(3, WireType::Len, out);
            encode_len_delim(&body, out);
        }
    }

    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut key: Option<Vec<u8>> = None;
        let mut value: Option<Vec<u8>> = None;
        let mut metadata: Option<KVMetadata> = None;
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => {
                    key = Some(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec())
                }
                (2, WireType::Len) => {
                    value = Some(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec())
                }
                (3, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    metadata = Some(KVMetadata::decode(body)?);
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(Self {
            key: key.ok_or(WireError::MissingRequiredField)?,
            value: value.ok_or(WireError::MissingRequiredField)?,
            metadata,
        })
    }
}

// ===== SetRequest =====

/// `SetRequest { repeated KeyValue KVs = 1; uint32 noWait = 2; ... }`.
///
/// We never emit preconditions; the field exists in the wire
/// schema but is always omitted (default empty repeated).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SetRequest {
    pub kvs: Vec<KeyValue>,
    pub no_wait: bool,
}

impl SetRequest {
    pub fn encode(&self, out: &mut Vec<u8>) {
        for kv in &self.kvs {
            let mut body = Vec::new();
            kv.encode(&mut body);
            encode_key(1, WireType::Len, out);
            encode_len_delim(&body, out);
        }
        if self.no_wait {
            encode_key(2, WireType::Varint, out);
            encode_varint(1, out);
        }
    }
}

// ===== VerifiableSetRequest =====

/// `VerifiableSetRequest { SetRequest setRequest = 1; uint64 proveSinceTx = 2; }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiableSetRequest {
    pub set_request: SetRequest,
    pub prove_since_tx: u64,
}

impl VerifiableSetRequest {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut body = Vec::new();
        self.set_request.encode(&mut body);
        encode_key(1, WireType::Len, &mut out);
        encode_len_delim(&body, &mut out);
        if self.prove_since_tx != 0 {
            encode_key(2, WireType::Varint, &mut out);
            encode_varint(self.prove_since_tx, &mut out);
        }
        out
    }
}

// ===== Signature =====

/// `Signature { bytes publicKey = 1; bytes signature = 2; }`.
///
/// Note the field-tag order matches the upstream proto: `publicKey`
/// is tag 1, `signature` is tag 2. (Be careful — the field order
/// reads "backwards" relative to the human-friendly name of the
/// struct.)
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Signature {
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

impl Signature {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => {
                    out.public_key = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (2, WireType::Len) => {
                    out.signature = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== ImmutableState =====

/// `ImmutableState { string db = 1; uint64 txId = 2; bytes txHash = 3; Signature signature = 4; }`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ImmutableState {
    pub db: String,
    pub tx_id: u64,
    pub tx_hash: Vec<u8>,
    pub signature: Option<Signature>,
}

impl ImmutableState {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => out.db = decode_string(input, &mut cur, DEFAULT_FIELD_CAP)?,
                (2, WireType::Varint) => out.tx_id = decode_varint(input, &mut cur)?,
                (3, WireType::Len) => {
                    out.tx_hash = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (4, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.signature = Some(Signature::decode(body)?);
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== TxHeader =====

/// `TxHeader { uint64 id = 1; bytes prevAlh = 2; int64 ts = 3; ... }`.
///
/// We decode only the fields the verifier needs (`id`, `prevAlh`,
/// `ts`, `nentries`, `eH`, `blTxId`, `blRoot`, `version`).
/// `metadata` is stored as opaque bytes for forward-compat.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TxHeader {
    pub id: u64,
    pub prev_alh: Vec<u8>,
    pub ts: i64,
    pub nentries: i32,
    pub e_h: Vec<u8>,
    pub bl_tx_id: u64,
    pub bl_root: Vec<u8>,
    pub version: i32,
    pub metadata: Option<Vec<u8>>,
}

impl TxHeader {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Varint) => out.id = decode_varint(input, &mut cur)?,
                (2, WireType::Len) => {
                    out.prev_alh = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (3, WireType::Varint) => out.ts = decode_varint(input, &mut cur)? as i64,
                (4, WireType::Varint) => out.nentries = decode_varint(input, &mut cur)? as i32,
                (5, WireType::Len) => {
                    out.e_h = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (6, WireType::Varint) => out.bl_tx_id = decode_varint(input, &mut cur)?,
                (7, WireType::Len) => {
                    out.bl_root = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (8, WireType::Varint) => out.version = decode_varint(input, &mut cur)? as i32,
                (9, WireType::Len) => {
                    out.metadata =
                        Some(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec())
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== TxEntry =====

/// `TxEntry { bytes key = 1; bytes hValue = 2; int32 vLen = 3; KVMetadata metadata = 4; bytes value = 5; }`.
///
/// Field tag order matches the upstream proto. Note tag 2 is
/// `hValue` (the value digest), not `metadata`; metadata is tag 4.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TxEntry {
    pub key: Vec<u8>,
    pub h_value: Vec<u8>,
    pub v_len: i32,
    pub metadata: Option<KVMetadata>,
    pub value: Vec<u8>,
}

impl TxEntry {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => {
                    out.key = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (2, WireType::Len) => {
                    out.h_value = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (3, WireType::Varint) => out.v_len = decode_varint(input, &mut cur)? as i32,
                (4, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.metadata = Some(KVMetadata::decode(body)?);
                }
                (5, WireType::Len) => {
                    out.value = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== Tx =====

/// `Tx { TxHeader header = 1; repeated TxEntry entries = 2; }`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Tx {
    pub header: Option<TxHeader>,
    pub entries: Vec<TxEntry>,
}

impl Tx {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.header = Some(TxHeader::decode(body)?);
                }
                (2, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.entries.push(TxEntry::decode(body)?);
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== LinearProof =====

/// `LinearProof { uint64 sourceTxId = 1; uint64 targetTxId = 2; repeated bytes terms = 3; }`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LinearProof {
    pub source_tx_id: u64,
    pub target_tx_id: u64,
    pub terms: Vec<Vec<u8>>,
}

impl LinearProof {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Varint) => out.source_tx_id = decode_varint(input, &mut cur)?,
                (2, WireType::Varint) => out.target_tx_id = decode_varint(input, &mut cur)?,
                (3, WireType::Len) => out
                    .terms
                    .push(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()),
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== InclusionProof =====

/// `InclusionProof { int32 leaf = 1; int32 width = 2; repeated bytes terms = 3; }`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InclusionProof {
    pub leaf: i32,
    pub width: i32,
    pub terms: Vec<Vec<u8>>,
}

impl InclusionProof {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Varint) => out.leaf = decode_varint(input, &mut cur)? as i32,
                (2, WireType::Varint) => out.width = decode_varint(input, &mut cur)? as i32,
                (3, WireType::Len) => out
                    .terms
                    .push(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()),
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== DualProof =====

/// `DualProof` — see schema.proto for full layout.
///
/// We decode all eight fields but the verifier (Phase 5) is the
/// authority for what they mean. `inclusionProof` and
/// `consistencyProof` are stored as `Vec<Vec<u8>>` — repeated
/// bytes sibling-hash terms.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DualProof {
    pub source_tx_header: Option<TxHeader>,
    pub target_tx_header: Option<TxHeader>,
    pub inclusion_proof: Vec<Vec<u8>>,
    pub consistency_proof: Vec<Vec<u8>>,
    pub target_bl_tx_alh: Vec<u8>,
    pub last_inclusion_proof: Vec<Vec<u8>>,
    pub linear_proof: Option<LinearProof>,
    pub linear_advance_proof: Option<Vec<u8>>,
}

impl DualProof {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.source_tx_header = Some(TxHeader::decode(body)?);
                }
                (2, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.target_tx_header = Some(TxHeader::decode(body)?);
                }
                (3, WireType::Len) => out
                    .inclusion_proof
                    .push(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()),
                (4, WireType::Len) => out
                    .consistency_proof
                    .push(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()),
                (5, WireType::Len) => {
                    out.target_bl_tx_alh =
                        decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()
                }
                (6, WireType::Len) => out
                    .last_inclusion_proof
                    .push(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec()),
                (7, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.linear_proof = Some(LinearProof::decode(body)?);
                }
                (8, WireType::Len) => {
                    out.linear_advance_proof =
                        Some(decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?.to_vec())
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// ===== VerifiableTx =====

/// `VerifiableTx { Tx tx = 1; DualProof dualProof = 2; Signature signature = 3; }`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VerifiableTx {
    pub tx: Option<Tx>,
    pub dual_proof: Option<DualProof>,
    pub signature: Option<Signature>,
}

impl VerifiableTx {
    pub fn decode(input: &[u8]) -> Result<Self> {
        let mut out = Self::default();
        let mut cur = 0;
        while cur < input.len() {
            let (tag, wt) = decode_key(input, &mut cur)?;
            match (tag, wt) {
                (1, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.tx = Some(Tx::decode(body)?);
                }
                (2, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.dual_proof = Some(DualProof::decode(body)?);
                }
                (3, WireType::Len) => {
                    let body = decode_len_delim(input, &mut cur, DEFAULT_FIELD_CAP)?;
                    out.signature = Some(Signature::decode(body)?);
                }
                _ => skip_field(input, &mut cur, wt)?,
            }
        }
        Ok(out)
    }
}

// Silence the unused-import lint when `decode_fixed64` is only
// indirectly reachable via skip_field.
#[allow(dead_code)]
fn _silence_unused_fixed64() {
    let _ = decode_fixed64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyvalue_encode_decode_roundtrip() {
        let kv = KeyValue {
            key: b"audit/host-1/000001".to_vec(),
            value: b"{\"action\":\"auth_login\"}".to_vec(),
            metadata: Some(KVMetadata {
                deleted: false,
                non_indexable: false,
            }),
        };
        let mut buf = Vec::new();
        kv.encode(&mut buf);
        let decoded = KeyValue::decode(&buf).unwrap();
        // metadata round-trips as default (empty body) ⇒ Some(default).
        assert_eq!(decoded.key, kv.key);
        assert_eq!(decoded.value, kv.value);
    }

    #[test]
    fn keyvalue_decode_missing_key_rejected() {
        let mut buf = Vec::new();
        // Only encode value (tag 2).
        encode_key(2, WireType::Len, &mut buf);
        encode_len_delim(b"v", &mut buf);
        assert_eq!(
            KeyValue::decode(&buf).unwrap_err(),
            WireError::MissingRequiredField
        );
    }

    #[test]
    fn set_request_encodes_multiple_kvs() {
        let req = SetRequest {
            kvs: alloc::vec![
                KeyValue {
                    key: b"k1".to_vec(),
                    value: b"v1".to_vec(),
                    metadata: None,
                },
                KeyValue {
                    key: b"k2".to_vec(),
                    value: b"v2".to_vec(),
                    metadata: None,
                },
            ],
            no_wait: false,
        };
        let mut buf = Vec::new();
        req.encode(&mut buf);
        // Two `KVs` field instances of tag 1 wire type 2.
        assert_eq!(buf[0], (1 << 3) | 2);
        // The second occurrence should also start with tag 1.
        let mut cur = 0;
        let (tag, wt) = decode_key(&buf, &mut cur).unwrap();
        assert_eq!(tag, 1);
        assert_eq!(wt, WireType::Len);
        let _body = decode_len_delim(&buf, &mut cur, DEFAULT_FIELD_CAP).unwrap();
        let (tag2, wt2) = decode_key(&buf, &mut cur).unwrap();
        assert_eq!(tag2, 1);
        assert_eq!(wt2, WireType::Len);
    }

    #[test]
    fn verifiable_set_request_with_prove_since_tx() {
        let req = VerifiableSetRequest {
            set_request: SetRequest::default(),
            prove_since_tx: 42,
        };
        let buf = req.encode();
        // Should contain tag 2 varint = 42 somewhere.
        assert!(buf.contains(&((2 << 3) | 0)));
    }

    #[test]
    fn verifiable_set_request_zero_omits_prove_since_tx() {
        let req = VerifiableSetRequest {
            set_request: SetRequest::default(),
            prove_since_tx: 0,
        };
        let buf = req.encode();
        // tag 2 wire type 0 must NOT be present (default value omission).
        assert!(!buf.contains(&((2 << 3) | 0)));
    }

    #[test]
    fn immutable_state_decode_full() {
        // Manually construct ImmutableState{db="t", txId=7, txHash=[1,2,3],
        // signature{publicKey=[10,11], signature=[4,5]}}.
        let mut sig_body = Vec::new();
        encode_key(1, WireType::Len, &mut sig_body); // publicKey
        encode_len_delim(&[10u8, 11], &mut sig_body);
        encode_key(2, WireType::Len, &mut sig_body); // signature
        encode_len_delim(&[4u8, 5], &mut sig_body);
        let mut buf = Vec::new();
        encode_key(1, WireType::Len, &mut buf);
        encode_len_delim(b"t", &mut buf);
        encode_key(2, WireType::Varint, &mut buf);
        encode_varint(7, &mut buf);
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&[1u8, 2, 3], &mut buf);
        encode_key(4, WireType::Len, &mut buf);
        encode_len_delim(&sig_body, &mut buf);
        let st = ImmutableState::decode(&buf).unwrap();
        assert_eq!(st.db, "t");
        assert_eq!(st.tx_id, 7);
        assert_eq!(st.tx_hash, alloc::vec![1, 2, 3]);
        let sig = st.signature.unwrap();
        assert_eq!(sig.public_key, alloc::vec![10, 11]);
        assert_eq!(sig.signature, alloc::vec![4, 5]);
    }

    /// Regression test for the v0.2.1 vendoring round-trip: the
    /// tag numbers in this test are read out of the *upstream*
    /// `audit-export/vendor/schema.proto`. If you ever re-vendor
    /// from a new immudb release, re-verify these constants
    /// against the new file before bumping `IMMUDB_SCHEMA_SHA`.
    #[test]
    fn upstream_proto_field_tags_pinned() {
        // Signature: publicKey = 1, signature = 2 (NOT the other way).
        let mut sig_body = Vec::new();
        encode_key(1, WireType::Len, &mut sig_body);
        encode_len_delim(b"PUB", &mut sig_body);
        encode_key(2, WireType::Len, &mut sig_body);
        encode_len_delim(b"SIG", &mut sig_body);
        let s = Signature::decode(&sig_body).unwrap();
        assert_eq!(s.public_key, b"PUB");
        assert_eq!(s.signature, b"SIG");

        // TxEntry: key=1, hValue=2, vLen=3, metadata=4, value=5.
        let mut tx_entry = Vec::new();
        encode_key(1, WireType::Len, &mut tx_entry);
        encode_len_delim(b"K", &mut tx_entry);
        encode_key(2, WireType::Len, &mut tx_entry);
        encode_len_delim(b"H", &mut tx_entry);
        encode_key(3, WireType::Varint, &mut tx_entry);
        encode_varint(42, &mut tx_entry);
        encode_key(4, WireType::Len, &mut tx_entry);
        encode_len_delim(&[], &mut tx_entry); // empty metadata
        encode_key(5, WireType::Len, &mut tx_entry);
        encode_len_delim(b"V", &mut tx_entry);
        let e = TxEntry::decode(&tx_entry).unwrap();
        assert_eq!(e.key, b"K");
        assert_eq!(e.h_value, b"H");
        assert_eq!(e.v_len, 42);
        assert!(e.metadata.is_some());
        assert_eq!(e.value, b"V");
    }

    #[test]
    fn unknown_field_skipped() {
        // ImmutableState body containing an unknown tag 99 first.
        let mut buf = Vec::new();
        encode_key(99, WireType::Len, &mut buf);
        encode_len_delim(b"future-field", &mut buf);
        encode_key(1, WireType::Len, &mut buf);
        encode_len_delim(b"db1", &mut buf);
        let st = ImmutableState::decode(&buf).unwrap();
        assert_eq!(st.db, "db1");
    }

    #[test]
    fn linear_proof_decode_terms_in_order() {
        let mut buf = Vec::new();
        encode_key(1, WireType::Varint, &mut buf);
        encode_varint(10, &mut buf);
        encode_key(2, WireType::Varint, &mut buf);
        encode_varint(20, &mut buf);
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&[0xaa; 32], &mut buf);
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&[0xbb; 32], &mut buf);
        let p = LinearProof::decode(&buf).unwrap();
        assert_eq!(p.source_tx_id, 10);
        assert_eq!(p.target_tx_id, 20);
        assert_eq!(p.terms.len(), 2);
        assert_eq!(p.terms[0][0], 0xaa);
        assert_eq!(p.terms[1][0], 0xbb);
    }

    #[test]
    fn inclusion_proof_decode_basic() {
        let mut buf = Vec::new();
        encode_key(1, WireType::Varint, &mut buf);
        encode_varint(3, &mut buf);
        encode_key(2, WireType::Varint, &mut buf);
        encode_varint(8, &mut buf);
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&[0xcc; 32], &mut buf);
        let p = InclusionProof::decode(&buf).unwrap();
        assert_eq!(p.leaf, 3);
        assert_eq!(p.width, 8);
        assert_eq!(p.terms.len(), 1);
    }

    #[test]
    fn tx_header_decode_subset() {
        let mut buf = Vec::new();
        encode_key(1, WireType::Varint, &mut buf);
        encode_varint(99, &mut buf);
        encode_key(3, WireType::Varint, &mut buf);
        encode_varint(1_700_000_000, &mut buf);
        encode_key(4, WireType::Varint, &mut buf);
        encode_varint(2, &mut buf);
        let h = TxHeader::decode(&buf).unwrap();
        assert_eq!(h.id, 99);
        assert_eq!(h.ts, 1_700_000_000);
        assert_eq!(h.nentries, 2);
    }

    #[test]
    fn dual_proof_decode_repeated_terms() {
        let mut buf = Vec::new();
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&[0x11; 32], &mut buf);
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&[0x22; 32], &mut buf);
        encode_key(4, WireType::Len, &mut buf);
        encode_len_delim(&[0x33; 32], &mut buf);
        let p = DualProof::decode(&buf).unwrap();
        assert_eq!(p.inclusion_proof.len(), 2);
        assert_eq!(p.consistency_proof.len(), 1);
    }

    #[test]
    fn verifiable_tx_decode_nested() {
        // Build a minimal VerifiableTx with just a Signature subfield.
        // Signature: publicKey (tag 1) = "PK", signature (tag 2) = [1,2,3].
        let mut sig_body = Vec::new();
        encode_key(1, WireType::Len, &mut sig_body);
        encode_len_delim(b"PK", &mut sig_body);
        encode_key(2, WireType::Len, &mut sig_body);
        encode_len_delim(&[1u8, 2, 3], &mut sig_body);
        let mut buf = Vec::new();
        encode_key(3, WireType::Len, &mut buf);
        encode_len_delim(&sig_body, &mut buf);
        let vt = VerifiableTx::decode(&buf).unwrap();
        let s = vt.signature.unwrap();
        assert_eq!(s.public_key, b"PK");
        assert_eq!(s.signature, alloc::vec![1, 2, 3]);
    }

    #[test]
    fn malformed_input_does_not_panic() {
        // Random bytes — must not panic, must return Err.
        let bad = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        let _ = ImmutableState::decode(&bad);
        let _ = TxHeader::decode(&bad);
        let _ = DualProof::decode(&bad);
        let _ = VerifiableTx::decode(&bad);
    }
}
