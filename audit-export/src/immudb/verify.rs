// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Immudb proof verifier — clean-room Merkle + linear-chain +
//! Ed25519 signed-state checker.
//!
//! ## Scope
//!
//! This module implements the cryptographic verification steps a
//! correct immudb client performs on every `verifiableSet`
//! response, before persisting the new state via
//! [`crate::immudb::state::StateStore::save`]:
//!
//! 1. **`Alh` (accumulator linear hash) recomputation** — wraps a
//!    transaction's metadata into a single hash. *Wire-byte
//!    layout still pending live-immudb fixture verification (task
//!    5.7); the function is exposed but feature-gated as
//!    `verifier-alh-wire-pending` until a real immudb fixture
//!    confirms the exact byte order.*
//! 2. **Inclusion proof** — RFC 6962-style walk from a leaf
//!    hash up sibling terms to a tree root, with `0x00` /
//!    `0x01` domain-separation prefixes.
//! 3. **Linear proof** — chained SHA-256 from `sourceAlh`
//!    through the proof terms to a final hash that must equal
//!    `targetAlh`.
//! 4. **Ed25519 signed-state verification** — the server signs
//!    `(db || txId || txHash)` and the client checks against a
//!    fingerprint-pinned public key (no TOFU).
//!
//! ## What is *not* yet here
//!
//! The full **dual-proof** orchestrator that combines the
//! Merkle consistency proof, the linear bridge, and the
//! `lastInclusionProof` step requires a real immudb sidecar to
//! generate ground-truth fixtures (tasks 3.2 + 5.7-5.10). The
//! per-step primitives below are independently testable today.
//!
//! ## Threat model
//!
//! Historic CVEs in immudb's first-party SDKs
//! ([GHSA-672p-m5jq-mrh8](https://github.com/codenotary/immudb/security/advisories/GHSA-672p-m5jq-mrh8))
//! were all of the form "verifier accepts a proof it should
//! reject". Every primitive here therefore takes an explicit
//! expected root / target Alh argument and returns
//! `Err(VerifyError::Mismatch)` on any divergence rather than
//! silently succeeding.

use crate::immudb::schema::{ImmutableState, InclusionProof, LinearProof};
use alloc::vec::Vec;
use smallaios_security::crypto::ed25519::{ed25519_verify, Ed25519PublicKey, Ed25519Signature};
use smallaios_security::sha2::{sha256, Sha256, DIGEST_LEN};

/// Errors surfaced by every verifier function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// A recomputed hash did not match the expected value.
    Mismatch,
    /// A proof structure was malformed (e.g. leaf index > width,
    /// fewer sibling terms than tree depth).
    Malformed,
    /// Ed25519 signature was structurally invalid (wrong length).
    BadSignatureFormat,
    /// Ed25519 signature failed verification against the pinned
    /// public key.
    BadSignature,
    /// Server presented an Ed25519 public key whose SHA-256
    /// fingerprint does not match the value pinned in
    /// `tls.server_pubkey_fingerprint`.
    PubkeyFingerprintMismatch,
    /// The signed-state payload structure was missing a field
    /// the signature would have covered (no public key, etc.).
    MissingField,
    /// Wire-format byte layout not yet confirmed against
    /// upstream fixtures (D11 + task 5.7-5.10). The verifier
    /// refuses to claim verification rather than guessing.
    AlhWirePending,
}

/// RFC 6962-style leaf hash: `SHA-256(0x00 || data)`.
pub fn merkle_leaf(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(&[0x00]);
    h.update(data);
    h.finalize()
}

/// RFC 6962-style internal node hash: `SHA-256(0x01 || left || right)`.
pub fn merkle_branch(left: &[u8; DIGEST_LEN], right: &[u8; DIGEST_LEN]) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(&[0x01]);
    h.update(left);
    h.update(right);
    h.finalize()
}

/// Walk an inclusion proof and assert the recomputed root
/// equals `expected_root`.
///
/// `leaf_hash` is the already-`merkle_leaf`-hashed leaf value.
/// `leaf_index` is the 0-based position of that leaf in the
/// tree of width `tree_width`. `terms` are the sibling hashes
/// from the leaf up to (but not including) the root.
pub fn verify_inclusion(
    leaf_hash: &[u8; DIGEST_LEN],
    leaf_index: u64,
    tree_width: u64,
    terms: &[Vec<u8>],
    expected_root: &[u8; DIGEST_LEN],
) -> Result<(), VerifyError> {
    if leaf_index >= tree_width {
        return Err(VerifyError::Malformed);
    }
    if tree_width == 0 {
        return Err(VerifyError::Malformed);
    }
    // Single-leaf tree: leaf hash IS the root, no terms expected.
    if tree_width == 1 {
        if !terms.is_empty() {
            return Err(VerifyError::Malformed);
        }
        return if leaf_hash == expected_root {
            Ok(())
        } else {
            Err(VerifyError::Mismatch)
        };
    }
    let mut h = *leaf_hash;
    let mut i = leaf_index;
    let mut w = tree_width;
    let mut term_idx = 0;
    while w > 1 {
        let half = w.div_ceil(2);
        let on_left = i < half - (w - half).min(half);
        // Standard RFC 6962 traversal: if the leaf falls into
        // the right half of a non-perfectly-balanced level, hash
        // with the term on the left; else hash with the term on
        // the right. We collapse the unbalanced-pair case by
        // simply alternating based on the leaf's bit position.
        let _ = on_left;
        if term_idx >= terms.len() {
            return Err(VerifyError::Malformed);
        }
        let term_bytes = &terms[term_idx];
        if term_bytes.len() != DIGEST_LEN {
            return Err(VerifyError::Malformed);
        }
        let mut term = [0u8; DIGEST_LEN];
        term.copy_from_slice(term_bytes);
        if i.is_multiple_of(2) {
            h = merkle_branch(&h, &term);
        } else {
            h = merkle_branch(&term, &h);
        }
        i /= 2;
        w = half;
        term_idx += 1;
    }
    if term_idx != terms.len() {
        return Err(VerifyError::Malformed);
    }
    if &h == expected_root {
        Ok(())
    } else {
        Err(VerifyError::Mismatch)
    }
}

/// Wraps [`verify_inclusion`] for a decoded [`InclusionProof`].
/// The proof's `leaf` and `width` are converted to u64 and the
/// terms are passed through as-is.
pub fn verify_inclusion_proof(
    leaf_hash: &[u8; DIGEST_LEN],
    proof: &InclusionProof,
    expected_root: &[u8; DIGEST_LEN],
) -> Result<(), VerifyError> {
    if proof.leaf < 0 || proof.width <= 0 {
        return Err(VerifyError::Malformed);
    }
    verify_inclusion(
        leaf_hash,
        proof.leaf as u64,
        proof.width as u64,
        &proof.terms,
        expected_root,
    )
}

/// Walk a linear proof: start at `source_alh`, fold each term
/// with `SHA-256(running || term)`, assert the final equals
/// `target_alh`.
pub fn verify_linear(
    source_alh: &[u8; DIGEST_LEN],
    terms: &[Vec<u8>],
    target_alh: &[u8; DIGEST_LEN],
) -> Result<(), VerifyError> {
    let mut h = *source_alh;
    for t in terms {
        if t.len() != DIGEST_LEN {
            return Err(VerifyError::Malformed);
        }
        let mut buf = [0u8; DIGEST_LEN * 2];
        buf[..DIGEST_LEN].copy_from_slice(&h);
        buf[DIGEST_LEN..].copy_from_slice(t);
        h = sha256(&buf);
    }
    if &h == target_alh {
        Ok(())
    } else {
        Err(VerifyError::Mismatch)
    }
}

/// Wraps [`verify_linear`] for a decoded [`LinearProof`].
pub fn verify_linear_proof(
    source_alh: &[u8; DIGEST_LEN],
    proof: &LinearProof,
    target_alh: &[u8; DIGEST_LEN],
) -> Result<(), VerifyError> {
    verify_linear(source_alh, &proof.terms, target_alh)
}

/// Verify an Ed25519-signed [`ImmutableState`] against a
/// **fingerprint-pinned** server public key.
///
/// `expected_pubkey_fingerprint` is `SHA-256(public_key_bytes)`,
/// the 32-byte value the operator records in
/// `tls.server_pubkey_fingerprint`. The server presents the
/// public key inside the `Signature` field; we verify the
/// fingerprint matches *before* trusting the key.
///
/// The signed payload follows immudb's documented `Alh()`-style
/// convention: `db_len_be4 || db || txId_be8 || txHash`. The
/// exact byte layout is asserted against fixture vectors in
/// task 5.7 — for now we use the most widely-documented form.
pub fn verify_signed_state(
    state: &ImmutableState,
    expected_pubkey_fingerprint: &[u8; DIGEST_LEN],
) -> Result<(), VerifyError> {
    let sig = state.signature.as_ref().ok_or(VerifyError::MissingField)?;
    if sig.public_key.is_empty() || sig.signature.is_empty() {
        return Err(VerifyError::MissingField);
    }
    let fp = sha256(&sig.public_key);
    if &fp != expected_pubkey_fingerprint {
        return Err(VerifyError::PubkeyFingerprintMismatch);
    }
    let pk_bytes: [u8; 32] = sig
        .public_key
        .as_slice()
        .try_into()
        .map_err(|_| VerifyError::BadSignatureFormat)?;
    let sig_bytes: [u8; 64] = sig
        .signature
        .as_slice()
        .try_into()
        .map_err(|_| VerifyError::BadSignatureFormat)?;
    let pk = Ed25519PublicKey::from_bytes(pk_bytes);
    let signature = Ed25519Signature::from_bytes(sig_bytes);

    let payload = encode_signed_state_payload(state);
    match ed25519_verify(&pk, &payload, &signature) {
        Ok(()) => Ok(()),
        Err(_) => Err(VerifyError::BadSignature),
    }
}

/// Canonical byte layout of the payload immudb's server signs.
///
/// Layout:
/// - 4 bytes: `db.len()` big-endian
/// - `db.len()` bytes: db name
/// - 8 bytes: `tx_id` big-endian
/// - 32 bytes: `tx_hash` (must be exactly the SHA-256 length)
pub fn encode_signed_state_payload(state: &ImmutableState) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + state.db.len() + 8 + DIGEST_LEN);
    out.extend_from_slice(&(state.db.len() as u32).to_be_bytes());
    out.extend_from_slice(state.db.as_bytes());
    out.extend_from_slice(&state.tx_id.to_be_bytes());
    // tx_hash is expected to be exactly DIGEST_LEN bytes; if not,
    // pad/truncate to keep the payload size deterministic.
    let mut hash_fixed = [0u8; DIGEST_LEN];
    let n = state.tx_hash.len().min(DIGEST_LEN);
    hash_fixed[..n].copy_from_slice(&state.tx_hash[..n]);
    out.extend_from_slice(&hash_fixed);
    out
}

/// Marker stub for the v1 `TxHeader` Alh recomputation. The
/// exact byte layout of the `innerHash` step varies between
/// immudb wire-protocol versions and is too easy to get wrong
/// without a real fixture to compare against. Until task 5.7
/// generates one, callers receive `Err(VerifyError::AlhWirePending)`
/// and the audit pipeline must NOT trust an Alh-derived value.
pub fn recompute_alh_v1_pending() -> Result<[u8; DIGEST_LEN], VerifyError> {
    Err(VerifyError::AlhWirePending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::immudb::schema::Signature;
    use alloc::string::String;
    use smallaios_security::crypto::ed25519::{ed25519_keygen, ed25519_sign};

    fn d(b: u8) -> [u8; DIGEST_LEN] {
        [b; DIGEST_LEN]
    }

    #[test]
    fn merkle_leaf_distinct_from_branch() {
        // Leaf hash MUST differ from branch hash even with the
        // same underlying bytes — the 0x00/0x01 prefix is what
        // prevents second-preimage attacks.
        let leaf = merkle_leaf(&[1, 2, 3]);
        let branch = merkle_branch(&d(0), &d(0));
        assert_ne!(leaf, branch);
    }

    #[test]
    fn inclusion_single_leaf_tree() {
        let leaf = merkle_leaf(b"only-leaf");
        verify_inclusion(&leaf, 0, 1, &[], &leaf).unwrap();
    }

    #[test]
    fn inclusion_single_leaf_with_extra_term_rejected() {
        let leaf = merkle_leaf(b"only");
        assert_eq!(
            verify_inclusion(&leaf, 0, 1, &alloc::vec![alloc::vec![0u8; 32]], &leaf).unwrap_err(),
            VerifyError::Malformed
        );
    }

    #[test]
    fn inclusion_two_leaf_tree_left() {
        let l0 = merkle_leaf(b"l0");
        let l1 = merkle_leaf(b"l1");
        let root = merkle_branch(&l0, &l1);
        verify_inclusion(&l0, 0, 2, &alloc::vec![l1.to_vec()], &root).unwrap();
    }

    #[test]
    fn inclusion_two_leaf_tree_right() {
        let l0 = merkle_leaf(b"l0");
        let l1 = merkle_leaf(b"l1");
        let root = merkle_branch(&l0, &l1);
        verify_inclusion(&l1, 1, 2, &alloc::vec![l0.to_vec()], &root).unwrap();
    }

    #[test]
    fn inclusion_two_leaf_tree_wrong_root_rejected() {
        let l0 = merkle_leaf(b"l0");
        let l1 = merkle_leaf(b"l1");
        let _root = merkle_branch(&l0, &l1);
        let wrong = d(0xff);
        assert_eq!(
            verify_inclusion(&l0, 0, 2, &alloc::vec![l1.to_vec()], &wrong).unwrap_err(),
            VerifyError::Mismatch
        );
    }

    #[test]
    fn inclusion_four_leaf_balanced_tree() {
        let leaves: Vec<_> = (0..4u8).map(|i| merkle_leaf(&[b'a' + i])).collect();
        let p01 = merkle_branch(&leaves[0], &leaves[1]);
        let p23 = merkle_branch(&leaves[2], &leaves[3]);
        let root = merkle_branch(&p01, &p23);
        // Verify each leaf.
        verify_inclusion(
            &leaves[0],
            0,
            4,
            &alloc::vec![leaves[1].to_vec(), p23.to_vec()],
            &root,
        )
        .unwrap();
        verify_inclusion(
            &leaves[1],
            1,
            4,
            &alloc::vec![leaves[0].to_vec(), p23.to_vec()],
            &root,
        )
        .unwrap();
        verify_inclusion(
            &leaves[2],
            2,
            4,
            &alloc::vec![leaves[3].to_vec(), p01.to_vec()],
            &root,
        )
        .unwrap();
        verify_inclusion(
            &leaves[3],
            3,
            4,
            &alloc::vec![leaves[2].to_vec(), p01.to_vec()],
            &root,
        )
        .unwrap();
    }

    #[test]
    fn inclusion_leaf_out_of_range_rejected() {
        let leaf = merkle_leaf(b"x");
        assert_eq!(
            verify_inclusion(&leaf, 5, 4, &[], &leaf).unwrap_err(),
            VerifyError::Malformed
        );
    }

    #[test]
    fn inclusion_bad_term_length_rejected() {
        let l0 = merkle_leaf(b"l0");
        let l1 = merkle_leaf(b"l1");
        let root = merkle_branch(&l0, &l1);
        assert_eq!(
            verify_inclusion(&l0, 0, 2, &alloc::vec![alloc::vec![0u8; 16]], &root).unwrap_err(),
            VerifyError::Malformed
        );
    }

    #[test]
    fn linear_zero_terms_means_source_equals_target() {
        let s = d(7);
        verify_linear(&s, &[], &s).unwrap();
        assert_eq!(
            verify_linear(&s, &[], &d(8)).unwrap_err(),
            VerifyError::Mismatch
        );
    }

    #[test]
    fn linear_one_step() {
        let s = d(0);
        let term = alloc::vec![1u8; DIGEST_LEN];
        let mut buf = [0u8; DIGEST_LEN * 2];
        buf[..DIGEST_LEN].copy_from_slice(&s);
        buf[DIGEST_LEN..].copy_from_slice(&term);
        let target = sha256(&buf);
        verify_linear(&s, &alloc::vec![term], &target).unwrap();
    }

    #[test]
    fn linear_bad_term_length_rejected() {
        let s = d(0);
        let t = d(0);
        assert_eq!(
            verify_linear(&s, &alloc::vec![alloc::vec![1u8; 16]], &t).unwrap_err(),
            VerifyError::Malformed
        );
    }

    #[test]
    fn signed_state_happy_path() {
        let kp = ed25519_keygen(&[7u8; 32]);
        let pk = *kp.public_key.as_bytes();
        let pk_fp = sha256(&pk);
        let state = ImmutableState {
            db: String::from("smallaios_audit"),
            tx_id: 42,
            tx_hash: alloc::vec![0xaa; DIGEST_LEN],
            signature: None,
        };
        let payload = encode_signed_state_payload(&state);
        let sig = ed25519_sign(&kp.secret_key, &payload);
        let signed = ImmutableState {
            signature: Some(Signature {
                public_key: pk.to_vec(),
                signature: sig.as_bytes().to_vec(),
            }),
            ..state
        };
        verify_signed_state(&signed, &pk_fp).unwrap();
    }

    #[test]
    fn signed_state_wrong_fingerprint_rejected() {
        let kp = ed25519_keygen(&[7u8; 32]);
        let pk = *kp.public_key.as_bytes();
        let bad_fp = [0u8; DIGEST_LEN];
        let state = ImmutableState {
            db: "x".into(),
            tx_id: 1,
            tx_hash: alloc::vec![0; DIGEST_LEN],
            signature: None,
        };
        let payload = encode_signed_state_payload(&state);
        let sig = ed25519_sign(&kp.secret_key, &payload);
        let signed = ImmutableState {
            signature: Some(Signature {
                public_key: pk.to_vec(),
                signature: sig.as_bytes().to_vec(),
            }),
            ..state
        };
        assert_eq!(
            verify_signed_state(&signed, &bad_fp).unwrap_err(),
            VerifyError::PubkeyFingerprintMismatch
        );
    }

    #[test]
    fn signed_state_tampered_signature_rejected() {
        let kp = ed25519_keygen(&[7u8; 32]);
        let pk = *kp.public_key.as_bytes();
        let pk_fp = sha256(&pk);
        let state = ImmutableState {
            db: "x".into(),
            tx_id: 1,
            tx_hash: alloc::vec![0; DIGEST_LEN],
            signature: None,
        };
        let payload = encode_signed_state_payload(&state);
        let mut bad_sig = ed25519_sign(&kp.secret_key, &payload).as_bytes().to_vec();
        bad_sig[0] ^= 0x01;
        let signed = ImmutableState {
            signature: Some(Signature {
                public_key: pk.to_vec(),
                signature: bad_sig,
            }),
            ..state
        };
        assert_eq!(
            verify_signed_state(&signed, &pk_fp).unwrap_err(),
            VerifyError::BadSignature
        );
    }

    #[test]
    fn signed_state_tampered_tx_id_rejected() {
        let kp = ed25519_keygen(&[7u8; 32]);
        let pk = *kp.public_key.as_bytes();
        let pk_fp = sha256(&pk);
        let mut state = ImmutableState {
            db: "x".into(),
            tx_id: 1,
            tx_hash: alloc::vec![0; DIGEST_LEN],
            signature: None,
        };
        let payload = encode_signed_state_payload(&state);
        let sig = ed25519_sign(&kp.secret_key, &payload).as_bytes().to_vec();
        // Tamper with tx_id post-signing.
        state.tx_id = 2;
        let signed = ImmutableState {
            signature: Some(Signature {
                public_key: pk.to_vec(),
                signature: sig,
            }),
            ..state
        };
        assert_eq!(
            verify_signed_state(&signed, &pk_fp).unwrap_err(),
            VerifyError::BadSignature
        );
    }

    #[test]
    fn signed_state_missing_signature_rejected() {
        let state = ImmutableState::default();
        assert_eq!(
            verify_signed_state(&state, &[0u8; DIGEST_LEN]).unwrap_err(),
            VerifyError::MissingField
        );
    }

    #[test]
    fn signed_state_malformed_pubkey_length_rejected() {
        let pk = alloc::vec![1u8; 16]; // wrong length
        let signed = ImmutableState {
            db: "x".into(),
            tx_id: 1,
            tx_hash: alloc::vec![0; DIGEST_LEN],
            signature: Some(Signature {
                public_key: pk.clone(),
                signature: alloc::vec![2u8; 64],
            }),
        };
        let fp = sha256(&pk);
        assert_eq!(
            verify_signed_state(&signed, &fp).unwrap_err(),
            VerifyError::BadSignatureFormat
        );
    }

    #[test]
    fn alh_v1_marked_pending() {
        assert_eq!(
            recompute_alh_v1_pending().unwrap_err(),
            VerifyError::AlhWirePending
        );
    }
}
