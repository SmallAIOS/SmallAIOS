// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Certificate-chain cross-vector corpus replay
//! (`tls-tcp-client-v1` task 5.8).
//!
//! The DER files under `tls-client/tests/corpus/` are real
//! OpenSSL 3-generated chains — ECDSA-P256 (`ecdsa-with-SHA256`),
//! RSA-2048/3072/4096 (RSASSA-PSS, SHA-256/MGF1-SHA-256/salt 32)
//! and Ed25519 — regenerable via `tests/corpus/gen_corpus.sh`.
//! Unlike the synthetic [`super::test_certs`] fixtures (Ed25519
//! only, hand-built DER), these exercise the parser + verifier
//! against independently produced encodings, cross-validating
//! the `security-ecdsa-p256-v1` / `security-rsa-pss-v1`
//! primitives at the X.509 layer.
//!
//! Every case pins the wall clock to `NOW` (2026-09-22), inside
//! the 10-year corpus validity window and past the
//! synced-clock sentinel; the expired case (`b06`) carries a
//! 1-day certificate generated 2026-07-16.

use super::verify::TrustStoreVerifier;
use super::{LeafPublicKey, ServerCertVerifier};
use crate::trust::TrustStore;
use crate::TlsClientError;
use alloc::vec::Vec;
use smallaios_security::sha2::sha256;

/// Injected verification time: 2026-09-22T01:33:20Z.
const NOW: i64 = 1_790_000_000;

macro_rules! corpus {
    ($name:literal) => {
        include_bytes!(concat!("../../tests/corpus/", $name, ".der")).as_slice()
    };
}

/// Run one corpus case: anchors into a fresh store (optionally
/// pinned to the SHA-256 of `pin_der`), then verify `chain`
/// (leaf first) for `host`.
fn run(
    anchors: &[&[u8]],
    pin_der: Option<&[u8]>,
    chain: &[&[u8]],
    host: Option<&str>,
) -> Result<LeafPublicKey, TlsClientError> {
    let mut store = TrustStore::new();
    for der in anchors {
        store
            .add_anchor_der(der)
            .expect("corpus anchor must load as a CA");
    }
    if let Some(der) = pin_der {
        store.set_pin(sha256(der));
    }
    let verifier = TrustStoreVerifier::new(&store, NOW, true);
    let chain: Vec<Vec<u8>> = chain.iter().map(|c| c.to_vec()).collect();
    verifier.verify_chain(&chain, host)
}

/// (name, anchor, chain leaf-first, hostname) — expected to verify.
type GoodCase = (
    &'static str,
    &'static [u8],
    &'static [&'static [u8]],
    &'static str,
);

const GOOD: &[GoodCase] = &[
    (
        "g01 ecdsa direct",
        corpus!("g01_root"),
        &[corpus!("g01_leaf")],
        "g01.corpus.test",
    ),
    (
        "g02 ecdsa two-link",
        corpus!("g02_root"),
        &[corpus!("g02_leaf"), corpus!("g02_int")],
        "g02.corpus.test",
    ),
    (
        "g03 rsa2048 direct",
        corpus!("g03_root"),
        &[corpus!("g03_leaf")],
        "g03.corpus.test",
    ),
    (
        "g04 rsa2048 two-link",
        corpus!("g04_root"),
        &[corpus!("g04_leaf"), corpus!("g04_int")],
        "g04.corpus.test",
    ),
    (
        "g05 ecdsa leaf under rsa path",
        corpus!("g05_root"),
        &[corpus!("g05_leaf"), corpus!("g05_int")],
        "g05.corpus.test",
    ),
    (
        "g06 rsa leaf under ecdsa path",
        corpus!("g06_root"),
        &[corpus!("g06_leaf"), corpus!("g06_int")],
        "g06.corpus.test",
    ),
    (
        "g07 ecdsa ip san",
        corpus!("g07_root"),
        &[corpus!("g07_leaf")],
        "203.0.113.7",
    ),
    (
        "g08 rsa3072 direct",
        corpus!("g08_root"),
        &[corpus!("g08_leaf")],
        "g08.corpus.test",
    ),
    (
        "g09 rsa4096 direct",
        corpus!("g09_root"),
        &[corpus!("g09_leaf")],
        "g09.corpus.test",
    ),
    (
        "g10 ecdsa wildcard san",
        corpus!("g10_root"),
        &[corpus!("g10_leaf")],
        "a.g10.corpus.test",
    ),
    (
        "g11 openssl ed25519",
        corpus!("g11_root"),
        &[corpus!("g11_leaf")],
        "g11.corpus.test",
    ),
    // g12 (positive pin) runs in its own test — it needs set_pin.
];

/// (name, anchor, chain, hostname, expected error).
type BadCase = (
    &'static str,
    &'static [u8],
    &'static [&'static [u8]],
    &'static str,
    TlsClientError,
);

const BAD: &[BadCase] = &[
    (
        "b01 tampered ecdsa leaf signature",
        corpus!("b01_root"),
        &[corpus!("b01_leaf")],
        "b01.corpus.test",
        TlsClientError::ChainUntrusted,
    ),
    (
        "b02 tampered rsa-pss leaf signature",
        corpus!("b02_root"),
        &[corpus!("b02_leaf")],
        "b02.corpus.test",
        TlsClientError::ChainUntrusted,
    ),
    (
        "b03 anchor not in trust store",
        corpus!("b03_other"),
        &[corpus!("b03_leaf")],
        "b03.corpus.test",
        TlsClientError::ChainUntrusted,
    ),
    (
        "b04 pkcs#1 v1.5 leaf refused by policy",
        corpus!("b04_root"),
        &[corpus!("b04_leaf")],
        "b04.corpus.test",
        TlsClientError::ChainUntrusted,
    ),
    (
        "b05 hostname mismatch",
        corpus!("b05_root"),
        &[corpus!("b05_leaf")],
        "b05-wrong.corpus.test",
        TlsClientError::NameMismatch,
    ),
    (
        "b06 expired leaf",
        corpus!("b06_root"),
        &[corpus!("b06_leaf")],
        "b06.corpus.test",
        TlsClientError::Expired,
    ),
    (
        "b07 non-ca intermediate",
        corpus!("b07_root"),
        &[corpus!("b07_leaf"), corpus!("b07_notca")],
        "b07.corpus.test",
        TlsClientError::ChainUntrusted,
    ),
    (
        "b08 leaf without san",
        corpus!("b08_root"),
        &[corpus!("b08_leaf")],
        "b08.corpus.test",
        TlsClientError::BadCertificate,
    ),
    (
        "b09 leaf without serverAuth eku",
        corpus!("b09_root"),
        &[corpus!("b09_leaf")],
        "b09.corpus.test",
        TlsClientError::BadCertificate,
    ),
    // b10 (wrong pin) runs in its own test — it needs set_pin.
];

#[test]
fn corpus_meets_task_floor() {
    // Task 5.8 floor: >= 10 known-good, >= 6 known-bad. The pin
    // cases below add one to each side.
    assert!(GOOD.len() + 1 >= 10, "good corpus shrank: {}", GOOD.len());
    assert!(BAD.len() + 1 >= 6, "bad corpus shrank: {}", BAD.len());
}

#[test]
fn good_chains_verify() {
    for (name, anchor, chain, host) in GOOD {
        let got = run(&[anchor], None, chain, Some(host));
        assert!(got.is_ok(), "{name}: expected Ok, got {got:?}");
    }
}

#[test]
fn bad_chains_reject_with_expected_error() {
    for (name, anchor, chain, host, want) in BAD {
        let got = run(&[anchor], None, chain, Some(host));
        assert_eq!(got.unwrap_err(), *want, "{name}: wrong outcome");
    }
}

#[test]
fn pinned_anchor_accepts_matching_chain() {
    let got = run(
        &[corpus!("g12_root")],
        Some(corpus!("g12_root")),
        &[corpus!("g12_leaf")],
        Some("g12.corpus.test"),
    );
    assert!(got.is_ok(), "g12 pin: expected Ok, got {got:?}");
}

#[test]
fn pinned_anchor_rejects_other_anchor() {
    // Store holds b10_root (which anchors the chain), but the pin
    // names b10_other — anchoring must fail the pin check.
    let got = run(
        &[corpus!("b10_root")],
        Some(corpus!("b10_other")),
        &[corpus!("b10_leaf")],
        Some("b10.corpus.test"),
    );
    assert_eq!(got.unwrap_err(), TlsClientError::ChainUntrusted, "b10 pin");
}

#[test]
fn leaf_keys_surface_correct_algorithms() {
    // The verifier must hand back the leaf SPKI in the right
    // LeafPublicKey arm for CertificateVerify.
    let ec = run(
        &[corpus!("g01_root")],
        None,
        &[corpus!("g01_leaf")],
        Some("g01.corpus.test"),
    )
    .unwrap();
    assert!(matches!(ec, LeafPublicKey::EcdsaP256(ref p) if p.len() == 65));

    let rsa = run(
        &[corpus!("g03_root")],
        None,
        &[corpus!("g03_leaf")],
        Some("g03.corpus.test"),
    )
    .unwrap();
    assert!(matches!(rsa, LeafPublicKey::Rsa(_)));

    let ed = run(
        &[corpus!("g11_root")],
        None,
        &[corpus!("g11_leaf")],
        Some("g11.corpus.test"),
    )
    .unwrap();
    assert!(matches!(ed, LeafPublicKey::Ed25519(_)));
}
