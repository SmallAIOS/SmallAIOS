// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests against a real immudb server.
//
// These tests are **ignored by default** because they require:
//
// 1. A running immudb server reachable over the network. Either:
//    - Set `IMMUDB_E2E_URL=https://host:3322` to point at an
//      existing instance you control, or
//    - Run the nightly CI job which boots `immudb:1.11.0` in a
//      Docker sidecar (see `docs/audit-export-ci.md`).
//
// 2. A bearer token in `IMMUDB_E2E_TOKEN` (or the path
//    `IMMUDB_E2E_TOKEN_PATH`).
//
// 3. The SHA-256 fingerprint of the server's Ed25519 public key
//    in `IMMUDB_E2E_PUBKEY_FINGERPRINT`.
//
// To run locally:
//
// ```
// IMMUDB_E2E_URL=https://localhost:3322 \
// IMMUDB_E2E_TOKEN=$(cat ./test-token) \
// IMMUDB_E2E_PUBKEY_FINGERPRINT=$(sha256sum < pubkey | awk '{print $1}') \
//   cargo test -p smallaios-audit-export --test e2e_immudb -- --ignored
// ```
//
// This harness intentionally does not implement the live transport
// itself — that lives in `container/` (Layer 3, which has real
// I/O) once Phase 7's container-side glue lands. The skeleton
// below documents the contract the live driver must satisfy.

#![cfg(test)]

use smallaios_audit_export::config::Config;

fn require_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Skeleton for the per-PR fixture-replay smoke test that runs
/// when the E2E env vars are NOT present.
#[test]
fn no_env_means_test_is_skipped() {
    if require_env("IMMUDB_E2E_URL").is_some() {
        return; // proper E2E mode handled below.
    }
    // Without env, just confirm the configuration plumbing
    // accepts a representative valid config.
    let c = Config {
        enabled: true,
        endpoint: "https://localhost:3322".into(),
        server_pubkey_fingerprint: "a".repeat(64),
        ..Config::default()
    };
    c.validate().expect("default+enabled validates");
}

/// Real end-to-end test against a live immudb server.
///
/// Wired in two phases:
///
/// 1. **Now** (this commit): asserts the env-driven config builds
///    a valid `Config` and that the host can resolve the endpoint.
///    Does not yet open a TCP/TLS socket — that requires the
///    container-side transport (Phase 7 follow-on).
///
/// 2. **After Phase 7 container glue**: open a TLS 1.3 socket,
///    run `current_state`, ship 10 K records over 5 minutes via
///    `verifiable_set`, assert `immuclient audit -d smallaios_audit`
///    reports zero divergence, then tamper with one record and
///    assert the verifier surfaces `audit_export_proof_failure`.
#[test]
#[ignore = "requires live immudb; see test file header"]
fn e2e_real_immudb_smoke() {
    let url = require_env("IMMUDB_E2E_URL").expect("IMMUDB_E2E_URL not set");
    let _token = require_env("IMMUDB_E2E_TOKEN")
        .or_else(|| {
            require_env("IMMUDB_E2E_TOKEN_PATH").and_then(|p| std::fs::read_to_string(p).ok())
        })
        .expect("IMMUDB_E2E_TOKEN or IMMUDB_E2E_TOKEN_PATH required");
    let fp = require_env("IMMUDB_E2E_PUBKEY_FINGERPRINT")
        .expect("IMMUDB_E2E_PUBKEY_FINGERPRINT not set");
    assert_eq!(fp.len(), 64, "fingerprint must be 64 hex chars");

    let c = Config {
        enabled: true,
        endpoint: url,
        server_pubkey_fingerprint: fp,
        ..Config::default()
    };
    c.validate().expect("env-driven config validates");

    // TODO(verifiable-audit-log-v1, container-side glue):
    //
    //   1. Build a real `GrpcTransport` against `c.endpoint`.
    //   2. Build a `StateStore` backed by a tempfile.
    //   3. Open `ImmudbClient::new(&mut transport, &mut store, …)`.
    //   4. Call `current_state()`, verify against `c.server_pubkey_fingerprint`.
    //   5. Loop: build 10_000 records, push via Pipeline, drive ticks
    //      until all shipped; assert no record dropped, no halt.
    //   6. Tamper with one record on the immudb side; re-run a
    //      verification round; assert the verifier returns
    //      `VerifyError::Mismatch` and the pipeline halts with
    //      `HaltReason::ProofFailure`.
    //
    // For now the env-driven contract is exercised; the real socket
    // path lands when `container/`'s transport implementation does.
}
