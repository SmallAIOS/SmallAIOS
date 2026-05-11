# `audit-export` CI flow

This document describes how the `verifiable-audit-log-v1`
test suite is staged across the existing SmallAIOS CI
pipeline.

## Per-PR jobs (blocking)

Run on every PR via `.github/workflows/ci.yml`:

| Job                              | What it does                                                                  |
|----------------------------------|-------------------------------------------------------------------------------|
| Format Check                     | `cargo fmt --check` over the workspace, including `audit-export/`.            |
| Clippy Lint                      | `cargo clippy -- -D warnings` against `smallaios-audit-export`.               |
| Unit Tests                       | `cargo test -p smallaios-audit-export` (135+ tests).                          |
| Fuzz Testing                     | 60 s each of `fuzz_audit_export_immudb_decode`, `fuzz_net_http2_frame`,       |
|                                  | `fuzz_net_http2_hpack`, `fuzz_net_http2_grpc`.                                |
| Cyclic Dependency Check          | `cargo-depgraph` rejects any cycle introduced via `audit-export`.             |
| API Semver Check                 | `cargo-semver-checks` against `audit-export`.                                 |
| Supply Chain Security            | `cargo-deny` advisory + license + ban over the new deps (none added).         |
| Dependency Audit                 | `cargo-vet` audit trail.                                                      |
| **Container size — feature OFF** | Build container with `--no-default-features` + every default except          |
|                                  | `audit-export`. Confirm `cargo bloat` shows zero audit-export symbols (D10    |
|                                  | Layer 1 zero-overhead invariant).                                             |
| **Container size — feature ON**  | Build with `--features audit-export` and default TOML (`enabled=false`).      |
|                                  | Confirms the runtime opt-in path also pays no I/O cost (D10 Layer 2).         |
| **Schema pin-check**             | Verify `audit-export/vendor/schema.proto` matches the upstream content at    |
|                                  | the SHA recorded in `audit-export/vendor/IMMUDB_SCHEMA_SHA`. Fails the build  |
|                                  | on drift.                                                                     |

## Nightly jobs (advisory)

Run via `.github/workflows/ci-nightly.yml`:

| Job                  | What it does                                                                 |
|----------------------|------------------------------------------------------------------------------|
| Immudb E2E sidecar   | Spins up `immudb:1.11.0` in a Docker sidecar, runs `tests/e2e_immudb.rs`,    |
|                      | asserts that the local verifier accepts the real proof traffic and that     |
|                      | `immuclient audit -d smallaios_audit` reports zero divergence.              |
| TLA+ verification    | TLC on `formal/tla/AuditExport.tla` plus the existing 19 models.            |
| Coverage             | `cargo-llvm-cov` includes `audit-export` in the workspace report. Target    |
|                      | ≥85 % line coverage on first pass.                                          |

## Why E2E is nightly, not per-PR

The immudb sidecar adds ~30 s of pull + ~10 s of startup
to every PR. Per-PR runs already cover the wire-format
contract via fixture-replay (unit tests + fuzz targets),
and the schema-pin-check catches upstream drift on every
PR. The E2E job is the empirical confirmation that the
clean-room client actually talks to a real server, which
is checked once per day and on every change to
`audit-export/src/immudb/**` paths.

## Generating proof-vector fixtures

The verifier ships with unit-test-only synthetic fixtures.
Real-world fixtures generated against a live immudb (tasks 5.7–5.10 in
`openspec/changes/verifiable-audit-log-v1/tasks.md`) require
a running immudb sidecar (any version-pinned 1.11.x
instance the developer controls) and land in
`audit-export/tests/proof_vectors/` once generated. The
harness in `audit-export/tests/scripts/gen_fixtures.rs` is
the canonical generator — a Rust binary that uses this
crate's own gRPC client against a live immudb instance and
writes the response blobs to disk. **SmallAIOS itself is
Rust-only; no Go toolchain is ever required.** The harness
produces:

- `inclusion_v1_*.bin` — ≥20 inclusion proofs across
  varying tree depths.
- `dual_v1_*.bin` — ≥10 dual-consistency proofs.
- `state_v1_*.bin` — ≥5 signed states (known good + 5 tampered
  variants).

Once these fixtures are checked into the repo, the
fixture-replay path in `audit-export/src/immudb/verify.rs`
becomes a blocking per-PR test (replaces the synthetic
fixtures used today).

## Required secrets

- `IMMUDB_E2E_DOCKER_TAG` (default `immudb:1.11.0`) — image
  the nightly job pulls.

No production credentials. The E2E job uses a throwaway
token generated inside the sidecar at startup.
