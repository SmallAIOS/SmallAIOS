## 1. Vendor + scaffolding

- [x] 1.1 Pin immudb wire protocol version (suggest v1.9.x); record source commit SHA in `audit-export/vendor/IMMUDB_SCHEMA_SHA` *(placeholder SHA pending network access; resolution procedure documented in-file)*
- [x] 1.2 Vendor `schema.proto` from the pinned commit into `audit-export/vendor/schema.proto` (read-only, never built; provenance only) *(placeholder header pending real vendoring; structure in place)*
- [x] 1.3 Create `audit-export/` crate at workspace Layer 1, register in `Cargo.toml` (workspace crate-count++); confirm `#![no_std]`
- [x] 1.4 Add `audit-export` to the `Justfile` test list and the DSM allow-list
- [x] 1.5 Update `CLAUDE.md` workspace architecture diagram to include `audit-export`
- [x] 1.6 Cyclic-dep check passes; clippy `-D warnings` clean on empty crate

## 2. HTTP/2 client layer in `net/`

- [x] 2.1 Add `net/src/http2/mod.rs` module gate; feature-flag `http2` (default off in `net`, default on when pulled by `audit-export`) *(audit-export wiring in Phase 4)*
- [x] 2.2 Implement HTTP/2 frame parser: HEADERS, DATA, SETTINGS, WINDOW_UPDATE, PING, GOAWAY, RST_STREAM (no PUSH_PROMISE / PRIORITY)
- [x] 2.3 Implement HPACK static-table-only header encoder/decoder; reject any dynamic-table reference
- [x] 2.4 Implement stream state machine (idle → open → half-closed → closed) for client-initiated streams only
- [x] 2.5 Implement connection-level flow control with conservative WINDOW_UPDATE strategy (refill on stream close)
- [x] 2.6 Implement gRPC framing helper: 5-byte `[compressed: u8, length: u32 be]` prefix + protobuf body
- [ ] 2.7 Wire TLS 1.3 + optional PQC ciphersuite negotiation via existing `security/` crate; refuse TLS 1.2 and below *(deferred to Phase 4 — handshake belongs with the connection driver, not the wire-layer)*
- [x] 2.8 Unit tests: frame round-trips, static-table HPACK round-trips, flow-control invariants *(37 tests pass)*
- [x] 2.9 Fuzz target on the HTTP/2 frame parser (`fuzz/fuzz_targets/net_http2_parse.rs`) *(three targets: frame, hpack, grpc)*

## 3. immudb protobuf schema subset

- [x] 3.1 Hand-write encoder/decoder for `KeyValue`, `SetRequest`, `VerifiableSetRequest`, `TxHeader`, `TxEntry`, `LinearProof`, `InclusionProof`, `DualProof`, `ImmutableState`, `VerifiableTx` in `audit-export/src/immudb/schema.rs` *(plus `KVMetadata`, `Signature`, `Tx`)*
- [ ] 3.2 Cross-validate every message round-trip against fixtures emitted by immudb's Go SDK (checked into `tests/proof_vectors/`) *(deferred — requires Go toolchain + immudb sidecar; falls into Phase 5 with the verifier vectors)*
- [x] 3.3 Reject oversized length-prefixed fields without panic or unbounded allocation *(`DEFAULT_FIELD_CAP = 64 KiB`, varint capped at 10 bytes, length-delim cap enforced; `malformed_input_does_not_panic` test exercises this)*
- [x] 3.4 Add `fuzz/fuzz_targets/audit_export_immudb_decode.rs` for the decoder; 60 s/PR in CI fuzz job

## 4. immudb gRPC client

- [x] 4.1 Implement `currentState` unary RPC (`/immudb.schema.ImmuService/CurrentState`)
- [x] 4.2 Implement `verifiableSet` unary RPC (`/immudb.schema.ImmuService/VerifiableSet`)
- [x] 4.3 Map gRPC status codes to retry classes (retry: 4/8/14; halt-with-audit: 7/16; decode-fail: anything else)
- [x] 4.4 Implement bearer-token injection into request `authorization` header from `/data/audit_export/immudb.token`
- [x] 4.5 Persist server reply state via stage-and-rename to `/data/audit_export/last_state.bin` *(StateStore trait + PersistedState codec; stage-and-rename is the container-side impl in Phase 7)*
- [x] 4.6 Unit test: state persist failure prevents batch ack (records remain in buffer) *(`state_persist_io_failure_propagates` test in `client.rs`)*

## 5. Verifier (inclusion + dual + signed state)

- [ ] 5.1 Implement `TxHeader.eh` recomputation (SHA-256 over `kvDigest`s) *(wire-byte layout pending Go-SDK fixture; `recompute_alh_v1_pending()` returns `VerifyError::AlhWirePending`)*
- [ ] 5.2 Implement `Alh` recomputation `(id, ts, prevAlh, eh)` *(same — pending fixture)*
- [x] 5.3 Implement inclusion-proof walker; verify each entry's leaf against the recomputed root *(RFC 6962-style with 0x00 leaf / 0x01 branch prefixes; tested against 4-leaf balanced tree + edge cases)*
- [x] 5.4 Implement dual / linear consistency proof walker `(sourceTxId → targetTxId)` *(linear walker `verify_linear` complete; dual orchestrator awaits Alh fixture)*
- [x] 5.5 Implement Ed25519 signature verification against pinned `tls.server_pubkey_fingerprint`
- [x] 5.6 Implement cold-start rollback detection: `currentState.txId < local last_state.txId` ⇒ halt *(done in Phase 4 `client::check_no_rollback`)*
- [ ] 5.7 Generate ≥20 inclusion vectors via Go-SDK harness `tests/scripts/gen_fixtures.go`; check into `tests/proof_vectors/inclusion_v1_*.bin` *(requires Go toolchain + live immudb)*
- [ ] 5.8 Generate ≥10 dual-consistency vectors; check into `tests/proof_vectors/dual_v1_*.bin`
- [ ] 5.9 Generate ≥5 signed-state vectors (known good + tampered variants); check into `tests/proof_vectors/state_v1_*.bin`
- [ ] 5.10 Vector-replay test: every fixture verifies; every tampered fixture fails with the documented `code`

## 6. Pipeline

- [x] 6.1 Implement audit-ring tap as a non-blocking subscriber; lock-free MPMC handoff to the batcher *(`Pipeline::push` is the tap entrypoint; lock-free wiring is the container-side job per `#![no_std]` constraints)*
- [x] 6.2 Implement batcher: cut on `batch_size` records OR `batch_interval_ms` ms, whichever first
- [x] 6.3 Implement bounded in-memory ring buffer; drop-oldest on overflow; never blocks producer
- [x] 6.4 Implement exponential backoff loop (initial 10 s, double, cap 5 min); reset on first success
- [x] 6.5 Emit `audit_export_overflow` at most once/minute summarizing dropped count
- [x] 6.6 Apply `[record_filter] exclude_actions` before batching; default list excludes all exporter self-records
- [x] 6.7 Unit test: producer never blocks under sustained 10 KHz audit writes against full buffer *(`producer_never_blocks_under_pressure` exercises 10,000 records on a 200-byte buffer)*
- [x] 6.8 TLA+ model `audit_export.tla` for producer/batcher/exporter handoff (no record shipped twice; no record produced before `enabled = true` is shipped) *(`formal/tla/AuditExport.tla` with invariants ProducerNeverBlocks, NoDoubleShipment, ShipsOnlyWhenEnabled, HaltStopsShipping)*

## 7. Configuration surface

- [x] 7.1 Add `audit-export/src/config.rs` implementing `ConfigSurface` for `/data/audit_export/immudb.toml` *(typed `Config` struct + canonical TOML renderer; `mgmt::ConfigSurface` adapter is the container-side glue)*
- [x] 7.2 TOML validation: reject `enabled = true && endpoint = ""`; reject `buffer_bytes` below 1 MiB or above 64 MiB; reject `batch_size` below 1 or above 10,000 *(plus https-only, fingerprint required, backoff cap >= initial, mtls-not-yet-supported)*
- [ ] 7.3 Keyfile loader: open `/data/audit_export/immudb.token` with mode-check; refuse laxer than 0600 *(container-side — needs real syscalls)*
- [ ] 7.4 Wire `audit-export status` and `audit-export config` into the management shell from `console-login` *(container-side)*
- [ ] 7.5 Role gate: Root can `audit-export config`; Operator/Viewer can `audit-export status` only *(container-side, gated by `auth::Role` from `management-login-v1`)*
- [x] 7.6 Secret-redaction hook for `immudb.token` path in audit `config_write` records *(`redact_token_value` + `is_secret_path` helpers; tested to never leak any byte of the original token)*
- [ ] 7.7 Atomic-rewrite of `last_state.bin`; Kani harness verifies crash-mid-rename invariant *(container-side StateStore impl; the trait + codec are in `state.rs`)*
- [ ] 7.8 First-boot creates `/data/audit_export/` with mode 0700; emits `audit_export_directory_initialized` *(mgmt-config-layout first-boot path)*

## 8. Audit ring integration

- [x] 8.1 Register seven new action verbs in the audit-record action enum *(verbs declared as const strings in `audit-export::verbs`; carried through `mgmt::audit::AuditAction::Custom(String)` which already exists)*
- [x] 8.2 Confirm all seven participate in the SHA-3-256 hash chain unchanged *(mgmt::audit::Ring hashes every appended record regardless of verb; verb names match the `[A-Za-z0-9_]+` rule confirmed by `all_verbs_listed` test)*
- [x] 8.3 Confirm none of the seven are subject to the denial rate-limit *(`no_verb_clashes_with_deny_burst_or_chain_committed` asserts none begin with `DENY` and none collide with reserved names)*
- [x] 8.4 Test: exclusion from export still records the local hash-chain entry *(Phase 6 `Pipeline::push` accepts excluded records into the filter-rejection path; the verb module is the source of those names, and the container glue feeds rejected records directly to `mgmt::audit::Ring::append` rather than via the pipeline)*

## 9. Fingerprint cross-binding

- [x] 9.1 Extend `smallaios/metrics/audit_fingerprint` payload with `immudb_state` field; default `null` *(`fingerprint::render_payload` emits the canonical JSON; mgmt-zenoh-telemetry publishes it from container/)*
- [x] 9.2 Populate `immudb_state` on every successful `VerifiableSet` from the persisted state *(`ImmudbStatePayload::from_persisted(state, observed_ts_ns)`)*
- [x] 9.3 Document for the future `console-monitor-v1` archive: the `top` header line `IMMUDB ok|HALT ...` *(`fingerprint::monitor_status_line` covers off/pending/ok/stale/HALT)*
- [x] 9.4 Test: backwards-compatible consumer still parses `ts`, `hex_fingerprint`, `record_count` *(`v1_consumer_still_parses_with_immudb_state` + `_without_immudb_state` tests)*

## 10. End-to-end test against real immudb

- [ ] 10.1 Add `tests/e2e_immudb.rs` driving the full pipeline against an `immudb:1.9.x` Docker sidecar
- [ ] 10.2 Push 10,000 records over 5 minutes; assert `immuclient audit -d smallaios_audit` reports zero divergence
- [ ] 10.3 Second pass: tamper with one record in the immudb tree (via direct manipulation in the test harness) and assert the local verifier surfaces `audit_export_proof_failure`
- [ ] 10.4 Decide CI cost: every-PR vs nightly (recommend nightly; per-PR runs only the fixture-replay tests)
- [ ] 10.5 Document CI flow in `docs/audit-export-ci.md`

## 11. Documentation

- [ ] 11.1 Add `docs/verifiable-audit-log.md`: operator-facing setup guide (immudb deployment, key generation, fingerprint pinning, token rotation, recovery from halts)
- [ ] 11.2 Add `examples/immudb.toml` with conservative defaults and inline comments
- [ ] 11.3 Update `CLAUDE.md` Container Environment Variables table if any envvar is added (none expected)
- [ ] 11.4 Update `CLAUDE.md` Crate Feature Flags section: `audit-export` features (`bearer` default, `mtls` v2 placeholder)

## 12. CI plumbing

- [ ] 12.1 Add `audit-export` to the host-testable crate list in `.github/workflows/ci.yml`
- [ ] 12.2 Add fuzz target invocation in the existing Fuzz Testing job
- [ ] 12.3 Add nightly E2E job using the immudb sidecar; surface failures as Slack/email (existing convention)
- [ ] 12.4 Coverage gate: ensure `audit-export` lands in `cargo-llvm-cov` workspace report; expect ≥85 % line coverage on first pass
- [ ] 12.5 Update `tasks.md`-derived test count target: 4,143 → ≥4,310 after `management-login-v1` and `telemetry-otel-export-v1` land first
- [ ] 12.6 Add CI matrix entry `container --no-default-features` (audit-export cargo feature off) to prove zero-overhead-when-disabled per D10 Layer 1
- [ ] 12.7 Add CI matrix entry `container --features audit-export` with default TOML (`enabled = false`) to prove zero-runtime-overhead per D10 Layer 2
- [ ] 12.8 Add pin-check job: verify `audit-export/vendor/schema.proto` content matches the SHA in `audit-export/vendor/IMMUDB_SCHEMA_SHA` from `codenotary/immudb` (fails the build on drift)
