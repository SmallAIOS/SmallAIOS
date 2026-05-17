## ADDED Requirements

### Requirement: Clean-room immudb gRPC client over HTTP/2 + TLS 1.3

The `audit-export/` crate SHALL provide a `#![no_std]` immudb client that speaks gRPC over HTTP/2 over TLS 1.3, without taking a dependency on `tokio`, `tonic`, `hyper`, `prost`, or any other `std`-only library. The client SHALL implement exactly the gRPC subset required for length-prefixed protobuf framing (5-byte length prefix + protobuf body) with end-of-stream signaling via `END_STREAM` HTTP/2 flag, and SHALL reject server push, stream priority, trailers-only, and dynamic-table HPACK on the inbound side.

The client SHALL pin to immudb wire protocol version 1.9.x. The protobuf message subset SHALL be hand-written from a checked-in `schema.proto` snapshot whose source commit SHA is recorded in `audit-export/vendor/IMMUDB_SCHEMA_SHA`.

#### Scenario: TLS 1.2 handshake rejected
- **WHEN** the configured immudb endpoint negotiates TLS 1.2 only
- **THEN** the client SHALL abort the handshake before sending any application data
- **AND** SHALL emit `audit_export_attempt` with `code = -EPROTONOSUPPORT`

#### Scenario: PQC hybrid offered first
- **WHEN** `tls.require_pqc = true` is set
- **THEN** the client's ClientHello SHALL list `X25519+ML-KEM-768` as the first key share
- **AND** SHALL refuse pure-classical fallback

#### Scenario: HTTP/2 server push refused
- **WHEN** the server sends a `PUSH_PROMISE` frame
- **THEN** the client SHALL respond `GOAWAY` with `PROTOCOL_ERROR`
- **AND** SHALL drop the connection

### Requirement: Immudb `verifiableSet` write path

Each export batch SHALL be sent via the immudb `VerifiableSet` RPC at path `/immudb.schema.ImmuService/VerifiableSet`. The request SHALL include `proveSinceTx` set to the last-known transaction id loaded from `/data/audit_export/last_state.bin`, or `0` on first-time use. The reply SHALL be parsed as a `VerifiableTx` protobuf message containing the new signed state and the dual proof.

Successful writes SHALL atomically replace `/data/audit_export/last_state.bin` with the new state via stage-and-rename. Failure to durably persist the new state SHALL prevent acknowledging the batch upstream, causing the records to remain in the in-memory buffer for retry.

#### Scenario: First-ever export uses proveSinceTx = 0
- **WHEN** the exporter starts on a fresh `/data/audit_export/` with no `last_state.bin`
- **THEN** the first `VerifiableSet` request SHALL carry `proveSinceTx: 0`
- **AND** the response SHALL be recorded with audit `action = "audit_export_state_init"`

#### Scenario: Subsequent batches use last-known tx id
- **WHEN** `/data/audit_export/last_state.bin` contains `{ txId: 42 }` at batch time
- **THEN** the next `VerifiableSet` request SHALL carry `proveSinceTx: 42`

#### Scenario: State persist failure prevents batch ack
- **WHEN** `VerifiableSet` succeeds but writing `last_state.bin` returns I/O error
- **THEN** the exporter SHALL keep the batch's records in the buffer
- **AND** SHALL retry the same batch with the same `proveSinceTx` on the next attempt

### Requirement: Inclusion + dual consistency + signed-state verification

Every `VerifiableTx` response SHALL be verified before its state is persisted. Verification SHALL include: (a) recomputing `TxHeader.eh` from entries, (b) recomputing `Alh`, (c) walking the inclusion proof for each record in the batch, (d) verifying the dual / linear consistency proof from the previously-stored state to the new state, and (e) verifying the Ed25519 signature over `(db, txId, txHash)` against the server public-key fingerprint configured in `tls.server_pubkey_fingerprint`.

Any verification step that fails SHALL cause the exporter to halt, emit an `audit_export_proof_failure` record with the failure reason in `code`, and refuse to advance until the operator explicitly acks. The Ed25519 trust anchor SHALL NOT be TOFU-trusted; absence of `tls.server_pubkey_fingerprint` SHALL cause the exporter to refuse to start.

#### Scenario: Tampered Ed25519 signature rejected
- **WHEN** the server's signed state has a signature that does not match the configured fingerprint
- **THEN** the verifier SHALL return failure
- **AND** the exporter SHALL halt with `audit_export_proof_failure code = -EBADMSG`

#### Scenario: Inclusion proof mismatch rejected
- **WHEN** the response includes an inclusion proof whose recomputed leaf hash differs from the entry's `kvDigest`
- **THEN** the verifier SHALL return failure
- **AND** the exporter SHALL halt with `audit_export_proof_failure code = -EBADE`

#### Scenario: Missing pubkey fingerprint refuses to start
- **WHEN** `[exporter] enabled = true` and `[tls] server_pubkey_fingerprint = ""` are configured together
- **THEN** boot of the exporter SHALL fail with a fatal log line
- **AND** no network connection SHALL be attempted

### Requirement: Cold-start rollback detection

On startup the exporter SHALL load `last_state.bin`, then issue `currentState` against the configured endpoint. If the server's reported `txId` is strictly less than the locally stored `txId`, the exporter SHALL treat the situation as a tampering signal: emit `audit_export_rollback_suspected` to the audit ring, refuse to ship further records, and require the operator to explicitly clear the failure via the management surface.

#### Scenario: Server txId behind local txId halts exporter
- **WHEN** `last_state.bin` records `txId = 1000` and the server replies with `currentState.txId = 500`
- **THEN** the exporter SHALL emit `audit_export_rollback_suspected`
- **AND** SHALL not issue any subsequent `VerifiableSet`
- **AND** the buffered records SHALL remain pending until operator ack

#### Scenario: Server txId equal or ahead accepted
- **WHEN** `last_state.bin` records `txId = 1000` and the server replies with `currentState.txId = 1042`
- **THEN** the exporter SHALL request a consistency proof from txId 1000 to 1042
- **AND** SHALL proceed once that proof verifies

### Requirement: Protobuf decoder fuzz target

The protobuf decoder for the immudb message subset SHALL ship with a `cargo-fuzz` target at `fuzz/fuzz_targets/audit_export_immudb_decode.rs`. The target SHALL execute under the existing CI fuzz job for at least 60 seconds per PR. Decoder panics, out-of-bounds reads, or unbounded allocations SHALL be treated as build-blocking defects.

#### Scenario: Decoder rejects oversized field
- **WHEN** a crafted input declares a length-prefixed field larger than the remaining buffer
- **THEN** the decoder SHALL return an error
- **AND** SHALL NOT panic or allocate beyond the input length
