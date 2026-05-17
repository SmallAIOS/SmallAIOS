## ADDED Requirements

### Requirement: Exporter-emitted audit record types

The audit ring SHALL accept and serialize the following new `action` verbs emitted by the `audit-export/` exporter:

- `audit_export_attempt` — one record per export attempt, `code = 0` on success or the gRPC-mapped errno on failure.
- `audit_export_overflow` — emitted at most once per minute, summarizing the count of records dropped from the exporter buffer in the prior interval. Contains an `extra.dropped` integer field.
- `audit_export_proof_failure` — emitted once when proof verification fails. Contains `extra.failure_reason` (one of `signature`, `inclusion`, `consistency`, `pubkey_mismatch`).
- `audit_export_rollback_suspected` — emitted once when the server's `currentState.txId` regresses below the locally stored value. Contains `extra.local_tx_id` and `extra.remote_tx_id`.
- `audit_export_decode_failure` — emitted once when the protobuf decoder rejects a server response. Contains `extra.byte_offset`.
- `audit_export_state_init` — emitted once on cold start when no `last_state.bin` is present. Contains `extra.initial_tx_id` from the server's first `currentState`.
- `immudb_state` — emitted on every successful `VerifiableSet`. Contains `extra.tx_id`, `extra.tx_hash_hex`, `extra.signature_hex`.

All seven verbs SHALL participate in the SHA-3-256 hash chain identically to existing verbs. They SHALL NOT be subject to the denial rate limit (they are bounded by the once-per-minute or once-per-event constraints documented above).

#### Scenario: Exporter success appends two records
- **WHEN** a `VerifiableSet` completes successfully
- **THEN** the audit ring SHALL gain one `audit_export_attempt code = 0` record
- **AND** one `immudb_state` record
- **AND** both SHALL appear in the on-disk JSONL within the normal flush window

#### Scenario: Proof failure halts and audits once
- **WHEN** the verifier rejects a server response with an Ed25519 signature mismatch
- **THEN** exactly one `audit_export_proof_failure` record SHALL be appended with `extra.failure_reason = "signature"`
- **AND** no further `audit_export_attempt` records SHALL be emitted until operator ack

### Requirement: Excluded actions still chain locally

Records carrying any of the seven new actions SHALL participate in the local SHA-3-256 hash chain regardless of whether they appear in `[record_filter] exclude_actions`. Exclusion applies only to the off-box export pipeline; the local chain remains complete and verifiable on-box.

#### Scenario: Excluded record still hashed locally
- **WHEN** `audit_export_attempt` is in `exclude_actions` and one is appended
- **THEN** the record SHALL participate in the local hash chain
- **AND** the next `audit_fingerprint` publication SHALL reflect the new chain head
- **AND** the record SHALL NOT be shipped to immudb
