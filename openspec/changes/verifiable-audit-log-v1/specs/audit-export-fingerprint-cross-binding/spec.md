## ADDED Requirements

### Requirement: Dual fingerprint publication

When `[exporter] enabled = true`, the `smallaios/metrics/audit_fingerprint` Zenoh keyspace publication SHALL carry both the existing local SHA-3-256 chain head and the latest immudb signed state. The JSON payload SHALL be additive over the existing schema and SHALL include the keys `local_sha3` (existing field, renamed for clarity in documentation; on-the-wire field name remains `hex_fingerprint` for backwards compatibility) and `immudb_state`.

The `immudb_state` object SHALL contain `{ db, tx_id, tx_hash_hex, signature_hex, observed_ts_ns }`. When the exporter is disabled or has never received a successful `VerifiableSet`, `immudb_state` SHALL be `null`.

#### Scenario: Exporter disabled publishes null immudb_state
- **WHEN** `[exporter] enabled = false`
- **THEN** every `audit_fingerprint` publication SHALL contain `immudb_state: null`
- **AND** the existing `hex_fingerprint` and `record_count` fields SHALL be unchanged

#### Scenario: After first successful export, immudb_state populated
- **WHEN** the exporter has successfully completed at least one `VerifiableSet`
- **THEN** subsequent `audit_fingerprint` publications SHALL contain `immudb_state.tx_id` matching the latest persisted state
- **AND** SHALL contain a non-empty `signature_hex`

### Requirement: Per-record dual digest

Every record shipped via `VerifiableSet` SHALL embed both the local SHA-3-256 digest (the source audit record's existing `hash` field) and the SHA-256 digest used by immudb's Merkle tree. The shipped value SHALL be JSON of the form:

```json
{
  "record": { /* original audit record */ },
  "local_sha3": "<64 hex chars>",
  "remote_sha256": "<64 hex chars>"
}
```

The 32-byte overhead per record (~32 B uncompressed) SHALL be considered acceptable in exchange for offline cross-verification: an auditor walking the immudb tree SHALL be able to recompute the on-box SHA-3 chain from the `local_sha3` values without contacting the box.

#### Scenario: Both digests present and self-consistent
- **WHEN** any record is decoded from the immudb tree
- **THEN** `local_sha3` SHALL equal SHA-3-256 of the canonical serialization of `record`
- **AND** `remote_sha256` SHALL equal SHA-256 of the same canonical serialization

#### Scenario: Chain head reconstructable from immudb scan
- **WHEN** an external auditor reads every shipped record from the immudb DB for one host_id ordered by `tx_id`
- **THEN** the auditor SHALL be able to recompute the local SHA-3 chain from `local_sha3` values
- **AND** SHALL match the latest `local_sha3` chain head published on `audit_fingerprint`

### Requirement: Console monitor surfaces both fingerprints

The `console-monitor-v1` `top` command SHALL render a one-line "audit export" status alongside the existing OTLP status when the exporter is enabled. The line SHALL show the immudb endpoint host (without scheme or port), the latest `tx_id`, and the age of the last successful `VerifiableSet`. When the exporter is in a halted state (proof failure or rollback suspected), the line SHALL render in the alert color scheme.

This requirement applies after `console-monitor-v1` archives. The integration is documented here so a future change archiving `console-monitor-v1` can pick up the binding without re-deriving it.

#### Scenario: Healthy export shown
- **WHEN** the exporter has shipped within the last 60 s and verification succeeded
- **THEN** the `top` header SHALL include a line like `IMMUDB ok tx=12345 last 4s`

#### Scenario: Halted export shown in alert color
- **WHEN** the exporter is halted with `audit_export_proof_failure`
- **THEN** the `top` header SHALL include `IMMUDB HALT proof_failure` in the alert color
