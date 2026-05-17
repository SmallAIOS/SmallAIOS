## ADDED Requirements

### Requirement: `audit_fingerprint` carries `immudb_state` sub-field

The `smallaios/metrics/audit_fingerprint` keyspace payload SHALL gain a top-level `immudb_state` field per the additive-schema rule. When the off-box exporter is disabled or has never succeeded, the field SHALL be `null`. When the exporter is enabled and has succeeded at least once, the field SHALL be an object containing `{ db, tx_id, tx_hash_hex, signature_hex, observed_ts_ns }`.

This addition SHALL NOT modify the existing `ts`, `hex_fingerprint`, or `record_count` fields.

#### Scenario: Backwards-compatible consumer ignores new field
- **WHEN** a subscriber written against the v1 `audit_fingerprint` schema receives a payload from a host running this change
- **THEN** the subscriber SHALL still parse `ts`, `hex_fingerprint`, and `record_count` correctly
- **AND** the unknown `immudb_state` field SHALL be ignorable per the additive-schema rule

#### Scenario: New consumer reads both fingerprints
- **WHEN** a subscriber written against the new schema receives a payload from a host with the exporter enabled and successful
- **THEN** it SHALL read both `hex_fingerprint` (local SHA-3) and `immudb_state.tx_hash_hex` (remote SHA-256) from the same publication
