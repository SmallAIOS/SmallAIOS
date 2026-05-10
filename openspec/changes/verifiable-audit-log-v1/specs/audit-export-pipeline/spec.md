## ADDED Requirements

### Requirement: Audit ring fan-out tap

The `audit-export/` exporter SHALL register as a subscriber to the `mgmt-audit-log` audit ring. New records SHALL fan out to the exporter in parallel with the existing JSONL flusher and the OTLP exporter. The tap SHALL be lock-free on the producer side; an exporter that cannot keep up SHALL drop records inside its own buffer, never block the producer or delay the JSONL flusher.

#### Scenario: Producer never blocks on exporter
- **WHEN** the exporter's in-memory buffer is full and a new audit record is appended
- **THEN** the audit producer SHALL return immediately
- **AND** the oldest record in the exporter buffer SHALL be evicted
- **AND** the JSONL flusher SHALL still receive the new record on its next tick

#### Scenario: Exporter disabled = zero overhead
- **WHEN** `[exporter] enabled = false`
- **THEN** the exporter SHALL not register a tap on the audit ring
- **AND** new audit records SHALL incur no additional CPU or memory beyond the existing fan-out

### Requirement: Batching policy

Records SHALL be batched for transmission. A batch SHALL be cut and dispatched when either of two conditions first holds: (a) `batch_size` records have accumulated since the previous dispatch (default 100, range 1–10,000); or (b) `batch_interval_ms` milliseconds have elapsed since the previous dispatch (default 1000, range 100–60,000). Both bounds SHALL be operator-configurable in `immudb.toml`.

#### Scenario: Size threshold triggers dispatch
- **WHEN** 100 records have been accepted by the batcher and only 200 ms have elapsed
- **THEN** a batch of 100 records SHALL be dispatched immediately

#### Scenario: Time threshold triggers dispatch
- **WHEN** 3 records have been accepted and 1000 ms have elapsed since the previous dispatch
- **THEN** a batch of 3 records SHALL be dispatched
- **AND** the elapsed-time clock SHALL reset

#### Scenario: Empty interval emits no batch
- **WHEN** no records have been accepted in the previous 1000 ms
- **THEN** no batch SHALL be dispatched
- **AND** the connection SHALL remain idle

### Requirement: Bounded in-memory ring buffer with drop-oldest

The exporter SHALL maintain a bounded in-memory ring buffer for batched records awaiting transmission. The buffer's byte capacity SHALL be configured via `buffer_bytes` (default 4,194,304 = 4 MiB, range 1 MiB – 64 MiB). When a new record cannot fit, the oldest record SHALL be evicted to make room; the producer SHALL NOT block.

The buffer SHALL emit a `audit_export_overflow` record back into the audit ring at most once per minute summarizing the count of dropped records over the prior interval.

#### Scenario: Sustained overflow logs once per minute
- **WHEN** records are appended at 10,000/s for 5 minutes and the buffer is at 4 MiB
- **THEN** at most 5 `audit_export_overflow` records SHALL be emitted (one per minute)
- **AND** each SHALL contain the count of records dropped during that minute

#### Scenario: Buffer below capacity does not drop
- **WHEN** 1,000 records accumulate against a 4 MiB buffer and the endpoint is reachable
- **THEN** all 1,000 records SHALL be transmitted in subsequent batches
- **AND** no `audit_export_overflow` SHALL be emitted

### Requirement: Exponential backoff on transport failure

On gRPC `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED`, or connection-level errors, the exporter SHALL apply exponential backoff with an initial delay of `backoff_initial_ms` (default 10,000) and a cap of `backoff_cap_ms` (default 300,000). The delay SHALL double on each consecutive failure and reset on the first success.

`UNAUTHENTICATED` and `PERMISSION_DENIED` SHALL NOT trigger retries; the exporter SHALL emit one `audit_export_attempt` record per minute with the corresponding gRPC code and SHALL stop attempting new batches until configuration is reloaded.

#### Scenario: Backoff doubles up to cap
- **WHEN** the endpoint is unreachable for 30 minutes
- **THEN** retry attempts SHALL occur at 10 s, 20 s, 40 s, 80 s, 160 s, then 300 s (cap), then every 300 s

#### Scenario: First success resets backoff
- **WHEN** retry attempt #5 succeeds after a 160 s delay
- **THEN** the next failure SHALL begin again at 10 s

#### Scenario: UNAUTHENTICATED stops retries
- **WHEN** the endpoint returns gRPC `UNAUTHENTICATED` on the first batch
- **THEN** the exporter SHALL NOT retry
- **AND** SHALL emit one `audit_export_attempt code = -EACCES` record per minute
- **AND** SHALL resume only when the token file changes or the config is reloaded

### Requirement: Self-feedback loop prevention

Records produced by the exporter itself — `audit_export_attempt`, `audit_export_overflow`, `audit_export_proof_failure`, `audit_export_rollback_suspected`, `audit_export_decode_failure`, `audit_export_state_init`, and `immudb_state` — SHALL be excluded from export by default via the `[record_filter] exclude_actions` list. They remain visible in the local audit ring and the local JSONL file; they are simply not shipped to immudb.

An operator MAY remove items from `exclude_actions` at their own risk; the documented warning SHALL note that doing so can produce unbounded export traffic in the presence of sustained failures.

#### Scenario: Default config excludes exporter self-records
- **WHEN** a fresh `immudb.toml` is generated
- **THEN** `exclude_actions` SHALL contain at minimum `["audit_export_attempt", "audit_export_overflow", "audit_export_proof_failure", "audit_export_rollback_suspected", "audit_export_decode_failure", "audit_export_state_init", "immudb_state"]`

#### Scenario: Excluded record never enters batch
- **WHEN** an `audit_export_attempt` record is appended to the audit ring
- **AND** that action is in `exclude_actions`
- **THEN** the batcher SHALL skip it
- **AND** SHALL NOT include it in any subsequent `VerifiableSet`
