## ADDED Requirements

### Requirement: Telemetry keyspace schema
The kernel SHALL publish runtime telemetry on the Zenoh keyspace `smallaios/metrics/<key>`. v1 keys SHALL include:

- `smallaios/metrics/cpu` — `{ ts, per_core: [{ id, util_pct }] }`
- `smallaios/metrics/mem` — `{ ts, heap_used_bytes, heap_cap_bytes, page_alloc_count, page_free_count }`
- `smallaios/metrics/inference` — `{ ts, per_model: [{ name, qps, p50_us, p99_us, error_count }] }`
- `smallaios/metrics/log` — streamed structured log records `{ ts, level, target, msg, fields }`
- `smallaios/metrics/audit_fingerprint` — the latest SHA-3-256 chain head `{ ts, hex_fingerprint, record_count }`

All payloads SHALL be JSON. The schema SHALL be additive: future fields appended; existing fields never renamed or repurposed.

#### Scenario: CPU metric round-trip
- **WHEN** a client subscribes to `smallaios/metrics/cpu` on a 4-core host
- **THEN** each received payload SHALL parse as the documented schema with `per_core` length 4

#### Scenario: Audit fingerprint streams
- **WHEN** a new audit record is appended to the chain
- **THEN** the next `smallaios/metrics/audit_fingerprint` publication SHALL contain the updated `record_count` and chain head

### Requirement: Per-key configurable cadence with adaptive option
Each metric key SHALL publish at a configurable interval, default 1 Hz. Bounds SHALL be enforced: minimum 100 ms, maximum 60 s. `mgmt/policy.toml` SHALL expose `metrics.<key>.interval_ms` per key.

An optional adaptive mode SHALL be supported per key via `metrics.<key>.adaptive = { threshold, fast_hz, slow_hz }`. When adaptive is enabled, the publisher SHALL upgrade to `fast_hz` when the chosen metric crosses `threshold` and SHALL revert to `slow_hz` when it falls below.

#### Scenario: Default 1 Hz cadence
- **WHEN** no override is configured
- **THEN** `smallaios/metrics/cpu` SHALL publish exactly once per 1000 ms ± 50 ms

#### Scenario: Operator tunes inference cadence
- **WHEN** `metrics.inference.interval_ms = 100` is set in `mgmt/policy.toml` and applied
- **THEN** `smallaios/metrics/inference` SHALL publish at 10 Hz

#### Scenario: Cadence below bound rejected
- **WHEN** `metrics.cpu.interval_ms = 50` is written to policy
- **THEN** the validator SHALL reject the value with `-EINVAL` and the previous interval SHALL remain in effect

#### Scenario: Adaptive mode upgrades on threshold cross
- **WHEN** `metrics.cpu.adaptive = { threshold: 80, fast_hz: 10, slow_hz: 1 }` is enabled
- **AND** any core's util_pct exceeds 80
- **THEN** the publisher SHALL switch to 10 Hz on `smallaios/metrics/cpu` until the metric falls back below threshold
