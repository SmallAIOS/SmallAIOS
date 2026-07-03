## ADDED Requirements

### Requirement: Configurable Push Interval

The exporter SHALL push on a configurable interval. The default SHALL be 10 seconds; the accepted range SHALL be 1 second to 10 minutes, configurable per-deployment via `push_interval_seconds` in `telemetry/otel.toml`.

#### Scenario: Default 10-second cadence

- **WHEN** the exporter is enabled with no `push_interval_seconds` override
- **THEN** export pushes SHALL occur every 10 seconds while the transport is healthy

#### Scenario: Out-of-range interval rejected

- **WHEN** an operator writes `push_interval_seconds = 0` or `push_interval_seconds = 601`
- **THEN** the config validator SHALL reject the value
- **AND** the previously effective interval SHALL remain in force

### Requirement: Bounded In-Memory Ring Buffer With Drop-Oldest Overflow

Unsent records SHALL be held in a bounded in-memory ring buffer, default 1 MiB, configurable via `buffer_bytes`. On overflow the buffer SHALL drop the oldest records first. Buffer overflow SHALL NOT block the producer: metric publishers and the audit ring SHALL continue unimpeded regardless of exporter state. There SHALL be no persistent on-disk WAL in v1; records unsent at reboot are lost (flagged for v2).

#### Scenario: Overflow drops oldest, keeps newest

- **WHEN** the ring buffer is at capacity and a new record is enqueued
- **THEN** the oldest buffered records SHALL be evicted to make room
- **AND** the new record SHALL be stored

#### Scenario: Producer never blocks on a full buffer

- **WHEN** the endpoint is unreachable and the ring buffer is at capacity
- **THEN** metric publishers and audit-ring appends SHALL proceed without stalling
- **AND** no producer-side call SHALL wait on exporter progress

#### Scenario: Buffer is volatile across reboot

- **WHEN** the unit reboots with unsent records in the ring buffer
- **THEN** those records SHALL NOT be exported after the reboot (in-memory only, no WAL in v1)

### Requirement: Exponential Backoff on Transport Failure

On a failed export the exporter SHALL back off exponentially: 10 s, 20 s, 40 s, doubling until a cap of 5 minutes. The ring buffer SHALL keep filling during backoff, evicting oldest records first on overflow. A successful export SHALL reset the schedule to the configured push interval.

#### Scenario: Backoff doubles to the 5-minute cap

- **WHEN** consecutive export attempts fail
- **THEN** the retry delays SHALL follow 10 s → 20 s → 40 s → 80 s → 160 s → 300 s
- **AND** subsequent delays SHALL remain capped at 300 s while failures continue

#### Scenario: Success resets the backoff

- **WHEN** an export succeeds after a period of backoff
- **THEN** the next push SHALL be scheduled at the configured `push_interval_seconds`

#### Scenario: Buffering continues during backoff

- **WHEN** the exporter is in backoff and producers keep generating records
- **THEN** new records SHALL accumulate in the ring buffer
- **AND** on overflow the oldest records SHALL be evicted first
