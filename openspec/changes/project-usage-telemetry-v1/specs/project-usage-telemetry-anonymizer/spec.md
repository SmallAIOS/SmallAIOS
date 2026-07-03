## ADDED Requirements

### Requirement: Anonymization Before Serialization

The on-box anonymizer in `telemetry/src/usage/` SHALL run before serialization: only the anonymizer's output type SHALL be accepted by the usage-telemetry serializer, so that a serializer bug cannot leak unanonymized fields (defense-in-depth). Raw, pre-anonymization data SHALL have no path to the wire encoder.

#### Scenario: Serializer accepts only anonymized records

- **WHEN** a reviewer reads the usage-telemetry serializer's public API
- **THEN** its input type SHALL be the anonymizer's output type
- **AND** passing a raw, pre-anonymization record SHALL fail to compile

#### Scenario: Ordering asserted by test

- **WHEN** the anonymizer-before-serializer ordering test runs
- **THEN** it SHALL assert that every serialized payload was produced from an anonymizer output
- **AND** the test SHALL fail if any code path serializes data that did not pass through the anonymizer

### Requirement: On-Box Schema Validator

The `telemetry/src/usage/` module SHALL include a schema validator that checks every outgoing payload against the documented schema before transmission. The validator SHALL be unit-tested so that every allowed field is accepted and every forbidden field is rejected.

#### Scenario: Every forbidden field has a rejection test

- **WHEN** the usage-telemetry unit-test suite runs
- **THEN** there SHALL be at least one test per forbidden-field category (model info, inference content, IP addresses, user/account data, hostnames, free text, `automotive/uds.toml`, non-bonded-mode `network/*.toml`, `auth/`/`mgmt/`/`update/` config) asserting the validator rejects it

#### Scenario: Every allowed field has an acceptance test

- **WHEN** the usage-telemetry unit-test suite runs
- **THEN** there SHALL be at least one test per allowed field asserting the validator accepts it

#### Scenario: Invalid payload never transmitted

- **WHEN** the validator rejects a payload
- **THEN** the payload SHALL NOT be handed to the HTTP transport
- **AND** no partial or stripped form of the payload SHALL be transmitted in its place

### Requirement: Counter Rounding and Bucketing

The anonymizer SHALL apply the documented rounding and bucketing rules to counters before validation and serialization: the inferences-run-since-install counter SHALL be rounded to the nearest power of 2, and the unique-sessions-opened counter SHALL be rounded per the documented bucketing rule. Exact counter values SHALL NOT appear in any payload.

#### Scenario: Inference count rounded to nearest power of 2

- **WHEN** the raw inferences-since-install counter is 1000
- **THEN** the anonymized payload SHALL carry 1024
- **AND** the raw value SHALL NOT appear anywhere in the serialized output

#### Scenario: Session count is rounded

- **WHEN** the unique-sessions-opened counter is anonymized
- **THEN** the emitted value SHALL be the rounded bucket value per the documented rule
- **AND** the exact session count SHALL NOT be recoverable from the payload

### Requirement: Install-ID and Host-ID Separation

The stable per-install random ID SHALL be a UUID generated at opt-in time and persisted across reboots. It SHALL NOT be the same UUID that `telemetry-otel-export-v1` uses for `host.id`; the two SHALL be independently generated random UUIDs so that joining the two telemetry streams cannot re-identify a host. An anonymizer unit test SHALL assert the separation.

#### Scenario: Install ID generated at opt-in

- **WHEN** the operator completes the opt-in flow on a system with an empty `install_id`
- **THEN** a new random UUID SHALL be generated and written to the `install_id` field of `telemetry/usage.toml`
- **AND** the same UUID SHALL be used across subsequent reboots

#### Scenario: Install ID differs from host.id

- **WHEN** both `telemetry-otel-export-v1` (with its `host.id` UUID) and usage telemetry are configured on the same system
- **THEN** the usage-telemetry `install_id` SHALL NOT equal the `host.id` UUID
- **AND** an anonymizer unit test SHALL assert the two identifiers are distinct
