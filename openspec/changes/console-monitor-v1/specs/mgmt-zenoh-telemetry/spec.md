## ADDED Requirements

### Requirement: Console monitor consumed-by contract

The `smallaios/metrics/*` keyspace SHALL be the canonical schema for the console monitor's data bindings. This change SHALL NOT modify the keyspace; it SHALL add a documented consumed-by relationship: the fields consumed by the console monitor SHALL be covered by the monitor's binding regression CI test, so a regression in a published field breaks the monitor's CI test instead of going unnoticed.

#### Scenario: Consumed fields are documented

- **WHEN** a reviewer reads the telemetry keyspace documentation after this change
- **THEN** each key consumed by the console monitor SHALL carry a documented consumed-by note naming the monitor

#### Scenario: Field regression breaks the monitor's CI test

- **WHEN** a published field consumed by the console monitor is renamed or removed
- **THEN** the console monitor's binding regression CI test SHALL fail
- **AND** the telemetry keyspace itself SHALL be otherwise unchanged by this change
