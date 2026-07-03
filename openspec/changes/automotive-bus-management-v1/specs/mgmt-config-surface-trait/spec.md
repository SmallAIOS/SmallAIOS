## ADDED Requirements

### Requirement: UDS Config Surface

The change SHALL add `UdsConfigSurface`, a fourth implementation of the `ConfigSurface` trait, dispatched by `container/src/mgmt_uds.rs` to the same admin core that serves Zenoh requests. `automotive/uds.toml` SHALL join the `Config` model under the `/data/` layout. By the universal-exposure invariant, every existing `Config` option SHALL be automatically reachable over the CAN bus with no per-feature UDS plumbing.

#### Scenario: UdsConfigSurface implements ConfigSurface

- **WHEN** a UDS request reads or writes a configuration option
- **THEN** `UdsConfigSurface` SHALL serve it through the `ConfigSurface` operations
- **AND** writes SHALL pass through the same apply lifecycle as the TOML, TTY, and Zenoh surfaces

#### Scenario: Universal-exposure walker covers the UDS surface

- **WHEN** a developer adds a new `Config` field without wiring a UDS handler
- **THEN** the build-time universal-exposure walker SHALL fail, naming the field and the `uds` surface
- **AND** no per-feature UDS plumbing SHALL be required for fields the walker accepts

#### Scenario: UDS requests reach the shared admin core

- **WHEN** `container/src/mgmt_uds.rs` receives a management request over ISO-TP
- **THEN** it SHALL dispatch to the same admin core that handles Zenoh requests

### Requirement: Equivalent Control Planes Share Verbs And Audit Log

Zenoh-on-IP and UDS-on-ISO-TP SHALL be two equivalent management control planes sharing the same verbs and the same audit log. A management action performed over UDS SHALL append a record to the shared audit log; the record format, including the `uds` value of the `surface` field, is specified by the `Audit record fields` requirement of the `mgmt-audit-log` capability, which this change modifies.

#### Scenario: UDS action lands in the shared audit log

- **WHEN** a management action is performed over UDS-on-ISO-TP
- **THEN** a record SHALL be appended to the same audit log used by the TTY, Zenoh, and TOML surfaces
- **AND** the record SHALL follow the `mgmt-audit-log` record format as modified by this change

#### Scenario: Verbs are equivalent across control planes

- **WHEN** a management verb is available over the Zenoh keyspace
- **THEN** the same verb SHALL be reachable over UDS-on-ISO-TP, unless the field declares a surface-only escape hatch
