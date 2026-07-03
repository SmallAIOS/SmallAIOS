## ADDED Requirements

### Requirement: Network metrics keyspace

The telemetry keyspace SHALL gain `smallaios/metrics/network/**` publishing, per interface and per bond: link state, byte and packet counters, and DHCP lease state.

#### Scenario: Link-state change is published

- **WHEN** `eth0` transitions from link-up to link-down
- **THEN** a publication under `smallaios/metrics/network/**` SHALL report the new link state for `eth0`

#### Scenario: Byte and packet counters published

- **WHEN** a subscriber listens on `smallaios/metrics/network/**`
- **THEN** it SHALL receive per-interface byte and packet counters

#### Scenario: DHCP lease state visible

- **WHEN** an interface holds a DHCP lease
- **THEN** the lease state (including expiry) SHALL be observable under `smallaios/metrics/network/**`
