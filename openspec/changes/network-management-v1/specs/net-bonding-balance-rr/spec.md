## ADDED Requirements

### Requirement: Round-Robin Transmit Rotation

The `net` crate SHALL provide a `balance-rr` bond mode (Linux mode 0) that rotates transmitted frames across all link-up slaves in round-robin order, targeting static-trunked switch uplinks. A slave with link down SHALL be excluded from the rotation until its link returns. The possibility of out-of-order delivery on some workloads SHALL be documented as a caveat of this mode.

#### Scenario: Frames alternate across slaves

- **WHEN** a bond in `balance-rr` mode with slaves `eth0` and `eth1` (both link-up) transmits four frames
- **THEN** the frames SHALL alternate between `eth0` and `eth1`

#### Scenario: Link-down slave skipped

- **WHEN** `eth1` reports link down on a two-slave `balance-rr` bond
- **THEN** all subsequent frames SHALL be transmitted on `eth0`
- **AND** `eth1` SHALL rejoin the rotation after a link-up notification

#### Scenario: Ordering caveat documented

- **WHEN** a reviewer reads the bonding documentation for `balance-rr`
- **THEN** it SHALL state that out-of-order delivery is possible on some workloads
