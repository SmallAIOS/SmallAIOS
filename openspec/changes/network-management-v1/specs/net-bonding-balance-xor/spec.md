## ADDED Requirements

### Requirement: L3+L4 Hash Transmit Selection

The `net` crate SHALL provide a `balance-xor` bond mode (Linux mode 2) that selects the transmit slave via a hash over L3 and L4 headers (source/destination IP and ports), giving per-flow stickiness with a deterministic tie-break. On a slave link-down, flows mapped to that slave SHALL be remapped deterministically across the remaining link-up slaves. Compatibility with static-trunked switches and MLAG pairs SHALL be documented.

#### Scenario: A flow sticks to one slave

- **WHEN** a bond in `balance-xor` mode transmits many frames belonging to one TCP flow
- **THEN** every frame of that flow SHALL egress via the same slave

#### Scenario: Hash uses L3 and L4 inputs

- **WHEN** two flows differ only in TCP source port
- **THEN** the transmit-slave hash SHALL treat them as distinct flows, allowing them to map to different slaves

#### Scenario: Link-down remaps flows deterministically

- **WHEN** the slave carrying a flow reports link down
- **THEN** the flow SHALL be remapped to a remaining link-up slave
- **AND** the remapping SHALL be deterministic for a given flow and slave set

#### Scenario: MLAG compatibility documented

- **WHEN** a reviewer reads the bonding documentation for `balance-xor`
- **THEN** it SHALL state that the mode works with static-trunked switches and MLAG pairs
