## ADDED Requirements

### Requirement: Longest-Prefix-Match Route Table

The `net` crate SHALL provide a route table (`net/src/route.rs`) replacing the previous default-route-only assumption. Route selection SHALL be by longest prefix match first, then lowest metric, then interface preference.

#### Scenario: More specific prefix wins

- **WHEN** the table contains a default route via `eth0` and a `10.1.0.0/16` route via `eth1`
- **AND** a packet is sent to `10.1.2.3`
- **THEN** the packet SHALL egress via `eth1`

#### Scenario: Metric breaks a prefix tie

- **WHEN** the table contains two default routes, metric 100 via `eth0` and metric 200 via `eth1`
- **THEN** packets with no more-specific match SHALL egress via `eth0`

#### Scenario: Interface preference breaks a metric tie

- **WHEN** two routes match with equal prefix length and equal metric
- **THEN** the table SHALL select one deterministically by interface preference
- **AND** repeated lookups for the same destination SHALL return the same route

### Requirement: Per-Flow ECMP Across Equal-Metric Routes

When multiple equal-metric default routes exist, the route table SHALL perform Equal-Cost Multi-Path selection using a per-flow hash so that all packets of a given flow (same TCP connection) always take the same uplink.

#### Scenario: A TCP connection sticks to one uplink

- **WHEN** two equal-metric default routes exist via `eth0` and `eth1`
- **AND** a TCP connection sends many segments
- **THEN** every segment of that connection SHALL egress via the same uplink

#### Scenario: Distinct flows spread across uplinks

- **WHEN** many flows with distinct 5-tuples are sent through two equal-metric default routes
- **THEN** the flows SHALL distribute across both uplinks per the ECMP hash distribution test

### Requirement: Deterministic ECMP Hash Across Reboots

The ECMP hash inputs and seed SHALL be deterministic so that the flow-to-uplink mapping is stable across reboots; a long-lived connection SHALL map to the same uplink after a restart as before it.

#### Scenario: Reboot preserves flow placement

- **WHEN** a flow maps to `eth1` under ECMP
- **AND** the unit reboots with the same interface and route configuration
- **THEN** the same flow SHALL map to `eth1` again after the reboot
