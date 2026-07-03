## ADDED Requirements

### Requirement: RFC 8415 IA_NA Address Acquisition

The `net` crate SHALL provide a clean-room `#![no_std]` DHCPv6 client in `net/src/dhcp/` implementing the RFC 8415 SOLICIT → ADVERTISE → REQUEST → REPLY exchange for non-temporary addresses (IA_NA) only. Prefix delegation (IA_PD) SHALL NOT be requested or processed in v1.

#### Scenario: Four-message exchange acquires an address

- **WHEN** the client starts on an interface with a reachable DHCPv6 server
- **THEN** it SHALL emit a SOLICIT, process the ADVERTISE, emit a REQUEST, and process the REPLY
- **AND** the IA_NA address from the REPLY SHALL be configured on the interface with its valid and preferred lifetimes

#### Scenario: IA_PD is never requested

- **WHEN** any SOLICIT or REQUEST emitted by the client is captured
- **THEN** it SHALL contain an IA_NA option
- **AND** it SHALL NOT contain an IA_PD option

#### Scenario: Golden vectors match wide-dhcpv6

- **WHEN** the DHCPv6 unit tests run against the golden message vectors cross-checked against `wide-dhcpv6`
- **THEN** the client's encoded messages and decoded server responses SHALL match the vectors byte-for-byte
