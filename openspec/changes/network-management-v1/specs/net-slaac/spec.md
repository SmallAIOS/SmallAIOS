## ADDED Requirements

### Requirement: RFC 4862 Address Autoconfiguration Mode

When an interface is configured with `mode = "slaac"`, the `net` crate SHALL perform RFC 4862 stateless address autoconfiguration: listen for Router Advertisements and form a global address from each advertised autonomous prefix, respecting the advertised valid and preferred lifetimes.

#### Scenario: Router Advertisement produces a global address

- **WHEN** an interface in `slaac` mode receives a Router Advertisement carrying an autonomous prefix
- **THEN** the stack SHALL form and assign a global IPv6 address in that prefix
- **AND** the address SHALL expire according to the advertised valid lifetime unless refreshed by a later advertisement

### Requirement: RFC 7217 Stable Interface Identifiers

SLAAC interface identifiers SHALL be generated per RFC 7217 (stable, opaque, per interface and prefix). The identifier SHALL NOT be derived from or embed the interface MAC address (no EUI-64), and SHALL be deterministic across reboots for the same interface and prefix.

#### Scenario: Interface identifier does not leak the MAC

- **WHEN** a SLAAC address is generated for an interface
- **THEN** the interface identifier SHALL NOT contain the interface's MAC address bytes in EUI-64 or any other recoverable form

#### Scenario: Identifier is stable across reboots

- **WHEN** the unit reboots and receives a Router Advertisement for the same prefix on the same interface
- **THEN** the generated SLAAC address SHALL be identical to the address generated before the reboot

#### Scenario: Different prefixes yield different identifiers

- **WHEN** the same interface autoconfigures addresses in two different advertised prefixes
- **THEN** the interface identifiers of the two addresses SHALL differ
