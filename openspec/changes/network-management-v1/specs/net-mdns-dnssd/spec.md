## ADDED Requirements

### Requirement: mDNS Hostname Responder

The `net` crate SHALL provide a clean-room RFC 6762 mDNS responder in `net/src/mdns/` that advertises `<hostname>.local.` with A and AAAA records on every configured interface where mDNS is enabled.

#### Scenario: A query answered with the interface address

- **WHEN** an mDNS A query for `<hostname>.local.` arrives on an mDNS-enabled interface
- **THEN** the responder SHALL answer with an A record carrying that interface's IPv4 address

#### Scenario: AAAA query answered

- **WHEN** an mDNS AAAA query for `<hostname>.local.` arrives on an mDNS-enabled interface
- **THEN** the responder SHALL answer with an AAAA record carrying that interface's IPv6 address

### Requirement: One-Interface-Per-Answer Rule

Per RFC 6762 §11, an mDNS answer sent on an interface SHALL contain only addresses reachable via that interface. The responder SHALL never advertise an address from a different interface back at the asker.

#### Scenario: Answer excludes other interfaces' addresses

- **WHEN** the unit has `eth0` on `192.168.1.0/24` and `eth1` on `10.0.0.0/24`
- **AND** an mDNS query for `<hostname>.local.` arrives on `eth0`
- **THEN** the answer SHALL contain only the `eth0` addresses
- **AND** SHALL NOT contain any `eth1` address

### Requirement: DNS-SD Service Publication

The responder SHALL implement RFC 6763 DNS-SD publication of two services: `_smallaios._tcp.local.` (the admin Zenoh endpoint) and `_smallaios-metrics._tcp.local.` (the metrics endpoint), each with PTR, SRV, and TXT records resolving to the unit.

#### Scenario: Discovery client finds the admin service

- **WHEN** a discovery client browses PTR records for `_smallaios._tcp.local.`
- **THEN** the responder SHALL answer with a PTR record for the unit's service instance
- **AND** the corresponding SRV record SHALL resolve to the unit's hostname and admin Zenoh port

#### Scenario: Metrics service is advertised separately

- **WHEN** a discovery client browses `_smallaios-metrics._tcp.local.`
- **THEN** the responder SHALL answer with the unit's metrics service instance

### Requirement: mDNS Resolver for Management Tooling

The `net` crate SHALL provide a client-side mDNS/DNS-SD resolver usable by the `mgmt` tooling to locate peer SmallAIOS units by hostname or service type rather than by IP address.

#### Scenario: Peer discovered by service type

- **WHEN** the resolver browses `_smallaios._tcp.local.` on a segment with another SmallAIOS unit advertising
- **THEN** it SHALL return the peer's service instance with resolved address and port, suitable for fleet telemetry aggregation

#### Scenario: Peer hostname resolved

- **WHEN** the resolver queries `<peer-hostname>.local.`
- **THEN** it SHALL return the peer's A/AAAA addresses learned via mDNS

### Requirement: Untrusted Interface Opt-Out and Response Rate Limiting

mDNS (responder and resolver) SHALL be disabled on any interface whose configuration sets `untrusted = true`; no mDNS listener SHALL be opened on such an interface. On enabled interfaces the responder SHALL rate-limit its responses to prevent broadcast amplification on a busy LAN.

#### Scenario: Untrusted interface stays silent

- **WHEN** `/data/network/eth1.toml` sets `untrusted = true`
- **THEN** no mDNS UDP listener SHALL be opened on `eth1`
- **AND** mDNS queries arriving on `eth1` SHALL receive no answer

#### Scenario: Query flood is rate-limited

- **WHEN** a burst of repeated identical mDNS queries arrives on an enabled interface
- **THEN** the responder SHALL cap its response rate rather than answering every query
- **AND** normal query answering SHALL resume once the burst subsides
