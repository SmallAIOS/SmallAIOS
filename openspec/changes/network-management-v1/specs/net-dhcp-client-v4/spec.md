## ADDED Requirements

### Requirement: RFC 2131 Address Acquisition

The `net` crate SHALL provide a clean-room `#![no_std]` DHCPv4 client in `net/src/dhcp/` implementing the full RFC 2131 DISCOVER → OFFER → REQUEST → ACK lifecycle, with no external DHCP daemon and no new external dependencies. Every DHCPDISCOVER and DHCPREQUEST SHALL carry the vendor class identifier option with the value `SmallAIOS/0.x` (matching the workspace version).

#### Scenario: Full lifecycle acquires a lease

- **WHEN** the client starts on an interface with a reachable DHCPv4 server
- **THEN** it SHALL emit a DHCPDISCOVER, process the DHCPOFFER, emit a DHCPREQUEST, and process the DHCPACK
- **AND** the interface SHALL be configured with the leased address, subnet mask, gateway, and DNS servers from the ACK

#### Scenario: Vendor class identifier present

- **WHEN** a DHCPDISCOVER or DHCPREQUEST emitted by the client is captured
- **THEN** it SHALL contain the vendor class identifier option set to `SmallAIOS/0.x`

#### Scenario: Golden vectors match dnsmasq

- **WHEN** the DHCPv4 unit tests run against the golden message vectors cross-checked against `dnsmasq`
- **THEN** the client's encoded messages and decoded server responses SHALL match the vectors byte-for-byte

### Requirement: T1/T2 Renewal Timers

The DHCPv4 client SHALL implement the RFC 2131 T1 (renewal) and T2 (rebinding) timers. At T1 the client SHALL unicast a DHCPREQUEST to the leasing server; if no reply arrives by T2 it SHALL broadcast the DHCPREQUEST; if the lease expires without renewal the address SHALL be removed from the interface and discovery SHALL restart.

#### Scenario: Renewal at T1 extends the lease

- **WHEN** the T1 timer fires and the leasing server answers the unicast DHCPREQUEST with a DHCPACK
- **THEN** the lease timers SHALL be reset from the new ACK
- **AND** the interface address SHALL remain unchanged and uninterrupted

#### Scenario: Rebind at T2 after silent server

- **WHEN** the leasing server does not answer between T1 and T2
- **THEN** at T2 the client SHALL broadcast the DHCPREQUEST to reach any server

#### Scenario: Lease expiry removes the address

- **WHEN** the lease expires with no DHCPACK received
- **THEN** the leased address SHALL be removed from the interface
- **AND** the client SHALL return to the DISCOVER state

### Requirement: Lease Persistence Across Reboots

The DHCPv4 client SHALL persist the active lease (address, server identifier, and timer state) to an on-disk lease file so a lease survives a reboot. On boot with an unexpired persisted lease, the client SHALL attempt to reclaim the persisted address via DHCPREQUEST before falling back to full discovery.

#### Scenario: Reboot within lease lifetime reclaims the address

- **WHEN** the unit reboots while its persisted lease is still valid
- **THEN** the client SHALL send a DHCPREQUEST for the persisted address instead of starting with DHCPDISCOVER
- **AND** on DHCPACK the same address SHALL be reinstated

#### Scenario: Server NAK falls back to discovery

- **WHEN** the reclaim DHCPREQUEST is answered with a DHCPNAK
- **THEN** the persisted lease SHALL be discarded
- **AND** the client SHALL perform a full DISCOVER → OFFER → REQUEST → ACK cycle
