## ADDED Requirements

### Requirement: Bond Virtual Interface

The network stack SHALL support bond devices that aggregate two or more physical interfaces into a single virtual interface, configured per bond at `/data/network/<bond>.toml` with a `mode` field selecting one of `active-backup`, `balance-rr`, `balance-xor`, or `802.3ad`. DHCP, mDNS, and the route table SHALL see only the bond, never the slave interfaces.

#### Scenario: DHCP runs on the bond, not the slaves

- **WHEN** `bond0` aggregates `eth0` and `eth1` and is configured with `mode = "dhcp"` addressing
- **THEN** the DHCP client SHALL run on `bond0`
- **AND** no DHCP client SHALL run on `eth0` or `eth1`

#### Scenario: Slaves invisible to routing and mDNS

- **WHEN** `eth0` and `eth1` are enslaved to `bond0`
- **THEN** the route table SHALL contain routes via `bond0` only
- **AND** mDNS SHALL advertise addresses of `bond0` only

#### Scenario: Bond mode selected from config

- **WHEN** `/data/network/bond0.toml` sets `mode = "active-backup"`
- **THEN** the bond SHALL operate in active-backup mode after the configuration is applied

### Requirement: Route Table Egress

Packet egress SHALL consult the longest-prefix-match route table (see `net-routing-multipath`) instead of the previous hard-coded single default route. Multiple default routes with distinct metrics SHALL coexist in the table.

#### Scenario: Egress follows the route table

- **WHEN** the table contains a default route via `eth0` and a more specific route via `eth1`
- **THEN** the stack SHALL choose the egress interface per route-table lookup for every outgoing packet

#### Scenario: Second default route does not clobber the first

- **WHEN** a second interface comes up and installs its own default route with a different metric
- **THEN** both default routes SHALL be present in the table
- **AND** the lower-metric route SHALL win selection

### Requirement: Role-Based Interface Selection

The stack SHALL honor per-interface `role` tags when placing traffic: interfaces tagged `role = "admin"` carry the Zenoh admin/telemetry plane, interfaces tagged `role = "data"` carry inference traffic, and `role = "any"` expresses no preference.

#### Scenario: Admin plane prefers the admin interface

- **WHEN** `eth0` is tagged `role = "admin"` and `eth1` is tagged `role = "data"`
- **THEN** Zenoh admin and telemetry traffic SHALL egress via `eth0`
- **AND** inference data traffic SHALL egress via `eth1`

#### Scenario: All-any falls back to metric ordering

- **WHEN** every interface is tagged `role = "any"`
- **THEN** traffic placement SHALL be decided by the route table's prefix and metric ordering alone

## MODIFIED Requirements

### Requirement: IPv4 Networking

The network stack SHALL implement IPv4 with static addressing, ARP, route-table-based routing (longest-prefix match, then metric — see `net-routing-multipath`), and ICMP echo. The route table replaces the previous hard-coded single default gateway.

#### Scenario: Send and receive IPv4 packets

- **WHEN** the network interface is configured with a static IPv4 address and subnet mask
- **THEN** the stack MUST construct valid IPv4 headers with correct checksum and TTL (default 64)
- **AND** MUST set the DF (Don't Fragment) bit on all outgoing packets

#### Scenario: ARP resolution

- **WHEN** the stack needs to send a packet to an IP address on the local subnet
- **AND** the destination MAC address is not in the ARP table
- **THEN** the stack MUST send an ARP request and cache the reply (timeout 300 seconds)
- **AND** the ARP table MUST be limited to 256 entries to prevent exhaustion

#### Scenario: Route-table gateway routing

- **WHEN** the stack needs to send a packet to an IP address outside the local subnet
- **THEN** the stack MUST select the next-hop gateway and egress interface via longest-prefix-match lookup in the route table, breaking ties by metric
- **AND** MUST forward the packet to the selected gateway via ARP resolution
- **AND** multiple default routes with distinct metrics MUST be allowed to coexist in the table

#### Scenario: ICMP echo reply

- **WHEN** the stack receives an ICMP Echo Request (ping)
- **THEN** the stack MUST reply with an ICMP Echo Reply containing the same identifier, sequence, and data
