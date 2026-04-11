## ADDED Requirements

### Requirement: DHCPv4 client implementation
The `net` crate SHALL provide a DHCPv4 client (RFC 2131) capable of obtaining an IPv4 address, subnet mask, default gateway, and DNS server from a DHCP server.

#### Scenario: Successful DHCP exchange
- **WHEN** the DHCP client is started on an interface with a DHCP server present
- **THEN** the client SHALL complete the DISCOVER → OFFER → REQUEST → ACK exchange and configure the interface with the assigned IP address, subnet mask, gateway, and DNS server

#### Scenario: No DHCP server available
- **WHEN** the DHCP client sends DISCOVER and receives no OFFER within 10 seconds (with 3 retries at exponential backoff)
- **THEN** the client SHALL report a timeout error and leave the interface unconfigured

### Requirement: Lease management
The DHCP client SHALL track the lease duration and attempt renewal at T1 (50% of lease time).

#### Scenario: Lease renewal
- **WHEN** T1 expires (50% of the lease duration)
- **THEN** the client SHALL send a DHCP REQUEST to the server to renew the lease

#### Scenario: Lease expiry without renewal
- **WHEN** the lease expires without successful renewal
- **THEN** the client SHALL deconfigure the interface IP and restart the DISCOVER process

### Requirement: DHCP options parsing
The client SHALL parse and apply at minimum these DHCP options: subnet mask (option 1), router/gateway (option 3), DNS server (option 6), and lease time (option 51).

#### Scenario: Options applied to interface
- **WHEN** a DHCP ACK is received with options 1, 3, 6, and 51
- **THEN** the interface SHALL be configured with the subnet mask, default route via the gateway, DNS resolver address, and lease timer set to the specified duration

### Requirement: Uses existing UDP stack
The DHCP client SHALL use the existing `net` crate UDP implementation for sending and receiving DHCP messages on port 67 (server) / port 68 (client).

#### Scenario: Broadcast DISCOVER
- **WHEN** the client sends a DHCP DISCOVER
- **THEN** it SHALL send a UDP broadcast packet (destination `255.255.255.255:67`) from `0.0.0.0:68` with the correct DHCP message format
