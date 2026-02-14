## ADDED Requirements

### Requirement: Compiled-in static IP configuration
The `net` crate SHALL support a compiled-in static IPv4 configuration as an alternative to DHCP, configured via Cargo feature flags or constants.

#### Scenario: Static IP at boot
- **WHEN** the kernel boots with static IP configured (e.g., `NET_STATIC_IP=192.168.1.100/24`)
- **THEN** the network interface SHALL be configured with the specified IP address and subnet mask immediately after the NIC driver initializes, without any DHCP exchange

### Requirement: Static gateway and DNS
The static configuration SHALL include a default gateway and DNS server address.

#### Scenario: Gateway configured
- **WHEN** a static gateway is specified (e.g., `NET_STATIC_GATEWAY=192.168.1.1`)
- **THEN** the routing table SHALL have a default route via the specified gateway

#### Scenario: DNS configured
- **WHEN** a static DNS server is specified (e.g., `NET_STATIC_DNS=8.8.8.8`)
- **THEN** DNS resolution SHALL use the specified server

### Requirement: DHCP fallback
The network configuration SHALL support a priority order: static IP (if configured) takes precedence over DHCP. If no static IP is compiled in, DHCP SHALL be attempted.

#### Scenario: Static overrides DHCP
- **WHEN** both static IP and DHCP are available
- **THEN** the static IP configuration SHALL be applied and DHCP SHALL NOT be started

#### Scenario: DHCP when no static config
- **WHEN** no static IP is compiled in
- **THEN** the DHCP client SHALL be started automatically after NIC initialization

### Requirement: IPv6 static address
The static configuration SHALL optionally support an IPv6 address in addition to IPv4.

#### Scenario: Dual-stack static config
- **WHEN** both IPv4 and IPv6 static addresses are configured
- **THEN** the interface SHALL be configured with both addresses and both protocol stacks SHALL be active
