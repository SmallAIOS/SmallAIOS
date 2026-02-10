# Delta for Native IPv4/IPv6 Networking

## ADDED Requirements

### Requirement: IPv4 Network Stack
The system SHALL implement IPv4 with static addressing, ARP, and single-gateway routing.

#### Scenario: Send and receive IPv4 packets
- **WHEN** the network stack is configured with a static IPv4 address and default gateway
- **THEN** it MUST send and receive IPv4 packets with correct headers and checksums
- **AND** it MUST resolve the gateway MAC address via ARP before sending

#### Scenario: Respond to ICMP echo requests
- **WHEN** the system receives an ICMPv4 echo request (ping)
- **THEN** it MUST respond with an ICMPv4 echo reply containing the same payload

### Requirement: IPv6 Network Stack
The system SHALL implement IPv6 with link-local addressing, NDP, and SLAAC.

#### Scenario: Auto-configure link-local address
- **WHEN** the network interface is initialized
- **THEN** the system MUST generate a link-local IPv6 address (fe80::/10)
- **AND** MUST perform Duplicate Address Detection (DAD) via NDP

#### Scenario: SLAAC global address configuration
- **WHEN** the system receives a Router Advertisement with a prefix
- **THEN** it MUST auto-configure a global IPv6 address using SLAAC
- **AND** MUST respect the advertised prefix lifetime

#### Scenario: Neighbor Discovery
- **WHEN** the system needs to send to an IPv6 address on the local link
- **THEN** it MUST resolve the link-layer address via Neighbor Solicitation
- **AND** MUST cache the result in the neighbor table with reachability tracking

### Requirement: TCP Implementation
The system SHALL implement TCP with reliable delivery, CUBIC congestion control, and SACK.

#### Scenario: Three-way handshake
- **WHEN** a client connects to the system's listening TCP socket
- **THEN** the system MUST complete the SYN, SYN-ACK, ACK three-way handshake
- **AND** transition to ESTABLISHED state

#### Scenario: Data transfer with flow control
- **WHEN** data is sent over an established TCP connection
- **THEN** the system MUST use sliding window flow control with the receiver's advertised window
- **AND** MUST support window scaling (RFC 7323) for windows larger than 64 KB

#### Scenario: CUBIC congestion control
- **WHEN** packet loss is detected (3 duplicate ACKs or RTO timeout)
- **THEN** the system MUST reduce the congestion window per CUBIC algorithm
- **AND** MUST perform fast retransmit on 3 duplicate ACKs
- **AND** MUST use SACK for efficient selective retransmission

#### Scenario: Connection close
- **WHEN** either end initiates connection close
- **THEN** the system MUST perform the four-way FIN handshake
- **AND** MUST enter TIME_WAIT state for 2*MSL (60 seconds default)

#### Scenario: SYN cookie protection
- **WHEN** the system is under SYN flood attack (SYN backlog exceeded)
- **THEN** it MUST use SYN cookies to respond without allocating state
- **AND** MUST reconstruct connection state from the cookie on ACK receipt

### Requirement: UDP Implementation
The system SHALL implement UDP for DNS resolution and NTP time synchronization.

#### Scenario: Send and receive UDP datagrams
- **WHEN** a UDP datagram is sent to a destination address and port
- **THEN** it MUST be encapsulated in an IPv4 or IPv6 packet with correct UDP checksum
- **AND** incoming UDP datagrams MUST be delivered to the correct listening socket

### Requirement: Built-in Packet Filter
The system SHALL include a configurable packet filter with default-deny ingress policy.

#### Scenario: Default deny ingress
- **WHEN** no firewall rules are configured
- **THEN** all incoming packets MUST be dropped except: responses to outgoing connections, ICMPv6 NDP
- **AND** a log entry MUST be generated for dropped packets (rate-limited)

#### Scenario: Allow IPC port
- **WHEN** the firewall is configured to allow TCP on the IPC port (default 7447)
- **THEN** incoming TCP connections to port 7447 MUST be accepted
- **AND** all other incoming TCP connections MUST be dropped

### Requirement: Network Device Driver — virtio-net
The system SHALL include a virtio-net driver for VM and container network connectivity.

#### Scenario: Initialize virtio-net device
- **WHEN** a virtio-net device is detected during boot (MMIO transport)
- **THEN** the driver MUST negotiate features, set up TX and RX virtqueues
- **AND** the device MUST be ready to send and receive Ethernet frames
