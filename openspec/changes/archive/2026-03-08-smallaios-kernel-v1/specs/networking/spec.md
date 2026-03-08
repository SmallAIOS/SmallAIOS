# Delta for Networking

## ADDED Requirements

### Requirement: IPv4 Networking
The network stack SHALL implement IPv4 with static addressing, ARP, single-gateway routing, and ICMP echo.

#### Scenario: Send and receive IPv4 packets
- WHEN the network interface is configured with a static IPv4 address and subnet mask
- THEN the stack MUST construct valid IPv4 headers with correct checksum and TTL (default 64)
- AND MUST set the DF (Don't Fragment) bit on all outgoing packets

#### Scenario: ARP resolution
- WHEN the stack needs to send a packet to an IP address on the local subnet
- AND the destination MAC address is not in the ARP table
- THEN the stack MUST send an ARP request and cache the reply (timeout 300 seconds)
- AND the ARP table MUST be limited to 256 entries to prevent exhaustion

#### Scenario: Default gateway routing
- WHEN the stack needs to send a packet to an IP address outside the local subnet
- THEN the stack MUST forward the packet to the configured default gateway via ARP resolution

#### Scenario: ICMP echo reply
- WHEN the stack receives an ICMP Echo Request (ping)
- THEN the stack MUST reply with an ICMP Echo Reply containing the same identifier, sequence, and data

### Requirement: IPv6 Networking
The network stack SHALL implement IPv6 with link-local addressing, NDP, SLAAC, and ICMP echo.

#### Scenario: Auto-generate link-local address
- WHEN the network interface is initialized
- THEN the stack MUST generate a link-local IPv6 address (fe80::/10)
- AND MUST perform Duplicate Address Detection before using the address

#### Scenario: Neighbor Discovery Protocol
- WHEN the stack needs to resolve an IPv6 address to a MAC address
- THEN the stack MUST send a Neighbor Solicitation and process the Neighbor Advertisement
- AND MUST maintain a neighbor cache with reachability tracking

#### Scenario: SLAAC address configuration
- WHEN the stack receives a Router Advertisement with a prefix
- THEN the stack MUST generate a global IPv6 address from the prefix and interface identifier
- AND MUST respect the valid and preferred lifetimes from the advertisement

#### Scenario: ICMPv6 echo reply
- WHEN the stack receives an ICMPv6 Echo Request
- THEN the stack MUST reply with an ICMPv6 Echo Reply

### Requirement: TCP Transport
The network stack SHALL implement TCP with 3-way handshake, reliable data transfer, CUBIC congestion control, SACK, and keepalive.

#### Scenario: TCP three-way handshake
- WHEN a client initiates a TCP connection
- THEN the stack MUST complete the SYN, SYN-ACK, ACK handshake
- AND MUST transition to the ESTABLISHED state upon receiving the final ACK

#### Scenario: TCP passive open (server)
- WHEN the IPC system binds a TCP listener on port 7447
- THEN the stack MUST accept incoming SYN packets and create connection state
- AND MUST support SYN cookies to protect against SYN flood attacks

#### Scenario: Reliable data transfer with SACK
- WHEN TCP segments are lost during transmission
- THEN the stack MUST detect loss via 3 duplicate ACKs and perform fast retransmit
- AND MUST use Selective Acknowledgment (SACK) for efficient loss recovery

#### Scenario: CUBIC congestion control
- WHEN a TCP connection experiences congestion
- THEN the stack MUST implement CUBIC congestion control with slow start, congestion avoidance, and fast recovery
- AND MUST support ECN (Explicit Congestion Notification)

#### Scenario: TCP keepalive
- WHEN a TCP connection is idle for the configured keepalive time (default 60 seconds)
- THEN the stack MUST send keepalive probes at the configured interval (default 10 seconds)
- AND MUST close the connection after the configured number of unanswered probes (default 5)

### Requirement: UDP Transport
The network stack SHALL implement minimal UDP for DNS resolution and NTP synchronization.

#### Scenario: Send and receive UDP datagrams
- WHEN the DNS resolver or NTP client sends a UDP datagram
- THEN the stack MUST construct a valid UDP header with correct checksum
- AND MUST deliver received datagrams to the correct port handler

#### Scenario: DNS stub resolution
- WHEN the stack needs to resolve a hostname
- THEN the DNS stub resolver MUST send a UDP query for A and AAAA records
- AND MUST parse the DNS response and return the resolved addresses

### Requirement: Built-in Packet Filter
The network stack SHALL include a built-in packet filter (firewall) with configurable allow/deny rules.

#### Scenario: Default deny policy
- WHEN the firewall is configured with default_input = "drop"
- AND no allow rule matches an incoming packet
- THEN the stack MUST silently drop the packet
- AND MUST NOT send any response

#### Scenario: Allow IPC port traffic
- WHEN an allow rule permits TCP traffic on port 7447
- AND an incoming TCP SYN arrives on port 7447
- THEN the firewall MUST allow the packet through to the TCP stack

#### Scenario: Allow ICMPv6 for NDP
- WHEN an allow rule permits ICMPv6 traffic
- THEN the firewall MUST allow Neighbor Solicitation and Router Solicitation messages
- AND MUST allow Router Advertisement messages required for SLAAC

### Requirement: Network Device Drivers
The network stack SHALL support virtio-net for VM deployments and Broadcom GENET for Raspberry Pi bare metal.

#### Scenario: virtio-net driver initialization
- WHEN SmallAIOS boots in a virtual machine with a virtio-net device
- THEN the driver MUST negotiate virtio features, set up TX and RX queues, and configure the MAC address
- AND the network interface MUST be ready to send and receive Ethernet frames

#### Scenario: Broadcom GENET driver for Raspberry Pi
- WHEN SmallAIOS boots on a Raspberry Pi 4 or 5 with a Broadcom GENET NIC
- THEN the driver MUST initialize the hardware, configure DMA rings, and register the network interface
- AND the interface MUST support standard Ethernet frame send and receive
