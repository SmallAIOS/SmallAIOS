# Spec 09: Native IPv4/IPv6 Networking

## Overview

SmallAIOS includes a **native TCP/IP network stack** supporting both IPv4 and IPv6.
The stack is minimal — it implements only the protocols needed for IPC transport,
health checks, and metrics endpoints — but it is a real network stack, not a
shim over a host OS.

In container mode (library OS), the native stack is bypassed in favor of host
sockets for compatibility. In VM mode and bare metal, the native stack is used.

## Protocol Support

### Implemented

| Layer | Protocol | Notes |
|---|---|---|
| L2 | Ethernet | Frame send/receive, MAC addressing |
| L3 | IPv4 | Addressing, routing (single default gateway) |
| L3 | IPv6 | Addressing, SLAAC, routing |
| L3 | ICMPv4 | Echo request/reply (ping), destination unreachable |
| L3 | ICMPv6 | Echo, neighbor solicitation/advertisement (NDP), router solicitation |
| L4 | TCP | Full implementation (3-way handshake, data transfer, FIN, RST) |
| L4 | UDP | For DNS resolution (optional) and NTP |
| L7 | DNS | Minimal stub resolver (UDP, A/AAAA records only) |
| L7 | NTP | SNTP client for wall clock synchronization (optional) |

### Not Implemented

- Raw sockets (not needed)
- SCTP, DCCP (not needed)
- IP fragmentation (we set DF bit; inference messages are over TCP)
- IPsec (TLS at application layer instead)
- Multicast (not needed for point-to-point inference)
- DHCP (use static config, SLAAC, or container-provided addresses)

## Architecture

```
┌──────────────────────────────────────────┐
│           Socket API (POSIX)              │
│    socket, bind, listen, accept, etc.     │
├──────────────────────────────────────────┤
│              TCP                          │
│  Connection management, flow control,     │
│  congestion control, retransmission       │
├──────────────────────────────────────────┤
│         UDP          │     ICMPv4/v6      │
├──────────────────────┼───────────────────┤
│    IPv4              │      IPv6          │
│  Routing, addressing │  SLAAC, NDP,       │
│  Header processing   │  routing           │
├──────────────────────┴───────────────────┤
│         ARP (IPv4) / NDP (IPv6)           │
├──────────────────────────────────────────┤
│             Ethernet                      │
│    Frame construction, MAC layer          │
├──────────────────────────────────────────┤
│          Network Device Driver            │
│   virtio-net (VM) / HW NIC (bare metal)  │
└──────────────────────────────────────────┘
```

## IPv4 Implementation

### Addressing
- Static configuration via `smallaios.toml` or environment variables
- Single IPv4 address per interface (no aliasing)
- Subnet mask and default gateway
- Loopback (127.0.0.1) for local connections

### Header Processing
- Construct and parse IPv4 headers
- Compute and verify header checksum
- Set DF (Don't Fragment) bit on all outgoing packets
- TTL: default 64, configurable

### ARP
- ARP table with timeout (300 seconds default)
- ARP request/reply handling
- Gratuitous ARP on interface up
- Table size limit (256 entries) to prevent exhaustion

### Routing
- Single default gateway (sufficient for container/edge deployments)
- Local subnet delivery (ARP resolution)
- No dynamic routing protocols (static configuration only)

## IPv6 Implementation

### Addressing
- Link-local address auto-generated from MAC (EUI-64) or random (RFC 7217)
- Global address via SLAAC (Stateless Address Autoconfiguration) or static config
- Loopback (::1) for local connections

### Neighbor Discovery Protocol (NDP)
- Neighbor Solicitation / Neighbor Advertisement (replaces ARP)
- Router Solicitation / Router Advertisement (for SLAAC)
- Duplicate Address Detection (DAD)
- Neighbor cache with reachability tracking

### SLAAC
- Process Router Advertisements
- Generate global address from prefix + interface ID
- Respect prefix lifetimes (valid, preferred)

### Header Processing
- Construct and parse IPv6 headers (40-byte fixed header)
- No extension headers (not needed for our use case)
- Hop limit: default 64

## TCP Implementation

### Connection Management
- Three-way handshake (SYN, SYN-ACK, ACK)
- Active open (client) and passive open (server/listener)
- Four-way close (FIN, ACK, FIN, ACK)
- RST handling (reset on error)
- TIME-WAIT state with timer (60 seconds)

### Data Transfer
- Sliding window flow control
- Receive window scaling (RFC 7323)
- Nagle's algorithm (configurable, default off for low-latency inference)
- Delayed ACK (configurable, default: 40ms or every 2 segments)

### Congestion Control
- **CUBIC** congestion control (Linux default, well-understood)
- Slow start, congestion avoidance, fast retransmit, fast recovery
- ECN (Explicit Congestion Notification) support

### Retransmission
- RTO calculation per RFC 6298 (Jacobson/Karels algorithm)
- Fast retransmit on 3 duplicate ACKs
- SACK (Selective Acknowledgment) for efficient loss recovery
- Maximum retransmission limit (configurable, default 15)

### Keep-Alive
- TCP keep-alive probes (configurable, default: 60s idle, 10s interval, 5 probes)
- For detecting dead IPC connections

### Socket Options
```
TCP_NODELAY      = true     // Disable Nagle (low-latency inference)
TCP_KEEPALIVE    = 60       // Seconds before first probe
SO_REUSEADDR     = true     // Allow port reuse on restart
SO_RCVBUF        = 262144   // 256 KB receive buffer
SO_SNDBUF        = 262144   // 256 KB send buffer
```

## UDP Implementation

Minimal UDP for DNS and NTP:
- Send/receive datagrams
- Port multiplexing
- Checksum computation and verification
- No fragmentation support (limit to MTU-sized datagrams)

## Network Device Drivers

### virtio-net (VM mode)

Primary network device for VM deployments:
- Virtio MMIO or PCI transport
- Single TX queue, single RX queue
- Checksum offload (if negotiated)
- No TSO/GRO (keep driver simple)

### Physical NIC Drivers (bare metal)

For bare metal edge deployment, support common server NICs:

| NIC | Driver | Use Case |
|---|---|---|
| virtio-net | Built-in | VM, containers with macvlan |
| Intel I210/I225 | `igb` | Jetson carrier boards, embedded |
| Broadcom GENET | `bcmgenet` | Raspberry Pi 4/5 |
| Realtek RTL8169 | `r8169` | Budget x86 boards |
| Mellanox ConnectX | Future | DGX, data center |

Priority: virtio-net first, then GENET (for Pi), then igb.

### DGX Spark Networking

The NVIDIA DGX Spark uses an integrated NIC (likely Mellanox/NVIDIA ConnectX family).
Initial support via virtio-net in VM mode; native ConnectX driver as a stretch goal.

## Security

### Network Attack Surface Mitigation

- **SYN cookies**: Protect against SYN flood without allocating state
- **Connection limits**: Maximum connections per source IP (configurable)
- **Rate limiting**: Packets per second limit on incoming connections
- **No raw sockets**: Cannot craft arbitrary packets
- **TLS required**: Configurable enforcement of TLS on all IPC connections
- **IPv6 privacy**: Random interface identifier (not MAC-based)

### Firewall Rules (Built-in)

SmallAIOS has a simple built-in packet filter:

```toml
[network.firewall]
# Default policy
default_input = "drop"
default_output = "allow"

# Allow rules
[[network.firewall.allow]]
direction = "input"
protocol = "tcp"
port = 7447          # IPC port

[[network.firewall.allow]]
direction = "input"
protocol = "icmpv6"  # Required for NDP

[[network.firewall.allow]]
direction = "input"
protocol = "icmpv4"
type = 8             # Echo request (optional, for debugging)
```

## Configuration

```toml
[network]
# Stack selection: "native" (VM/bare metal) or "host" (container mode, use host sockets)
stack = "native"

[network.ipv4]
enabled = true
address = "10.0.0.2/24"      # Static config
gateway = "10.0.0.1"
# OR
# dhcp = true                 # Future: DHCP client

[network.ipv6]
enabled = true
# address = "fd00::2/64"     # Static config (optional)
slaac = true                  # Auto-configure from router advertisements
privacy_extensions = true     # RFC 7217 stable privacy addresses

[network.dns]
# servers = ["8.8.8.8", "2001:4860:4860::8888"]  # Optional
# Usually not needed — IPC uses IP addresses directly

[network.tcp]
nagle = false                 # Disable for low latency
keepalive_idle = 60
keepalive_interval = 10
keepalive_count = 5
max_connections = 1024
syn_cookies = true

[network.ntp]
enabled = false
# server = "pool.ntp.org"    # For wall-clock sync
```

## Crate Structure

```
net/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── ethernet.rs      # Ethernet frame handling
    ├── arp.rs           # ARP table and protocol
    ├── ipv4.rs          # IPv4 header, routing
    ├── ipv6.rs          # IPv6 header, SLAAC
    ├── ndp.rs           # Neighbor Discovery Protocol
    ├── icmp.rs          # ICMPv4 and ICMPv6
    ├── tcp/
    │   ├── mod.rs
    │   ├── connection.rs  # TCP state machine
    │   ├── segment.rs     # TCP segment parsing/construction
    │   ├── timer.rs       # Retransmission timers
    │   ├── congestion.rs  # CUBIC congestion control
    │   └── listen.rs      # Listener (passive open)
    ├── udp.rs           # UDP send/receive
    ├── dns.rs           # Minimal DNS stub resolver
    ├── socket.rs        # Socket API (POSIX-compatible)
    ├── firewall.rs      # Packet filter
    └── driver/
        ├── mod.rs
        ├── virtio_net.rs  # virtio-net driver
        ├── bcmgenet.rs    # Broadcom GENET (Raspberry Pi)
        └── igb.rs         # Intel I210/I225
```
