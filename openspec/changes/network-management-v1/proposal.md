## Why

The `net` crate already implements production-quality IPv4/IPv6,
TCP, ARP, NDP, and QUIC/HTTP3 with TLS 1.3 + PQC. What it lacks
is the **operational layer** that turns a working netstack into
a deployable appliance:

1. **Address management** — today every interface needs a
   compile-time / boot-arg static IP. Real deployments get an
   address from DHCP on the office LAN, fall back to a static
   address on a vehicle or industrial bus, or auto-configure
   via SLAAC on an IPv6-only segment.
2. **Discovery** — once `management-login-v1` lands, an
   operator who wants to log in needs to *find* the box first.
   Hardcoding IPs across a fleet is brittle; mDNS / DNS-SD is
   how every other modern appliance solves this.
3. **Multiple interfaces** — Jetson Orin and most server-class
   x86 boxes have ≥2 NICs. Today the netstack treats them
   independently with no policy about which one carries
   inference data, which one carries the Zenoh admin plane, or
   what happens when one cable is cut.
4. **Bonding / link aggregation** — the switch on the other
   end dictates what is possible. Some deployments have a
   single managed switch (LACP works), some have two redundant
   switches without MLAG (active-backup is the only safe mode),
   some have a static-trunked uplink (balance-xor or
   balance-rr), and some have two unmanaged switches on
   different segments (no switch cooperation possible). A
   single mode does not fit the deployment matrix; we need to
   support several.

This change adds the address-management, discovery,
multi-interface routing, and bonding/load-balancing layer that
turns SmallAIOS into a network-deployable appliance.

## What Changes

### Per-interface address management

- New `network` config (TOML) with one section per interface.
  Per interface:
  - `mode = "dhcp" | "static" | "slaac" | "dhcp_then_static"`
  - For `static`: `ipv4`, `ipv6`, `gateway`, `dns`, `mtu`.
  - For `dhcp_then_static`: try DHCP for N seconds, then apply
    a fallback static address — covers the "office LAN with
    DHCP, vehicle bus with hardcoded address" reuse case.
- Configuration lives at `/data/network/<iface>.toml`.
- Live config changes via the `management-login-v1` admin
  surface (`smallaios/admin/network/<iface>`). Apply on commit;
  rollback on failure (same atomic-rewrite pattern as the
  shadow file).

### DHCP client (clean-room)

- Clean-room `no_std` implementation of:
  - **DHCPv4** (RFC 2131): full DISCOVER → OFFER → REQUEST →
    ACK lifecycle, T1/T2 renewal timers, lease persistence
    across reboots, vendor class identifier (`SmallAIOS/0.x`).
  - **DHCPv6** (RFC 8415, IA_NA only — no IA_PD/prefix
    delegation in v1): SOLICIT → ADVERTISE → REQUEST → REPLY.
  - **SLAAC** (RFC 4862): listen for Router Advertisements,
    pick a stable interface ID per RFC 7217 (no MAC-leaking).
- ~600 LOC for v4, ~400 for v6, ~200 for SLAAC. No external
  DHCP daemon.

### mDNS / DNS-SD responder + resolver

- **Responder** (RFC 6762): advertise `<hostname>.local.` on
  every configured interface, A and AAAA records, respect
  RFC 6762 §11 (one-interface-per-answer rule — never
  advertise an unreachable address back at the asker).
- **DNS-SD service publication** (RFC 6763): advertise
  `_smallaios._tcp.local.` (admin Zenoh endpoint) and
  `_smallaios-metrics._tcp.local.` so a discovery client can
  find a box by service rather than by IP.
- **Resolver**: client side for the `mgmt` tooling to find
  peers — useful when SmallAIOS units talk to each other (e.g.
  fleet telemetry aggregation).
- Off-by-default on interfaces flagged `untrusted = true` in
  config (mDNS expands the L2 attack surface; explicit opt-out
  on hostile segments).
- ~400 LOC clean-room.

### Multi-interface routing

- Per-interface metadata in config:
  - `role = "admin" | "data" | "any"` — `admin` carries
    Zenoh admin/telemetry, `data` carries inference traffic,
    `any` is the default if no preference.
  - `metric = u32` — routing-table priority.
- New `route` table in the netstack: longest-prefix match,
  then metric, then interface preference. Replaces the current
  "default-route-only" assumption.
- **ECMP** (Equal-Cost Multi-Path) across multiple equal-metric
  default routes — per-flow hash so packets in the same TCP
  connection always take the same uplink. ~300 LOC.

### Bonding / link aggregation (multiple modes)

A single bond device aggregates two or more physical
interfaces. Mode is selected per-bond in config:

- **`active-backup`** (Linux mode 1) — one slave is primary,
  others are standby; failover on link-down. No switch
  cooperation. ~250 LOC.
- **`balance-rr`** (mode 0) — round-robin TX; works with any
  static-trunked switch. Caveat: out-of-order delivery
  possible on some workloads (documented). ~150 LOC.
- **`balance-xor`** (mode 2) — TX selection via L3+L4 hash;
  per-flow stickiness; works with static-trunked switches and
  MLAG pairs. ~200 LOC.
- **`802.3ad LACP`** (IEEE 802.1AX, mode 4) — full LACPDU
  state machine (Mux, Selection, Receive, Periodic, Churn
  detection), slow-protocols MAC `01:80:c2:00:00:02`,
  short / long timeout negotiation, key matching. ~1200 LOC
  clean-room. Required for any deployment with a managed
  switch.

A bond exposes a single virtual interface; DHCP, mDNS, and the
routing table see only the bond, not the slaves.

### LLDP receiver (RFC-compliant TLV parser)

- Listen-only LLDP (IEEE 802.1AB) on every interface: parse
  `Chassis ID`, `Port ID`, `System Name`, `System
  Capabilities`, `Management Address`. Log what we hear.
- Used in v1 for **diagnostics** ("what switch is this plugged
  into?"). Not yet used for auto-mode selection (that is a v2
  nice-to-have once we have field data).
- ~250 LOC clean-room.

### Out of scope for v1 (flagged)

- **`balance-tlb`** (mode 5) and **`balance-alb`** (mode 6)
  — adaptive load balancing via ARP rewriting; fragile,
  weakly-supported on modern switches, and
  switch-implementation-dependent. Documented as deferred,
  not "rejected forever."
- **`broadcast`** (mode 3) — niche; revisit if a specific
  industrial / avionics protocol needs duplicate-frame
  semantics.
- **VRRP** (RFC 5798) — gateway redundancy; SmallAIOS today
  is rarely a router. Defer to v2.
- **MPTCP** (RFC 8684) — clean active-active at the TCP
  layer; large implementation cost (~2000 LOC) and the QUIC
  path arguably already handles multi-path via connection
  migration. Defer.
- **DHCPv6 prefix delegation** (IA_PD) — only relevant if
  SmallAIOS hosts downstream networks. Defer.
- **IPv6 Router Advertisement *sending*** — likewise, only
  relevant if we are a router. Defer.
- **mDNS proxy** (advertise on behalf of non-mDNS peers).
  Out of scope.
- **LACP transmit** — v1 LACP is full bidirectional (we
  must speak LACPDUs to participate); LLDP is receive-only.
- **Auto-mode selection from LLDP** — v2 once we have field
  data on which deployments speak LLDP cleanly.
- **Network namespaces / VRFs** — single global routing
  table for v1.

## Capabilities

### New Capabilities

- `net-address-management`: per-interface mode, DHCP-then-
  static fallback semantics, lease persistence, atomic config
  rewrite, role tagging (admin / data / any).
- `net-dhcp-client-v4`: RFC 2131 state machine, T1/T2 timers,
  vendor class identifier, lease format on disk.
- `net-dhcp-client-v6`: RFC 8415 IA_NA flow.
- `net-slaac`: RFC 4862 + RFC 7217 stable interface ID rules.
- `net-mdns-dnssd`: RFC 6762 / 6763 responder + resolver,
  `_smallaios._tcp.local.` service definition,
  one-interface-per-answer rule, `untrusted = true` opt-out.
- `net-routing-multipath`: longest-prefix-match table,
  per-flow ECMP hash, metric ordering.
- `net-bonding-active-backup`: failover criteria, MAC handling,
  link-monitor cadence.
- `net-bonding-balance-rr`: TX rotation rules, ordering caveat.
- `net-bonding-balance-xor`: hash inputs (L3+L4), tie-break,
  MLAG compatibility notes.
- `net-bonding-lacp`: full IEEE 802.1AX state machine, LACPDU
  format, slow-protocols handling, partner-key matching, churn
  detection.
- `net-lldp-receive`: TLV parser, on-disk neighbor table.

### Modified Capabilities

- `net-stack`: gains a route table (replacing the hard-coded
  default route), the bond virtual-interface abstraction, and
  per-interface role tagging.
- `mgmt-zenoh-admin`: adds `smallaios/admin/network/**` for
  config CRUD and `smallaios/metrics/network/**` for link
  state, byte/packet counters, and DHCP lease state.
- `peripheral-ethernet`: gains link-state-change notifications
  (needed for active-backup failover and LACP timing).
- `console-login`: adds `ip`, `ip route`, `ip link`, `bond` —
  small read-mostly diagnostic commands.

## Impact

- **Code:**
  - New `net/src/dhcp/` (v4 + v6 + SLAAC).
  - New `net/src/mdns/` (responder + resolver).
  - New `net/src/route.rs` (table + ECMP).
  - New `net/src/bond/` (one module per mode).
  - New `net/src/lldp.rs` (receive-only).
  - Config plumbing in `container/` for `/data/network/`.
  - Zenoh handler in `container/src/mgmt_network.rs`.
- **Tests:** ~150 new tests targeted: DHCPv4 golden vectors
  cross-checked against `dnsmasq`, DHCPv6 vs `wide-dhcpv6`,
  SLAAC stable-ID determinism, mDNS one-answer-per-iface
  rule, ECMP hash distribution, each bond mode under simulated
  link-down, LACP state machine fuzzed against the IEEE
  reference state diagram, LLDP TLV round-trip. Aim for
  ~4,200 → ≥4,350 passing.
- **Boot footprint:** ~80 KB added to the netstack. mDNS adds
  one extra UDP listener per interface; cheap.
- **Container image:** unchanged — all of this is in the
  netstack, no new external deps.
- **Downstream:** unblocks fleet deployment (DHCP), discovery
  (mDNS), redundancy (active-backup), and throughput (LACP /
  balance-xor) for any networked SmallAIOS unit. Required for
  `management-login-v1` to be useful in any non-static-IP
  environment.
- **Dependencies:** `management-login-v1` (admin surface for
  live config changes); none in the other direction (this
  change can ship in parallel and is not blocked by the
  power-control or update changes).
- **Risks:**
  (1) LACP is the largest single piece (~1200 LOC) and has
  several timing-sensitive state machines; mis-implementation
  silently drops slaves out of the bond. The TLA+ model
  inventory should include a Promela / TLA+ model of the LACP
  Selection Logic before we ship.
  (2) mDNS broadcast amplification — a misbehaving responder
  on a busy LAN gets us blocked by other implementations.
  Need rate-limiting and the §11 one-answer rule from
  day one.
  (3) DHCP-then-static fallback timing: too short and we
  blackhole a slow-DHCP LAN; too long and a static-only
  segment waits N seconds at every boot. Default 30 s with
  per-interface override.
  (4) ECMP per-flow hashing must be stable across reboots —
  a hash that changes breaks long-lived connections after a
  restart. Hash inputs and seed must be deterministic.

## Open Questions

1. **Bond mode default**: when an operator says "bond these
   two interfaces" without specifying a mode, which is the
   safe default? Leaning `active-backup` (works everywhere,
   never breaks even if the switch is wrong); could argue for
   "auto-detect via LLDP" but that is v2.
2. **LACP first or balance-xor first**? balance-xor is ~200
   LOC and gets us most of the active-active value with a
   static-trunked switch; LACP is ~1200 LOC but is the
   industry standard. Phasing options: (a) ship both in v1
   (preferred), (b) ship balance-xor in v1 and LACP as
   `network-management-v2`, (c) ship LACP only and skip
   balance-xor.
3. **mDNS default**: on by default everywhere, on by default
   only on interfaces tagged `role=admin`, or off by default
   and explicit opt-in? Leaning on-by-default-on-admin-only —
   keeps the data plane quiet.
4. **DHCPv6 vs SLAAC** when both are offered (the M and O
   bits in the RA): follow the bits strictly, or prefer SLAAC
   regardless? Leaning follow the bits (RFC-compliant).
5. **Route metric defaults**: if every interface defaults to
   metric 100, the order of `ip` configuration determines the
   primary uplink — fragile. Leaning on a deterministic
   ordering (alphabetical interface name) for ties.
6. **LLDP TX in v1?** Easy to add (~50 extra LOC) and useful
   for the operator-side switch to identify our box. Could
   creep into v1 cleanly. Currently flagged out-of-scope but
   reconsider.
