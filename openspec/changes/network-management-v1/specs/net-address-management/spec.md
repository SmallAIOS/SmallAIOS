## ADDED Requirements

### Requirement: Per-Interface Network Configuration File

The `net` crate SHALL be configured through one TOML file per interface at `/data/network/<iface>.toml`. Each file SHALL accept a `mode` field taking exactly one of `"dhcp"`, `"static"`, `"slaac"`, or `"dhcp_then_static"`. For `mode = "static"` the file SHALL accept `ipv4`, `ipv6`, `gateway`, `dns`, and `mtu` fields.

#### Scenario: Static configuration applied at boot

- **WHEN** `/data/network/eth0.toml` contains `mode = "static"` with `ipv4`, `gateway`, `dns`, and `mtu` set
- **THEN** at boot `eth0` SHALL be configured with exactly those values
- **AND** the configured gateway SHALL be installed in the route table
- **AND** no DHCP traffic SHALL be emitted on `eth0`

#### Scenario: DHCP mode delegates to the DHCP client

- **WHEN** `/data/network/eth0.toml` contains `mode = "dhcp"`
- **THEN** the DHCPv4 client SHALL be started on `eth0`
- **AND** no static address SHALL be applied to `eth0`

#### Scenario: Unrecognized mode is rejected

- **WHEN** a configuration is committed with `mode = "bridged"` (not one of the four defined modes)
- **THEN** the commit SHALL be rejected
- **AND** the previously active configuration for that interface SHALL remain in effect

### Requirement: DHCP-Then-Static Fallback

For `mode = "dhcp_then_static"`, the interface SHALL attempt DHCP for a configurable window (default 30 seconds, overridable per interface) and, if no lease is acquired within the window, SHALL apply the fallback static address configured in the same file.

#### Scenario: DHCP succeeds within the window

- **WHEN** an interface in `dhcp_then_static` mode receives a DHCPACK 5 seconds after boot
- **THEN** the leased address SHALL be used
- **AND** the fallback static address SHALL NOT be applied

#### Scenario: Fallback applied after the window expires

- **WHEN** an interface in `dhcp_then_static` mode receives no DHCP lease within the default 30-second window
- **THEN** the fallback static address SHALL be applied at window expiry
- **AND** the interface SHALL be usable with the static address without a reboot

#### Scenario: Per-interface window override

- **WHEN** the interface configuration overrides the DHCP window to 5 seconds
- **THEN** the fallback static address SHALL be applied 5 seconds after DHCP starts, not 30

### Requirement: Management Config Model Integration

`network/<iface>.toml` and `network/<bond>.toml` SHALL be added to the `mgmt` `Config` model and served through the existing TOML, TTY, and Zenoh `ConfigSurface` implementations from `management-login-v1`. Live configuration changes SHALL apply on commit and SHALL roll back on apply failure, using the same atomic-rewrite pattern as the shadow file.

#### Scenario: Committed change applies live

- **WHEN** an operator commits a change to `network/eth0.toml` through any `ConfigSurface`
- **THEN** the new configuration SHALL be applied to `eth0` without a reboot
- **AND** the file at `/data/network/eth0.toml` SHALL be rewritten atomically

#### Scenario: Failed apply rolls back

- **WHEN** a committed network configuration fails to apply
- **THEN** the previous configuration SHALL be restored on the interface
- **AND** the on-disk file SHALL still contain the previous (pre-commit) contents

#### Scenario: Interrupted rewrite leaves prior config intact

- **WHEN** power is lost mid-way through a configuration rewrite
- **THEN** on the next boot the interface SHALL come up with either the complete old or the complete new configuration, never a partial file

### Requirement: Interface Role and Metric Tagging

Each interface configuration SHALL accept `role = "admin" | "data" | "any"` (defaulting to `"any"` when absent) and `metric = u32`. The `role` value SHALL classify the interface for traffic placement (`admin` carries the Zenoh admin/telemetry plane, `data` carries inference traffic) and the `metric` value SHALL feed the routing-table priority for routes via that interface.

#### Scenario: Role defaults to any

- **WHEN** `/data/network/eth1.toml` omits the `role` field
- **THEN** the interface SHALL be treated as `role = "any"`

#### Scenario: Metric propagates to route entries

- **WHEN** `eth0` is configured with `metric = 50` and `eth1` with `metric = 200`
- **THEN** routes installed via `eth0` SHALL carry metric 50 and routes via `eth1` SHALL carry metric 200 in the route table
