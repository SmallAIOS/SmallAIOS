## ADDED Requirements

### Requirement: Active-Backup Failover

The `net` crate SHALL provide an `active-backup` bond mode (Linux mode 1) in `net/src/bond/`: one slave is primary and carries all traffic, remaining slaves are standby, and on a link-down notification for the primary the bond SHALL fail over to a standby slave. The mode SHALL require no switch cooperation and SHALL emit no bonding control-protocol frames.

#### Scenario: Primary link-down triggers failover

- **WHEN** a bond in `active-backup` mode has primary `eth0` and standby `eth1`
- **AND** the ethernet driver delivers a link-down notification for `eth0`
- **THEN** the bond SHALL promote `eth1` to carry traffic
- **AND** traffic through the bond SHALL continue without upper-layer reconfiguration

#### Scenario: No switch cooperation required

- **WHEN** a bond in `active-backup` mode operates against two unmanaged switches on different segments
- **THEN** the bond SHALL function without emitting LACPDUs or any other switch-cooperation frames

### Requirement: Bond Identity Preserved Across Failover

In `active-backup` mode, failover SHALL preserve the bond's virtual-interface identity: the bond's MAC address, addresses, and DHCP lease SHALL remain unchanged so upper layers (DHCP, mDNS, routing) observe no interface change.

#### Scenario: MAC and lease survive failover

- **WHEN** a failover from `eth0` to `eth1` occurs on a bond holding a DHCP lease
- **THEN** the bond SHALL continue to present the same MAC address
- **AND** the DHCP lease SHALL remain in effect without re-acquisition
- **AND** the route table entries via the bond SHALL be unchanged
