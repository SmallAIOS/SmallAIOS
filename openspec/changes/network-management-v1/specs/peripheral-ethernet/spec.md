## ADDED Requirements

### Requirement: Link-State-Change Notifications

Ethernet drivers SHALL deliver link-state-change notifications (link-up and link-down events) to network-stack subscribers. These notifications are required by bond active-backup failover and by LACP timing.

#### Scenario: Cable pull delivers link-down

- **WHEN** carrier is lost on an ethernet interface (e.g., cable unplugged)
- **THEN** the driver SHALL deliver a link-down notification for that interface to its subscribers
- **AND** a bond enslaving the interface SHALL be able to trigger failover from that notification

#### Scenario: Link restoration delivers link-up

- **WHEN** carrier returns on an ethernet interface
- **THEN** the driver SHALL deliver a link-up notification for that interface
- **AND** a bond enslaving the interface SHALL be able to return it to service
