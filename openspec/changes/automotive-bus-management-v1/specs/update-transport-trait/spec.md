## ADDED Requirements

### Requirement: UDS ISO-TP Update Transport

The `automotive` crate SHALL provide `UdsIsoTpTransport`, a third implementation of the `update::Transport` trait from `remote-update-v1`, backed by the UDS `0x34 Request Download` / `0x36 Transfer Data` / `0x37 Request Transfer Exit` flow over ISO-TP. The trait itself SHALL NOT be modified; the existing A/B-slot machinery SHALL handle a UDS-driven update unchanged.

#### Scenario: UdsIsoTpTransport implements the unchanged trait

- **WHEN** a reviewer reads the public API of the `automotive` crate
- **THEN** `UdsIsoTpTransport` SHALL implement `update::Transport`
- **AND** the `update::Transport` trait definition SHALL be unchanged by this change

#### Scenario: A/B update over loopback CAN uses the shared slot machinery

- **WHEN** the end-to-end test drives an update through `UdsIsoTpTransport` over loopback CAN
- **THEN** the image SHALL be staged and applied by the same A/B-slot machinery used by the other `remote-update-v1` transports
- **AND** no UDS-specific slot-handling code path SHALL exist
