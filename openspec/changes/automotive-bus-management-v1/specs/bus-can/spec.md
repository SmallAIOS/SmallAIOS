## ADDED Requirements

### Requirement: Management And Inference Coexistence On A Shared CAN Interface

The management ISO-TP listener and the existing CAN inference bridge SHALL be able to coexist on the same physical CAN interface by using disjoint CAN-ID ranges: frames carrying the configured diagnostic request CAN ID SHALL be routed to the ISO-TP/UDS handler, and frames matching the inference routing table SHALL be routed to the inference bridge.

#### Scenario: Both planes operate on one interface

- **WHEN** the inference bridge and the ISO-TP management listener are both configured on the same CAN interface with disjoint CAN-ID ranges
- **THEN** inference frames SHALL continue to be processed by the routing table
- **AND** diagnostic requests on the configured diagnostic CAN ID SHALL be answered by the UDS handler

#### Scenario: Frames are never cross-delivered

- **WHEN** a frame arrives carrying the diagnostic request CAN ID
- **THEN** it SHALL NOT be delivered to the inference routing table
- **AND** a frame matching an inference route SHALL NOT be delivered to the ISO-TP listener
