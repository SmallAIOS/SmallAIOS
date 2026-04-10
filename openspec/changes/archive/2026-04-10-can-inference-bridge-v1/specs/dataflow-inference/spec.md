## ADDED Requirements

### Requirement: CAN Transport Backend
The dataflow inference runner SHALL support CAN as a message transport via the `CanInferenceAdapter`.

#### Scenario: CAN backend processes inference round-trip
- **WHEN** the runner is started with a CAN adapter and routing table
- **AND** CAN frames matching the routing table arrive
- **THEN** the adapter MUST batch them, the runner MUST process the batch through `Session::run()`, and the result MUST be published as outbound CAN frames
