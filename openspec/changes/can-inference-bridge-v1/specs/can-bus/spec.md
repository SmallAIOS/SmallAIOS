## ADDED Requirements

### Requirement: CanController Adapter Hook
The CAN bus crate SHALL provide a generic `CanFrameSink` trait that adapter modules (like the inference bridge) can implement to receive frames from any CanController.

#### Scenario: Sink receives frames from controller
- **WHEN** a `CanController` implementation receives a frame
- **AND** a `CanFrameSink` is attached
- **THEN** the controller MUST call `sink.on_frame(&frame)` for each received frame

#### Scenario: Multiple sinks supported
- **WHEN** a controller has multiple sinks attached
- **THEN** all sinks MUST receive each frame in registration order
