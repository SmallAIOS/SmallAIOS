## ADDED Requirements

### Requirement: Runner Thread Lifecycle
The container SHALL manage the dataflow runner thread lifecycle alongside the HTTP server.

#### Scenario: Runner thread spawned on bus backend activation
- **WHEN** the container activates a bus backend (zenoh/dds/can)
- **THEN** a background thread MUST be spawned with an `Arc<AtomicBool>` shutdown flag
- **AND** the thread MUST loop until the shutdown flag is set

#### Scenario: Runner processes messages in its loop iteration
- **WHEN** the runner loop iterates
- **THEN** it MUST drain pending messages from its transport
- **AND** for each message call `DataflowRunner::process_message()` or equivalent
- **AND** publish results back to the transport
- **AND** sleep briefly if no messages are available (avoid busy-loop)
