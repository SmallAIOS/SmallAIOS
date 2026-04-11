## MODIFIED Requirements

### Requirement: Bus Backend Selection
The container binary SHALL start a dataflow runner thread alongside the HTTP server based on environment configuration and actually process inference requests via the selected transport.

#### Scenario: Start with Zenoh bus backend
- **WHEN** `SMALLAIOS_BUS_BACKEND=zenoh` is set
- **THEN** the container MUST load all registered models into `Session` instances
- **AND** MUST instantiate a `DataflowRunner` containing those sessions
- **AND** MUST spawn a background thread running `serve_dataflow_runner()` against an in-process pub/sub subscriber
- **AND** the HTTP server MUST also remain available on the same process
- **AND** both MUST share the same loaded models

#### Scenario: Start with DDS bus backend
- **WHEN** `SMALLAIOS_BUS_BACKEND=dds` is set
- **THEN** the container MUST start the dataflow runner with the DDS-Zenoh adapter bridging inference topics

#### Scenario: Start with CAN bus backend
- **WHEN** `SMALLAIOS_BUS_BACKEND=can` is set
- **AND** `SMALLAIOS_CAN_DEVICE` specifies a valid device (loopback/mcp2515/axi)
- **THEN** the container MUST instantiate the CAN controller
- **AND** MUST load the routing table from `SMALLAIOS_CAN_ROUTING` if specified
- **AND** MUST spawn a thread that feeds received frames through the `CanInferenceAdapter` and the runner, then transmits result frames

#### Scenario: HTTP-only by default
- **WHEN** `SMALLAIOS_BUS_BACKEND` is unset or set to `none`
- **THEN** only the HTTP server MUST start
- **AND** no dataflow runner MUST be initialized
- **AND** no background threads MUST be spawned for bus processing

#### Scenario: Graceful shutdown of bus runner
- **WHEN** the container receives SIGTERM
- **THEN** both the HTTP server and any active dataflow runner MUST stop accepting new requests
- **AND** the main thread MUST join runner threads before exiting
- **AND** in-flight inference requests MUST be allowed to complete

#### Scenario: No models loaded
- **WHEN** the bus backend is configured but `ModelManager` has zero loaded models
- **THEN** the container MUST log a warning
- **AND** MUST fall back to HTTP-only mode
- **AND** MUST NOT crash
