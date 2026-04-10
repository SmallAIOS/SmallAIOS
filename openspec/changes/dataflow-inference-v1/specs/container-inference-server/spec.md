## ADDED Requirements

### Requirement: Bus Backend Selection
The container binary SHALL support starting a dataflow runner alongside the HTTP server based on environment configuration.

#### Scenario: Start with Zenoh bus backend
- **WHEN** `SMALLAIOS_BUS_BACKEND=zenoh` is set
- **THEN** the container MUST start the dataflow runner with Zenoh transport
- **AND** the HTTP server MUST also remain available
- **AND** both MUST share the same `ModelManager` instance

#### Scenario: Start with DDS bus backend
- **WHEN** `SMALLAIOS_BUS_BACKEND=dds` is set
- **THEN** the container MUST start the dataflow runner with DDS transport via the Zenoh adapter

#### Scenario: HTTP-only by default
- **WHEN** `SMALLAIOS_BUS_BACKEND` is unset or set to `none`
- **THEN** only the HTTP server MUST start
- **AND** no dataflow runner MUST be initialized

#### Scenario: Graceful shutdown of bus runner
- **WHEN** the container receives SIGTERM
- **THEN** both the HTTP server and dataflow runner MUST stop accepting new requests
- **AND** in-flight inference requests MUST complete before exit
