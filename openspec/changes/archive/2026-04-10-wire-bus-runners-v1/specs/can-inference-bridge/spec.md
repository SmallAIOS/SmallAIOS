## ADDED Requirements

### Requirement: CAN Runner Instantiation
The container SHALL instantiate a CAN controller and inference adapter when the CAN bus backend is active.

#### Scenario: Loopback device spec creates MockCanController
- **WHEN** `SMALLAIOS_CAN_DEVICE=loopback`
- **THEN** the container MUST instantiate `MockCanController::new()`
- **AND** attach it to the `CanInferenceAdapter`

#### Scenario: Hardware device specs produce warnings
- **WHEN** `SMALLAIOS_CAN_DEVICE=mcp2515:/dev/spidev0.0` or `axi:0x40000000`
- **THEN** the container MAY log a warning that hardware support is not yet wired
- **AND** MAY fall back to `MockCanController` until real hardware initialization is implemented
- **AND** MUST NOT crash

#### Scenario: Routing file missing
- **WHEN** `SMALLAIOS_CAN_ROUTING` points to a file that does not exist
- **THEN** the container MUST log an error
- **AND** the CAN runner MUST not start
- **AND** the container MUST fall back to HTTP-only mode
