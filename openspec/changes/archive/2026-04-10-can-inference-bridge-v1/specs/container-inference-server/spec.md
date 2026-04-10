## ADDED Requirements

### Requirement: CAN Backend Selection
The container binary SHALL support `SMALLAIOS_BUS_BACKEND=can` with device and routing configuration via environment variables.

#### Scenario: CAN loopback for testing
- **WHEN** `SMALLAIOS_BUS_BACKEND=can` and `SMALLAIOS_CAN_DEVICE=loopback` are set
- **THEN** the container MUST instantiate a loopback CanController
- **AND** MUST attach the CanInferenceAdapter to it
- **AND** MUST connect the adapter to the dataflow runner

#### Scenario: MCP2515 SPI controller
- **WHEN** `SMALLAIOS_CAN_DEVICE=mcp2515:/dev/spidev0.0` is set
- **THEN** the container MUST instantiate the MCP2515 driver attached to that SPI device
- **AND** MUST initialize it before starting the runner

#### Scenario: AXI CAN FPGA controller
- **WHEN** `SMALLAIOS_CAN_DEVICE=axi:0x40000000` is set
- **THEN** the container MUST instantiate the AXI CAN driver at the specified MMIO base address

#### Scenario: Routing table loaded from file
- **WHEN** `SMALLAIOS_CAN_ROUTING=/etc/smallaios/can-routes.toml` is set
- **THEN** the container MUST parse the TOML file to build the input and output routing tables
- **AND** MUST fail startup with an error if the file is missing or malformed

#### Scenario: Unknown CAN device falls back gracefully
- **WHEN** `SMALLAIOS_CAN_DEVICE` has an invalid format
- **THEN** the container MUST log an error and disable the CAN backend
- **AND** MUST NOT crash
