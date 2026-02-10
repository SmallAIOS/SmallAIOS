# Delta for CAN Bus Protocol

## ADDED Requirements

### Requirement: CAN 2.0A Standard Frame
The CAN driver SHALL encode and decode CAN 2.0A standard frames with 11-bit identifiers, 0-8 byte payloads, and CRC-15 checksums per ISO 11898.

#### Scenario: Encode a standard CAN frame
- WHEN the application submits a message with an 11-bit identifier and up to 8 bytes of payload
- THEN the encoder MUST produce a valid CAN 2.0A frame with correct SOF, arbitration field, control field (DLC), data field, CRC-15, ACK slot, and EOF
- AND the CRC MUST be computed using the ISO 11898 generator polynomial (0x4599)

#### Scenario: Decode a standard CAN frame
- WHEN a valid CAN 2.0A frame is received from the bus
- THEN the decoder MUST extract the 11-bit identifier, DLC, and 0-8 byte data payload
- AND MUST verify the CRC-15 and reject frames with incorrect CRC

#### Scenario: Reject oversized payload
- WHEN the application attempts to encode a CAN 2.0A frame with more than 8 bytes of payload
- THEN the encoder MUST return an error indicating payload exceeds the maximum 8-byte limit

### Requirement: CAN 2.0B Extended Frame
The CAN driver SHALL encode and decode CAN 2.0B extended frames with 29-bit identifiers per ISO 11898.

#### Scenario: Encode an extended CAN frame
- WHEN the application submits a message with a 29-bit extended identifier
- THEN the encoder MUST produce a valid CAN 2.0B frame with the IDE bit set, the 11-bit base identifier, the SRR bit, and the 18-bit extension identifier
- AND the CRC-15 MUST cover the entire frame including the extended arbitration field

#### Scenario: Decode an extended CAN frame
- WHEN a valid CAN 2.0B frame with IDE bit set is received from the bus
- THEN the decoder MUST reconstruct the full 29-bit identifier from the base and extension fields
- AND MUST extract the DLC and data payload correctly

#### Scenario: Distinguish standard from extended frames
- WHEN the decoder processes an incoming frame
- THEN it MUST use the IDE bit to determine whether the frame is CAN 2.0A (11-bit) or CAN 2.0B (29-bit)
- AND MUST report the frame type to the application layer

### Requirement: CAN FD Frame Support
The CAN driver SHALL support CAN FD frames with payloads up to 64 bytes and bit rate switching per ISO 11898-1:2015.

#### Scenario: Encode a CAN FD frame with bit rate switching
- WHEN the application submits a CAN FD message with the BRS flag set and up to 64 bytes of payload
- THEN the encoder MUST produce a valid CAN FD frame with the FDF bit set, BRS bit set, and ESI bit reflecting the error state
- AND the DLC MUST encode the payload length using the CAN FD DLC mapping (12, 16, 20, 24, 32, 48, 64 bytes)

#### Scenario: Decode a CAN FD frame
- WHEN a valid CAN FD frame is received from the bus
- THEN the decoder MUST detect the FDF bit, extract the BRS and ESI flags, and decode the payload up to 64 bytes
- AND MUST use the CAN FD CRC (CRC-17 for payloads <= 16 bytes, CRC-21 for payloads > 16 bytes)

#### Scenario: Reject CAN FD frames on classic-only controller
- WHEN a CAN FD frame is received but the controller is configured for classic CAN only
- THEN the driver MUST generate an error frame and discard the FD frame

### Requirement: Bus State Machine
The CAN driver SHALL implement the CAN bus error state machine with Error Active, Error Passive, and Bus Off states, including automatic recovery.

#### Scenario: Transition from Error Active to Error Passive
- WHEN either the transmit error counter (TEC) or receive error counter (REC) exceeds 127
- THEN the controller MUST transition to the Error Passive state
- AND MUST send passive error flags instead of active error flags

#### Scenario: Transition from Error Passive to Bus Off
- WHEN the transmit error counter exceeds 255
- THEN the controller MUST transition to the Bus Off state
- AND MUST cease all bus activity immediately

#### Scenario: Bus Off recovery
- WHEN the controller is in Bus Off state and 128 occurrences of 11 consecutive recessive bits have been observed
- THEN the controller MUST transition back to the Error Active state with TEC and REC reset to 0
- AND MUST notify the application layer of the recovery event

#### Scenario: Error counter decrement on successful operation
- WHEN a frame is successfully transmitted or received
- THEN the relevant error counter MUST be decremented by 1 (if greater than 0)

### Requirement: Acceptance Filtering
The CAN driver SHALL support hardware mask-based filtering and software-configurable acceptance filters for frame reception.

#### Scenario: Hardware mask filter match
- WHEN a hardware acceptance filter is configured with an ID mask and match value
- AND an incoming frame's identifier matches (frame_id AND mask) == match_value
- THEN the frame MUST be accepted and delivered to the receive buffer

#### Scenario: Hardware mask filter reject
- WHEN an incoming frame's identifier does not match any configured hardware filter
- THEN the frame MUST be silently discarded without consuming receive buffer space

#### Scenario: Software filter refinement
- WHEN a frame passes the hardware filter but does not match any registered software filter callback
- THEN the frame MUST be discarded by the software filter layer
- AND MUST NOT be delivered to the application

#### Scenario: Filter reconfiguration at runtime
- WHEN the application requests a change to the acceptance filter configuration
- THEN the driver MUST apply the new filter settings without losing frames that are currently being received

### Requirement: CAN Controller Driver Abstraction
The CAN driver SHALL provide a hardware abstraction layer supporting PS-CAN (Zynq), AXI CAN (FPGA), and MCP2515 (SPI) controllers.

#### Scenario: Initialize PS-CAN controller on Zynq
- WHEN SmallAIOS boots on a Zynq platform with a PS-CAN peripheral
- THEN the driver MUST initialize the PS-CAN registers, configure the baud rate, set up TX/RX FIFOs, and enable interrupts
- AND the CAN interface MUST be ready to send and receive frames

#### Scenario: Initialize AXI CAN controller on FPGA
- WHEN an AXI CAN IP core is detected in the FPGA fabric
- THEN the driver MUST configure the AXI CAN registers via memory-mapped I/O and enable the controller
- AND MUST support both standard and extended frame formats

#### Scenario: Initialize MCP2515 over SPI
- WHEN an MCP2515 CAN controller is connected via SPI bus
- THEN the driver MUST initialize the MCP2515 via SPI commands, configure bit timing, and set up receive filters
- AND MUST use interrupt-driven reception to minimize latency

#### Scenario: Portable frame send across controllers
- WHEN the application sends a CAN frame through the abstraction layer
- THEN the abstraction layer MUST dispatch the frame to the correct controller driver without the application needing controller-specific knowledge

### Requirement: Zenoh Transport Adapter for CAN
The CAN driver SHALL provide a Zenoh transport adapter mapping CAN frame identifiers to Zenoh key expressions using the pattern `can/{bus_id}/{frame_id}`.

#### Scenario: Publish received CAN frame to Zenoh
- WHEN a CAN frame with identifier 0x1A3 is received on bus 0
- THEN the adapter MUST publish the frame payload to Zenoh key expression `can/0/0x1A3`
- AND the payload MUST include the raw data bytes and a metadata header with DLC, timestamp, and frame type

#### Scenario: Subscribe to Zenoh and transmit on CAN bus
- WHEN a Zenoh subscriber matches key expression `can/0/0x100`
- AND a Zenoh publication is received on that key expression
- THEN the adapter MUST decode the payload and transmit the corresponding CAN frame on bus 0 with identifier 0x100

#### Scenario: Wildcard subscription for all frames on a bus
- WHEN a Zenoh subscriber registers for key expression `can/0/**`
- THEN the adapter MUST deliver all CAN frames received on bus 0 to the subscriber

### Requirement: CANaerospace Profile Support
The CAN driver SHALL support the CANaerospace protocol profile for civil aviation CAN applications, including standard message identifiers and data type mappings.

#### Scenario: Encode a CANaerospace normal operation data message
- WHEN the application publishes an air data parameter (e.g., indicated airspeed) via the CANaerospace profile
- THEN the encoder MUST format the message using the CANaerospace NOD (Normal Operation Data) message type with the correct node ID, data type, and service code fields

#### Scenario: Decode a CANaerospace emergency event message
- WHEN a CANaerospace EED (Emergency Event Data) message is received
- THEN the decoder MUST extract the emergency event code, the originating node ID, and the associated data
- AND MUST deliver the decoded event to the application with elevated priority

#### Scenario: Node service protocol exchange
- WHEN the application initiates a CANaerospace Node Service (NSS) request to identify a remote node
- THEN the driver MUST send the NSS request and process the NSS response containing the remote node's hardware and software revision

### Requirement: TLA+ Model for CAN Arbitration
The project SHALL provide a TLA+ formal model verifying CAN arbitration correctness under concurrent transmission.

#### Scenario: Verify priority-based arbitration
- WHEN the TLA+ model simulates two or more nodes transmitting simultaneously
- THEN the model MUST verify that the node with the numerically lowest identifier always wins arbitration
- AND all losing nodes MUST detect the loss and retry without error

#### Scenario: Verify no message loss under full bus load
- WHEN the TLA+ model simulates sustained maximum bus utilization
- THEN the model MUST verify that every transmitted message is eventually delivered (liveness property)
- AND no message is silently dropped due to arbitration failure

### Requirement: Clean-Room Implementation
All CAN protocol implementations SHALL be clean-room developed from the ISO 11898 public specification without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the CAN module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the ISO 11898 public specification, publicly available errata, and BOSCH CAN 2.0 specification as reference sources
- AND MUST NOT contain code derived from proprietary CAN stack implementations
