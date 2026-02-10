# Delta for SpaceWire (ECSS-E-ST-50-12C)

## ADDED Requirements

### Requirement: SpaceWire Packet Encode/Decode
The SpaceWire driver SHALL encode and decode SpaceWire packets with destination address, cargo (payload), and end-of-packet markers (EOP/EEP) per the publicly available ECSS-E-ST-50-12C standard.

#### Scenario: Encode a SpaceWire packet
- WHEN the application submits a destination address and payload for transmission
- THEN the encoder MUST produce a valid SpaceWire packet with the destination address byte(s) as the leading cargo bytes, followed by the payload data, terminated by an EOP (End of Packet) marker
- AND the destination address MUST support both path addressing (single byte consumed at each router) and logical addressing (single byte, not consumed)

#### Scenario: Decode a SpaceWire packet
- WHEN a complete SpaceWire packet terminated by EOP or EEP is received on a link
- THEN the decoder MUST extract the destination address and payload cargo
- AND MUST distinguish between normal packet termination (EOP) and error packet termination (EEP)

#### Scenario: Handle error end of packet
- WHEN a packet is received with an EEP (Error End of Packet) marker
- THEN the decoder MUST deliver the partial payload to the application with an error indication
- AND MUST increment a per-link EEP counter

### Requirement: Link Interface State Machine
The SpaceWire driver SHALL implement the link interface state machine with states ErrorReset, ErrorWait, Ready, Started, Connecting, and Run per ECSS-E-ST-50-12C.

#### Scenario: Normal link initialization sequence
- WHEN both ends of a SpaceWire link are powered on and enabled
- THEN the link state machine MUST transition through ErrorReset, ErrorWait, Ready, Started, Connecting, and finally Run
- AND data transfer MUST only be permitted in the Run state

#### Scenario: Link disconnect detection
- WHEN the link is in the Run state and a disconnect timeout (850 ns) elapses without receiving any character
- THEN the state machine MUST transition to ErrorReset
- AND MUST notify the application of the link disconnect event

#### Scenario: ErrorReset recovery
- WHEN the link enters the ErrorReset state
- THEN the state machine MUST reset the link interface, wait for the reset timeout (6.4 us)
- AND MUST then transition to ErrorWait to begin the reconnection sequence

#### Scenario: Link autostart
- WHEN the autostart flag is enabled and a valid NULL character is received in the Ready state
- THEN the state machine MUST transition directly to Started
- AND MUST begin transmitting NULL characters to facilitate link establishment

### Requirement: Character-Level Encoding
The SpaceWire driver SHALL implement character-level encoding for data characters, control characters (FCT, EOP, EEP, ESC), and NULL characters.

#### Scenario: Encode a data character
- WHEN the link transmits a data byte
- THEN the encoder MUST produce a 10-bit data character with 1 parity bit, 1 data flag bit (0), and 8 data bits
- AND the parity MUST cover the previous character's parity bit and the current character

#### Scenario: Encode a control character
- WHEN the link needs to transmit a flow control token (FCT)
- THEN the encoder MUST produce a 4-bit control character with 1 parity bit, 1 data flag bit (1), and 2-bit control code (FCT = 00)
- AND FCT characters MUST be used for credit-based flow control

#### Scenario: NULL character generation
- WHEN the link is in Started, Connecting, or Run state with no data to send
- THEN the transmitter MUST continuously send NULL characters (ESC + FCT) to maintain link synchronization
- AND NULL characters MUST be sent at least every 850 ns to prevent disconnect timeout

#### Scenario: Flow control credit management
- WHEN the receiver has buffer space available for at least 8 N-Chars
- THEN the link MUST send an FCT to grant one credit (8 N-Chars) to the remote transmitter
- AND the transmitter MUST NOT send data characters when its credit count is zero

### Requirement: Time-Code Distribution
The SpaceWire driver SHALL support 6-bit time-code broadcast for time synchronization across the SpaceWire network.

#### Scenario: Transmit a time-code
- WHEN the time master node generates a time tick
- THEN the driver MUST transmit a time-code character containing a 2-bit control field and 6-bit time counter value
- AND time-codes MUST be transmitted with the highest priority, pre-empting any packet in progress

#### Scenario: Receive and propagate a time-code
- WHEN a time-code is received on a link
- THEN the driver MUST compare the received time value with the local time counter
- AND MUST accept the time-code if the received value equals (local_time + 1) mod 64
- AND MUST propagate the time-code on all other links (if acting as a router)

#### Scenario: Reject invalid time-code
- WHEN a received time-code value does not equal (local_time + 1) mod 64
- THEN the driver MUST discard the time-code
- AND MUST increment a time-code error counter

### Requirement: RMAP Protocol Support
The SpaceWire driver SHALL implement the Remote Memory Access Protocol (RMAP) for read and write commands over SpaceWire links.

#### Scenario: RMAP write command
- WHEN the application issues an RMAP write to a remote target at a specified memory address
- THEN the driver MUST construct an RMAP write command packet with the target logical address, key, memory address, and write data
- AND MUST include the RMAP CRC-8 for header and data integrity

#### Scenario: RMAP read command
- WHEN the application issues an RMAP read for a specified memory address and length on a remote target
- THEN the driver MUST construct an RMAP read command packet with the target logical address, key, memory address, and read length
- AND MUST process the RMAP read reply containing the requested data

#### Scenario: RMAP write reply verification
- WHEN an RMAP write is configured to require acknowledgment
- THEN the driver MUST wait for the RMAP write reply from the target
- AND MUST verify the reply status field indicates successful completion (status = 0)
- AND MUST report any error status to the application

#### Scenario: RMAP CRC verification
- WHEN an RMAP reply packet is received
- THEN the driver MUST verify the CRC-8 of both the header and data sections
- AND MUST reject packets with CRC mismatches and report the error

### Requirement: Link Speed Configuration
The SpaceWire driver SHALL support link speed configuration in the range of 2-400 Mbps.

#### Scenario: Configure link speed
- WHEN the application configures a SpaceWire link for 200 Mbps operation
- THEN the driver MUST set the transmit clock divisor to achieve the requested bit rate within 1% accuracy
- AND MUST negotiate the link speed during the initialization sequence

#### Scenario: Minimum speed operation
- WHEN the link is configured for 2 Mbps (minimum speed)
- THEN the driver MUST operate at 2 Mbps for the initial connection handshake
- AND MUST support speed increase after link establishment if both ends support higher rates

#### Scenario: Reject out-of-range speed
- WHEN the application configures a link speed below 2 Mbps or above 400 Mbps
- THEN the driver MUST return an error indicating the speed is out of the supported range

### Requirement: Zenoh Transport Adapter for SpaceWire
The SpaceWire driver SHALL provide a Zenoh transport adapter mapping SpaceWire destinations to Zenoh key expressions using the pattern `spw/{link}/{dest}`.

#### Scenario: Publish received SpaceWire packet to Zenoh
- WHEN a SpaceWire packet is received on link 0 with logical destination address 42
- THEN the adapter MUST publish the packet cargo to Zenoh key expression `spw/0/42`
- AND the payload MUST include the raw cargo data and metadata (source path, EOP/EEP status, timestamp)

#### Scenario: Subscribe to Zenoh and transmit SpaceWire packet
- WHEN a Zenoh subscriber matches key expression `spw/1/10`
- AND a Zenoh publication is received on that key expression
- THEN the adapter MUST construct a SpaceWire packet with destination address 10 and transmit it on link 1

#### Scenario: Wildcard subscription for all destinations on a link
- WHEN a Zenoh subscriber registers for key expression `spw/0/**`
- THEN the adapter MUST deliver all SpaceWire packets received on link 0 to the subscriber

### Requirement: Hardware Interface Abstraction
The SpaceWire driver SHALL provide a hardware abstraction layer supporting LVDS PHY transceivers and FPGA SpaceWire codec IP cores.

#### Scenario: Initialize LVDS PHY SpaceWire transceiver
- WHEN SmallAIOS boots with a dedicated SpaceWire LVDS PHY connected
- THEN the driver MUST initialize the PHY, configure data strobe encoding, and enable the link interface
- AND the link state machine MUST begin the initialization sequence

#### Scenario: Initialize FPGA SpaceWire codec IP
- WHEN an FPGA-based SpaceWire codec IP core is detected via memory-mapped I/O
- THEN the driver MUST configure the codec registers, set the link speed, and enable interrupt-driven operation
- AND MUST support simultaneous operation of multiple SpaceWire links

#### Scenario: Portable packet interface
- WHEN the application sends or receives SpaceWire packets through the abstraction layer
- THEN the abstraction MUST provide a uniform API regardless of the underlying SpaceWire hardware

### Requirement: Clean-Room Implementation
All SpaceWire implementations SHALL be clean-room developed from the publicly available ECSS-E-ST-50-12C standard without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the SpaceWire module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the publicly available ECSS-E-ST-50-12C standard, ECSS-E-ST-50-52C (RMAP), and publicly available ESA technical notes as reference sources
- AND MUST NOT contain code derived from proprietary SpaceWire stack implementations
