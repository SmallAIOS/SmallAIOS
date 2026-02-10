# Delta for MIL-STD-1553B Military Avionics Data Bus

## ADDED Requirements

### Requirement: Command Word Encode/Decode
The MIL-STD-1553B driver SHALL encode and decode 20-bit command words with RT address, transmit/receive bit, subaddress, word count or mode code, and sync pattern per the MIL-STD-1553B public specification.

#### Scenario: Encode a command word
- WHEN the Bus Controller prepares a command for RT address 5, subaddress 3, with word count 8 and transmit/receive bit set to receive
- THEN the encoder MUST produce a valid 20-bit command word with 3-bit sync (command sync), 5-bit RT address, 1-bit T/R, 5-bit subaddress, 5-bit word count, and odd parity bit
- AND the command sync pattern MUST be distinguishable from the data sync pattern

#### Scenario: Decode a command word
- WHEN a 20-bit word with command sync pattern is received from the bus
- THEN the decoder MUST extract the RT address, T/R bit, subaddress, and word count or mode code
- AND MUST verify the odd parity bit and reject words with parity errors

#### Scenario: Mode code command detection
- WHEN the subaddress field is 0b00000 or 0b11111
- THEN the decoder MUST interpret the word count field as a mode code instead of a data word count
- AND MUST route the command to the mode code handler

### Requirement: Status Word Encode/Decode
The MIL-STD-1553B driver SHALL encode and decode 20-bit status words with RT address, message error, instrumentation, service request, broadcast received, busy, subsystem flag, dynamic bus control acceptance, and terminal flag fields.

#### Scenario: Encode a status word
- WHEN the Remote Terminal generates a response status
- THEN the encoder MUST produce a valid 20-bit status word with 3-bit sync (data sync), 5-bit RT address, and 11 status bits including message error, instrumentation, service request, reserved, broadcast received, busy, subsystem flag, dynamic bus control acceptance, terminal flag, and odd parity

#### Scenario: Decode a status word
- WHEN a 20-bit word with data sync pattern is received in the status word position of a message
- THEN the decoder MUST extract the RT address and all status flag bits
- AND MUST verify odd parity and report any set error flags to the Bus Controller

#### Scenario: Detect busy RT
- WHEN a decoded status word has the busy bit set
- THEN the Bus Controller MUST schedule a retry for the command after the configured retry interval
- AND MUST NOT treat the busy response as a message error

### Requirement: Data Word Encode/Decode
The MIL-STD-1553B driver SHALL encode and decode 20-bit data words with 3-bit sync, 16-bit payload, and odd parity.

#### Scenario: Encode a data word
- WHEN the application provides a 16-bit data value for transmission
- THEN the encoder MUST produce a valid 20-bit data word with data sync pattern, 16-bit data field, and odd parity bit

#### Scenario: Decode a data word
- WHEN a 20-bit word with data sync pattern is received in a data word position
- THEN the decoder MUST extract the 16-bit payload
- AND MUST verify odd parity and reject words with parity errors

#### Scenario: Multi-word data transfer
- WHEN a command specifies word count N (1-32)
- THEN the encoder/decoder MUST handle exactly N consecutive data words following the command
- AND each data word MUST be independently parity-checked

### Requirement: Bus Controller Mode
The MIL-STD-1553B driver SHALL implement Bus Controller (BC) mode with command scheduling, response timeout detection, and retry logic.

#### Scenario: Schedule and execute a BC-to-RT transfer
- WHEN the BC schedules a command to write data to RT 7, subaddress 2
- THEN the BC MUST transmit the command word followed by the data words
- AND MUST wait for the RT status word response within the 14-microsecond response timeout

#### Scenario: Schedule and execute an RT-to-BC transfer
- WHEN the BC schedules a command to read data from RT 3, subaddress 5 with word count 4
- THEN the BC MUST transmit the receive command word
- AND MUST wait for the RT to respond with a status word followed by 4 data words within the response timeout

#### Scenario: Response timeout handling
- WHEN an RT fails to respond within 14 microseconds of the command word transmission
- THEN the BC MUST flag a response timeout error for that command
- AND MUST proceed to the next scheduled command after recording the timeout

#### Scenario: RT-to-RT transfer
- WHEN the BC schedules an RT-to-RT transfer from RT 3 subaddress 1 to RT 7 subaddress 2
- THEN the BC MUST transmit the receive command (to RT 7) followed by the transmit command (to RT 3)
- AND MUST verify that both RTs respond correctly within the specified timeouts

### Requirement: Remote Terminal Mode
The MIL-STD-1553B driver SHALL implement Remote Terminal (RT) mode with command recognition, response generation, and subaddress-based data management.

#### Scenario: Respond to a receive command
- WHEN the RT receives a valid command word addressed to its RT address with T/R bit set to receive
- THEN the RT MUST accept the following data words, store them in the appropriate subaddress buffer
- AND MUST transmit a status word within 4-12 microseconds of the last data word

#### Scenario: Respond to a transmit command
- WHEN the RT receives a valid command word addressed to its RT address with T/R bit set to transmit
- THEN the RT MUST transmit a status word followed by the requested number of data words from the specified subaddress buffer
- AND the response MUST begin within 4-12 microseconds of the command word

#### Scenario: Ignore commands for other RT addresses
- WHEN a command word is received with an RT address that does not match this terminal's configured address
- THEN the RT MUST ignore the command entirely
- AND MUST NOT transmit any response on the bus

#### Scenario: Broadcast command handling
- WHEN a command word with RT address 31 (broadcast) is received
- THEN the RT MUST process the command data but MUST NOT transmit a status word response
- AND MUST set the broadcast received bit in the next status word

### Requirement: Dual-Redundant Bus Management
The MIL-STD-1553B driver SHALL support dual-redundant bus operation with Bus A and Bus B.

#### Scenario: BC selects active bus
- WHEN the Bus Controller determines that Bus A has experienced a communication failure
- THEN the BC MUST switch to Bus B for subsequent commands to affected RTs
- AND MUST log the bus switchover event

#### Scenario: RT listens on both buses
- WHEN the Remote Terminal is initialized
- THEN the RT MUST listen on both Bus A and Bus B simultaneously
- AND MUST respond on the same bus from which the command was received

#### Scenario: Bus health monitoring
- WHEN the BC monitors bus health for both Bus A and Bus B
- THEN the BC MUST track per-bus error rates, response timeouts, and message error counts
- AND MUST declare a bus failed when the error rate exceeds the configured threshold

### Requirement: Mode Codes
The MIL-STD-1553B driver SHALL support standard mode codes including transmit status word, synchronize, transmitter shutdown, override transmitter shutdown, reset remote terminal, and transmit built-in test word.

#### Scenario: Transmit status word mode code
- WHEN the BC sends mode code 00010 (transmit status word) with T/R bit set to transmit
- THEN the RT MUST respond with its status word and the last command word received as a data word

#### Scenario: Synchronize mode code
- WHEN the BC sends mode code 00001 (synchronize without data) to all RTs via broadcast
- THEN all RTs MUST synchronize their internal time references
- AND MUST NOT transmit a status word response (broadcast)

#### Scenario: Transmitter shutdown mode code
- WHEN the BC sends mode code 00100 (transmitter shutdown) to a specific RT
- THEN the RT MUST inhibit its transmitter on the commanded bus
- AND MUST respond with a status word before shutting down the transmitter

### Requirement: Zenoh Transport Adapter for MIL-STD-1553B
The MIL-STD-1553B driver SHALL provide a Zenoh transport adapter mapping RT address and subaddress to Zenoh key expressions using the pattern `mil1553/{bus}/{rt}/{sa}`.

#### Scenario: Publish received RT data to Zenoh
- WHEN data is received from RT 5, subaddress 3 on Bus A
- THEN the adapter MUST publish the data words to Zenoh key expression `mil1553/a/5/3`
- AND the payload MUST include the raw 16-bit data words, message timestamp, and status word flags

#### Scenario: Subscribe to Zenoh for BC command scheduling
- WHEN a Zenoh subscriber matches key expression `mil1553/a/7/2`
- AND a Zenoh publication is received on that key expression
- THEN the adapter MUST schedule a BC-to-RT write command to RT 7, subaddress 2 on Bus A with the published data

#### Scenario: Wildcard subscription for an RT
- WHEN a Zenoh subscriber registers for key expression `mil1553/a/5/**`
- THEN the adapter MUST deliver all data received from RT 5 on Bus A across all subaddresses

### Requirement: Hardware Interface Abstraction
The MIL-STD-1553B driver SHALL provide a hardware abstraction layer supporting dedicated MIL-STD-1553B transceiver hardware.

#### Scenario: Initialize 1553 transceiver hardware
- WHEN SmallAIOS boots with a dedicated MIL-STD-1553B transceiver connected
- THEN the driver MUST initialize the transceiver, configure BC or RT mode, set the RT address (if RT mode), and enable dual-bus operation
- AND the interface MUST be ready to send and receive 1553 messages

#### Scenario: Transceiver interrupt handling
- WHEN the 1553 transceiver generates an interrupt indicating message completion
- THEN the driver MUST read the completed message from the transceiver buffer within the inter-message gap time
- AND MUST dispatch the message to the appropriate BC or RT handler

#### Scenario: Portable message interface
- WHEN the application sends or receives 1553 messages through the abstraction layer
- THEN the abstraction MUST provide a uniform API regardless of the underlying transceiver hardware

### Requirement: Clean-Room Implementation
All MIL-STD-1553B implementations SHALL be clean-room developed from the MIL-STD-1553B public specification without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the MIL-STD-1553B module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the MIL-STD-1553B public specification and publicly available MIL-HDBK-1553A handbook as reference sources
- AND MUST NOT contain code derived from proprietary 1553 stack implementations
