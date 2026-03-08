# Delta for ARINC 429 Avionics Data Bus

## ADDED Requirements

### Requirement: ARINC 429 Word Encode/Decode
The ARINC 429 driver SHALL encode and decode 32-bit ARINC 429 words with correct label, SDI, data, SSM, and parity fields per the publicly available ARINC 429 specification.

#### Scenario: Encode a 32-bit ARINC 429 word
- WHEN the application submits label, SDI, data, and SSM values
- THEN the encoder MUST produce a valid 32-bit word with label in bits 1-8 (reversed bit order), SDI in bits 9-10, data in bits 11-29, SSM in bits 30-31, and odd parity in bit 32
- AND the label octal encoding MUST follow ARINC 429 bit-reversal convention

#### Scenario: Decode a 32-bit ARINC 429 word
- WHEN a 32-bit word is received from the ARINC 429 bus
- THEN the decoder MUST extract the label (reversing bit order to octal), SDI, data field, and SSM
- AND MUST verify odd parity on bit 32 and reject words with incorrect parity

#### Scenario: Parity error detection
- WHEN a received word fails the odd parity check on bit 32
- THEN the decoder MUST discard the word and increment a parity error counter
- AND MUST notify the application of the parity failure

### Requirement: BNR Data Format
The ARINC 429 driver SHALL support Binary (BNR) data format encoding and decoding with configurable resolution and range.

#### Scenario: Encode a BNR value
- WHEN the application provides a floating-point value and a label definition specifying BNR format with sign bit, MSB resolution, and range
- THEN the encoder MUST convert the value to two's complement binary representation scaled to the defined resolution
- AND MUST place the encoded value in bits 11-29 of the ARINC 429 word

#### Scenario: Decode a BNR value
- WHEN a received ARINC 429 word has a label registered as BNR format
- THEN the decoder MUST extract the binary data from bits 11-29, apply the sign from bit 29, and scale the result according to the configured resolution
- AND MUST return the decoded engineering-unit value to the application

#### Scenario: Out-of-range BNR value
- WHEN the application attempts to encode a BNR value that exceeds the configured range for the label
- THEN the encoder MUST return an error indicating the value is out of range
- AND MUST NOT transmit the invalid word

### Requirement: BCD Data Format
The ARINC 429 driver SHALL support Binary Coded Decimal (BCD) data format encoding and decoding.

#### Scenario: Encode a BCD value
- WHEN the application provides a decimal value and a label definition specifying BCD format with digit positions
- THEN the encoder MUST convert each decimal digit to its 4-bit BCD representation and pack the digits into bits 11-29 of the ARINC 429 word
- AND the sign/status MUST be encoded in the SSM field (bits 30-31)

#### Scenario: Decode a BCD value
- WHEN a received ARINC 429 word has a label registered as BCD format
- THEN the decoder MUST extract each 4-bit BCD digit from the data field and reconstruct the decimal value
- AND MUST validate that each nibble contains a valid BCD digit (0-9) and reject words with invalid BCD encoding

### Requirement: Discrete Data Words
The ARINC 429 driver SHALL support discrete data words where individual bits represent independent boolean states.

#### Scenario: Encode discrete bits
- WHEN the application sets individual discrete bit states for a label defined as discrete format
- THEN the encoder MUST pack the specified bits into the data field (bits 11-29) at their defined positions
- AND MUST set the SSM field to indicate normal operation status

#### Scenario: Decode discrete bits
- WHEN a received ARINC 429 word has a label registered as discrete format
- THEN the decoder MUST extract each individual bit from the data field and report its boolean state
- AND MUST check the SSM field and report any failure warning or no computed data status to the application

### Requirement: Label-Based Filtering and Routing
The ARINC 429 driver SHALL support label-based filtering to accept or reject words based on their 8-bit label field, and route accepted words to registered handlers.

#### Scenario: Accept words matching a registered label
- WHEN the application registers interest in label 0o310 (octal)
- AND a word with label 0o310 is received
- THEN the driver MUST deliver the word to the registered handler for that label

#### Scenario: Reject words not matching any registered label
- WHEN a word is received with a label that has no registered handler
- THEN the driver MUST silently discard the word without consuming application buffer space

#### Scenario: Multiple handlers for different labels
- WHEN handlers are registered for labels 0o310, 0o311, and 0o312 on the same receive channel
- THEN each received word MUST be routed to the correct handler based on its label
- AND routing MUST be completed within the inter-word gap time

### Requirement: Fixed-Rate Transmit Scheduler
The ARINC 429 driver SHALL provide a fixed-rate transmit scheduler with per-label configurable transmission rates.

#### Scenario: Schedule periodic label transmission
- WHEN a label is configured for periodic transmission at 50 Hz
- THEN the scheduler MUST transmit the most recent value for that label every 20 ms (+/- 1 ms jitter)
- AND MUST maintain the configured rate regardless of other label schedules

#### Scenario: Multiple labels with different rates
- WHEN label 0o310 is configured at 50 Hz and label 0o311 is configured at 12.5 Hz
- THEN the scheduler MUST interleave transmissions to meet both rate requirements
- AND MUST NOT violate the minimum 4-bit-time inter-word gap

#### Scenario: Update scheduled label data
- WHEN the application updates the data value for a periodically scheduled label
- THEN the new value MUST be used starting with the next scheduled transmission
- AND the transmission schedule timing MUST NOT be disrupted by the data update

### Requirement: Hardware Interface Abstraction
The ARINC 429 driver SHALL provide a hardware abstraction layer supporting SPI-based ARINC 429 transceivers and FPGA soft-IP implementations.

#### Scenario: Initialize SPI-based ARINC 429 transceiver
- WHEN SmallAIOS detects an ARINC 429 transceiver on the SPI bus
- THEN the driver MUST initialize the transceiver via SPI commands, configure the bus speed, and enable TX and RX channels
- AND the interface MUST be ready for word transmission and reception

#### Scenario: Initialize FPGA soft-IP ARINC 429 core
- WHEN an FPGA-based ARINC 429 IP core is detected via memory-mapped I/O
- THEN the driver MUST configure the core registers, set the bus speed, and enable interrupt-driven operation
- AND MUST support simultaneous operation of multiple TX and RX channels

#### Scenario: Portable word send across hardware
- WHEN the application transmits an ARINC 429 word through the abstraction layer
- THEN the abstraction layer MUST route the word to the correct hardware driver without application-level hardware knowledge

### Requirement: Zenoh Transport Adapter for ARINC 429
The ARINC 429 driver SHALL provide a Zenoh transport adapter mapping ARINC 429 labels to Zenoh key expressions using the pattern `arinc429/{channel}/{label}`.

#### Scenario: Publish received ARINC 429 word to Zenoh
- WHEN an ARINC 429 word with label 0o310 is received on channel 1
- THEN the adapter MUST publish the decoded data to Zenoh key expression `arinc429/1/0o310`
- AND the payload MUST include the decoded engineering-unit value, SSM status, SDI, and receive timestamp

#### Scenario: Subscribe to Zenoh and transmit ARINC 429 word
- WHEN a Zenoh subscriber matches key expression `arinc429/0/0o205`
- AND a Zenoh publication is received on that key expression
- THEN the adapter MUST encode the payload into an ARINC 429 word and schedule it for transmission on channel 0 with label 0o205

#### Scenario: Wildcard subscription for all labels on a channel
- WHEN a Zenoh subscriber registers for key expression `arinc429/1/**`
- THEN the adapter MUST deliver all decoded ARINC 429 words received on channel 1 to the subscriber

### Requirement: Bus Speed Support
The ARINC 429 driver SHALL support both low speed (12.5 kbps) and high speed (100 kbps) bus operation.

#### Scenario: Configure high speed operation
- WHEN the application configures an ARINC 429 channel for high speed
- THEN the driver MUST set the bit rate to 100 kbps (+/- 1%)
- AND MUST configure rise/fall times per the ARINC 429 high speed electrical specification

#### Scenario: Configure low speed operation
- WHEN the application configures an ARINC 429 channel for low speed
- THEN the driver MUST set the bit rate to 12.5 kbps (+/- 1%)
- AND MUST configure rise/fall times per the ARINC 429 low speed electrical specification

#### Scenario: Reject invalid speed configuration
- WHEN the application attempts to configure a speed other than 12.5 kbps or 100 kbps
- THEN the driver MUST return an error indicating the speed is not supported

### Requirement: Clean-Room Implementation
All ARINC 429 implementations SHALL be clean-room developed from the publicly available ARINC 429 specification without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the ARINC 429 module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the publicly available ARINC 429 specification and publicly available technical references as source material
- AND MUST NOT contain code derived from proprietary ARINC 429 stack implementations
