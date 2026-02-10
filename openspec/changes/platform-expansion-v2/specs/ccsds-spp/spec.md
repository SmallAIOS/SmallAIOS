# Delta for CCSDS Space Packet Protocol (CCSDS 133.0-B)

## ADDED Requirements

### Requirement: Space Packet Encode/Decode
The CCSDS SPP driver SHALL encode and decode Space Packets with version, type, APID, sequence control, data length, and user data fields per CCSDS 133.0-B.

#### Scenario: Encode a Space Packet
- WHEN the application submits a packet type (TM or TC), APID, and user data payload
- THEN the encoder MUST produce a valid Space Packet with a 6-byte primary header and the user data field
- AND the primary header MUST contain version (000), type indicator, secondary header flag, APID, sequence flags, sequence count, and packet data length

#### Scenario: Decode a Space Packet
- WHEN a byte stream containing one or more Space Packets is received
- THEN the decoder MUST parse the 6-byte primary header, extract all header fields, and deliver the user data payload to the application
- AND MUST validate that the packet data length field matches the actual remaining bytes

#### Scenario: Reject malformed packet
- WHEN a received byte stream has a primary header with an invalid version number (not 000)
- THEN the decoder MUST reject the packet and increment a malformed packet counter
- AND MUST attempt to resynchronize to the next valid packet boundary

### Requirement: Primary Header Format
The CCSDS SPP driver SHALL implement the 6-byte primary header with packet version number (3 bits), packet type (1 bit), secondary header flag (1 bit), APID (11 bits), sequence flags (2 bits), sequence count (14 bits), and packet data length (16 bits).

#### Scenario: Set packet identification fields
- WHEN encoding a telemetry packet for APID 0x1A3 with a secondary header present
- THEN the primary header bits 0-2 MUST be 000 (version 1)
- AND bit 3 MUST be 0 (telemetry)
- AND bit 4 MUST be 1 (secondary header present)
- AND bits 5-15 MUST contain APID 0x1A3

#### Scenario: Set sequence control fields
- WHEN encoding a standalone packet (not part of a segmentation group) with sequence count 1042
- THEN the sequence flags (bits 0-1 of the third header word) MUST be 11 (unsegmented)
- AND the sequence count (bits 2-15) MUST be 1042
- AND the sequence count MUST increment modulo 16384 for each packet on the same APID

#### Scenario: Set packet data length
- WHEN the user data payload is 256 bytes and there is no secondary header
- THEN the packet data length field MUST be set to 255 (number of octets in the packet data field minus 1)
- AND the total packet size MUST be 262 bytes (6-byte header + 256-byte data)

### Requirement: Telemetry Transfer Frame Support
The CCSDS SPP driver SHALL support encapsulation of Space Packets within CCSDS Telemetry (TM) Transfer Frames per CCSDS 132.0-B.

#### Scenario: Pack Space Packets into a TM Transfer Frame
- WHEN one or more Space Packets are ready for downlink on a virtual channel
- THEN the driver MUST pack the packets into a TM Transfer Frame with the correct frame header (version, spacecraft ID, virtual channel ID, frame counter)
- AND MUST set the First Header Pointer to indicate the start of the first packet that begins in this frame

#### Scenario: Handle packet spanning two TM frames
- WHEN a Space Packet is too large to fit in the remaining space of the current TM Transfer Frame
- THEN the driver MUST split the packet across two consecutive frames
- AND the first frame MUST contain the beginning of the packet and the second frame MUST set the First Header Pointer to indicate where the next new packet begins

#### Scenario: Idle frame generation
- WHEN no Space Packets are available for a scheduled TM frame transmission
- THEN the driver MUST generate an idle frame filled with the idle data pattern
- AND the First Header Pointer MUST be set to 0x7FF (no packet starts in this frame)

### Requirement: Telecommand Transfer Frame Support
The CCSDS SPP driver SHALL support encapsulation of Space Packets within CCSDS Telecommand (TC) Transfer Frames per CCSDS 232.0-B.

#### Scenario: Encapsulate a TC Space Packet in a TC Transfer Frame
- WHEN a telecommand Space Packet is ready for uplink
- THEN the driver MUST encapsulate it in a TC Transfer Frame with the correct frame header (version, bypass flag, control command flag, spacecraft ID, virtual channel ID, frame length, frame sequence number)
- AND MUST compute and append the Frame Error Control Field (FECF) using CRC-16

#### Scenario: Extract TC Space Packet from a TC Transfer Frame
- WHEN a TC Transfer Frame is received
- THEN the driver MUST verify the FECF CRC-16
- AND MUST extract the enclosed Space Packet(s) and deliver them to the APID router
- AND MUST reject frames with invalid CRC

#### Scenario: Bypass flag handling
- WHEN a TC Transfer Frame is received with the bypass flag set (Type-B)
- THEN the driver MUST deliver the frame contents immediately without sequence checking
- AND MUST NOT update the expected sequence counter for that virtual channel

### Requirement: APID-Based Routing and Filtering
The CCSDS SPP driver SHALL route and filter received Space Packets based on their 11-bit Application Process Identifier (APID).

#### Scenario: Route packet to registered APID handler
- WHEN a Space Packet with APID 0x1A3 is received and a handler is registered for APID 0x1A3
- THEN the router MUST deliver the packet to the registered handler
- AND delivery MUST preserve packet ordering within the same APID

#### Scenario: Discard packet for unregistered APID
- WHEN a Space Packet is received with an APID that has no registered handler
- THEN the router MUST silently discard the packet
- AND MUST increment a per-APID discard counter

#### Scenario: Idle packet filtering
- WHEN a Space Packet with APID 0x7FF (idle packet) is received
- THEN the router MUST discard the idle packet without delivering it to any handler
- AND MUST NOT count idle packets as discarded-for-no-handler

#### Scenario: Multiple handlers for different APIDs
- WHEN handlers are registered for APIDs 0x100, 0x1A3, and 0x200
- THEN each received packet MUST be routed to exactly the handler matching its APID
- AND no packet MUST be delivered to more than one handler

### Requirement: CLTU Encoding for Uplink
The CCSDS SPP driver SHALL optionally support Communications Link Transmission Unit (CLTU) encoding for telecommand uplink per CCSDS 231.0-B.

#### Scenario: Encode a CLTU
- WHEN a TC Transfer Frame is ready for uplink via CLTU
- THEN the encoder MUST prepend the CLTU start sequence (0xEB90)
- AND MUST segment the frame data into code blocks with BCH(63,56) encoding applied to each block
- AND MUST append the CLTU tail sequence (0xC5C5C5C5C5C5C579)

#### Scenario: Decode a CLTU
- WHEN a CLTU is received from the uplink
- THEN the decoder MUST verify the start sequence, decode each BCH code block (correcting single-bit errors)
- AND MUST detect the tail sequence to determine the end of the CLTU
- AND MUST extract and reassemble the TC Transfer Frame

#### Scenario: BCH error correction
- WHEN a BCH code block within a CLTU contains a single-bit error
- THEN the decoder MUST correct the error using the BCH syndrome
- AND MUST deliver the corrected data to the TC frame layer

### Requirement: Zenoh Transport Adapter for CCSDS SPP
The CCSDS SPP driver SHALL provide a Zenoh transport adapter mapping APIDs to Zenoh key expressions using the pattern `ccsds/{apid}`.

#### Scenario: Publish received Space Packet to Zenoh
- WHEN a Space Packet with APID 0x1A3 is received and decoded
- THEN the adapter MUST publish the user data payload to Zenoh key expression `ccsds/0x1A3`
- AND the payload MUST include the user data and metadata (packet type, sequence count, secondary header flag, timestamp)

#### Scenario: Subscribe to Zenoh and transmit Space Packet
- WHEN a Zenoh subscriber matches key expression `ccsds/0x100`
- AND a Zenoh publication is received on that key expression
- THEN the adapter MUST encode the payload as a Space Packet with APID 0x100 and submit it for transmission

#### Scenario: Wildcard subscription for all APIDs
- WHEN a Zenoh subscriber registers for key expression `ccsds/**`
- THEN the adapter MUST deliver all received Space Packet payloads to the subscriber with APID metadata

### Requirement: Clean-Room Implementation
All CCSDS SPP implementations SHALL be clean-room developed from the publicly available CCSDS Blue Books (133.0-B, 132.0-B, 231.0-B, 232.0-B) without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the CCSDS SPP module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the CCSDS 133.0-B (Space Packet Protocol), CCSDS 132.0-B (TM Space Data Link Protocol), CCSDS 232.0-B (TC Space Data Link Protocol), and CCSDS 231.0-B (TC Synchronization and Channel Coding) Blue Books as reference sources
- AND MUST NOT contain code derived from proprietary CCSDS stack implementations
