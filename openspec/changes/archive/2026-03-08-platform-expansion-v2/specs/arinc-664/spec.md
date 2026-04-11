# Delta for ARINC 664 Part 7 (AFDX)

## ADDED Requirements

### Requirement: Virtual Link Configuration
The AFDX driver SHALL support Virtual Link (VL) configuration with VL ID, BAG interval (1-128 ms), and Lmax frame size per the publicly available ARINC 664 Part 7 description.

#### Scenario: Configure a Virtual Link
- WHEN the application defines a Virtual Link with VL ID 1001, BAG interval 32 ms, and Lmax 512 bytes
- THEN the driver MUST allocate resources for the VL and enforce the specified BAG and Lmax parameters
- AND the VL MUST be assigned a multicast destination MAC address derived from the VL ID per AFDX conventions

#### Scenario: Reject VL with invalid BAG interval
- WHEN the application attempts to configure a VL with a BAG interval that is not a power of 2 in the range 1-128 ms
- THEN the driver MUST return an error indicating the BAG interval is invalid
- AND MUST NOT create the VL

#### Scenario: Reject frame exceeding Lmax
- WHEN the application submits a frame for a VL that exceeds the configured Lmax
- THEN the driver MUST reject the frame with an error
- AND MUST NOT transmit the oversized frame onto the network

### Requirement: BAG Traffic Shaping and Policing
The AFDX driver SHALL implement Bandwidth Allocation Gap (BAG) traffic shaping on the transmit side and BAG policing on the receive side for each Virtual Link.

#### Scenario: Transmit-side BAG enforcement
- WHEN the application submits frames for a VL with BAG interval 16 ms
- THEN the driver MUST ensure that no two consecutive frames for that VL are transmitted less than 16 ms apart
- AND frames submitted faster than the BAG rate MUST be queued or dropped according to the configured overflow policy

#### Scenario: Receive-side BAG policing
- WHEN frames arrive on a VL faster than the configured BAG interval allows (accounting for jitter tolerance of 500 us)
- THEN the policing function MUST discard the excess frames
- AND MUST increment a per-VL policing violation counter

#### Scenario: Jitter tolerance within BAG window
- WHEN frames arrive on a VL with inter-frame gaps within the BAG interval +/- 500 us jitter tolerance
- THEN the policing function MUST accept the frames as valid
- AND MUST NOT count them as violations

### Requirement: Sequence Number Management
The AFDX driver SHALL generate and check per-VL sequence numbers to detect lost, duplicate, and out-of-order frames.

#### Scenario: Sequence number generation on transmit
- WHEN a frame is transmitted on a VL
- THEN the driver MUST insert a monotonically incrementing sequence number (0-255, wrapping) in the AFDX trailer
- AND each VL MUST maintain an independent sequence counter

#### Scenario: Detect lost frame on receive
- WHEN a received frame's sequence number indicates one or more missing sequence values since the last accepted frame
- THEN the driver MUST report the number of lost frames to the application
- AND MUST accept the current frame and update the expected sequence counter

#### Scenario: Detect and discard duplicate frame
- WHEN a received frame's sequence number matches the last accepted sequence number for that VL
- THEN the driver MUST discard the duplicate frame
- AND MUST increment a per-VL duplicate frame counter

### Requirement: Dual-Network Redundancy Management
The AFDX driver SHALL implement dual-network redundancy with integrity checking across Network A and Network B.

#### Scenario: Transmit on both networks
- WHEN the application submits a frame for transmission on a redundant VL
- THEN the driver MUST transmit identical copies on both Network A and Network B
- AND both copies MUST carry the same sequence number

#### Scenario: Receive with redundancy selection
- WHEN identical frames are received on both Network A and Network B
- THEN the redundancy management function MUST deliver only the first valid frame to the application
- AND MUST discard the duplicate from the second network

#### Scenario: Network failover
- WHEN frames from a VL are received only on Network A and Network B produces no frames for that VL within the configured timeout
- THEN the redundancy manager MUST continue delivering frames from Network A without interruption
- AND MUST log the Network B failure for that VL

#### Scenario: Integrity check failure
- WHEN a frame fails the integrity check (invalid FCS or sequence number anomaly) on one network but is valid on the other
- THEN the redundancy manager MUST discard the corrupted frame and deliver the valid copy
- AND MUST increment the integrity failure counter for the affected network

### Requirement: Sub-VL Scheduling
The AFDX driver SHALL support sub-VL scheduling using round-robin within a Virtual Link to multiplex multiple data flows.

#### Scenario: Round-robin scheduling across sub-VLs
- WHEN a VL has three sub-VLs (A, B, C) with pending frames
- THEN the scheduler MUST transmit frames in round-robin order: A, B, C, A, B, C, ...
- AND each sub-VL MUST receive a fair share of the VL's BAG-limited bandwidth

#### Scenario: Skip empty sub-VL in round-robin
- WHEN sub-VL B has no pending frames but sub-VLs A and C do
- THEN the scheduler MUST skip sub-VL B and transmit from A and C in alternation
- AND MUST NOT waste a BAG slot on an empty sub-VL

#### Scenario: Single sub-VL uses full VL bandwidth
- WHEN only one sub-VL is configured for a VL
- THEN that sub-VL MUST receive the full BAG-limited bandwidth of the VL
- AND the scheduling overhead MUST be negligible

### Requirement: Frame Filtering
The AFDX driver SHALL filter received frames based on VL ID and destination MAC address matching.

#### Scenario: Accept frame matching configured VL
- WHEN a frame is received with a destination MAC address and VL ID matching a locally configured VL
- THEN the filter MUST accept the frame and pass it to the redundancy management layer

#### Scenario: Reject frame for unconfigured VL
- WHEN a frame is received with a VL ID that is not configured on this end system
- THEN the filter MUST silently discard the frame
- AND MUST NOT deliver it to any application

#### Scenario: Reject frame with wrong destination MAC
- WHEN a frame's destination MAC address does not match the expected AFDX multicast MAC for any configured VL
- THEN the filter MUST discard the frame at the earliest stage to minimize processing overhead

### Requirement: Zenoh Transport Adapter for AFDX
The AFDX driver SHALL provide a Zenoh transport adapter mapping Virtual Link IDs to Zenoh key expressions using the pattern `afdx/{vl_id}`.

#### Scenario: Publish received AFDX frame to Zenoh
- WHEN an AFDX frame is received on VL 1001 and passes redundancy management
- THEN the adapter MUST publish the frame payload to Zenoh key expression `afdx/1001`
- AND the payload MUST include the application data and metadata (VL ID, sequence number, receive network, timestamp)

#### Scenario: Subscribe to Zenoh and transmit on AFDX VL
- WHEN a Zenoh subscriber matches key expression `afdx/2002`
- AND a Zenoh publication is received on that key expression
- THEN the adapter MUST encapsulate the payload and transmit it on VL 2002 through the BAG shaper

#### Scenario: Wildcard subscription for all VLs
- WHEN a Zenoh subscriber registers for key expression `afdx/**`
- THEN the adapter MUST deliver all received AFDX application payloads to the subscriber with VL ID metadata

### Requirement: Ethernet/IP Stack Integration
The AFDX driver SHALL ride on the existing SmallAIOS Ethernet and IP stack for physical frame transport.

#### Scenario: Frame encapsulation over Ethernet
- WHEN an AFDX frame is submitted for transmission
- THEN the driver MUST encapsulate it as a standard Ethernet frame with the AFDX multicast destination MAC, source MAC, and EtherType
- AND MUST use the existing Ethernet driver for physical transmission

#### Scenario: Frame extraction from Ethernet
- WHEN an Ethernet frame is received matching an AFDX multicast MAC address
- THEN the existing Ethernet driver MUST deliver the frame to the AFDX filter layer
- AND the AFDX layer MUST process the frame independently from non-AFDX Ethernet traffic

### Requirement: Clean-Room Implementation
All AFDX implementations SHALL be clean-room developed from the publicly available ARINC 664 Part 7 description without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the AFDX module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the publicly available ARINC 664 Part 7 description, ARINC 664 overviews, and IEEE 802.3 standard as reference sources
- AND MUST NOT contain code derived from proprietary AFDX stack implementations
