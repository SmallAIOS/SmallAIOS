# Delta for IPC Messaging

## ADDED Requirements

### Requirement: Bus Protocol Transport Types

The IPC system SHALL support CAN (2.0A/B, CAN FD), ARINC 429, ARINC 664 Part 7 (AFDX), MIL-STD-1553B, SpaceWire (ECSS-E-ST-50-12C), and CCSDS Space Packet Protocol as Zenoh transport adapters alongside existing TCP, shared memory, and intra-kernel transports. Each bus transport adapter SHALL implement the `ZenohTransport` trait and map protocol-native addressing to Zenoh key expressions using the following scheme:

- CAN: `can/{bus_id}/{frame_id}`
- ARINC 429: `arinc429/{channel}/{label}`
- ARINC 664: `afdx/{vl_id}`
- MIL-STD-1553: `mil1553/{bus}/{rt}/{sa}`
- SpaceWire: `spw/{link}/{dest}`
- CCSDS: `ccsds/{apid}`

#### Scenario: CAN frame received and delivered as Zenoh sample

- WHEN a CAN 2.0B frame with ID 0x1A3 arrives on bus controller 0
- AND a subscriber is registered for key expression `can/0/0x1A3`
- THEN the CAN transport adapter MUST decode the frame and deliver it as a Zenoh sample to the subscriber
- AND the sample payload MUST contain the CAN data field (up to 8 bytes for CAN 2.0, up to 64 bytes for CAN FD)

#### Scenario: ARINC 429 word received and delivered as Zenoh sample

- WHEN an ARINC 429 32-bit word with label 0o215 arrives on channel 1
- AND a subscriber is registered for key expression `arinc429/1/0o215`
- THEN the ARINC 429 transport adapter MUST decode the word (BNR, BCD, or discrete) and deliver it as a Zenoh sample
- AND the sample payload MUST include the decoded data field and SDI/SSM bits

#### Scenario: MIL-STD-1553 message received and delivered as Zenoh sample

- WHEN a MIL-STD-1553B data message is received from RT address 5, subaddress 3 on bus A
- AND a subscriber is registered for key expression `mil1553/A/5/3`
- THEN the MIL-STD-1553 transport adapter MUST deliver the message data words as a Zenoh sample
- AND the adapter MUST validate the parity bit on each received word

#### Scenario: SpaceWire packet received and delivered as Zenoh sample

- WHEN a SpaceWire packet arrives on link 0 with destination address 42
- AND a subscriber is registered for key expression `spw/0/42`
- THEN the SpaceWire transport adapter MUST deliver the packet cargo as a Zenoh sample

#### Scenario: CCSDS Space Packet received and delivered as Zenoh sample

- WHEN a CCSDS Space Packet with APID 0x0100 is received
- AND a subscriber is registered for key expression `ccsds/0x0100`
- THEN the CCSDS transport adapter MUST extract the packet data field and deliver it as a Zenoh sample
- AND the adapter MUST validate the packet version number (version 1) and sequence count

#### Scenario: ARINC 664 Virtual Link received and delivered as Zenoh sample

- WHEN an AFDX frame arrives on Virtual Link 1024
- AND a subscriber is registered for key expression `afdx/1024`
- THEN the ARINC 664 transport adapter MUST validate the sequence number, apply redundancy management across dual networks, and deliver the payload as a Zenoh sample

### Requirement: Transport Auto-Discovery

SmallAIOS SHALL auto-detect available bus transports by parsing DTB (Device Tree Blob) and/or ACPI tables at boot time and SHALL register each discovered bus controller with the Zenoh router as an available transport. When no hardware controller is found for a given bus protocol, that transport MUST NOT be registered, and attempts to publish or subscribe on its key prefix MUST return an error indicating the transport is unavailable.

#### Scenario: CAN controller discovered via DTB

- WHEN the DTB contains a compatible node matching a supported CAN controller (e.g., `xlnx,axi-can-1.00.a` or `microchip,mcp2515`)
- THEN SmallAIOS MUST initialize the CAN controller driver
- AND MUST register the CAN transport adapter with the Zenoh router under the `can/` key prefix
- AND the transport MUST be available for pub/sub within 100 ms of boot completion

#### Scenario: No ARINC 429 hardware present

- WHEN the DTB and ACPI tables contain no nodes matching a supported ARINC 429 transceiver
- THEN SmallAIOS MUST NOT register an ARINC 429 transport adapter
- AND a subscriber attempting to register for `arinc429/**` MUST receive a `TransportUnavailable` error

#### Scenario: Multiple bus controllers of same type discovered

- WHEN the DTB contains two CAN controller nodes (bus 0 and bus 1)
- THEN SmallAIOS MUST initialize both controllers
- AND MUST register them as separate transports under `can/0/` and `can/1/` key prefixes
- AND messages on bus 0 MUST NOT be delivered to subscribers on bus 1 unless explicitly bridged

### Requirement: Transport-Agnostic Pub/Sub Routing

Messages published on one transport SHALL be routable to subscribers on any other transport via the Zenoh router. The router MUST perform transparent protocol translation, converting the payload format between transport-native encodings as needed. A single `put()` call by an application MUST be sufficient to reach all matching subscribers regardless of their underlying transport.

#### Scenario: Inference result published over TCP delivered to CAN subscriber

- WHEN an inference task publishes a result on key expression `smallaios/v1/models/detector/output`
- AND a CAN transport subscriber is registered for `smallaios/v1/models/detector/output` with a configured mapping to CAN frame ID 0x200 on bus 0
- THEN the Zenoh router MUST route the message to the CAN transport adapter
- AND the adapter MUST serialize the payload into one or more CAN frames (segmenting if payload exceeds CAN data length)
- AND the frames MUST be transmitted on the CAN bus

#### Scenario: CAN message routed to TCP subscriber

- WHEN a CAN frame with ID 0x100 is received on bus 0
- AND an external TCP client is subscribed to key expression `can/0/0x100`
- THEN the Zenoh router MUST deliver the CAN frame payload to the TCP subscriber
- AND the delivery MUST use the standard Zenoh wire protocol DATA frame

#### Scenario: Cross-bus routing between MIL-STD-1553 and ARINC 429

- WHEN a MIL-STD-1553 message is received on `mil1553/A/5/3`
- AND a bridge rule maps `mil1553/A/5/3` to `arinc429/0/0o310`
- THEN the Zenoh router MUST route the message to the ARINC 429 transport adapter
- AND the adapter MUST encode the data into a valid ARINC 429 word with label 0o310
- AND the word MUST be queued for transmission on the configured transmit schedule

#### Scenario: Pub/sub across shared memory and SpaceWire transports

- WHEN an internal kernel component publishes a telemetry sample via shared memory transport on key expression `smallaios/v1/telemetry/attitude`
- AND a SpaceWire transport subscriber is registered for `smallaios/v1/telemetry/**`
- THEN the Zenoh router MUST route the sample to the SpaceWire transport adapter
- AND the adapter MUST encapsulate the payload in a SpaceWire packet and transmit it on the configured link

#### Scenario: DDS topic received and delivered as Zenoh sample

- WHEN a DDS DataWriter publishes a sample on Topic "SensorData" in domain 0
- AND a subscriber is registered for key expression `dds/0/SensorData`
- THEN the DDS transport adapter MUST deserialize the CDR-encoded sample and deliver it as a Zenoh sample
- AND the sample payload MUST contain the deserialized data fields

#### Scenario: DDS topic published via Zenoh

- WHEN a Zenoh publisher puts a sample on key expression `dds/0/CommandData`
- AND a DDS DataReader is subscribed to Topic "CommandData" on domain 0
- THEN the DDS transport adapter MUST serialize the payload in CDR format and inject it into the DDS domain
- AND the DataReader MUST receive the sample with correct QoS handling

#### Scenario: DDS-to-CAN cross-transport routing

- WHEN a DDS DataWriter publishes a sample on Topic "BrakeCommand" in domain 0
- AND a bridge rule maps `dds/0/BrakeCommand` to `can/0/0x200`
- THEN the Zenoh router MUST route the message to the CAN transport adapter
- AND the CAN adapter MUST serialize the payload into CAN frame(s) and transmit on bus 0
