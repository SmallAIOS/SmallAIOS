# Delta for DDS (Data Distribution Service)

## ADDED Requirements

#### Requirement: DCPS API — Domain Participant

SmallAIOS SHALL implement the DDS DCPS DomainParticipant entity, which represents a node's participation in a DDS domain. Each DomainParticipant SHALL be identified by a domain ID (0-232) and a globally unique participant GUID. The implementation SHALL support creating multiple DomainParticipants within a single SmallAIOS instance for domain isolation.

#### Scenario: Create a DomainParticipant

- WHEN a task calls `dds_create_participant(domain_id=0, qos=default)`
- THEN the DDS subsystem MUST allocate a DomainParticipant with a unique GUID
- AND the participant MUST begin SPDP (Simple Participant Discovery Protocol) announcements on the configured RTPS multicast group
- AND the participant MUST be discoverable by other DDS participants on the same domain within 5 seconds

#### Scenario: Create DomainParticipant with invalid domain ID

- WHEN a task calls `dds_create_participant(domain_id=0xFFFFFFFF, qos=default)`
- THEN the DDS subsystem MUST return `InvalidArgument` error
- AND no resources MUST be allocated

#### Scenario: Multiple DomainParticipants on different domains

- WHEN two DomainParticipants are created with domain_id=0 and domain_id=1
- THEN topics published on domain 0 MUST NOT be visible to subscribers on domain 1
- AND SPDP announcements for domain 0 MUST NOT be sent on domain 1's multicast group

### Requirement: DCPS API — Topic, DataWriter, DataReader

SmallAIOS SHALL implement the DDS DCPS Topic, DataWriter, and DataReader entities. A Topic defines a named data type within a domain. DataWriters publish samples to a Topic; DataReaders subscribe to samples from a Topic. The implementation SHALL enforce type consistency: a DataReader and DataWriter on the same Topic name MUST agree on the data type.

#### Scenario: Create Topic and publish a sample

- WHEN a DomainParticipant creates a Topic named "SensorData" with type "SensorMsg"
- AND a DataWriter is created for that Topic
- AND the DataWriter calls `write(sample)`
- THEN the sample MUST be serialized using CDR (Common Data Representation) encoding
- AND the sample MUST be delivered to all matched DataReaders on the same Topic and domain
- AND the sample MUST be transmitted via the RTPS wire protocol to remote participants

#### Scenario: DataReader receives published sample

- WHEN a DataReader is subscribed to Topic "SensorData" on domain 0
- AND a remote DataWriter publishes a sample on "SensorData" domain 0
- THEN the DataReader MUST receive the sample via RTPS
- AND the sample MUST be deserialized from CDR encoding
- AND the DataReader's listener or wait-set MUST be notified of the new sample

#### Scenario: Type mismatch between DataWriter and DataReader

- WHEN a DataWriter publishes on Topic "SensorData" with type "SensorMsg"
- AND a DataReader subscribes to Topic "SensorData" with type "OtherMsg"
- THEN the DDS subsystem MUST detect the type mismatch via SEDP (Simple Endpoint Discovery Protocol)
- AND the DataWriter and DataReader MUST NOT be matched
- AND no samples MUST be delivered between them

#### Scenario: DataWriter with no matched readers

- WHEN a DataWriter publishes a sample on Topic "SensorData"
- AND no DataReaders are matched for that Topic
- THEN the DDS subsystem MUST retain the sample according to the DURABILITY QoS policy
- AND the DataWriter MUST report PUBLICATION_MATCHED status with current_count=0

### Requirement: RTPS Wire Protocol

SmallAIOS SHALL implement the RTPS (Real-Time Publish-Subscribe) wire protocol version 2.3 as specified in OMG DDSI-RTPS. The implementation SHALL support UDP/IPv4 transport with unicast and multicast. RTPS enables wire-level interoperability with any compliant DDS implementation (FastDDS, CycloneDDS, RTI Connext).

#### Scenario: RTPS message exchange between SmallAIOS and external DDS node

- WHEN a SmallAIOS DDS participant is on the same network as an external DDS participant (e.g., ROS 2 node using FastDDS)
- AND both are on domain 0
- THEN SmallAIOS MUST discover the external participant via SPDP multicast
- AND SmallAIOS MUST exchange endpoint information via SEDP
- AND matched DataWriters/DataReaders MUST exchange DATA submessages over RTPS

#### Scenario: RTPS reliable delivery

- WHEN a DataWriter with RELIABILITY=RELIABLE publishes a sample
- AND the sample is lost in transit (no ACKNACK received within heartbeat period)
- THEN the DataWriter MUST retransmit the sample via a GAP or DATA submessage
- AND delivery MUST be guaranteed to all matched reliable DataReaders
- AND the DataWriter MUST continue heartbeating until all readers have acknowledged

#### Scenario: RTPS best-effort delivery

- WHEN a DataWriter with RELIABILITY=BEST_EFFORT publishes a sample
- THEN the sample MUST be sent once without waiting for acknowledgment
- AND if the sample is lost, it MUST NOT be retransmitted
- AND the DataReader MUST accept samples with non-contiguous sequence numbers

#### Scenario: SPDP participant discovery

- WHEN a new DomainParticipant is created
- THEN it MUST send SPDP announcements to the domain's multicast address (239.255.0.1, port derived from domain_id)
- AND it MUST repeat SPDP announcements at the configured LEASE_DURATION / 3 interval
- AND when a remote participant's SPDP announcement is received, the participant MUST be added to the discovered participants list
- AND when a participant's lease expires without renewal, it MUST be removed from the discovered list

#### Scenario: SEDP endpoint discovery

- WHEN a DomainParticipant discovers a remote participant via SPDP
- THEN it MUST exchange SEDP information (publications and subscriptions)
- AND matching DataWriters and DataReaders MUST be paired based on Topic name, type, and compatible QoS
- AND matched endpoints MUST begin data exchange immediately

### Requirement: QoS Policies

SmallAIOS SHALL implement the following DDS QoS policies. QoS policies control the behavior of data distribution and MUST be enforced by the DDS subsystem.

#### Scenario: RELIABILITY QoS — Reliable

- WHEN a DataWriter is configured with `RELIABILITY = RELIABLE, max_blocking_time = 100ms`
- AND the DataWriter's send buffer is full
- THEN the write call MUST block for up to 100ms waiting for buffer space
- AND if space is not available within 100ms, the write MUST return a TIMEOUT error

#### Scenario: RELIABILITY QoS — Best Effort

- WHEN a DataWriter is configured with `RELIABILITY = BEST_EFFORT`
- THEN write calls MUST never block
- AND samples that cannot be sent immediately MUST be silently dropped

#### Scenario: DURABILITY QoS — Transient Local

- WHEN a DataWriter is configured with `DURABILITY = TRANSIENT_LOCAL`
- AND the DataWriter has published 5 samples
- AND a new DataReader joins and matches
- THEN the DataReader MUST receive all 5 previously published samples (subject to HISTORY depth)
- AND samples MUST be delivered in order before any new samples

#### Scenario: DEADLINE QoS

- WHEN a DataWriter is configured with `DEADLINE = 100ms`
- AND the DataWriter does not publish a new sample within 100ms
- THEN the DDS subsystem MUST trigger an OFFERED_DEADLINE_MISSED status callback
- AND when a DataReader has `DEADLINE = 100ms` and no sample is received within 100ms
- THEN the DDS subsystem MUST trigger a REQUESTED_DEADLINE_MISSED status callback

#### Scenario: LIVELINESS QoS — Automatic

- WHEN a DataWriter is configured with `LIVELINESS = AUTOMATIC, lease_duration = 1s`
- THEN the DDS subsystem MUST automatically assert liveliness on behalf of the writer at intervals ≤ lease_duration
- AND if the writer's process crashes, remote readers MUST detect liveliness loss within lease_duration

#### Scenario: OWNERSHIP QoS — Exclusive

- WHEN two DataWriters publish on the same Topic with `OWNERSHIP = EXCLUSIVE`
- AND Writer A has `OWNERSHIP_STRENGTH = 10` and Writer B has `OWNERSHIP_STRENGTH = 5`
- THEN DataReaders MUST only deliver samples from Writer A (highest strength)
- AND if Writer A's liveliness is lost, DataReaders MUST switch to Writer B

#### Scenario: HISTORY QoS — Keep Last N

- WHEN a DataWriter is configured with `HISTORY = KEEP_LAST, depth = 5`
- AND the DataWriter publishes 10 samples
- THEN only the most recent 5 samples MUST be retained for late joiners
- AND samples 1-5 MUST be discarded

#### Scenario: QoS compatibility check

- WHEN a DataWriter offers `RELIABILITY = BEST_EFFORT`
- AND a DataReader requests `RELIABILITY = RELIABLE`
- THEN the QoS policies MUST be deemed incompatible
- AND the DataWriter and DataReader MUST NOT be matched
- AND both MUST report INCOMPATIBLE_QOS status

### Requirement: DDS-Security — Authentication and Access Control

SmallAIOS SHALL implement DDS-Security authentication and access control plugins as specified in OMG DDS-Security 1.1. The implementation SHALL use SmallAIOS's existing post-quantum cryptographic primitives (ML-KEM-768, ML-DSA-65) as the underlying algorithms rather than classical RSA/ECDSA.

#### Scenario: Mutual authentication between participants

- WHEN two DDS participants attempt to communicate
- AND both have authentication enabled with valid identity certificates
- THEN the DDS-Security authentication plugin MUST perform mutual authentication via the handshake protocol
- AND both participants MUST validate each other's identity certificates against a configured CA
- AND authentication MUST complete before any application data is exchanged

#### Scenario: Authentication failure — invalid certificate

- WHEN a remote participant presents an expired or untrusted identity certificate
- THEN the DDS-Security authentication plugin MUST reject the handshake
- AND the remote participant MUST NOT be allowed to communicate
- AND the rejection MUST be logged in the audit log

#### Scenario: Access control — topic authorization

- WHEN a DataWriter attempts to publish on Topic "ClassifiedData"
- AND the participant's access control permissions do not include publish rights for "ClassifiedData"
- THEN the DDS-Security access control plugin MUST deny the operation
- AND the DataWriter creation MUST fail with a SECURITY_ERROR

#### Scenario: Access control — domain authorization

- WHEN a participant attempts to join domain 42
- AND the participant's governance document does not authorize domain 42
- THEN the DDS-Security access control plugin MUST deny domain participation
- AND the DomainParticipant creation MUST fail with a SECURITY_ERROR

### Requirement: Zenoh Transport Adapter

SmallAIOS SHALL implement a DDS-to-Zenoh transport adapter that maps DDS domain/topic addressing to Zenoh key expressions. This adapter enables DDS topics to be accessible via the SmallAIOS Zenoh router, bridging DDS applications with other Zenoh-based transports (CAN, ARINC, SpaceWire, etc.).

#### Scenario: DDS topic mapped to Zenoh key expression

- WHEN a DataWriter publishes on Topic "SensorData" in domain 0
- AND the DDS-Zenoh adapter is active
- THEN the sample MUST be available on Zenoh key expression `dds/0/SensorData`
- AND Zenoh subscribers on `dds/0/SensorData` MUST receive the sample payload

#### Scenario: Zenoh publisher bridged to DDS topic

- WHEN a Zenoh publisher puts a sample on key expression `dds/0/CommandData`
- AND a DDS DataReader is subscribed to Topic "CommandData" on domain 0
- THEN the DDS-Zenoh adapter MUST inject the sample into the DDS domain
- AND the DataReader MUST receive the sample as if published by a native DDS DataWriter

#### Scenario: DDS wildcard subscription via Zenoh

- WHEN a Zenoh subscriber registers for `dds/0/**`
- THEN the subscriber MUST receive samples from ALL topics on DDS domain 0
- AND each sample's key expression MUST indicate the originating topic name

#### Scenario: Cross-transport routing — DDS to CAN

- WHEN a DDS DataWriter publishes on Topic "BrakeCommand" in domain 0
- AND a bridge rule maps `dds/0/BrakeCommand` to `can/0/0x200`
- THEN the Zenoh router MUST route the DDS sample to the CAN transport adapter
- AND the CAN adapter MUST serialize the payload into CAN frame(s) and transmit on bus 0

#### Scenario: ROS 2 interoperability via RTPS

- WHEN a ROS 2 node publishes on topic `/sensor_data` using FastDDS with domain 0
- AND a SmallAIOS DDS participant is on the same network with domain 0
- THEN SmallAIOS MUST discover the ROS 2 node via SPDP/SEDP
- AND SmallAIOS MUST receive the published messages via RTPS
- AND the messages MUST be accessible on Zenoh key expression `dds/0/rt/sensor_data` (ROS 2 topic mangling)

#### Scenario: AUTOSAR Adaptive interoperability

- WHEN an AUTOSAR Adaptive Platform application publishes a service event via DDS/SOME-IP binding
- AND a SmallAIOS DDS participant is on the same domain
- THEN SmallAIOS MUST discover and match the AUTOSAR DDS endpoints
- AND data exchange MUST use standard RTPS/CDR encoding

### Requirement: CDR Serialization

SmallAIOS SHALL implement CDR (Common Data Representation) v2 serialization as defined in the OMG XCDR specification. CDR is the default serialization format for DDS and is required for RTPS interoperability.

#### Scenario: Serialize and deserialize a struct

- WHEN a data type `SensorMsg { timestamp: u64, value: f64, id: u32 }` is defined
- AND a DataWriter serializes an instance via CDR
- THEN the serialized bytes MUST follow CDR v2 encoding rules (alignment, endianness flag, padding)
- AND a DataReader on any compliant DDS implementation MUST be able to deserialize the bytes

#### Scenario: CDR endianness

- WHEN a sample is serialized on a little-endian system
- THEN the CDR encapsulation header MUST indicate little-endian byte order (0x0001)
- AND a big-endian receiver MUST correctly decode all fields by respecting the endianness flag

#### Scenario: CDR alignment

- WHEN serializing a struct with mixed field sizes (u8, u32, u64)
- THEN each field MUST be aligned to its natural alignment boundary per CDR rules
- AND padding bytes MUST be inserted as needed between fields
- AND the total serialized size MUST match CDR specification requirements
