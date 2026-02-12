# Delta for Security Boundaries

## ADDED Requirements

### Requirement: Trust Domain Boundary Documentation
The system SHALL document all trust domain boundaries: kernel boundary (capability-protected syscalls), K8s boundary (Virtual Kubelet management API over mutual TLS), network boundary (TLS 1.3 termination, stateful firewall), bus protocol boundaries (CAN/ARINC/1553/SpaceWire/CCSDS transport isolation), and GPU boundary (DMA memory restrictions, command validation).

#### Scenario: Document kernel trust boundary
- WHEN the security boundary documentation is generated
- THEN it MUST include the kernel boundary definition specifying capability-protected syscalls as the enforcement mechanism
- AND MUST enumerate all ~49 syscall entry points as the boundary interface
- AND MUST describe the single-address-space unikernel model with capability checks replacing ring transitions

#### Scenario: Document K8s trust boundary
- WHEN the security boundary documentation is generated
- THEN it MUST include the K8s boundary definition specifying the Virtual Kubelet management API as the boundary interface
- AND MUST document mutual TLS (mTLS) with post-quantum hybrid key exchange as the transport protection
- AND MUST describe the pod lifecycle management operations that cross this boundary

#### Scenario: Document network trust boundary
- WHEN the security boundary documentation is generated
- THEN it MUST include the network boundary definition specifying TLS 1.3 termination as the inbound protection mechanism
- AND MUST document the stateful firewall rules governing ingress and egress traffic
- AND MUST describe connection tracking state machine and packet filtering criteria

#### Scenario: Document bus protocol trust boundaries
- WHEN the security boundary documentation is generated
- THEN it MUST include separate boundary definitions for each bus protocol: CAN (ISO 11898), ARINC 429/664, MIL-STD-1553, SpaceWire (ECSS-E-ST-50-12C), and CCSDS SPP
- AND each bus boundary MUST document the transport isolation mechanism that prevents cross-bus data leakage
- AND MUST describe the framing and validation applied at each bus protocol boundary

#### Scenario: Document GPU trust boundary
- WHEN the security boundary documentation is generated
- THEN it MUST include the GPU boundary definition specifying DMA memory restrictions that confine GPU access to designated tensor memory regions
- AND MUST document the command validation mechanism that inspects GPU command buffers before submission
- AND MUST describe the supported GPU architectures (Maxwell through Blackwell) and any architecture-specific boundary enforcement differences

### Requirement: Trust Boundary Specification Attributes
Each trust boundary SHALL specify: protocol/mechanism, authentication method, authorization model, data flow direction, and maximum data rate.

#### Scenario: Kernel boundary attributes
- WHEN the kernel trust boundary specification is reviewed
- THEN it MUST specify the protocol/mechanism as capability-protected syscall dispatch
- AND MUST specify the authentication method as capability token validation (unforgeable kernel-issued tokens)
- AND MUST specify the authorization model as capability-based access control with per-resource capability grants
- AND MUST specify the data flow direction as bidirectional (syscall arguments inbound, return values outbound)
- AND MUST specify the maximum data rate as bounded by syscall dispatch latency and task scheduling throughput

#### Scenario: Network boundary attributes
- WHEN the network trust boundary specification is reviewed
- THEN it MUST specify the protocol/mechanism as TLS 1.3 with hybrid X25519+ML-KEM-768 key exchange
- AND MUST specify the authentication method as mutual TLS with ML-DSA-65 certificate verification
- AND MUST specify the authorization model as connection-level authorization via certificate identity mapping
- AND MUST specify the data flow direction for each service endpoint (inbound inference requests, outbound inference results, bidirectional management)
- AND MUST specify the maximum data rate per interface (configurable, with defaults documented per deployment class)

#### Scenario: Bus protocol boundary attributes
- WHEN any bus protocol trust boundary specification is reviewed
- THEN it MUST specify the protocol/mechanism as the bus-specific framing and encoding (e.g., CAN 2.0A/B/FD, ARINC 429 word format, 1553 command/data words, SpaceWire packets, CCSDS SPP)
- AND MUST specify the authentication method (bus protocol framing integrity for physical buses, capability check for software interface)
- AND MUST specify the authorization model as capability-gated bus access (one capability per bus instance)
- AND MUST specify the data flow direction per bus role (e.g., bus controller vs. remote terminal for 1553)
- AND MUST specify the maximum data rate per bus type (CAN: 1 Mbit/s classic / 8 Mbit/s FD, ARINC 429: 100 kbit/s, 1553: 1 Mbit/s, SpaceWire: 200 Mbit/s, CCSDS: link-dependent)

#### Scenario: GPU boundary attributes
- WHEN the GPU trust boundary specification is reviewed
- THEN it MUST specify the protocol/mechanism as validated GPU command buffer submission with DMA region confinement
- AND MUST specify the authentication method as capability token for GPU access (one capability per GPU context)
- AND MUST specify the authorization model as capability-based with per-tensor-region DMA permissions
- AND MUST specify the data flow direction as bidirectional (model weights and input tensors to GPU, inference results from GPU)
- AND MUST specify the maximum data rate as bounded by PCIe/NVLink bandwidth for the target GPU architecture

### Requirement: Information Flow Enforcement
The system SHALL enforce information flow rules: ONNX runtime SHALL NOT access network resources; IPC router SHALL NOT access GPU; bus protocol handlers SHALL NOT access ONNX models -- all enforced via the capability system.

#### Scenario: ONNX runtime network isolation
- WHEN the ONNX runtime requests a capability granting access to any network resource (socket, TLS session, firewall configuration)
- THEN the capability system MUST deny the request
- AND MUST log the denial as a security event with the requesting task ID, requested resource, and timestamp
- AND the ONNX runtime task MUST NOT be granted network capabilities at any point during its lifecycle

#### Scenario: IPC router GPU isolation
- WHEN the IPC router requests a capability granting access to any GPU resource (command buffer, DMA region, GPU context)
- THEN the capability system MUST deny the request
- AND MUST log the denial as a security event
- AND the IPC router task MUST NOT be granted GPU capabilities at any point during its lifecycle

#### Scenario: Bus protocol handler model isolation
- WHEN any bus protocol handler (CAN, ARINC, 1553, SpaceWire, CCSDS) requests a capability granting access to ONNX model data (model files, tensor buffers, inference contexts)
- THEN the capability system MUST deny the request
- AND MUST log the denial as a security event
- AND no bus protocol handler task MUST be granted ONNX model capabilities at any point during its lifecycle

#### Scenario: Capability grant audit for information flow
- WHEN any capability is granted to a task
- THEN the capability system MUST verify the grant does not violate the information flow enforcement matrix
- AND MUST reject the grant and generate a security audit event if the flow would be disallowed
- AND the information flow matrix MUST be defined at build time and immutable at runtime

### Requirement: PlantUML Diagrams Traceable to Sphinx-needs
Security boundary documentation SHALL include PlantUML diagrams traceable to Sphinx-needs requirements.

#### Scenario: Generate boundary topology diagram
- WHEN the security boundary documentation is built via Sphinx
- THEN it MUST include a PlantUML component diagram showing all trust domains (kernel, K8s, network, bus protocols, GPU) and their interconnections
- AND each component in the diagram MUST reference a Sphinx-needs requirement ID using the `<<req_id>>` stereotype or equivalent traceability annotation

#### Scenario: Generate data flow diagrams per boundary
- WHEN the security boundary documentation is built via Sphinx
- THEN it MUST include a PlantUML sequence diagram for each trust boundary showing the data flow, authentication exchange, and authorization check
- AND each message and decision point in the sequence diagram MUST be traceable to a specific Sphinx-needs requirement

#### Scenario: Generate information flow enforcement diagram
- WHEN the security boundary documentation is built via Sphinx
- THEN it MUST include a PlantUML diagram depicting the information flow enforcement matrix (ONNX runtime, IPC router, bus handlers, GPU, network) with allowed and denied flows
- AND denied flows MUST be visually distinguished (e.g., dashed red lines) and annotated with the enforcing capability rule

#### Scenario: Traceability validation
- WHEN the Sphinx documentation build completes
- THEN every PlantUML diagram element referencing a Sphinx-needs requirement ID MUST resolve to a valid requirement in the needs database
- AND the build MUST emit a warning for any diagram element that references a nonexistent requirement ID

### Requirement: Attack Surface Inventory
The system SHALL maintain an attack surface inventory per boundary listing: entry points, data formats accepted, validation mechanisms, and known limitations.

#### Scenario: Kernel boundary attack surface inventory
- WHEN the attack surface inventory for the kernel boundary is reviewed
- THEN it MUST list every syscall entry point (all ~49 syscalls) with its parameter types and valid ranges
- AND MUST document the data formats accepted by each syscall (capability tokens, buffer pointers, size parameters, flags)
- AND MUST document the validation mechanism for each parameter (capability check, bounds check, type check)
- AND MUST document known limitations (e.g., single address space implies shared memory visibility)

#### Scenario: Network boundary attack surface inventory
- WHEN the attack surface inventory for the network boundary is reviewed
- THEN it MUST list every listening port and protocol endpoint
- AND MUST document the data formats accepted (TLS 1.3 records, Zenoh protocol messages, management API payloads)
- AND MUST document the validation mechanisms (TLS handshake verification, message schema validation, rate limiting)
- AND MUST document known limitations (e.g., TLS termination CPU cost, maximum concurrent connections)

#### Scenario: Bus protocol boundary attack surface inventory
- WHEN the attack surface inventory for any bus protocol boundary is reviewed
- THEN it MUST list every bus interface endpoint (physical port or virtual bus instance)
- AND MUST document the data formats accepted (frame formats per protocol specification)
- AND MUST document the validation mechanisms (CRC verification, frame format checks, acceptance filtering, capability gating)
- AND MUST document known limitations (e.g., CAN lacks native authentication, 1553 bus controller trust assumption)

#### Scenario: GPU boundary attack surface inventory
- WHEN the attack surface inventory for the GPU boundary is reviewed
- THEN it MUST list every GPU interaction endpoint (command buffer submission, DMA mapping, context management)
- AND MUST document the data formats accepted (GPU command opcodes, tensor data layouts, DMA descriptors)
- AND MUST document the validation mechanisms (command buffer inspection, DMA region bounds checking, capability verification)
- AND MUST document known limitations (e.g., GPU firmware opacity, side-channel risks in shared GPU)

#### Scenario: Inventory update on change
- WHEN a new entry point, data format, or validation mechanism is added to any trust boundary
- THEN the attack surface inventory for that boundary MUST be updated before the change is merged
- AND the change control process MUST require attack surface inventory review as a merge gate

### Requirement: Cross-Boundary Data Flow Protection
All cross-boundary data flows SHALL be authenticated and integrity-protected: capability check for kernel boundary, mutual TLS for network boundary, and bus protocol framing for bus boundaries.

#### Scenario: Kernel boundary data flow authentication
- WHEN a task invokes a syscall that transfers data across the kernel boundary
- THEN the syscall dispatcher MUST verify the caller holds a valid capability for the requested operation
- AND MUST validate all data parameters (buffer pointer bounds, size limits, type correctness) before processing
- AND MUST reject the syscall with an appropriate error code if the capability check or validation fails

#### Scenario: Network boundary data flow authentication
- WHEN data is transmitted or received across the network boundary
- THEN the data flow MUST be protected by a mutual TLS 1.3 session with both client and server certificate verification
- AND the TLS session MUST use hybrid X25519+ML-KEM-768 key exchange and AES-256-GCM authenticated encryption
- AND MUST reject any data received on an unauthenticated or integrity-unprotected channel

#### Scenario: Bus protocol boundary data flow integrity
- WHEN data is transmitted or received across a bus protocol boundary
- THEN the data MUST be framed according to the bus protocol specification (CAN CRC-15/17/21, ARINC 429 parity, 1553 parity and sync patterns, SpaceWire EOP markers, CCSDS CRC-16)
- AND the receiving side MUST verify the frame integrity check before delivering data to the application layer
- AND frames failing integrity checks MUST be discarded and logged as security events

#### Scenario: GPU boundary data flow protection
- WHEN data is transferred across the GPU boundary (host-to-device or device-to-host DMA)
- THEN the transfer MUST be authorized by a valid GPU capability held by the requesting task
- AND the DMA region MUST be bounds-checked to ensure it falls within the task's allocated tensor memory
- AND any DMA request targeting memory outside the authorized region MUST be denied and logged as a security event

#### Scenario: Reject unprotected cross-boundary flow
- WHEN any component attempts to send data across a trust boundary without the required authentication or integrity protection
- THEN the boundary enforcement mechanism MUST reject the data flow
- AND MUST generate a security audit event recording the source, destination, boundary, and reason for rejection
