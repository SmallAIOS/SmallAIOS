# Delta for QUIC Transport

## ADDED Requirements

### Requirement: QUIC v1 Connection Establishment

SmallAIOS SHALL implement QUIC version 1 as specified in RFC 9000 and RFC 9001 (QUIC-TLS). The QUIC stack SHALL support both client and server roles. The TLS 1.3 handshake SHALL use ML-KEM-768 hybrid key exchange (classical X25519 + post-quantum ML-KEM-768) by default, with fallback to X25519-only for interoperability with non-PQ peers. Connection establishment SHALL complete within a single round trip (1-RTT) for new connections.

#### Scenario: 1-RTT connection establishment

- WHEN a SmallAIOS QUIC client initiates a connection to a remote server
- AND the client and server have not previously communicated
- THEN the QUIC handshake MUST complete in a single round trip (1-RTT)
- AND the TLS 1.3 handshake MUST negotiate ML-KEM-768 hybrid key exchange if both sides support it
- AND the connection MUST be usable for application data after the first server reply

#### Scenario: Server accepts incoming QUIC connection

- WHEN an external QUIC client connects to SmallAIOS's QUIC server endpoint
- THEN SmallAIOS MUST validate the client's TLS 1.3 ClientHello
- AND MUST complete the handshake with server certificate authentication
- AND MUST allocate connection state with a unique connection ID

#### Scenario: Version negotiation

- WHEN a client sends an Initial packet with an unsupported QUIC version
- THEN SmallAIOS MUST reply with a Version Negotiation packet listing QUIC v1 (0x00000001)
- AND MUST NOT allocate any connection state for the unsupported version

#### Scenario: TLS handshake failure — untrusted certificate

- WHEN a remote peer presents a TLS certificate not signed by a trusted CA
- THEN the QUIC handshake MUST fail with a TLS alert
- AND the connection MUST be closed with a CRYPTO_ERROR transport error code
- AND the failure MUST be logged in the audit log

### Requirement: 0-RTT Session Resumption

SmallAIOS SHALL support QUIC 0-RTT session resumption as specified in RFC 9000 Section 7.4 and RFC 8446 Section 2.3. When a client has previously connected to a server and possesses a valid session ticket, the client SHALL be able to send application data in the first flight (0-RTT) without waiting for a server reply. The server MUST validate the 0-RTT data against replay protection.

#### Scenario: 0-RTT resumption with valid session ticket

- WHEN a SmallAIOS QUIC client reconnects to a server it has previously connected to
- AND the client possesses a valid TLS session ticket from the prior connection
- THEN the client MUST send 0-RTT application data in the Initial flight
- AND the server MUST accept and process the 0-RTT data if the ticket is valid and not expired
- AND the total handshake latency MUST be zero round trips for the 0-RTT data

#### Scenario: 0-RTT rejected by server

- WHEN a server rejects 0-RTT data (e.g., ticket expired, anti-replay triggered)
- THEN the client MUST fall back to 1-RTT handshake
- AND the 0-RTT application data MUST be retransmitted after the 1-RTT handshake completes
- AND no application data MUST be lost

#### Scenario: 0-RTT replay protection

- WHEN the server receives 0-RTT data
- THEN the server MUST check the data against a replay cache (timestamp-based or strike register)
- AND if a replay is detected, the server MUST reject the 0-RTT data
- AND MUST proceed with a 1-RTT handshake instead

### Requirement: Connection Migration

SmallAIOS SHALL support QUIC connection migration as specified in RFC 9000 Section 9. When a peer's IP address or port changes (e.g., vehicular platform switching networks, mobile handoff), the connection SHALL continue without requiring a new handshake. Path validation SHALL be performed before migrating to confirm reachability of the new path.

#### Scenario: Client migrates to a new network address

- WHEN a SmallAIOS QUIC client's IP address changes (e.g., network interface failover)
- THEN the client MUST send packets from the new address with the existing connection ID
- AND the server MUST initiate path validation on the new path (PATH_CHALLENGE/PATH_RESPONSE)
- AND after successful validation, both peers MUST continue using the new path
- AND no application data MUST be lost during migration

#### Scenario: Path validation failure after migration

- WHEN path validation fails on a new path (no PATH_RESPONSE received within timeout)
- THEN the QUIC stack MUST revert to the previous path if it is still available
- AND if no valid path exists, the connection MUST be closed with a NO_VIABLE_PATH error
- AND the application MUST be notified of the connection loss

#### Scenario: NAT rebinding — port change without IP change

- WHEN a NAT device assigns a new source port to an existing connection
- THEN the server MUST recognize the connection via the connection ID
- AND MUST perform path validation on the new source port
- AND the connection MUST continue without interruption

### Requirement: Multiplexed Streams

SmallAIOS SHALL implement QUIC multiplexed bidirectional and unidirectional streams as specified in RFC 9000 Sections 2-3. Streams SHALL be independently flow-controlled. Loss on one stream MUST NOT block data delivery on other streams (no head-of-line blocking). The implementation SHALL support concurrent streams for mixed inference telemetry, model delivery, and management traffic on a single QUIC connection.

#### Scenario: Concurrent bidirectional streams

- WHEN a QUIC connection is established
- AND the application opens 3 bidirectional streams (inference results, telemetry, management)
- THEN all 3 streams MUST transmit and receive data concurrently
- AND packet loss on the telemetry stream MUST NOT delay delivery on the inference stream
- AND each stream MUST have independent flow control windows

#### Scenario: Unidirectional stream for model delivery

- WHEN a remote server needs to push an ONNX model to SmallAIOS
- THEN the server MUST open a unidirectional stream
- AND the model data MUST be delivered reliably in order on that stream
- AND other streams on the same connection MUST NOT be blocked during model transfer

#### Scenario: Stream concurrency limits

- WHEN the peer has advertised `initial_max_streams_bidi = 100`
- AND the application attempts to open stream 101
- THEN the QUIC stack MUST block or return a STREAM_LIMIT_REACHED error
- AND MUST NOT violate the peer's advertised stream limit
- AND when the peer increases the limit via MAX_STREAMS frame, the blocked stream MUST proceed

#### Scenario: Stream reset

- WHEN the application aborts a stream (e.g., cancelled inference request)
- THEN the QUIC stack MUST send a RESET_STREAM frame to the peer
- AND the peer MUST stop delivering data for that stream
- AND other streams on the connection MUST NOT be affected

### Requirement: Flow Control and Congestion Control

SmallAIOS SHALL implement QUIC flow control (stream-level and connection-level) and congestion control as specified in RFC 9000 Section 4 and RFC 9002. The congestion controller SHALL prevent overwhelming the network path. Flow control limits MUST be enforced to prevent unbounded memory consumption.

#### Scenario: Stream-level flow control

- WHEN a sender has transmitted data up to the receiver's stream flow control limit
- THEN the sender MUST stop sending on that stream until the receiver sends a MAX_STREAM_DATA frame
- AND other streams MUST NOT be affected by one stream's flow control limit

#### Scenario: Connection-level flow control

- WHEN the total data across all streams reaches the connection-level flow control limit
- THEN the sender MUST stop sending on ALL streams until the receiver sends a MAX_DATA frame
- AND the sender MUST NOT exceed the advertised connection-level limit

#### Scenario: Congestion window exceeded

- WHEN the number of bytes in flight exceeds the congestion window
- THEN the QUIC stack MUST queue packets until the congestion window allows transmission
- AND lost packets MUST trigger congestion window reduction per RFC 9002 (NewReno or similar)

### Requirement: Zenoh Session Transport Integration

SmallAIOS SHALL implement QUIC as a Zenoh session transport. Zenoh sessions between SmallAIOS nodes and between SmallAIOS and external Zenoh routers SHALL be able to use QUIC instead of TCP. The QUIC transport MUST be selectable via Zenoh locator syntax (e.g., `quic/192.168.1.1:7447`).

#### Scenario: Zenoh session over QUIC

- WHEN a Zenoh session is configured with locator `quic/192.168.1.1:7447`
- THEN the Zenoh transport layer MUST establish a QUIC connection to the specified address
- AND Zenoh protocol messages (SCOUT, HELLO, OPEN, FRAME, etc.) MUST be carried over QUIC streams
- AND the session MUST provide the same pub/sub/queryable semantics as a TCP-backed session

#### Scenario: Zenoh session resilience via QUIC connection migration

- WHEN a Zenoh session is running over QUIC
- AND the underlying network path changes (e.g., SmallAIOS node moves to a new network)
- THEN the QUIC connection MUST migrate to the new path
- AND the Zenoh session MUST remain active without requiring session re-establishment
- AND no published samples MUST be lost during migration

#### Scenario: Zenoh multiplexing over QUIC streams

- WHEN multiple Zenoh subscribers and publishers are active on a single QUIC-backed session
- THEN different Zenoh key expressions MAY be mapped to separate QUIC streams
- AND head-of-line blocking on one key expression MUST NOT delay delivery of others

### Requirement: Standalone QUIC Endpoint API

SmallAIOS SHALL provide a standalone QUIC endpoint API independent of Zenoh. This API SHALL allow creating QUIC server and client endpoints for use cases such as HTTP/3 management API, OTA model delivery, and cloud telemetry upload. The API SHALL expose stream-level operations (open, read, write, close, reset).

#### Scenario: HTTP/3 management endpoint

- WHEN the SmallAIOS management API is configured to serve over HTTP/3
- THEN SmallAIOS MUST accept QUIC connections on the configured port
- AND MUST handle HTTP/3 requests (GET /health, GET /metrics, POST /deploy) over QUIC streams
- AND responses MUST use HTTP/3 framing (HEADERS, DATA frames) per RFC 9114

#### Scenario: OTA model delivery over QUIC

- WHEN a remote model registry pushes a new ONNX model to SmallAIOS
- THEN the model data MUST be transferred over a QUIC stream
- AND the transfer MUST use TLS 1.3 encryption with ML-KEM-768
- AND the transfer MUST support resumption if interrupted (via QUIC 0-RTT or stream offset)

#### Scenario: Telemetry upload to cloud

- WHEN SmallAIOS sends inference telemetry to a cloud endpoint
- THEN the telemetry MUST be sent over a QUIC unidirectional stream
- AND if the connection is lost, 0-RTT resumption MUST be attempted on reconnect
- AND telemetry samples MUST be buffered locally until confirmed delivered

### Requirement: QUIC Packet Protection

SmallAIOS SHALL implement QUIC packet protection as specified in RFC 9001. All QUIC packets after the Initial phase SHALL be encrypted using the TLS 1.3 negotiated keys. Header protection SHALL be applied to prevent middlebox ossification. The implementation SHALL use SmallAIOS's existing cryptographic primitives for AES-128-GCM and ChaCha20-Poly1305 AEAD ciphers.

#### Scenario: Packet encryption with AES-128-GCM

- WHEN QUIC packets are sent after the handshake completes
- THEN each packet's payload MUST be encrypted with AES-128-GCM using the current traffic keys
- AND the packet number MUST be protected using AES-ECB header protection
- AND the authentication tag MUST cover the QUIC header and payload

#### Scenario: Key update

- WHEN either peer initiates a key update (RFC 9001 Section 6)
- THEN both peers MUST transition to the new traffic keys
- AND packets encrypted with old keys MUST still be accepted during the transition period
- AND the old keys MUST be discarded after the transition is confirmed

#### Scenario: ChaCha20-Poly1305 cipher negotiation

- WHEN the TLS handshake negotiates TLS_CHACHA20_POLY1305_SHA256
- THEN all subsequent QUIC packet protection MUST use ChaCha20-Poly1305
- AND header protection MUST use ChaCha20-based header protection per RFC 9001 Section 5.4.4
