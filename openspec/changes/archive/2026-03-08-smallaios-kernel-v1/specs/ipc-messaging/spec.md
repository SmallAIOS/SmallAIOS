# Delta for IPC Messaging

## ADDED Requirements

### Requirement: Key Expression Matching
The IPC router SHALL support hierarchical key expressions with wildcard matching for resource addressing.

#### Scenario: Single-level wildcard match
- WHEN a subscriber registers for key expression "smallaios/models/*/infer"
- AND a publisher publishes to "smallaios/models/resnet50/infer"
- THEN the router MUST deliver the message to the subscriber
- AND MUST NOT deliver messages published to "smallaios/models/resnet50/metadata"

#### Scenario: Multi-level wildcard match
- WHEN a subscriber registers for key expression "smallaios/models/**"
- AND a publisher publishes to "smallaios/models/resnet50/infer"
- THEN the router MUST deliver the message to the subscriber
- AND MUST also deliver messages to "smallaios/models/yolo/metadata"

#### Scenario: Exact match
- WHEN a subscriber registers for "smallaios/health" (no wildcards)
- THEN the router MUST deliver only messages published to exactly "smallaios/health"

### Requirement: Pub/Sub Pattern
The IPC system SHALL implement a fire-and-forget publish/subscribe pattern.

#### Scenario: Publish to multiple subscribers
- WHEN two subscribers are registered for the same key expression
- AND a publisher publishes a message to a matching key
- THEN the router MUST deliver the message to both subscribers
- AND delivery order between subscribers is unspecified

#### Scenario: No subscriber available
- WHEN a publisher publishes a message and no subscribers match the key expression
- THEN the message MUST be silently dropped
- AND the publisher MUST NOT receive an error

### Requirement: Request/Reply Pattern
The IPC system SHALL implement a synchronous request/reply pattern via queryable endpoints.

#### Scenario: Successful query and reply
- WHEN a queryable is registered on key "smallaios/v1/models/resnet50/infer"
- AND a client sends a query with input tensor data
- THEN the queryable MUST receive the query and process it
- AND the reply MUST be delivered back to the requesting client

#### Scenario: Query timeout
- WHEN a client sends a query and the queryable does not reply within the configured timeout
- THEN the client MUST receive a timeout error
- AND the query MUST be cancelled

### Requirement: Shared Memory Zero-Copy Transport
The IPC system SHALL support zero-copy shared memory transport for intra-kernel and container-to-container communication.

#### Scenario: Intra-kernel zero-copy delivery
- WHEN a publisher and subscriber are within the same kernel address space
- THEN the router MUST deliver messages via shared buffer reference without copying data
- AND MUST use reference-counted buffers to manage lifetime

#### Scenario: Lock-free ring buffer delivery
- WHEN a single producer publishes to a single consumer
- THEN the transport MUST use a lock-free SPSC ring buffer
- AND MUST NOT block the publisher if the buffer is full (backpressure via drop or error)

### Requirement: TCP Transport
The IPC system SHALL support TCP transport for external client communication on a configurable port (default 7447).

#### Scenario: Accept external TCP connection
- WHEN an external client connects to the IPC TCP listener port
- THEN the IPC system MUST accept the connection
- AND MUST process Zenoh-compatible wire protocol frames

### Requirement: TLS 1.3 Transport with PQC
The IPC system SHALL support TLS 1.3 with post-quantum key exchange for encrypted external communication.

#### Scenario: Establish PQC TLS connection
- WHEN TLS is enabled and an external client initiates a connection
- THEN the IPC system MUST perform a TLS 1.3 handshake using hybrid X25519+ML-KEM-768 key exchange
- AND MUST encrypt all subsequent data with AES-256-GCM
- AND MUST reject connections attempting TLS versions below 1.3

### Requirement: Built-in Health Endpoint
The IPC system SHALL expose a health check queryable at "smallaios/v1/health".

#### Scenario: Health check query
- WHEN a client queries "smallaios/v1/health"
- THEN the system MUST reply with a JSON payload containing status "ok" and uptime in nanoseconds

### Requirement: Built-in Metrics Endpoint
The IPC system SHALL publish Prometheus-format metrics periodically on "smallaios/v1/metrics".

#### Scenario: Metrics publication
- WHEN the configured metrics interval elapses (default 5 seconds)
- THEN the system MUST publish current metrics including inference count, latency percentiles, memory usage, and active connections

### Requirement: Built-in Inference Endpoint
The IPC system SHALL expose model inference as a queryable at "smallaios/v1/models/{model_name}/infer".

#### Scenario: Inference request via IPC
- WHEN a client sends a query to "smallaios/v1/models/resnet50/infer" with valid input tensors
- THEN the system MUST execute inference and reply with output tensors
- AND the request/response MUST use the binary inference protocol format

### Requirement: Built-in Logs Endpoint
The IPC system SHALL publish kernel log messages on "smallaios/v1/logs".

#### Scenario: Log message publication
- WHEN the kernel emits a log message
- THEN the IPC system MUST publish it on the logs key expression
- AND MUST include timestamp, severity level, and message text

### Requirement: Inference Protocol Binary Format
The IPC system SHALL use a binary request/response format with a 16-byte header for inference messages.

#### Scenario: Parse inference request
- WHEN the system receives an inference request with magic bytes 0x534D4149 ("SMAI")
- THEN it MUST parse the header to extract version, input count, metadata length, and tensor data length
- AND MUST validate that tensor data offsets and lengths do not exceed the payload

#### Scenario: Reject invalid magic bytes
- WHEN the system receives a message with incorrect magic bytes
- THEN it MUST return an error response with error code and descriptive message
- AND MUST NOT attempt to parse the payload
