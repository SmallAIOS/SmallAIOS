## ADDED Requirements

### Requirement: Pub/Sub Inference Runner
The system SHALL provide a dataflow runner that subscribes to input tensor topics, executes inference, and publishes output tensors to result topics.

#### Scenario: Subscribe and process input
- **WHEN** the runner is started with a model name
- **THEN** it MUST subscribe to `smallaios/inference/<model>/input`
- **AND** for each received message, decode the binary inference protocol
- **AND** call `Session::run()` with the input tensors
- **AND** publish the result to `smallaios/inference/<model>/output`

#### Scenario: Publish errors to dedicated topic
- **WHEN** inference execution fails
- **THEN** the runner MUST publish an error message to `smallaios/inference/<model>/error`
- **AND** MUST NOT publish a partial output
- **AND** MUST continue processing subsequent input messages

#### Scenario: Wildcard model subscription
- **WHEN** the runner is started with topic pattern `smallaios/inference/*/input`
- **THEN** it MUST handle multiple models concurrently
- **AND** dispatch each input to the model named in the topic path

### Requirement: Backpressure Handling
The dataflow runner SHALL handle the case where inference cannot keep up with input rate.

#### Scenario: Drop oldest on queue overflow
- **WHEN** the input queue is full and a new message arrives
- **THEN** the runner MUST drop the oldest queued message
- **AND** MUST increment a `inference_dropped_messages_total` counter

#### Scenario: Configurable queue depth
- **WHEN** the runner is created with a `max_queue_depth` parameter
- **THEN** the queue MUST not exceed that depth
- **AND** the default MUST be 16 messages

### Requirement: DDS Topic Compatibility
The dataflow runner SHALL be usable from DDS clients via the existing DDS↔Zenoh adapter.

#### Scenario: DDS client publishes to inference topic
- **WHEN** a DDS DataWriter publishes to topic `smallaios/inference/<model>/input` via the DDS-Zenoh adapter
- **THEN** the runner subscribed to the equivalent Zenoh key expression MUST receive the message
- **AND** the response MUST be deliverable to a DDS DataReader on the output topic
