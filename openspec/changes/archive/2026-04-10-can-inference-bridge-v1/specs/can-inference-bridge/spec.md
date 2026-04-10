## ADDED Requirements

### Requirement: CAN Frame to Tensor Mapping
The CAN inference adapter SHALL map incoming CAN frames to positions within an inference input tensor based on a configurable routing table.

#### Scenario: Single sensor frame
- **WHEN** a CAN frame with ID `0x100` arrives and the routing table maps it to topic `smallaios/inference/perception/input` at offset 0
- **THEN** the adapter MUST copy the frame payload to the corresponding offset in the topic's batch buffer

#### Scenario: Frame for unknown CAN ID
- **WHEN** a CAN frame arrives with an ID not in the routing table
- **THEN** the adapter MUST drop the frame
- **AND** MUST increment a `can_unrouted_frames_total` counter

### Requirement: Batch Trigger
The adapter SHALL flush a batched tensor to the dataflow runner when the configured trigger fires.

#### Scenario: Frame count trigger
- **WHEN** the routing config specifies `BatchTrigger::FrameCount(N)` and N frames have been received for a topic
- **THEN** the adapter MUST return the (topic, payload) pair for the caller to publish

#### Scenario: Time window trigger
- **WHEN** the routing config specifies `BatchTrigger::TimeWindow(T)` and T microseconds have elapsed since the first frame in the batch
- **THEN** the adapter MUST flush the batch even if not all expected frames arrived
- **AND** MUST increment a `can_partial_batches_total` counter

#### Scenario: Trigger frame
- **WHEN** the routing config specifies `BatchTrigger::OnFrame(trigger_id)` and a frame with that ID is received
- **THEN** the adapter MUST flush the batch immediately

### Requirement: Inference Output to CAN Frames
The adapter SHALL convert inference output tensors back into CAN frames using a symmetric output routing table.

#### Scenario: Single output value to CAN frame
- **WHEN** an inference output is received with topic `smallaios/inference/control/output`
- **AND** the output routing maps offset 0 size 4 to CAN ID `0x200`
- **THEN** the adapter MUST produce a CAN frame with ID `0x200` and the 4 bytes from offset 0

### Requirement: CANaerospace Decoder
The adapter SHALL support decoding CANaerospace-formatted frames into typed tensor values.

#### Scenario: CANaerospace float32 sensor
- **WHEN** a CANaerospace NOD frame arrives with data_type=FLOAT and a float32 payload
- **AND** the routing table specifies `FrameDecoder::CanAerospaceFloat32`
- **THEN** the adapter MUST extract the float value per ARINC 825 §3.3
- **AND** write it to the tensor at the configured offset

#### Scenario: Reject mismatched CANaerospace data type
- **WHEN** a CANaerospace frame arrives with a data_type that does not match the configured decoder
- **THEN** the adapter MUST drop the frame and increment a counter
- **AND** MUST NOT corrupt the batch buffer

### Requirement: Stale Frame Detection
The adapter SHALL detect and reject inference batches built from stale sensor data.

#### Scenario: All required frames arrive within freshness window
- **WHEN** all routed CAN IDs receive frames within the configured freshness window
- **THEN** the batch MUST be published normally

#### Scenario: Some frames are stale
- **WHEN** the time window elapses but one or more required frames have timestamps older than the freshness window
- **THEN** the batch MUST be dropped without running inference
- **AND** a `can_stale_batch_drops_total` counter MUST be incremented
