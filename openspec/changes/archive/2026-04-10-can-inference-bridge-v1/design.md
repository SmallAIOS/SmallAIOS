## Context

The `bus::can` crate provides CAN frame I/O via a `CanController` trait. Implementations exist for `loopback` (in-process), `mcp2515` (Microchip SPI controller), and `axi_can` (Xilinx FPGA). Frames are 11-bit (CAN 2.0A), 29-bit (CAN 2.0B), or CAN FD (up to 64-byte payload).

The `ipc::dataflow_runner::DataflowRunner` from PR #62 processes inference messages identified by topic name. Each message is binary-encoded via `inference_proto::InferenceRequest`. The runner is transport-agnostic — it just provides `process_message(model, payload) -> Result<Vec<u8>>`.

The gap: there's no code that reads CAN frames, batches them into tensors, calls the runner, and writes results back to CAN. CAN frames are 8 bytes (or 64 with FD), so a single frame can't carry a meaningful inference input — they need to be aggregated.

## Goals / Non-Goals

**Goals:**
- Bidirectional CAN ↔ inference: subscribe to a set of CAN IDs, batch them into a tensor, run inference, write results back as CAN frames
- Support CAN 2.0A, 2.0B, and CAN FD
- ID-to-topic mapping with configurable routing tables
- CANaerospace decoding for typed sensor data (float32, int16/32 per ARINC 825)
- Frame batching with time window OR frame count triggers
- Loopback testing (no real hardware needed for CI)
- Pure Rust, no FFI

**Non-Goals:**
- DBC file parsing (Vector database for CAN signals) — could be added later
- ISO-TP / DoIP — those are higher-level transport protocols on CAN, separate change
- J1939 — automotive truck protocol, separate change if needed
- Hardware bring-up testing in CI — only loopback in CI; hardware tests run on dev boards

## Decisions

### D1: CanInferenceAdapter Holds Configuration, Not the Controller

```rust
pub struct CanInferenceAdapter {
    config: CanInferenceConfig,
    routing_table: BTreeMap<u32, RouteSpec>,  // CAN ID → topic + tensor slot
    batch_buffers: BTreeMap<String, BatchBuffer>,  // topic → accumulated frames
}

pub struct RouteSpec {
    pub topic: String,           // e.g., "smallaios/inference/perception/input"
    pub tensor_offset: usize,    // byte offset in the input tensor
    pub frame_size: usize,       // bytes per frame to copy (8 for CAN 2.0, up to 64 for FD)
    pub decoder: FrameDecoder,   // raw bytes, CANaerospace, or custom
}

pub enum FrameDecoder {
    RawBytes,
    CanAerospaceFloat32,
    CanAerospaceInt32,
    Custom(fn(&CanFrame) -> Vec<u8>),
}

pub enum BatchTrigger {
    /// Flush after N frames received
    FrameCount(usize),
    /// Flush after T microseconds since first frame in batch
    TimeWindow(u64),
    /// Flush on receipt of a specific "trigger" CAN ID
    OnFrame(u32),
}
```

The adapter takes raw frames in via `on_frame(&CanFrame) -> Option<(String, Vec<u8>)>` — if a batch completes, it returns the topic + payload to publish to the runner. This keeps the adapter pure data transformation; the caller owns the controller and the runner.

### D2: Tensor Layout — Caller Provides Schema

The adapter doesn't know what shape the model expects. The routing table maps each CAN ID to a (topic, offset, size) tuple. The caller pre-allocates a buffer the size of the model's input tensor, frames write into it via offset, and when the batch trigger fires, the buffer is wrapped in an `InferenceRequest` and published.

Example: a perception model with input shape `[1, 6]` (6 float32 = 24 bytes) reading 6 sensors:
```
CAN ID 0x100 → offset 0,  size 4 (steering angle f32)
CAN ID 0x101 → offset 4,  size 4 (vehicle speed f32)
CAN ID 0x102 → offset 8,  size 4 (yaw rate f32)
CAN ID 0x103 → offset 12, size 4 (lateral accel f32)
CAN ID 0x104 → offset 16, size 4 (longitudinal accel f32)
CAN ID 0x105 → offset 20, size 4 (brake pressure f32)
```

When all 6 frames arrive (or the time window expires), the buffer is published as the model input.

### D3: Inverse Mapping — Inference Output to CAN Frames

After inference, the result tensor needs to be written back as CAN frames. Symmetric routing table:

```rust
pub struct OutputRouteSpec {
    pub source_topic: String,
    pub tensor_offset: usize,
    pub frame_size: usize,
    pub can_id: u32,
    pub encoder: FrameEncoder,
}
```

`adapter.on_inference_output(topic, payload) -> Vec<CanFrame>` returns the frames to transmit.

### D4: CANaerospace Integration

CANaerospace messages have a structured 4-byte payload: `[node_id, data_type, service_code, message_code]` followed by typed data. The `FrameDecoder::CanAerospaceFloat32` variant extracts the data field as f32 bytes (handling endianness per ARINC 825 §3.3).

This makes the adapter directly usable with CANaerospace-compliant avionics buses without manual byte fiddling.

### D5: Container `SMALLAIOS_BUS_BACKEND=can`

```bash
SMALLAIOS_BUS_BACKEND=can
SMALLAIOS_CAN_DEVICE=loopback                    # in-process testing
SMALLAIOS_CAN_DEVICE=mcp2515:/dev/spidev0.0      # MCP2515 on SPI
SMALLAIOS_CAN_DEVICE=axi:0x40000000              # AXI CAN at MMIO base
SMALLAIOS_CAN_ROUTING=/etc/smallaios/can-routes.toml  # routing table file
```

The container loads the routing table at startup, instantiates the adapter, attaches it to the configured CAN controller, and feeds the dataflow runner.

## Risks / Trade-offs

**[Risk] Frame loss during batching** — If CAN messages drop, the input tensor is incomplete. Mitigation: each route has a "freshness" timestamp; if any required input is stale beyond the window, the batch is dropped (with a metric counter) rather than running inference on stale data.

**[Risk] Real-time jitter** — Inference latency may exceed CAN message period. Mitigation: the runner already has backpressure (drop oldest). For hard real-time, the OperatorBudget mechanism enforces per-operator time limits.

**[Trade-off] Static routing table vs. dynamic** — A TOML file must be updated to add new sensors. Acceptable for safety-critical deployments where routing is part of the certified configuration. Dynamic discovery is a non-goal for DAL A.

## Open Questions

- **Q1:** Should the routing table support per-frame normalization (e.g., raw int16 sensor → normalized float32)? *Leaning toward: yes, via the FrameDecoder::Custom variant.*
- **Q2:** How to handle 29-bit extended IDs vs 11-bit standard — separate routing tables or unified? *Leaning toward: unified, store full 32-bit ID with extended flag in upper bit.*
