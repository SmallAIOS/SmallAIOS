## Why

SmallAIOS has a complete pure-Rust CAN stack (`bus/src/can/`) — CAN 2.0A/B, CAN FD, MCP2515 + AXI CAN drivers, CANaerospace (ARINC 825) avionics protocol — and a working dataflow inference runner (PR #62) that processes pub/sub messages. The natural combination is **AI inference on CAN networks**: read sensor frames, run a model, write actuator commands. This is the dominant pattern for safety-critical edge AI in automotive ADAS, industrial robotics, drones, and avionics.

Today, the dataflow runner only speaks Zenoh-style pub/sub and DDS. To deploy SmallAIOS as a real-time AI controller on a CAN network, we need a CAN-to-runner adapter that maps CAN frames to inference inputs and inference outputs back to CAN frames.

## What Changes

- Add `bus::can::adapter::CanInferenceAdapter` that bridges CAN frames to the dataflow runner's pub/sub interface
- Map CAN IDs to topic names via configurable rules (e.g., CAN ID `0x100` → `smallaios/inference/perception/input`)
- Frame batching: aggregate N CAN frames over a time window into one input tensor
- CANaerospace integration: typed sensor messages (NOD/EED) decoded into semantically-labeled tensor inputs
- Container `SMALLAIOS_BUS_BACKEND=can` mode that wires a CAN controller (loopback or hardware) to the runner
- End-to-end test using the existing `loopback.rs` CAN transport: publish frames → runner runs inference → result frames published back

## Capabilities

### New Capabilities
- `can-inference-bridge`: Bidirectional CAN ↔ ONNX inference adapter with frame batching, ID-to-topic mapping, and CANaerospace decoding

### Modified Capabilities
- `can-bus`: Add adapter trait for higher-level data routing (currently just frame transport)
- `dataflow-inference`: Accept CAN as a transport backend alongside Zenoh/DDS
- `container-inference-server`: `SMALLAIOS_BUS_BACKEND=can` plus `SMALLAIOS_CAN_DEVICE` (e.g., `loopback`, `mcp2515:/dev/spidev0.0`, `axi:0x40000000`)

## Impact

- **Code:** New `bus/src/can/inference_adapter.rs` (~400 lines), CAN config parsing in `container/src/main.rs`, optional `onnx` feature on `bus` crate
- **Pure Rust:** Zero new external dependencies — uses existing `bus::can`, `ipc::dataflow_runner`, `onnx-rt`
- **Hardware support:** Loopback for testing, MCP2515 for SPI-attached CAN, AXI CAN for FPGA. Real hardware untested in CI but the abstraction supports it.
- **Use cases unlocked:** Automotive ADAS (sensor fusion → steering), industrial control (CAN PLC → policy network), drones (IMU/GPS → flight controller), avionics (CANaerospace → DO-178C-compliant inference)
