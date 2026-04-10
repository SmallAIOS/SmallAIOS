## Why

SmallAIOS has three pure-Rust messaging stacks (`net/quic`, `bus/dds`, `ipc`) and a working ONNX inference pipeline (loaded via PR #61), but they're not connected. The HTTP server is the only way to submit inference requests today. For real-time/streaming workloads — robotics, autonomous systems, sensor pipelines — HTTP request/response has too much overhead. The natural fit is dataflow: a client publishes input tensors to a topic, the runtime subscribes, runs inference, and publishes results to an output topic. This is how ROS 2 (DDS) and Zenoh-based systems already work.

This change wires the existing IPC/DDS/Zenoh stacks to the ONNX runtime so SmallAIOS can serve inference via pub/sub message buses, not just HTTP.

## What Changes

- Wire `ipc::endpoints::inference` to actually call `Session::run()` (currently a stub that echoes)
- Add a `dataflow_runner` module that subscribes to an input topic, runs inference, publishes to an output topic
- Add a Zenoh inference example: client publishes tensor to `smallaios/inference/<model>/input`, runner publishes result to `smallaios/inference/<model>/output`
- Add a DDS inference adapter using the existing `DdsZenohAdapter` bridge — same dataflow pattern, DDS topic names
- End-to-end integration test: spin up runner + client in-process, send tensor, verify output
- Container binary: optional `--bus` mode that starts the dataflow runner alongside (or instead of) the HTTP server

## Capabilities

### New Capabilities
- `dataflow-inference`: Pub/sub-based inference pipeline subscribing to input tensor topics, executing models, publishing outputs

### Modified Capabilities
- `ipc-messaging`: Wire the inference endpoint stub to real `Session::run()` calls
- `container-inference-server`: Add `--bus` mode and `SMALLAIOS_BUS_BACKEND` env var (zenoh/dds/none)

## Impact

- **Code:** New `ipc/src/dataflow_runner.rs` (~250 lines), update `ipc/src/endpoints/inference.rs` to call ONNX runtime, update `container/src/main.rs` for `--bus` mode
- **Cross-crate dep:** `ipc` may need an optional `smallaios-onnx-rt` dep behind a feature flag (or invert: container wires them)
- **Pure Rust goal:** Zero new external dependencies — uses existing pure-Rust QUIC, DDS, Zenoh implementations
- **Testing:** End-to-end test using the existing loopback transport (no real network needed)
