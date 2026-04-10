## 1. IPC Crate Inference Wiring

- [ ] 1.1 Add optional `onnx` feature to `ipc/Cargo.toml` with `dep:smallaios-onnx-rt`
- [ ] 1.2 Update `ipc/src/endpoints/inference.rs`: replace stub with real `Session::run()` call when feature is enabled
- [ ] 1.3 Implement tensor decode/encode between IPC binary protocol (`inference_proto.rs`) and ONNX `Tensor` type
- [ ] 1.4 Handle errors: unknown model → `IpcError::NotFound`, shape mismatch → `IpcError::InvalidProtocol`
- [ ] 1.5 Unit tests: encode tensor → endpoint → decode → verify round-trip

## 2. Dataflow Runner Module

- [ ] 2.1 Create `ipc/src/dataflow_runner.rs` with `DataflowRunner` struct: holds subscriber, publisher, model manager ref, bounded queue
- [ ] 2.2 Implement `run()` method: subscribe to input topic, dequeue messages, dispatch to inference endpoint, publish results
- [ ] 2.3 Implement backpressure: bounded queue with drop-oldest semantics, atomic counter for dropped messages
- [ ] 2.4 Implement wildcard subscription: accept `smallaios/inference/*/input` pattern, dispatch by model name from topic path
- [ ] 2.5 Add `DataflowRunnerConfig` struct: `max_queue_depth` (default 16), `topic_prefix` (default "smallaios/inference")

## 3. Zenoh Transport Integration

- [ ] 3.1 Wire `DataflowRunner` to existing `ipc::pubsub` Zenoh-style pub/sub
- [ ] 3.2 Use `ipc::key_expr` for hierarchical topic matching
- [ ] 3.3 Test: in-process loopback transport, publish input → runner processes → verify output published

## 4. DDS Transport Integration

- [ ] 4.1 Wire `DataflowRunner` to DDS via `bus::dds::DdsZenohAdapter`
- [ ] 4.2 Map DDS topic names to Zenoh key expressions (already implemented in adapter)
- [ ] 4.3 Test: DDS DataWriter publishes input → adapter bridges → runner processes → result reaches DDS DataReader

## 5. Container Binary Bus Mode

- [ ] 5.1 Add `SMALLAIOS_BUS_BACKEND` env var parsing (zenoh/dds/none) in `container/src/main.rs`
- [ ] 5.2 Add `enable_dataflow_runner()` function: starts runner in background thread sharing the ModelManager Arc
- [ ] 5.3 Wire signal handler to stop the runner on SIGTERM alongside HTTP shutdown
- [ ] 5.4 Update Dockerfile/docker-compose.yml with `SMALLAIOS_BUS_BACKEND` documented

## 6. End-to-End Testing

- [ ] 6.1 Integration test: in-process Zenoh transport, runner subscribes, client publishes Relu input, verify Relu output published to result topic
- [ ] 6.2 Integration test: backpressure — flood input topic, verify dropped counter increments and runner doesn't crash
- [ ] 6.3 Integration test: wildcard subscription — multiple models on the same runner
- [ ] 6.4 Integration test: DDS round-trip via adapter
- [ ] 6.5 Integration test: container binary with `SMALLAIOS_BUS_BACKEND=zenoh`, verify both HTTP and bus paths work simultaneously

## 7. Validation and Documentation

- [ ] 7.1 Run `just fmt`, `just clippy`, `just test` — all green
- [ ] 7.2 Update `docs/scheduling-model.md` with dataflow runner as a System-class task (subscribes/publishes are I/O, run on the inference scheduler class)
- [ ] 7.3 Update CLAUDE.md with `SMALLAIOS_BUS_BACKEND` env var documentation
- [ ] 7.4 Add a `docs/inference-bus.md` showing example client code (Zenoh + DDS) for invoking inference
