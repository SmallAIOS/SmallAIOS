## 1. CanFrameSink Trait

- [ ] 1.1 Add `CanFrameSink` trait to `bus/src/can/controller.rs`: `fn on_frame(&mut self, frame: &CanFrame)`
- [ ] 1.2 Update `CanController` trait to support attaching one or more sinks
- [ ] 1.3 Wire `loopback.rs` to call sinks on each transmitted frame
- [ ] 1.4 Wire `mcp2515.rs` to call sinks from its receive ISR/poll path
- [ ] 1.5 Wire `axi_can.rs` similarly
- [ ] 1.6 Unit tests: dummy sink counts received frames across all controllers

## 2. CanInferenceAdapter Core

- [ ] 2.1 Create `bus/src/can/inference_adapter.rs` with `CanInferenceAdapter` struct
- [ ] 2.2 Define `RouteSpec`, `OutputRouteSpec`, `BatchTrigger`, `FrameDecoder`, `FrameEncoder` types
- [ ] 2.3 Implement `routing_table_from_toml()` parser for `/etc/smallaios/can-routes.toml`
- [ ] 2.4 Implement `on_frame()`: lookup CAN ID, decode payload, write to batch buffer at offset
- [ ] 2.5 Implement batch trigger evaluation: FrameCount, TimeWindow, OnFrame
- [ ] 2.6 Implement `flush_batch()`: returns `(topic, payload)` for the runner
- [ ] 2.7 Implement `on_inference_output()`: convert tensor back to CAN frames via output routing
- [ ] 2.8 Add `dropped_frames_total`, `partial_batches_total`, `unrouted_frames_total` atomic counters
- [ ] 2.9 Unit tests for each batch trigger type
- [ ] 2.10 Unit test for stale frame detection

## 3. CANaerospace Decoder Integration

- [ ] 3.1 Implement `FrameDecoder::CanAerospaceFloat32`: extract f32 from CANaerospace data field per ARINC 825
- [ ] 3.2 Implement `FrameDecoder::CanAerospaceInt32`: extract i32
- [ ] 3.3 Implement encoder counterparts for output routing
- [ ] 3.4 Reject frames with mismatched data_type
- [ ] 3.5 Unit tests with real CANaerospace frame bytes

## 4. Dataflow Runner Integration

- [ ] 4.1 Add `bus` crate optional `onnx` feature pulling `smallaios-onnx-rt` (mirrors ipc crate pattern)
- [ ] 4.2 Connect `CanInferenceAdapter` to `ipc::dataflow_runner::DataflowRunner` via the existing `serve_dataflow_runner()` pattern (or new helper)
- [ ] 4.3 End-to-end test using loopback CAN: publish 6 sensor frames → batch fills → runner runs Relu/MatMul → output frames produced

## 5. Container CAN Backend

- [ ] 5.1 Add `SMALLAIOS_CAN_DEVICE` env var parsing in `container/src/main.rs`
- [ ] 5.2 Add `SMALLAIOS_CAN_ROUTING` env var for routing table file path
- [ ] 5.3 Implement `start_can_dataflow_runner()`: parse device spec, instantiate controller, load routing, attach adapter
- [ ] 5.4 Wire `SMALLAIOS_BUS_BACKEND=can` case in `enable_dataflow_runner()`
- [ ] 5.5 Update Dockerfile, docker-compose.yml with new env vars documented
- [ ] 5.6 Update CLAUDE.md env var table

## 6. Configuration File Format

- [ ] 6.1 Define TOML routing table schema (input routes, output routes, decoders, batch triggers)
- [ ] 6.2 Add example file `examples/can-routes.toml` showing automotive ADAS sensor mapping
- [ ] 6.3 Document the schema in `docs/inference-bus.md` (extend existing file)

## 7. Documentation

- [ ] 7.1 Create `docs/can-inference.md`: architecture diagram, use cases (ADAS, robotics, drones, avionics), CANaerospace integration notes
- [ ] 7.2 Add CAN inference example to README
- [ ] 7.3 Document hardware requirements (MCP2515 SPI wiring, AXI CAN MMIO setup)

## 8. End-to-End Testing

- [ ] 8.1 Loopback CAN test: 6 sensor frames → adapter → runner → output frames, verify values match expected inference output
- [ ] 8.2 Stale frame test: send 5 fresh + 1 stale frame, verify batch dropped
- [ ] 8.3 Backpressure test: flood frames faster than inference can process, verify drop counter and no crash
- [ ] 8.4 CANaerospace round-trip: NOD frames in → typed tensor → inference → output as NOD frames
- [ ] 8.5 Verify `just test`, `just clippy`, `just fmt-check` all pass
