## Why

PRs #62 (dataflow-inference-v1) and #63 (can-inference-bridge-v1) built all the infrastructure for pub/sub-based inference: the `DataflowRunner`, the `serve_dataflow_runner()` helper, the CAN adapter, the container `SMALLAIOS_BUS_BACKEND` env var parsing. But the container's `enable_dataflow_runner()` function still has TODO placeholders — it logs "starting runner" but doesn't actually start anything. The dataflow runner is never instantiated. This change connects the dots: when a user sets `SMALLAIOS_BUS_BACKEND=zenoh/dds/can`, the container actually spawns a runner thread, loads models into it, and processes pub/sub inference requests.

## What Changes

- Replace the 3 TODO placeholders in `container/src/main.rs` with real runner initialization
- `start_zenoh_dataflow_runner(manager)` — instantiate `DataflowRunner` with loaded models, spawn a background thread that drains the in-process pub/sub subscriber and publishes results
- `start_dds_dataflow_runner(manager)` — same pattern, using the `DdsZenohAdapter` bridge
- `start_can_dataflow_runner(manager, device, routing)` — instantiate the CAN controller (loopback/mcp2515/axi), parse the routing TOML, create the `CanInferenceAdapter`, feed it frames, publish results back as CAN frames
- Add shutdown coordination: runners must stop when the HTTP server's `AtomicBool` shutdown flag is set
- Activate the 4 `#[ignore]` e2e tests in `container/tests/e2e_bus.rs` and the 3 in `container/tests/e2e_can.rs`
- End-to-end test: container binary with `SMALLAIOS_BUS_BACKEND=zenoh` + in-process Zenoh client → inference response

## Capabilities

### Modified Capabilities
- `container-inference-server`: runners actually start when bus backend is configured
- `dataflow-inference`: runner lifecycle managed by the container binary
- `can-inference-bridge`: CAN controller actually instantiated from config

## Impact

- **Code:** `container/src/main.rs` (remove TODOs, add 3 runner start functions ~150 lines)
- **Tests:** 7 currently-ignored e2e tests become active
- **Behavior:** `SMALLAIOS_BUS_BACKEND=zenoh` now does what the docs say
- **Dependencies:** Enable `ipc/onnx` feature from container crate
- **Threading:** Runners run in background threads sharing the `Arc<ModelManager>` with the HTTP server
