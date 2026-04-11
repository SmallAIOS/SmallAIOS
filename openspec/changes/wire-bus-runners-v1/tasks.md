## 1. Session Loading

- [ ] 1.1 Add `container/Cargo.toml` dependency: `smallaios-ipc = { path = "../ipc", features = ["onnx"] }`
- [ ] 1.2 Add `load_sessions(manager: &ModelManager) -> BTreeMap<String, Session>` helper in `main.rs`
- [ ] 1.3 For each loaded `ModelInfo`, read the file bytes, call `session::load_model()`, call `Session::initialize()`, insert into the map
- [ ] 1.4 Log successes and failures individually
- [ ] 1.5 Unit test: load_sessions with a tempdir containing a valid minimal ONNX model

## 2. Zenoh Runner Thread

- [ ] 2.1 Read `ipc::pubsub` to understand the in-process pub/sub factory API
- [ ] 2.2 Implement `start_zenoh_dataflow_runner(manager, shutdown) -> Option<JoinHandle<()>>`
- [ ] 2.3 Build `DataflowRunner` from loaded sessions via `register_session`
- [ ] 2.4 Spawn a thread that drains a subscriber, calls `serve_dataflow_runner`, publishes results, sleeps 10ms between iterations
- [ ] 2.5 Thread exits when `shutdown` is set
- [ ] 2.6 Store the join handle in a `Vec<JoinHandle>` owned by main, join before exit

## 3. DDS Runner Thread

- [ ] 3.1 Implement `start_dds_dataflow_runner()` — same shape as Zenoh but uses `DdsZenohAdapter` to bridge topic names
- [ ] 3.2 For the initial wire-up, the DDS runner can share the same in-process pub/sub as Zenoh (the adapter just translates topic names)
- [ ] 3.3 Log a clear message indicating DDS real wire protocol is not active

## 4. CAN Runner Thread

- [ ] 4.1 Implement `start_can_dataflow_runner(manager, shutdown, device_spec, routing_file)`
- [ ] 4.2 Parse routing TOML if specified (use a simple hand-rolled parser or leverage existing toml parsing if available)
- [ ] 4.3 Match `CanDeviceSpec::Loopback` → instantiate `MockCanController::new()`
- [ ] 4.4 For Mcp2515 and AxiCan: log a warning and fall back to MockCanController for now
- [ ] 4.5 Spawn a thread that polls `controller.receive()`, processes frames via `CanInferenceAdapter::process_frame()`, runs inference, transmits response frames
- [ ] 4.6 Use a tick counter for timestamps (or `std::time::Instant` since we're in container mode)

## 5. Wire enable_dataflow_runner

- [ ] 5.1 Update `enable_dataflow_runner()` to take the shutdown Arc and return `Vec<JoinHandle<()>>`
- [ ] 5.2 Replace the zenoh, dds, can TODO placeholders with calls to the new start_* functions
- [ ] 5.3 Store the returned handles in main and join them on exit
- [ ] 5.4 Remove the "placeholder — enable once ipc ships the onnx feature" messages

## 6. Activate E2E Tests

- [ ] 6.1 Remove `#[ignore]` from the 4 tests in `container/tests/e2e_bus.rs`
- [ ] 6.2 Remove `#[ignore]` from the 3 tests in `container/tests/e2e_can.rs`
- [ ] 6.3 Ensure tests start the container with a valid test model directory
- [ ] 6.4 Add at least one test that submits an inference request via pub/sub and verifies output

## 7. Validation

- [ ] 7.1 `just fmt` clean
- [ ] 7.2 `just clippy` clean
- [ ] 7.3 `just test` all passing (all 3,200+ tests + 7 newly activated e2e tests)
- [ ] 7.4 Manual smoke test: `SMALLAIOS_BUS_BACKEND=zenoh` starts without crashing and the container logs show the runner is active
