## Context

Current state in `container/src/main.rs::enable_dataflow_runner()`:

```rust
"zenoh" => {
    println!("Bus: Zenoh dataflow runner requested (placeholder — ...)");
    // TODO(dataflow-inference-v1 §5.2): start_zenoh_dataflow_runner(_manager);
}
```

The runner primitives exist and are tested:
- `ipc::dataflow_runner::DataflowRunner` (behind `ipc/onnx` feature) — holds `BTreeMap<String, Session>`, processes messages
- `ipc::dataflow_runner::serve_dataflow_runner(runner, subscriber)` — drains a subscriber, calls runner, returns (topic, payload) pairs to publish
- `ipc::pubsub::Publisher` / `Subscriber` — in-process pub/sub primitives
- `bus::dds::DdsZenohAdapter` — DDS↔Zenoh bridge
- `bus::can::CanInferenceAdapter` + `CanFrameSink` — CAN frame batching
- `bus::can::controller::{MockCanController, CanController}` — CAN controller abstraction

The gap: nothing in `container/src/main.rs` instantiates these. The `ModelManager` tracks `ModelInfo` (name, path, loaded flag) but the container doesn't actually call `Session::load_model()` and `Session::initialize()` to build Sessions — that's only done in tests.

## Goals / Non-Goals

**Goals:**
- When `SMALLAIOS_BUS_BACKEND=zenoh`, spawn a runner thread that processes inference requests from an in-process Zenoh-style pub/sub
- When `SMALLAIOS_BUS_BACKEND=dds`, same behavior with the DDS adapter bridging
- When `SMALLAIOS_BUS_BACKEND=can`, instantiate a CAN controller + adapter, feed frames through the runner, emit response frames
- Load real ONNX models from `ModelManager` into the runner's Sessions at startup
- Clean shutdown: runners stop when SIGTERM fires
- All 7 currently-ignored e2e tests pass

**Non-Goals:**
- Real hardware CAN bring-up (MCP2515/AXI require actual hardware — stick with loopback in CI)
- Remote Zenoh clients (keep pub/sub in-process for now; external networking is a future change)
- Multi-threaded inference within a single runner (one request at a time per runner)
- QUIC transport for pub/sub (exists in `net/quic` but wiring is a separate change)

## Decisions

### D1: Load Sessions at Startup, Share Runner Across Threads

Modify `container/src/main.rs` to actually parse models from disk and build Sessions:

```rust
fn load_sessions(manager: &ModelManager) -> BTreeMap<String, Session> {
    let mut sessions = BTreeMap::new();
    for info in manager.list_models() {
        if !info.loaded { continue; }
        let bytes = match std::fs::read(&info.file_path) {
            Ok(b) => b,
            Err(e) => { eprintln!("  failed to read {}: {}", info.file_path, e); continue; }
        };
        let model = match smallaios_onnx_rt::session::load_model(&bytes) {
            Ok(m) => m,
            Err(e) => { eprintln!("  failed to parse {}: {}", info.name, e); continue; }
        };
        let mut session = Session::new(SessionConfig::default());
        if let Err(e) = session.initialize(&model) {
            eprintln!("  failed to initialize {}: {}", info.name, e);
            continue;
        }
        sessions.insert(info.name.clone(), session);
        println!("  Session ready: {}", info.name);
    }
    sessions
}
```

The runner is built once from these sessions and passed to whichever backend is active.

### D2: Runner Runs in a Background Thread Polling an Input Queue

In container mode, we have `std::thread`. Each runner spawns a thread:

```rust
fn start_zenoh_dataflow_runner(
    manager: Arc<ModelManager>,
    shutdown: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    let sessions = load_sessions(&manager);
    if sessions.is_empty() {
        eprintln!("WARNING: no models loaded, Zenoh runner disabled");
        return None;
    }
    let mut runner = DataflowRunner::new(Default::default());
    for (name, session) in sessions {
        runner.register_session(name, session);
    }

    let handle = std::thread::spawn(move || {
        let (mut publisher, mut subscriber) = ipc::pubsub::channel();
        // Subscribe to the input wildcard
        let _ = subscriber.subscribe("smallaios/inference/*/input");
        
        while !shutdown.load(Ordering::Relaxed) {
            // Drain subscriber, run inference, publish results
            let results = ipc::dataflow_runner::serve_dataflow_runner(&runner, &mut subscriber);
            for (topic, payload) in results {
                let _ = publisher.publish(&topic, payload);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    Some(handle)
}
```

**Caveat:** The exact pub/sub API in `ipc::pubsub` may not expose a way to bind a publisher and subscriber to the same in-process broker. I need to read the actual API and adapt. If there's no ready-made channel factory, the simplest approach is to use a shared `Arc<PubSubBroker>` or directly expose inference handlers at the HTTP layer and skip the pub/sub indirection for in-process testing.

### D3: CAN Runner Uses MockCanController for Default Loopback

For `SMALLAIOS_CAN_DEVICE=loopback`:

```rust
fn start_can_dataflow_runner(
    manager: Arc<ModelManager>,
    shutdown: Arc<AtomicBool>,
    device: &str,
    routing_file: Option<&str>,
) -> Option<JoinHandle<()>> {
    let sessions = load_sessions(&manager);
    if sessions.is_empty() { return None; }
    
    let routing = match routing_file {
        Some(path) => load_can_routing(path)?,
        None => {
            eprintln!("WARNING: no SMALLAIOS_CAN_ROUTING, using empty routing");
            AdapterConfig::default()
        }
    };
    
    let mut adapter = CanInferenceAdapter::new(routing);
    let mut runner = DataflowRunner::new(Default::default());
    for (name, session) in sessions { runner.register_session(name, session); }
    
    // For loopback: a MockCanController that also receives what we transmit
    let spec = parse_can_device(device).ok()?;
    let mut controller = match spec {
        CanDeviceSpec::Loopback => MockCanController::new(),
        CanDeviceSpec::Mcp2515(_) => {
            eprintln!("WARNING: MCP2515 not yet wired, using loopback");
            MockCanController::new()
        }
        CanDeviceSpec::AxiCan(_) => {
            eprintln!("WARNING: AXI CAN not yet wired, using loopback");
            MockCanController::new()
        }
    };
    
    let handle = std::thread::spawn(move || {
        let mut timestamp_us = 0u64;
        while !shutdown.load(Ordering::Relaxed) {
            // Poll for incoming CAN frames
            while let Some(frame) = controller.receive().ok().flatten() {
                if let Some((topic, payload)) = adapter.process_frame(&frame, timestamp_us) {
                    // Run inference via the runner
                    let model_name = adapter.extract_model_name(&topic).unwrap_or_default();
                    if let Ok(output_bytes) = runner.process_message(&model_name, &payload) {
                        // Convert result back to CAN frames and transmit
                        let output_topic = format!("smallaios/inference/{}/output", model_name);
                        let frames = adapter.on_inference_output(&output_topic, &output_bytes);
                        for frame in frames {
                            let _ = controller.transmit(&frame);
                        }
                    }
                }
                timestamp_us += 100; // coarse tick
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    Some(handle)
}
```

### D4: Shutdown Coordination

All runner threads check `shutdown: Arc<AtomicBool>` on each poll iteration. The existing SIGTERM handler sets this flag. Runners clean up when the flag is true. The main thread joins runner handles before exiting.

### D5: Keep DDS Minimal for Now

The DDS runner uses the existing `DdsZenohAdapter` but the actual DDS RTPS wire protocol doesn't need to be active in-process. For the initial wire-up, `SMALLAIOS_BUS_BACKEND=dds` can use the same in-process pub/sub as `zenoh` (via the adapter) — the distinction is mostly for demonstration that the adapter works. Real DDS networking is a future change.

## Risks / Trade-offs

**[Risk] Model loading may fail** — If protobuf parsing or session initialization fails, the runner has no models. Mitigation: Log errors clearly, fall back to HTTP-only mode if no sessions are loaded.

**[Risk] Shared mutable Session state across threads** — Sessions aren't Sync by default. The runner owns them in one thread, so no sharing. Good.

**[Risk] Polling loops waste CPU when idle** — Each runner polls every 10ms. Acceptable for a single-instance unikernel; would revisit for dense multi-tenant deployments.

**[Trade-off] In-process only for now** — No external Zenoh clients can actually connect. This change proves the pipeline works end-to-end inside the container; external networking comes next.

## Open Questions

- **Q1:** What's the exact `ipc::pubsub` API for creating a bound publisher/subscriber pair? May need to adapt the design based on actual primitives.
- **Q2:** Should `load_sessions` be part of `ModelManager` itself (e.g., `ModelManager::build_sessions()`) rather than a free function in main.rs? *Leaning toward: yes, but refactor in a follow-up — this change keeps it local to keep the diff small.*
