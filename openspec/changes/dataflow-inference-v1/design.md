## Context

The IPC crate has `endpoints/inference.rs` with a stub `handle_inference_request()` that decodes the binary protocol and returns a not-implemented response. The ONNX runtime (`onnx-rt`) has a working `Session::run()` after PR #61. The DDS crate has `DdsZenohAdapter` that bridges DDS topics to Zenoh key expressions. The container binary has an HTTP server running the inference handler.

The gap: nothing connects these. The IPC inference endpoint doesn't call the ONNX runtime. There's no pub/sub-based runner. The container binary only knows about HTTP.

The architectural constraint: keep `ipc` at Layer 1 (Core Services) and `onnx-rt` also at Layer 1. They're peers. The wiring happens at Layer 3 (the container binary) or via a new feature flag on `ipc` that opt-in pulls `onnx-rt`.

## Goals / Non-Goals

**Goals:**
- Pub/sub inference pipeline: subscribe to input topic, run model, publish output
- Reuse existing IPC binary inference protocol (no new wire format)
- Zenoh-style key expressions for topic naming: `smallaios/inference/<model>/{input,output}`
- DDS topic compatibility via `DdsZenohAdapter`
- End-to-end test using loopback transport (no real network)
- Zero new external crates

**Non-Goals:**
- New wire protocol — reuse existing `inference_proto.rs` binary format
- Streaming inference (continuous output from a single input) — future work
- Multi-model routing on one topic — each model gets its own topic
- gRPC/Protobuf service — not Rust-friendly enough, would add tonic dependency

## Decisions

### D1: Dataflow Runner in `ipc` Crate Behind Feature Flag

Add an optional `onnx` feature to `ipc/Cargo.toml`:
```toml
[features]
onnx = ["dep:smallaios-onnx-rt"]
```

The `dataflow_runner` module is `#[cfg(feature = "onnx")]`. This keeps `ipc` usable without ONNX (e.g., for non-inference IPC) while allowing the container to opt in.

**Why feature flag over Layer 3 wiring:** The runner logic (subscribe → run → publish) is generic and reusable — any consumer of the IPC crate that wants inference dataflow can enable it. Putting it in container/main.rs would couple the binary to the runner implementation.

### D2: Topic Naming Convention

```
smallaios/inference/<model_name>/input    ← client publishes input tensors here
smallaios/inference/<model_name>/output   ← runner publishes result tensors here
smallaios/inference/<model_name>/error    ← runner publishes errors here
smallaios/inference/_meta/models          ← runner publishes available models list
```

Wildcards work naturally: `smallaios/inference/*/input` subscribes to all models.

### D3: Wire Format — Reuse Existing Binary Protocol

`ipc/src/inference_proto.rs` already defines:
```rust
pub const INFERENCE_MAGIC: u32 = 0x4F4E4E58; // "ONNX"
pub enum InferenceRequestType { LoadModel, UnloadModel, RunInference, GetModelInfo }
```

Each pub/sub message is one of these binary requests. The runner decodes, dispatches to ONNX, encodes the response, publishes. No JSON, no protobuf for the wire — just the existing binary format.

### D4: Container Binary `--bus` Mode

Add to `container/src/main.rs`:
```rust
let bus_backend = env::var("SMALLAIOS_BUS_BACKEND").unwrap_or_else(|_| "none".to_string());
match bus_backend.as_str() {
    "zenoh" => start_zenoh_runner(&model_manager),
    "dds"   => start_dds_runner(&model_manager),
    "none"  => {} // HTTP only
    _ => warn!("unknown bus backend: {}", bus_backend),
}
```

The runner runs alongside the HTTP server in a separate thread. Same model manager, same lifecycle.

### D5: Loopback Transport for Tests

The IPC crate already has shared-memory and TCP transports. For end-to-end tests, use the existing loopback transport (`bus/src/dds/loopback.rs` for DDS, in-process channels for Zenoh-style IPC). No real network — fast, deterministic tests.

## Risks / Trade-offs

**[Risk] Cyclic dependency** — If `ipc` depends on `onnx-rt`, and `onnx-rt` ever needs to publish telemetry via IPC, we have a cycle. Mitigation: feature flag is one-way (`ipc` optionally pulls `onnx-rt`, never vice versa). Document in CLAUDE.md.

**[Risk] DDS QoS complexity** — DDS has 22+ QoS policies. Mitigation: start with sensible defaults (Reliable, KeepLast, Volatile), expose only critical ones in the runner config.

**[Trade-off] Binary protocol vs. JSON** — Binary is faster and smaller, but harder to debug than JSON. Mitigation: HTTP server still uses JSON for human-friendly debugging. Pub/sub uses binary for performance.

## Open Questions

- **Q1:** Should the runner support multiple models concurrently, or one model per runner instance? *Leaning toward: one runner can handle multiple models, dispatch by topic name.*
- **Q2:** Backpressure — what happens if inference can't keep up with input topic rate? Drop oldest? Block publisher? *Leaning toward: drop oldest with metric counter, configurable.*
