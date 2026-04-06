## Context

The container crate has two layers: a `no_std` library (`lib.rs`) with boot, health, metrics, config, and shutdown modules, and a `std` binary (`main.rs`) that currently does nothing. The library was designed to be used from both kernel mode and container mode — the binary is the container-specific glue.

After `onnx-cpu-runtime-v1`, `Session::run()` works. After `compute-abstraction-v1`, GPU backends are available. This change wraps them in an HTTP server that Docker/Kubernetes can interact with.

Existing infrastructure already handles:
- Boot phases: ConfigLoaded → MemoryReady → SchedulerReady → SecurityReady → NetworkReady → IpcReady → RuntimeReady → ModelsLoaded → Ready
- Health status: Healthy / Degraded / Unhealthy with per-component registration
- Readiness: Ready / NotReady with dependency tracking
- Metrics: Counter, Gauge, Histogram with Prometheus text format export
- Config: TOML + environment variable overrides
- Shutdown: SIGTERM handler with component drain ordering

## Goals / Non-Goals

**Goals:**
- Minimal HTTP/1.1 server for inference requests and Kubernetes probes
- Model loading from Docker volume-mounted paths
- Integration with existing boot/health/metrics/shutdown infrastructure
- Configuration via environment variables matching Kubernetes patterns
- Graceful shutdown: stop accepting connections, drain in-flight requests, exit

**Non-Goals:**
- HTTP/2 or gRPC (future — QUIC/HTTP3 stack exists in `net/` but is overkill for MVP)
- Model hot-reload without restart (future optimization)
- Authentication/authorization on inference endpoints (defer to service mesh / API gateway)
- Batched inference (single-request-at-a-time initially)
- TLS termination (handled by ingress controller / sidecar in K8s)
- WebSocket streaming for long-running inference

## Decisions

### D1: Hand-Written HTTP/1.1 Server — No External Crates

Use `std::net::TcpListener` with a simple request parser. The server needs to handle only 4 endpoints with JSON payloads. A minimal HTTP parser (read headers, parse Content-Length, read body) is ~200 lines and avoids pulling in `hyper`, `actix`, or `tokio`.

**Why not `hyper`/`axum`:** Size constraint (<15 MB container). These frameworks pull in `tokio`, `mio`, `pin-project`, etc. The inference server is single-threaded (one request at a time in the cooperative model), so async isn't needed.

**Why not the `net/` crate's HTTP3 stack:** The `net/` crate implements QUIC/HTTP3 for the kernel's network stack. In container mode, we use the host kernel's TCP via `std::net`. Different layers.

### D2: JSON Serialization — Minimal Hand-Written

Implement a tiny JSON serializer/deserializer for the inference API types. The request format is simple:

```json
// Request: POST /v1/inference
{
  "model": "model-name",
  "inputs": {
    "input_0": { "shape": [1, 3, 224, 224], "data": [0.1, 0.2, ...], "dtype": "float32" }
  }
}

// Response
{
  "outputs": {
    "output_0": { "shape": [1, 1000], "data": [0.001, 0.003, ...], "dtype": "float32" }
  },
  "timing_ms": 42.5
}
```

**Why not `serde_json`:** Binary size. `serde` + `serde_json` add ~300 KB to the binary. The inference API has a fixed schema — a hand-rolled parser for this specific format is smaller and sufficient.

**Alternative considered:** Feature-flag `serde_json` for development convenience, hand-rolled for release. Adds build complexity. Start with hand-rolled, revisit if maintenance burden is high.

### D3: Model Manager — Load at Boot, Serve by Name

A `ModelManager` holds a `BTreeMap<String, Session>` of loaded models. At boot:
1. Read `SMALLAIOS_MODEL_DIR` env var (default: `/models`)
2. Scan directory for `.onnx` files
3. Load each into a `Session` via `Session::load_model()`
4. Register model name (filename without extension) in the map

Inference requests specify `"model": "name"` to select which session to use.

### D4: Thread Model — Single-Threaded Event Loop

The server runs a blocking accept loop on a single thread:
1. Accept connection
2. Read HTTP request
3. Route to handler
4. Execute inference (blocking — cooperative within the ONNX executor)
5. Write HTTP response
6. Close connection (no keep-alive initially)

This matches the unikernel's cooperative scheduling model. For throughput, scale horizontally (more containers), not vertically (more threads).

### D5: Wire Existing Boot Sequence

The existing `boot.rs` phase machine drives startup:
1. `ConfigLoaded` — parse config from env vars
2. `MemoryReady` — no-op in container mode (host manages memory)
3. `SchedulerReady` — no-op (no kernel scheduler in container mode)
4. `SecurityReady` — optional formal gate check
5. `NetworkReady` — bind TCP listener
6. `IpcReady` — no-op
7. `RuntimeReady` — initialize ONNX session(s) and GPU backend
8. `ModelsLoaded` — load models from `MODEL_DIR`
9. `Ready` — start accepting connections, set readiness probe to true

## Risks / Trade-offs

**[Risk] Single-threaded blocking model limits throughput** — One inference at a time. Mitigation: This matches the unikernel design (cooperative, single-core). For production throughput, use K8s horizontal pod autoscaling. Async can be added later.

**[Risk] Hand-written HTTP parser may have edge cases** — Malformed requests, chunked encoding, etc. Mitigation: Only support `Content-Length` (no chunked), reject anything that doesn't parse cleanly. The attack surface is minimized by running behind an ingress controller.

**[Risk] Large model files may slow container startup** — Loading a 500 MB model at boot blocks readiness. Mitigation: The readiness probe returns `NotReady` until `ModelsLoaded` phase completes. K8s won't route traffic until ready. Log progress during model loading.

**[Trade-off] No keep-alive connections** — Each request opens/closes a TCP connection. Simpler implementation at the cost of connection overhead. Acceptable for inference workloads (compute dominates latency, not connection setup).

## Open Questions

- **Q1:** Should the inference endpoint accept raw bytes (protobuf-encoded tensors) in addition to JSON? JSON is convenient but inefficient for large tensors. A binary format could use Content-Type negotiation.
- **Q2:** Should the `/metrics` endpoint use Prometheus text format (existing `metrics.rs` supports this) or OpenMetrics?
