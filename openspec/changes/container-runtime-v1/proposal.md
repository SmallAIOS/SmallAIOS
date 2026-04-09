## Why

SmallAIOS can build and run as a Docker container but does nothing useful — `main.rs` prints a version string and parks the thread. The container crate has substantial `no_std` infrastructure (boot sequence with 8 phases, health checker, metrics exporter, config system, shutdown handler) but none of it is wired into the binary. After `onnx-cpu-runtime-v1` enables CPU inference and `compute-abstraction-v1` adds GPU dispatch, there needs to be an HTTP server to accept inference requests and return results. This change builds the container runtime: an HTTP inference server that loads ONNX models from volumes, serves predictions via REST API, and integrates with Kubernetes health/readiness probes.

## What Changes

- Implement a minimal HTTP/1.1 server in `main.rs` using `std::net::TcpListener` (container mode has `std`)
- Add REST API endpoints: `POST /v1/inference` (submit input tensors, get predictions), `GET /health` (liveness), `GET /ready` (readiness), `GET /metrics` (Prometheus)
- Implement ONNX model loading from filesystem paths (Docker volume mounts)
- Wire the existing boot sequence, health checker, metrics, config, and shutdown modules into the server lifecycle
- Add SIGTERM/SIGINT signal handling for graceful shutdown
- Support configuration via environment variables (model path, port, GPU backend selection)

## Capabilities

### New Capabilities
- `container-inference-server`: HTTP REST API for ONNX inference, model loading, and Kubernetes probe endpoints
- `container-model-loading`: Load ONNX models from filesystem paths at startup and on-demand

### Modified Capabilities
- `onnx-runtime`: Add requirement for file-based model loading (read `.onnx` files from disk)

## Impact

- **Code:** `container/src/main.rs` (HTTP server), new `container/src/server.rs` (request routing), new `container/src/model_manager.rs` (model lifecycle)
- **APIs:** New REST endpoints — this is the user-facing interface for container deployments
- **Docker:** Update `Dockerfile` and `docker-compose.yml` with model volume mount, port exposure, environment variables
- **Dependencies:** No new external crates — HTTP server uses `std::net`, JSON serialization via minimal hand-written serializer (or `serde_json` behind feature flag if size allows)
