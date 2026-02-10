# Phase 5: IPC, Container Integration, and Production Readiness

## Objective

Complete the IPC messaging system, container/Kubernetes integration, and
production-hardening to make SmallAIOS deployable for real inference workloads.

## Dependencies

- Phase 3 complete (ONNX runtime, CPU inference)
- Phase 4 recommended but not required (GPU is optional)

## Tasks

### 5.1 IPC Core
- [ ] Key expression parser and matcher (wildcards: `*`, `**`)
- [ ] Message router: match incoming messages to subscriptions/queryables
- [ ] Pub/sub: publisher, subscriber, message delivery
- [ ] Request/reply: queryable registration, query dispatch, reply routing
- [ ] Lock-free SPSC ring buffers for internal channels
- [ ] Async integration: IPC events wake sleeping tasks

### 5.2 TCP Transport
- [ ] Minimal TCP stack (or use host TCP in container mode via POSIX sockets)
- [ ] TCP listener on configurable port
- [ ] Connection management (accept, close, timeout)
- [ ] Framing: length-prefixed messages (Zenoh wire protocol)
- [ ] Backpressure: flow control via TCP window

### 5.3 TLS Transport (PQC)
- [ ] TLS 1.3 handshake with ML-KEM-768 hybrid key exchange
- [ ] Certificate loading and verification
- [ ] Mutual TLS authentication
- [ ] AES-256-GCM record encryption
- [ ] Integration with TCP transport

### 5.4 Inference Protocol
- [ ] Binary request format: header + tensor metadata + tensor data
- [ ] Binary response format: same structure for outputs
- [ ] Error response format
- [ ] Zero-copy path: TCP recv directly into tensor buffer
- [ ] Batching: accumulate requests for batch inference

### 5.5 Built-in Endpoints
- [ ] `smallaios/v1/health`: health check queryable
- [ ] `smallaios/v1/models`: list loaded models
- [ ] `smallaios/v1/models/{name}/infer`: inference endpoint per model
- [ ] `smallaios/v1/models/{name}/metadata`: model metadata
- [ ] `smallaios/v1/metrics`: Prometheus metrics publisher
- [ ] `smallaios/v1/logs`: log stream publisher
- [ ] `smallaios/v1/control/shutdown`: graceful shutdown

### 5.6 HTTP Compatibility Layer
- [ ] Minimal HTTP/1.1 parser (GET only, for K8s probes)
- [ ] `GET /health` → maps to `smallaios/v1/health`
- [ ] `GET /metrics` → maps to `smallaios/v1/metrics`
- [ ] All other paths → 404
- [ ] No HTTP POST (use native IPC for inference)

### 5.7 POSIX Compatibility Layer
- [ ] File descriptor table implementation
- [ ] Virtual filesystem (read-only: /models/, /config/, /dev/, /proc/self/)
- [ ] mmap implementation (MAP_ANONYMOUS, MAP_PRIVATE for model files)
- [ ] pthread subset (create, join, mutex, condvar, rwlock)
- [ ] epoll implementation (for async I/O)
- [ ] Socket API (TCP client/server)
- [ ] clock_gettime, nanosleep
- [ ] Minimal signal handling (SIGTERM for shutdown)
- [ ] getrandom (backed by PQC-grade CSPRNG)

### 5.8 Container Image
- [ ] Multi-stage Dockerfile (build + scratch runtime)
- [ ] Multi-architecture build (x86-64, ARM64)
- [ ] OCI multi-arch manifest
- [ ] Image size optimization (strip, LTO, opt-level=z)
- [ ] Example docker-compose.yml (CPU only)
- [ ] Example docker-compose.yml (with GPU)

### 5.9 Kubernetes Manifests
- [ ] Deployment manifest
- [ ] Service manifest (ClusterIP for internal, LoadBalancer for external)
- [ ] Readiness and liveness probes
- [ ] Resource requests and limits
- [ ] HPA configuration (scale on inference latency)
- [ ] GPU resource request (`nvidia.com/gpu`)
- [ ] Network policy (restrict ingress to IPC port only)
- [ ] Pod security standards (restricted profile)

### 5.10 Security Hardening
- [ ] Capability system fully integrated (all syscalls check capabilities)
- [ ] Model signature verification (ML-DSA-65)
- [ ] TLS mutual authentication
- [ ] Rate limiting on IPC connections
- [ ] Resource limits enforced (memory, connections, requests)
- [ ] CPU hardening features enabled (NX, SMEP/SMAP, IBRS, CET, PAC, BTI, MTE)
- [ ] Security audit of all `unsafe` blocks
- [ ] Fuzzing: IPC protocol, ONNX parser, syscall interface

### 5.11 Observability
- [ ] Prometheus metrics: inference count, latency histograms, error rates
- [ ] Prometheus metrics: memory usage, GPU utilization, CPU utilization
- [ ] Structured logging with request correlation IDs
- [ ] Boot time measurement and reporting
- [ ] Model load time and optimization time metrics

### 5.12 Documentation
- [ ] Deployment guide (Docker, Kubernetes)
- [ ] Configuration reference
- [ ] IPC API reference (endpoints, wire format)
- [ ] Security guide (TLS setup, model signing, capability model)
- [ ] Performance tuning guide
- [ ] Contributing guide

## Exit Criteria

- External Zenoh client can connect and run inference
- Health check and metrics endpoints work with Kubernetes probes
- TLS with ML-KEM-768 hybrid key exchange functional
- Model signature verification enforced
- Container image < 15 MB (CPU only, no model)
- Kubernetes deployment runs stable for 24h under load
- All security features enabled, `unsafe` blocks audited
- Fuzzer runs for 24h with no crashes
- Prometheus metrics collected by monitoring stack
- Documentation complete for deployment and API usage
