# Spec 07: Container Interface

## Overview

SmallAIOS is designed to run **inside** containers, not as a container runtime.
It is packaged as an OCI container image and deployed via Docker or Kubernetes
like any other containerized application — but internally it is a unikernel that
bypasses the host Linux kernel's process abstraction.

## Deployment Modes

### Mode 1: Standard Container (Primary)

SmallAIOS runs as a **normal Linux process** inside a container. The host Linux
kernel provides memory, scheduling, and device access. SmallAIOS's "kernel"
functions as a user-space runtime library.

```
┌─────────────────────────────┐
│  SmallAIOS (user process)   │
│  ┌───────────────────────┐  │
│  │ ONNX Runtime          │  │
│  │ IPC Layer             │  │
│  │ POSIX Compat Layer    │  │
│  │ SmallAIOS "Kernel"    │  │  ← Library OS mode
│  └───────────────────────┘  │
├─────────────────────────────┤
│  Container Runtime (runc)   │
├─────────────────────────────┤
│  Host Linux Kernel          │
└─────────────────────────────┘
```

Benefits:
- No special container runtime needed
- Standard Docker/Kubernetes deployment
- GPU passthrough via NVIDIA Container Toolkit
- Easiest to adopt

### Mode 2: MicroVM (Advanced)

SmallAIOS runs as a **guest kernel** in a lightweight VM (Firecracker, Cloud
Hypervisor, QEMU microvm). The container image is converted to a VM image.

```
┌─────────────────────────────┐
│  SmallAIOS (guest kernel)   │
│  ┌───────────────────────┐  │
│  │ ONNX Runtime          │  │
│  │ IPC Layer             │  │
│  │ SmallAIOS Kernel      │  │  ← Real kernel mode
│  └───────────────────────┘  │
├─────────────────────────────┤
│  VMM (Firecracker/QEMU)    │
├─────────────────────────────┤
│  Host Linux Kernel + KVM   │
└─────────────────────────────┘
```

Benefits:
- True kernel-level isolation
- Smaller attack surface (no host kernel syscall surface)
- Faster boot (no Linux init overhead)

### Mode 3: Bare Metal (Edge)

SmallAIOS boots directly on hardware via UEFI. For edge inference appliances.

```
┌─────────────────────────────┐
│  SmallAIOS (bare metal)     │
│  ┌───────────────────────┐  │
│  │ ONNX Runtime          │  │
│  │ IPC Layer             │  │
│  │ SmallAIOS Kernel      │  │
│  └───────────────────────┘  │
├─────────────────────────────┤
│  Hardware (x86/ARM/GPU)     │
└─────────────────────────────┘
```

## OCI Container Image

### Image Structure

```dockerfile
FROM scratch
COPY smallaios /smallaios          # Single static binary
COPY models/ /models/              # ONNX model files
COPY smallaios.toml /config/       # Runtime configuration
EXPOSE 7447                        # IPC port
ENTRYPOINT ["/smallaios"]
```

The image is built `FROM scratch` — no base OS, no libc, no shell. The entire
container image is:
1. The SmallAIOS binary (statically linked, ~5-15 MB depending on features)
2. ONNX model files
3. A TOML configuration file

### Multi-Architecture Images

Built and published as OCI multi-arch manifests:

```
smallaios/runtime:latest
├── linux/amd64    (x86-64 binary)
├── linux/arm64    (ARM64 binary)
└── linux/amd64    (x86-64 + NVIDIA GPU binary)
```

### Image Size Targets

| Configuration | Target Size |
|---|---|
| CPU only (x86-64) | < 10 MB |
| CPU only (ARM64) | < 10 MB |
| CPU + GPU (x86-64) | < 20 MB |
| + small model (MobileNet) | + 14 MB |
| + medium model (ResNet50) | + 98 MB |
| + large model (BERT-base) | + 440 MB |

## Kubernetes Integration

### Pod Specification

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference-server
  labels:
    app: smallaios
spec:
  containers:
  - name: inference
    image: smallaios/runtime:latest
    ports:
    - containerPort: 7447
      name: ipc
    resources:
      requests:
        memory: "512Mi"
        cpu: "2"
      limits:
        memory: "2Gi"
        cpu: "4"
        nvidia.com/gpu: "1"     # Optional GPU
    readinessProbe:
      tcpSocket:
        port: 7447
      initialDelaySeconds: 1
      periodSeconds: 5
    livenessProbe:
      httpGet:
        path: /health           # Mapped from smallaios/v1/health
        port: 7447
      initialDelaySeconds: 5
      periodSeconds: 10
    securityContext:
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
```

### Health and Readiness

SmallAIOS exposes health information via two mechanisms:

1. **IPC query**: `smallaios/v1/health` — native, for Zenoh clients
2. **HTTP endpoint**: `GET /health` on the IPC port — for Kubernetes probes

The HTTP endpoint is a minimal handler (not a full HTTP server) that responds
to `GET /health` and `GET /metrics` only. All other HTTP requests return 404.

### Metrics (Prometheus)

Published at `GET /metrics` and via IPC at `smallaios/v1/metrics`:

```
# HELP smallaios_inference_requests_total Total inference requests
# TYPE smallaios_inference_requests_total counter
smallaios_inference_requests_total{model="resnet50"} 12345

# HELP smallaios_inference_latency_seconds Inference latency
# TYPE smallaios_inference_latency_seconds histogram
smallaios_inference_latency_seconds_bucket{model="resnet50",le="0.001"} 100
smallaios_inference_latency_seconds_bucket{model="resnet50",le="0.01"} 11000
smallaios_inference_latency_seconds_bucket{model="resnet50",le="0.1"} 12340
smallaios_inference_latency_seconds_bucket{model="resnet50",le="+Inf"} 12345

# HELP smallaios_memory_used_bytes Current memory usage
# TYPE smallaios_memory_used_bytes gauge
smallaios_memory_used_bytes{region="heap"} 52428800
smallaios_memory_used_bytes{region="tensor_pool"} 419430400
smallaios_memory_used_bytes{region="gpu"} 1073741824

# HELP smallaios_uptime_seconds Time since boot
# TYPE smallaios_uptime_seconds gauge
smallaios_uptime_seconds 3600.5
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: smallaios_inference_latency_seconds
      target:
        type: AverageValue
        averageValue: "50m"   # 50ms p50 latency target
```

## GPU Passthrough in Containers

### NVIDIA Container Toolkit

In standard container mode, GPU access requires:
1. NVIDIA drivers installed on the host
2. NVIDIA Container Toolkit (`nvidia-ctk`) configured
3. Pod requests `nvidia.com/gpu` resource

SmallAIOS in library OS mode uses the host's GPU driver through:
- `/dev/nvidia*` device files (passed through by container toolkit)
- NVIDIA UVM (Unified Virtual Memory) for managed memory
- CUDA driver API loaded via `dlopen` at startup

### Direct GPU Access (MicroVM mode)

In MicroVM mode, the GPU is passed through via VFIO:
- GPU assigned to VM via VFIO-PCI
- SmallAIOS uses its own minimal GPU driver (Spec 05)
- No host driver dependency

## Container Build Pipeline

```
┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│  Rust source  │───→│ Cross-compile│───→│  Strip +     │
│  (Cargo)      │    │  (per arch)  │    │  Optimize    │
└───────────────┘    └──────────────┘    └──────┬───────┘
                                                │
┌───────────────┐    ┌──────────────┐    ┌──────▼───────┐
│  ONNX models  │───→│  Validate +  │───→│  Build OCI   │
│  (.onnx)      │    │  Optimize    │    │  Image       │
└───────────────┘    └──────────────┘    └──────┬───────┘
                                                │
┌───────────────┐                        ┌──────▼───────┐
│  Config       │───────────────────────→│  Push to     │
│  (.toml)      │                        │  Registry    │
└───────────────┘                        └──────────────┘
```

### Dockerfile (multi-stage)

```dockerfile
# Stage 1: Build
FROM rust:nightly AS builder
WORKDIR /build
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl
RUN strip target/x86_64-unknown-linux-musl/release/smallaios

# Stage 2: Runtime
FROM scratch
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/smallaios /smallaios
COPY models/ /models/
COPY smallaios.toml /config/smallaios.toml
EXPOSE 7447
ENTRYPOINT ["/smallaios"]
```

## Crate Structure

```
container/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── config.rs        # Container configuration loading
    ├── health.rs        # Health check endpoint
    ├── metrics.rs       # Prometheus metrics exporter
    └── http.rs          # Minimal HTTP handler (health + metrics only)
```
