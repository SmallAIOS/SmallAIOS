# Spec 04: IPC and Messaging System

## Overview

SmallAIOS uses a **Zenoh-inspired pub/sub messaging system** as its primary
inter-component communication mechanism. Instead of traditional UNIX IPC (pipes,
signals, shared memory segments), all communication flows through a unified
key-expression-based message router.

This serves as the interface between the outside world and the ONNX runtime:
external clients publish inference requests and subscribe to results.

## Why Zenoh-Inspired (Not Zenoh Itself)

Zenoh is an excellent protocol, but embedding the full Zenoh implementation would:
- Pull in a large Rust dependency tree (tokio, async runtime, crypto, etc.)
- Include features irrelevant to a unikernel (routing peers, scouting, plugins)
- Violate the minimal-dependency principle

Instead, SmallAIOS implements the **core Zenoh abstractions** (key expressions,
pub/sub, queryables) with a minimal implementation optimized for the unikernel's
single-address-space architecture.

The wire protocol is **Zenoh-compatible** so external Zenoh clients can communicate
with SmallAIOS directly.

## Key Abstractions

### Key Expressions

Resources are addressed by hierarchical key expressions with wildcard support:

```
smallaios/models/resnet50/infer        # Specific inference endpoint
smallaios/models/*/infer               # All model inference endpoints
smallaios/models/**                    # All model-related resources
smallaios/health                       # Health check
smallaios/metrics/**                   # All metrics
```

Wildcard semantics:
- `*` matches a single path segment
- `**` matches zero or more path segments

### Communication Patterns

**Pub/Sub** — Fire-and-forget data distribution:
```
Publisher → [key] → Subscriber(s)
```

**Request/Reply (Queryable)** — Synchronous request-response:
```
Requester → [key + query] → Queryable → [reply] → Requester
```

**Pull** — Subscriber-initiated data retrieval:
```
Subscriber → [pull request] → Publisher → [data] → Subscriber
```

## Architecture

```
┌──────────────────────────────────────────────┐
│            External Network Interface         │
│         (TCP/TLS listener on port 7447)       │
├──────────────────────────────────────────────┤
│              Message Router                    │
│    (key-expression matching + dispatch)        │
├──────┬──────────┬──────────┬─────────────────┤
│ Pub/ │ Request/ │ Shared   │  Built-in       │
│ Sub  │ Reply    │ Memory   │  Endpoints      │
│      │          │ Transport│  (health, etc)  │
├──────┴──────────┴──────────┴─────────────────┤
│           Serialization Layer                  │
│    (zero-copy, raw bytes, optional codec)     │
└──────────────────────────────────────────────┘
```

## Built-in Endpoints

SmallAIOS exposes these key expressions by default:

### Inference

```
smallaios/v1/models/{model_name}/infer    [Queryable]
  Request:  Serialized input tensors
  Reply:    Serialized output tensors

smallaios/v1/models/{model_name}/metadata [Queryable]
  Request:  (empty)
  Reply:    Model metadata (inputs, outputs, opset)

smallaios/v1/models                       [Queryable]
  Request:  (empty)
  Reply:    List of loaded models
```

### System

```
smallaios/v1/health                       [Queryable]
  Reply:    { "status": "ok", "uptime_ns": 12345 }

smallaios/v1/metrics                      [Publisher]
  Publishes: Prometheus-format metrics periodically

smallaios/v1/metrics/inference            [Publisher]
  Publishes: Per-inference latency, throughput metrics

smallaios/v1/config                       [Queryable]
  Reply:    Current runtime configuration

smallaios/v1/logs                         [Publisher]
  Publishes: Kernel log messages (ring buffer)
```

## Inference Protocol

### Request Format

Inference requests use a simple binary protocol:

```
┌────────────┬──────────┬────────────────────────────────┐
│ Header     │ Metadata │ Tensor Data                    │
│ (16 bytes) │ (var)    │ (var)                          │
└────────────┴──────────┴────────────────────────────────┘

Header:
  [0:4]   Magic: 0x534D4149 ("SMAI")
  [4:6]   Version: 0x0001
  [6:8]   Num inputs: u16
  [8:12]  Metadata length: u32
  [12:16] Total tensor data length: u32

Metadata (per input, repeated num_inputs times):
  Name length: u16
  Name: [u8; name_length]
  Data type: u8 (ONNX TensorProto.DataType enum)
  Num dimensions: u8
  Dimensions: [i64; num_dimensions]
  Data offset: u32 (offset into tensor data section)
  Data length: u32

Tensor Data:
  Raw tensor bytes, contiguous, in row-major order
```

### Response Format

Same format as request, with output tensor names and data.

### Error Response

```
Header with num_inputs = 0, metadata contains:
  Error code: u32
  Error message: UTF-8 string
```

## Transport Mechanisms

### Intra-kernel (shared memory)

For communication between internal components (ONNX runtime ↔ scheduler):
- Zero-copy: Publisher writes to shared buffer, subscriber reads directly
- Reference-counted buffers prevent premature deallocation
- Lock-free SPSC (single producer, single consumer) ring buffers

### External TCP

For communication with external clients:
- TCP listener on configurable port (default 7447)
- Optional TLS with mutual authentication
- Zenoh wire protocol compatibility
- Backpressure via TCP flow control

### External Shared Memory (container mode)

For high-performance container-to-container communication:
- Memory-mapped files in shared volumes
- POSIX shared memory semantics
- Zero-copy tensor transfer between containers

## Wire Protocol

Compatible with Zenoh's wire protocol for interoperability:

```
Frame:
  [0:1]  Frame type (DATA, DECLARE, QUERY, REPLY, PULL, PING, PONG)
  [1:5]  Frame length (u32, big-endian)
  [5..]  Frame payload (type-specific)

DATA frame payload:
  Key expression (length-prefixed string)
  Encoding (u16)
  Payload (remaining bytes)
```

## Serialization

SmallAIOS supports multiple serialization formats for tensor data:

| Format | ID | Description |
|---|---|---|
| Raw | 0x00 | Raw bytes, no encoding (default, zero-copy) |
| NumPy | 0x01 | NumPy `.npy` format (for Python interop) |
| Arrow | 0x02 | Apache Arrow IPC format (for data pipelines) |

The default is raw bytes for minimum overhead. Clients can request specific
encoding via the request metadata.

## Concurrency Model

- The message router runs as a dedicated async task on a pinned core.
- Subscriptions are lock-free: a subscriber registers a callback that the router
  invokes inline (for internal) or via async channel (for external).
- Multiple inference requests can be in-flight concurrently, limited by a
  configurable concurrency semaphore.

## Crate Structure

```
ipc/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── router.rs           # Key-expression matching and dispatch
    ├── key_expr.rs         # Key expression parsing and matching
    ├── pubsub.rs           # Publisher/Subscriber types
    ├── queryable.rs        # Request/Reply pattern
    ├── transport/
    │   ├── mod.rs
    │   ├── shm.rs          # Shared memory transport
    │   ├── tcp.rs          # TCP transport
    │   └── tls.rs          # TLS wrapper
    ├── protocol.rs         # Wire protocol codec
    ├── inference.rs        # Inference request/response types
    └── builtins.rs         # Built-in endpoints (health, metrics)
```

## Configuration

```toml
[ipc]
# External listener
listen = "tcp://0.0.0.0:7447"
# tls_cert = "/config/cert.pem"
# tls_key = "/config/key.pem"

# Concurrency
max_concurrent_requests = 64
request_timeout_ms = 30000

# Shared memory (container mode)
shm_enabled = false
shm_size = "256M"

# Metrics publishing interval
metrics_interval_ms = 5000
```
