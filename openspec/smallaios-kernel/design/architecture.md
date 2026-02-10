# Architecture Design Document

## System Architecture

SmallAIOS is a **single-purpose unikernel** for AI inference. Every architectural
decision is driven by one question: "Does this serve ONNX model inference?"

### Architecture Type: Library OS / Unikernel

SmallAIOS uses the library OS model: the "kernel" is a Rust library linked directly
with the application (ONNX runtime + IPC). There is no user/kernel boundary, no
process isolation, and no ring transitions in the primary execution path.

```
Traditional OS:                    SmallAIOS:

┌─────────────┐                   ┌─────────────────────────────┐
│ Application │ ring 3            │  SmallAIOS                  │
├─────────────┤ ← syscall trap    │  ┌───────────────────────┐  │
│ Kernel      │ ring 0            │  │ ONNX Runtime          │  │
└─────────────┘                   │  │ IPC + Function Iface  │  │
                                  │  │ POSIX Compat          │  │
                                  │  │ Kernel Core           │  │
                                  │  │ HAL                   │  │
                                  │  └───────────────────────┘  │
                                  └─────────────────────────────┘
                                  Single binary, single address space
```

### Why Not Microkernel

A microkernel (separating ONNX runtime, IPC, etc. into isolated processes) would:
- Add IPC overhead between components (~1-5us per message passing)
- Require full process isolation infrastructure (page tables per process, context switching)
- Increase memory usage (each process needs its own stack, heap, page tables)
- Gain little security benefit: all components are equally trusted (they ship together)

### Why Not Monolithic

A monolithic kernel (like Linux) would:
- Include thousands of features irrelevant to inference
- Have a huge attack surface
- Be impossible to build clean-room in any reasonable timeframe

## Data Flow Architecture

### Inference Request Flow

```
External Client
    │
    ▼
[TCP/TLS Listener]
    │
    ▼
[IPC Message Router]  ← key expression matching
    │
    ▼
[Inference Dispatcher]  ← concurrency control, queueing
    │
    ▼
[ONNX Session]
    │
    ├─── [Graph Optimizer] (first run only, cached)
    │
    ▼
[Execution Planner]
    │
    ├─── CPU operators ──→ [CPU EP] ──→ SIMD kernels
    │
    └─── GPU operators ──→ [CUDA EP] ──→ GPU kernels
                              │
                              ├── DMA transfer (input)
                              ├── GPU kernel execution
                              └── DMA transfer (output)
    │
    ▼
[Result serialization]
    │
    ▼
[IPC Reply]
    │
    ▼
External Client
```

### Zero-Copy Data Path

The critical performance optimization is minimizing data copies:

```
1. Network receive: TCP stack writes directly to tensor buffer
   (scatter-gather I/O, no intermediate copy)

2. Tensor buffer is in the dedicated tensor pool region
   (pre-allocated, aligned for SIMD/DMA)

3. For GPU inference:
   a. Tensor buffer is in pinned memory (DMA-capable)
   b. GPU DMA engine copies directly from tensor buffer to GPU VRAM
   c. No CPU involvement in the transfer

4. Result tensor written directly to output buffer
   (which is the IPC reply buffer — zero-copy to network send)
```

Ideal case: **2 copies** (network → tensor buffer, tensor buffer → network).
With GPU: **4 copies** (network → pinned buffer → GPU, GPU → pinned buffer → network).

## Concurrency Architecture

### Async Runtime

SmallAIOS uses an **async/await** concurrency model built on Rust futures:

```
┌─────────────────────────────────────────────────────┐
│                    Executor                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Worker 0 │  │ Worker 1 │  │ Worker N │  ...      │
│  │ (CPU 0)  │  │ (CPU 1)  │  │ (CPU N)  │          │
│  ├──────────┤  ├──────────┤  ├──────────┤          │
│  │ Local Q  │  │ Local Q  │  │ Local Q  │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       └──────────────┴─────────────┘                │
│                  Work stealing                       │
└─────────────────────────────────────────────────────┘
```

- One worker thread per CPU core (pinned, no migration)
- Each worker has a local run queue (lock-free LIFO for cache locality)
- Workers steal from other workers' queues when idle (lock-free FIFO)
- GPU completion callbacks are posted to the worker with GPU affinity

### Why Not OS Threads

- Cooperative scheduling eliminates context switch overhead
- Rust futures are zero-allocation (state machine transform)
- Natural fit: inference is a pipeline of async steps
- Simpler to reason about than preemptive multitasking

## Module Dependency Graph

```
                 ┌──────────┐
                 │ container│ (entry point, config, health)
                 └────┬─────┘
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
         ┌────────┐ ┌────┐ ┌──────┐
         │onnx-rt │ │ipc │ │posix │
         └───┬────┘ └──┬─┘ └──┬───┘
             │         │      │
             └────┬────┘      │
                  ▼           │
             ┌────────┐      │
             │security│      │
             └───┬────┘      │
                 │           │
                 └─────┬─────┘
                       ▼
                  ┌────────┐
                  │ kernel │ (core: mem, sched, syscall)
                  └───┬────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     ┌────────┐  ┌────────┐  ┌────────┐
     │x86_64  │  │aarch64 │  │nvidia  │
     │  HAL   │  │  HAL   │  │  HAL   │
     └────────┘  └────────┘  └────────┘
```

## Error Handling Strategy

SmallAIOS uses Rust's `Result<T, E>` for all fallible operations. Panics are
reserved for unrecoverable states (out of memory, hardware failure).

### Error Categories

| Category | Response | Example |
|---|---|---|
| Input error | Return error to client | Invalid ONNX model, bad tensor shape |
| Resource exhaustion | Return error, log warning | Out of tensor pool memory |
| Hardware error | Log error, degrade | GPU timeout, PCIe error |
| Internal bug | Panic (kernel halt) | Assertion failure, invariant violation |

### Graceful Degradation

- GPU failure → fall back to CPU execution provider
- Network error → retry with backoff, then drop connection
- Memory pressure → reject new requests, complete in-flight ones

## Configuration Architecture

```
┌──────────────────────────────────────────────────┐
│                Build-time Config                  │
│  (Cargo features: target arch, GPU support)       │
├──────────────────────────────────────────────────┤
│                Image-time Config                  │
│  (smallaios.toml embedded in container image)     │
├──────────────────────────────────────────────────┤
│              Runtime Config (optional)             │
│  (Environment variables for overrides)            │
└──────────────────────────────────────────────────┘
```

Environment variable overrides:
```
SMALLAIOS_LOG_LEVEL=debug
SMALLAIOS_IPC_LISTEN=tcp://0.0.0.0:8080
SMALLAIOS_ONNX_THREADS=4
SMALLAIOS_TENSOR_POOL_SIZE=2G
```

## Performance Targets

| Metric | Target | Notes |
|---|---|---|
| Boot to ready | < 50 ms | Container mode |
| Inference overhead | < 1% vs native | Compared to onnxruntime on Linux |
| Memory overhead | < 8 MB | Kernel + runtime, excluding models |
| IPC latency | < 10 us | Shared memory transport |
| IPC latency | < 100 us | TCP loopback transport |
| Max throughput | Limited by hardware | Should saturate CPU/GPU |
