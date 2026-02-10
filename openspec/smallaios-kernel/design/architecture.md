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

## Real-Time Scheduling Model

### Classification: Soft Real-Time Unikernel

SmallAIOS is **not** a traditional hard-RTOS. ONNX inference is inherently variable-time:
model sizes range from 1 MB (MobileNet, ~2ms) to multi-GB (LLM, seconds or minutes).
Individual operator execution times are data-dependent (sparse vs dense tensors), and
GPU offload introduces non-deterministic latency (PCIe transfers, kernel launch overhead).

Instead, SmallAIOS is a **soft real-time unikernel** with:
- **Hard-RT class** for system-critical tasks (health checks, diagnostics, watchdog)
- **Soft-RT class** for inference with per-operator time budgets and observability
- **Priority preemption** of inference by system tasks at operator boundaries

| Property | Hard RTOS | SmallAIOS | General OS |
|---|---|---|---|
| Deadline guarantees | Hard | Soft / best-effort | None |
| Scheduling | Preemptive priority | Cooperative + priority preemption | Preemptive fair |
| Memory model | Static allocation | Pool + bounded dynamic | Fully dynamic |
| Task count | Fixed at compile time | Fixed at boot time | Unlimited |
| WCET analysis | Required | Per-operator budgets | Not done |

### Scheduling Classes

Tasks are assigned to one of three scheduling classes, listed in descending priority:

```
┌─────────────────────────────────────────────────────────────┐
│  Class 0: SYSTEM (Hard-RT)                                   │
│  - Watchdog servicing                                        │
│  - Health check / readiness probe responses                  │
│  - Syslog flush                                              │
│  - Timer interrupt bottom halves                             │
│  Guarantee: Always preempts lower classes at yield points    │
│  Budget: Must complete within 1 ms                           │
├─────────────────────────────────────────────────────────────┤
│  Class 1: IPC (Soft-RT, low latency)                         │
│  - IPC message routing                                       │
│  - Model reload requests                                     │
│  - TCP connection management                                 │
│  - Network packet processing                                 │
│  Guarantee: Preempts inference at next operator boundary     │
│  Budget: Target < 10 ms response                             │
├─────────────────────────────────────────────────────────────┤
│  Class 2: INFERENCE (Soft-RT, throughput)                     │
│  - ONNX model execution                                     │
│  - GPU command submission / completion                       │
│  - Tensor allocation / deallocation                          │
│  Guarantee: Best-effort, time-budgeted per operator          │
│  Budget: Configurable per-model (default: no hard limit)     │
└─────────────────────────────────────────────────────────────┘
```

### Operator-Level Yield Points

The ONNX runtime inserts mandatory yield points between every operator in the execution
graph. At each yield point, the scheduler checks:

1. **Pending SYSTEM tasks?** → Execute immediately (preempt inference)
2. **Pending IPC tasks?** → Execute before resuming inference
3. **Operator time budget exceeded?** → Log warning via syslog, continue
4. **Watchdog deadline approaching?** → Service watchdog, then resume
5. **No higher-priority work?** → Continue with next operator

```
Inference execution timeline:

  ┌───────┐  yield  ┌──────┐  yield  ┌──────┐  yield  ┌────────┐
  │ Conv  │───────→ │ BN   │───────→ │ Relu │───────→ │ MatMul │
  └───────┘    │    └──────┘    │    └──────┘    │    └────────┘
               │                │                │
               ▼                ▼                ▼
          [Check for       [IPC msg         [Watchdog
           SYSTEM tasks]    arrives →        service →
                            handle it]       resume]
```

### Per-Operator Time Budgets

Each operator execution is timed. If an operator exceeds its configurable budget:

- **Warning**: Log to syslog with operator name, actual time, budget
- **Soft limit exceeded (2x budget)**: Emit metric, flag for profiling
- **Hard limit exceeded (10x budget or configurable)**: Abort inference, return timeout error

Default budgets (configurable per-model):

| Operator Class | Default Budget | Rationale |
|---|---|---|
| Elementwise (Relu, Add, etc.) | 1 ms | Should be SIMD-fast |
| Reduction (ReduceMean, Softmax) | 10 ms | Data-dependent |
| GEMM (MatMul, Conv) | 100 ms | Dominant cost, size-dependent |
| Attention (MultiHeadAttention) | 500 ms | Quadratic in sequence length |
| GPU kernel | 1000 ms | Includes DMA + compute |

These are soft budgets for observability, not hard deadlines.

### Hardware Watchdog

SmallAIOS supports a hardware watchdog timer (HPET/ACPI on x86-64, SP805/SBSA on ARM64,
or virtualized watchdog via virtio). The watchdog:

- Is initialized during boot with a configurable timeout (default: 30 seconds)
- Must be serviced (pet/kicked) by the SYSTEM scheduling class
- Triggers a system reset if not serviced (indicates hang/deadlock)
- Watchdog servicing is the highest-priority task in the system

```
Boot → [Watchdog init: 30s timeout]
         │
         ├─── Every operator yield point:
         │      if (time_since_last_pet > timeout/2):
         │          pet_watchdog()
         │
         └─── SYSTEM class task runs every 5s:
                pet_watchdog()
                emit_health_metrics()
```

### WCET Analysis for Edge Targets

For constrained hardware (Jetson Nano, Raspberry Pi), operators can be profiled
during model load to establish per-operator worst-case execution time estimates:

1. **Calibration run**: Execute each operator once with representative input
2. **WCET estimate**: Measured time × safety factor (default: 3x)
3. **Budget assignment**: Use WCET estimates as operator budgets
4. **Runtime monitoring**: Track actual vs estimated, adjust factors

This enables predictable inference latency on edge devices while remaining
best-effort on high-performance hardware (DGX Spark, Xeon).

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
