# Spec 15: Watchdog and Real-Time Scheduling

## Overview

SmallAIOS is a **soft real-time unikernel**. ONNX inference execution time varies
enormously based on model size (1 MB MobileNet ≈ 2ms vs multi-GB transformer ≈ seconds).
This spec defines the watchdog timer, scheduling class hierarchy, and per-operator
time budget system that together provide predictable system responsiveness despite
variable inference workloads.

## Classification: Soft Real-Time

SmallAIOS is **not** a hard RTOS. It does not guarantee worst-case execution time
for inference. Instead, it provides:

- Hard-RT guarantees for system-critical tasks (watchdog, health checks)
- Soft-RT targets for IPC responsiveness
- Best-effort with time budgets and observability for inference
- Priority preemption at ONNX operator boundaries

## Hardware Watchdog Timer

### Supported Hardware

| Platform | Watchdog Hardware | Interface |
|---|---|---|
| x86-64 (QEMU) | i6300esb PCI watchdog | PCI MMIO |
| x86-64 (bare metal) | ACPI WDAT / TCO watchdog | MMIO / I/O port |
| ARM64 (QEMU virt) | SP805 / SBSA Generic Watchdog | MMIO |
| ARM64 (Jetson/RPi) | Platform-specific | MMIO |
| Virtualized | virtio-watchdog | virtqueue |

### Watchdog Behavior

- Initialized during boot with configurable timeout (default: 30 seconds)
- The SYSTEM scheduling class services (pets) the watchdog at regular intervals
- Failure to service triggers a system reset (indicates hang or deadlock)
- Watchdog timeout must be > 2× the longest expected operator execution time

### Watchdog API

```rust
/// Initialize the hardware watchdog with the given timeout.
pub fn watchdog_init(timeout_secs: u32) -> Result<(), WatchdogError>;

/// Service (pet/kick) the watchdog timer, resetting the countdown.
pub fn watchdog_pet() -> Result<(), WatchdogError>;

/// Disable the watchdog (only during controlled shutdown).
pub fn watchdog_disable() -> Result<(), WatchdogError>;

/// Get remaining time before watchdog fires.
pub fn watchdog_remaining() -> Result<u32, WatchdogError>;
```

## Scheduling Classes

### Class 0: SYSTEM (Hard Real-Time)

Tasks in this class have the highest priority and always preempt lower classes
at the next yield point.

**Tasks:**
- `WatchdogTask`: Service hardware watchdog (runs every `timeout/4`)
- `HealthTask`: Respond to Kubernetes health/readiness probes
- `SyslogTask`: Flush syslog buffer to output
- `TimerBottomHalf`: Process timer interrupt work items

**Guarantee:** Must complete within 1 ms. If a SYSTEM task exceeds 1 ms, a
kernel warning is logged (potential design issue, not a runtime error).

### Class 1: IPC (Soft Real-Time, Low Latency)

Tasks that handle external communication. Preempt INFERENCE at the next
operator boundary.

**Tasks:**
- `IpcRouterTask`: Route pub/sub messages to subscribers
- `TcpListenerTask`: Accept new TCP connections
- `TcpConnectionTask`: Handle per-connection I/O
- `ModelReloadTask`: Hot-reload ONNX model from new data

**Target:** < 10 ms response time for IPC messages.

### Class 2: INFERENCE (Soft Real-Time, Throughput)

The primary workload. Runs when no SYSTEM or IPC tasks are pending.

**Tasks:**
- `InferenceTask`: Execute ONNX model operator graph
- `GpuSubmitTask`: Submit GPU command buffers
- `GpuCompletionTask`: Handle GPU completion callbacks
- `TensorAllocTask`: Bulk tensor pre-allocation

**Guarantee:** Best-effort. Per-operator time budgets for observability.

## Per-Operator Time Budgets

### Default Budgets

| Operator Category | Default Soft Budget | Default Hard Limit | Examples |
|---|---|---|---|
| Elementwise | 1 ms | 10 ms | Relu, Add, Mul, Sigmoid |
| Reduction | 10 ms | 100 ms | Softmax, ReduceMean, LayerNorm |
| GEMM | 100 ms | 1000 ms | MatMul, Conv, Gemm |
| Attention | 500 ms | 5000 ms | MultiHeadAttention |
| GPU kernel | 1000 ms | 10000 ms | Any GPU-dispatched operator |
| I/O | 50 ms | 500 ms | Model load chunk, DMA transfer |

### Budget Behavior

1. **Within budget**: No action, continue normally
2. **Soft budget exceeded (1x)**: Log warning to syslog with operator name and timing
3. **Soft limit exceeded (2x)**: Emit metric counter, flag session for profiling
4. **Hard limit exceeded**: Abort inference, return `OnnxError::OperatorTimeout`

### Configurable Scaling

Budgets can be scaled per-session via `SessionOptions::operator_budget_scale`:
- `0.5`: Strict budgets (latency-sensitive edge deployment)
- `1.0`: Default
- `0.0`: Disable budget enforcement (infinite budgets, logging only)
- `10.0`: Generous budgets (large model, relaxed latency)

### Whole-Inference Timeout

`SessionOptions::inference_timeout_ms` sets an absolute wall-clock timeout for
the entire `onnx_run` call. Default: 0 (no limit). When set, the scheduler
checks elapsed time at each yield point and aborts if exceeded.

## WCET Calibration (Edge Mode)

For constrained hardware (Jetson Nano, Raspberry Pi 4/5, Snapdragon 845):

### Calibration Process

1. During `create_session` with `calibrate_wcet: true`:
   a. Allocate representative input tensors (random data, correct shapes)
   b. Execute each operator individually, measuring wall-clock time
   c. Compute WCET estimate: `measured_time × wcet_safety_factor`
   d. Store estimates in the session's operator budget table
2. At runtime, use calibrated budgets instead of category defaults
3. Track actual vs estimated; log if safety factor is consistently too low

### Safety Factor Guidance

| Hardware Class | Recommended Factor | Rationale |
|---|---|---|
| RPi 4 (Cortex-A72) | 4.0 | Thermal throttling, shared memory bus |
| Jetson Nano (Maxwell) | 3.0 | GPU clock varies with power mode |
| Jetson Orin (Ampere) | 2.0 | More predictable performance |
| x86 Desktop (i7/Ryzen) | 2.0 | Turbo boost variability |
| Server (Xeon/EPYC/DGX) | 1.5 | Stable clocks, dedicated cooling |

## Syslog Diagnostics

The real-time scheduling system emits structured syslog messages:

```
[SmallAIOS] SCHED: operator=Conv_0 elapsed=15ms budget=10ms class=INFERENCE status=OVER_BUDGET
[SmallAIOS] SCHED: watchdog pet, remaining=28s
[SmallAIOS] SCHED: class=IPC task=TcpConnection preempted INFERENCE at operator=MatMul_3
[SmallAIOS] SCHED: inference session=0x1234 total=245ms operators=47 timeouts=0 over_budget=2
[SmallAIOS] SCHED: WCET calibration complete, operators=47 total_calibration=89ms safety_factor=3.0
```

## Configuration

```toml
[scheduler]
worker_threads = 0              # 0 = auto-detect from CPU count
watchdog_timeout_secs = 30      # Hardware watchdog timeout
system_task_budget_ms = 1       # Max time for SYSTEM class tasks
ipc_response_target_ms = 10     # Target IPC response time

[scheduler.budgets]
elementwise_ms = 1
reduction_ms = 10
gemm_ms = 100
attention_ms = 500
gpu_kernel_ms = 1000

[onnx.session]
operator_budget_scale = 1.0     # Budget multiplier
inference_timeout_ms = 0        # 0 = no whole-inference timeout
calibrate_wcet = false          # Enable WCET calibration
wcet_safety_factor = 3.0        # WCET multiplier
```

## Rust Crate Impact

The watchdog and scheduling class system lives in the kernel crate:

```
kernel/src/sched/
├── mod.rs          # Scheduler with priority classes
├── task.rs         # Task struct with scheduling class field
├── executor.rs     # Async executor with priority dequeue
├── queue.rs        # Lock-free work-stealing queue (priority-aware)
├── watchdog.rs     # Hardware watchdog abstraction
└── budget.rs       # Per-operator time budget tracking
```

New syscalls:
- `sys_watchdog_pet()` (0x58): Service watchdog timer
- `sys_watchdog_remaining()` (0x59): Query remaining watchdog time
- `task_set_class(id, class)` (0x17): Set task scheduling class

## Testing Strategy

- Unit tests: Verify priority ordering, budget enforcement, timeout behavior
- Integration tests: Run inference in QEMU with artificial delays to trigger budgets
- Formal verification: TLA+ model extended with scheduling classes and preemption
- Stress testing: Concurrent SYSTEM + IPC + INFERENCE tasks, verify no starvation
- Watchdog testing: Intentionally block to verify reset trigger
