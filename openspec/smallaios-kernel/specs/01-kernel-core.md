# Spec 01: Kernel Core

## Overview

The SmallAIOS kernel core provides the minimal set of services required to execute
AI inference workloads: memory management, task scheduling, interrupt handling, and
the syscall interface. It is a single-address-space unikernel — all components run
in the same privilege level with capability-based isolation rather than hardware
memory protection between processes.

## Architecture Decision: Unikernel

SmallAIOS uses a **unikernel architecture** rather than a traditional monolithic or
microkernel design. Rationale:

- **Single purpose**: Only one application (the ONNX runtime) ever runs.
- **No context switch overhead**: Single address space eliminates TLB flushes and
  ring transitions for syscalls.
- **Smaller binary**: No need for process isolation infrastructure.
- **Container fit**: Unikernels map naturally to the one-process-per-container model.

The trade-off is that multiple separate ONNX models run as cooperative tasks within
the same address space, not as isolated processes. This is acceptable because all
models are loaded by the same trusted operator.

## Memory Management

### Physical Memory Allocator

- **Buddy allocator** for page-granularity allocations (4 KiB pages).
- **Slab allocator** layered on top for sub-page kernel object allocations.
- Support for 2 MiB and 1 GiB huge pages for tensor buffer allocations.
- Architecture-specific page table management delegated to HAL.

### Virtual Memory Layout (x86-64 example)

```
0x0000_0000_0000_0000 - 0x0000_7FFF_FFFF_FFFF  User/application space (unused in unikernel)
0xFFFF_8000_0000_0000 - 0xFFFF_8FFF_FFFF_FFFF  Direct physical memory map
0xFFFF_9000_0000_0000 - 0xFFFF_9FFF_FFFF_FFFF  Kernel heap
0xFFFF_A000_0000_0000 - 0xFFFF_AFFF_FFFF_FFFF  MMIO region (device memory)
0xFFFF_B000_0000_0000 - 0xFFFF_BFFF_FFFF_FFFF  GPU memory mappings
0xFFFF_C000_0000_0000 - 0xFFFF_CFFF_FFFF_FFFF  Tensor buffer pool
0xFFFF_D000_0000_0000 - 0xFFFF_DFFF_FFFF_FFFF  IPC shared memory
0xFFFF_E000_0000_0000 - 0xFFFF_EFFF_FFFF_FFFF  Stack space
0xFFFF_F000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF  Kernel code + read-only data
```

### Tensor Memory Pool

Dedicated region for tensor allocations with:
- Pre-allocated aligned buffers for common tensor sizes
- NUMA-aware allocation on multi-socket systems
- DMA-capable allocations for GPU data transfers
- Reference-counted buffers for zero-copy sharing between operators

## Task Scheduler

### Model: Soft Real-Time Cooperative Scheduling

SmallAIOS is a **soft real-time unikernel**. It uses cooperative multitasking with
async/await (Rust futures) combined with **priority-based preemption at operator
boundaries**. ONNX inference is inherently variable-time (model sizes range from
1 MB edge models to multi-GB transformers), so SmallAIOS provides soft deadlines
with observability rather than hard real-time guarantees.

Tasks yield at ONNX operator boundaries. At each yield point, the scheduler checks
for higher-priority pending work (system health, IPC) and preempts inference if needed.

### Scheduling Classes

| Class | Name | Priority | Preempts | Budget |
|---|---|---|---|---|
| 0 | `SYSTEM` | Highest | All others | < 1 ms |
| 1 | `IPC` | Medium | INFERENCE | < 10 ms target |
| 2 | `INFERENCE` | Normal | — | Per-operator budgets |

**SYSTEM class** (hard-RT): Watchdog servicing, health check responses, syslog flush,
timer interrupt bottom halves. Must always complete within 1 ms.

**IPC class** (soft-RT, low latency): IPC message routing, model reload requests,
TCP connection management, network packet processing. Preempts inference at the
next operator boundary.

**INFERENCE class** (soft-RT, throughput): ONNX model execution, GPU command
submission/completion, tensor allocation. Best-effort with per-operator time budgets.

### Task Types

| Type | Scheduling Class | Description |
|---|---|---|
| `WatchdogTask` | SYSTEM | Service hardware watchdog timer |
| `HealthTask` | SYSTEM | Respond to health/readiness probes |
| `SyslogTask` | SYSTEM | Flush diagnostic log buffer |
| `TimerTask` | SYSTEM | Timeout and deadline management |
| `IpcRouterTask` | IPC | Route pub/sub messages |
| `TcpTask` | IPC | TCP connection handling |
| `ModelReloadTask` | IPC | Hot-reload ONNX model |
| `InferenceTask` | INFERENCE | ONNX model execution |
| `GPUTask` | INFERENCE | GPU command submission/completion |

### Operator Yield Points

The ONNX runtime inserts a mandatory scheduler yield between every operator.
At each yield point:

1. Check for pending SYSTEM tasks → execute immediately
2. Check for pending IPC tasks → execute before resuming inference
3. Check operator time budget → log warning if exceeded
4. Service watchdog if deadline approaching
5. No higher-priority work → continue with next operator

### Per-Operator Time Budgets

Each operator execution is timed against a configurable budget:

| Operator Class | Default Budget | Rationale |
|---|---|---|
| Elementwise (Relu, Add) | 1 ms | SIMD-fast |
| Reduction (Softmax, ReduceMean) | 10 ms | Data-dependent |
| GEMM (MatMul, Conv) | 100 ms | Size-dependent dominant cost |
| Attention (MultiHeadAttention) | 500 ms | Quadratic in sequence length |
| GPU kernel | 1000 ms | Includes DMA + compute |

Budget behavior:
- **Warning** (1x): Log operator name, actual time, budget to syslog
- **Soft limit** (2x): Emit metric, flag for profiling
- **Hard limit** (10x or configurable): Abort inference with timeout error

### Hardware Watchdog

SmallAIOS initializes a hardware watchdog timer during boot:
- Default timeout: 30 seconds (configurable)
- Serviced by SYSTEM class task at highest priority
- Triggers system reset if not serviced (indicates hang/deadlock)
- Watchdog service also occurs at operator yield points if deadline approaches

### WCET Calibration (Edge Targets)

For constrained hardware (Jetson Nano, Raspberry Pi), the runtime performs a
calibration run during model load:
1. Execute each operator once with representative input
2. Estimate WCET: measured time × safety factor (default: 3x)
3. Assign WCET as operator budget
4. Monitor actual vs estimated at runtime, adjust factors

### Executor

- Work-stealing executor with one worker per CPU core.
- Per-core run queues with lock-free stealing.
- Priority queues: SYSTEM and IPC tasks always dequeued before INFERENCE.
- GPU tasks pinned to cores with NVIDIA GPU affinity.
- Idle cores enter low-power state (HLT on x86, WFI on ARM64).

## Interrupt Handling

### Model

- Bottom-half / top-half split.
- Top half: Minimal interrupt acknowledgment, enqueue work item.
- Bottom half: Async task processes the interrupt data.

### Required Interrupts

| Interrupt | Source | Purpose |
|---|---|---|
| Timer | APIC/GIC | Scheduler ticks, timeouts |
| IPI | CPU | Cross-core task migration |
| PCIe MSI-X | GPU | GPU command completion |
| Virtio | Hypervisor | Container I/O (block, net) |

## Syscall Interface

SmallAIOS exposes a minimal syscall interface. In unikernel mode these are direct
function calls (no ring transition). In VM mode they use `syscall`/`svc` instructions.

### Syscall Categories

See [Spec 08: Security Model](08-security-model.md) for capability requirements.

**Memory** (~8 syscalls):
- `mem_alloc(size, align, flags) -> *mut u8`
- `mem_free(ptr, size)`
- `mem_map(phys, virt, size, flags)` — MMIO mapping
- `mem_protect(ptr, size, flags)` — change permissions
- `tensor_alloc(shape, dtype) -> TensorHandle`
- `tensor_free(handle)`
- `tensor_map_gpu(handle, device) -> GpuPtr`
- `tensor_unmap_gpu(handle, device)`

**Task** (~7 syscalls):
- `task_spawn(entry, arg) -> TaskId`
- `task_yield()`
- `task_exit(code)`
- `task_join(id) -> ExitCode`
- `task_set_priority(id, priority)`
- `task_set_class(id, class)` — set scheduling class (SYSTEM/IPC/INFERENCE)
- `task_current() -> TaskId`

**IPC** (~8 syscalls):
- `ipc_publish(key, data, len)`
- `ipc_subscribe(key_expr) -> SubHandle`
- `ipc_recv(handle, buf, len) -> usize`
- `ipc_query(key, data, len, reply_buf, reply_len) -> usize`
- `ipc_channel_create() -> (SendHandle, RecvHandle)`
- `ipc_channel_send(handle, data, len)`
- `ipc_channel_recv(handle, buf, len) -> usize`
- `ipc_channel_close(handle)`

**ONNX** (~6 syscalls):
- `onnx_load(data, len) -> ModelHandle`
- `onnx_unload(handle)`
- `onnx_create_session(model, opts) -> SessionHandle`
- `onnx_run(session, inputs, num_inputs, outputs, num_outputs)`
- `onnx_get_metadata(model, buf, len) -> usize`
- `onnx_list_providers() -> ProviderList`

**Device** (~5 syscalls):
- `dev_enumerate() -> DeviceList`
- `dev_open(id) -> DevHandle`
- `dev_close(handle)`
- `dev_ioctl(handle, cmd, arg) -> isize`
- `dev_dma_alloc(size, align) -> DmaBuffer`

**System** (~7 syscalls):
- `sys_info() -> SystemInfo`
- `sys_time() -> u64` (nanoseconds since boot)
- `sys_shutdown(code)`
- `sys_log(level, msg, len)`
- `sys_random(buf, len)` — CSPRNG
- `sys_watchdog_pet()` — service hardware watchdog
- `sys_watchdog_remaining() -> u32` — query remaining watchdog time

**Total: ~41 syscalls** (vs. Linux ~450)

## Boot Sequence

1. **Firmware/container runtime** hands control to kernel entry point
2. **Early init**: Set up stack, BSS, page tables, GDT/IDT (x86) or exception vectors (ARM64)
3. **HAL init**: Detect CPU features, initialize interrupt controller
4. **Memory init**: Build physical memory map, initialize allocators
5. **Watchdog init**: Initialize hardware watchdog timer (default: 30s timeout)
6. **Scheduler init**: Create idle task, initialize per-core priority queues, start SYSTEM tasks
7. **Device init**: Enumerate PCIe devices, initialize GPU if present
7. **ONNX init**: Initialize runtime, register execution providers
8. **IPC init**: Start pub/sub message router
9. **Model load**: Load ONNX model(s) from container image or virtio-blk
10. **Ready**: Begin accepting inference requests via IPC

## Kernel Configuration

Build-time configuration via Cargo features:

```toml
[features]
default = ["x86_64", "cpu-inference"]
x86_64 = []
aarch64 = []
nvidia_gpu = []
cpu-inference = []
gpu-inference = ["nvidia_gpu"]
verbose-boot = []
```

Runtime configuration via a minimal TOML config embedded in the container image:

```toml
[kernel]
log_level = "info"
heap_size = "256M"
tensor_pool_size = "1G"

[scheduler]
worker_threads = 0              # 0 = auto-detect from CPU count
watchdog_timeout_secs = 30      # Hardware watchdog timeout
system_task_budget_ms = 1       # Max time for SYSTEM class tasks
ipc_response_target_ms = 10     # Target IPC response time

[scheduler.budgets]
elementwise_ms = 1              # Relu, Add, Mul, Sigmoid
reduction_ms = 10               # Softmax, ReduceMean, LayerNorm
gemm_ms = 100                   # MatMul, Conv, Gemm
attention_ms = 500              # MultiHeadAttention
gpu_kernel_ms = 1000            # Any GPU-dispatched operator

[onnx]
models = ["model.onnx"]
execution_providers = ["cpu"]   # or ["cuda", "cpu"]
operator_budget_scale = 1.0     # Budget multiplier (0 = disable enforcement)
inference_timeout_ms = 0        # 0 = no whole-inference timeout
calibrate_wcet = false          # Enable WCET calibration for edge targets
wcet_safety_factor = 3.0        # WCET multiplier

[ipc]
listen = "tcp://0.0.0.0:7447"
```

## Rust Crate Structure

```
kernel/
├── Cargo.toml
└── src/
    ├── lib.rs          # Kernel entry, initialization
    ├── mem/
    │   ├── mod.rs
    │   ├── buddy.rs    # Buddy allocator
    │   ├── slab.rs     # Slab allocator
    │   ├── heap.rs     # Kernel heap (#[global_allocator])
    │   ├── tensor.rs   # Tensor buffer pool
    │   └── page.rs     # Page table abstractions
    ├── sched/
    │   ├── mod.rs
    │   ├── task.rs      # Task struct with scheduling class
    │   ├── executor.rs  # Async executor with priority dequeue
    │   ├── queue.rs     # Lock-free work-stealing queue (priority-aware)
    │   ├── watchdog.rs  # Hardware watchdog abstraction
    │   └── budget.rs    # Per-operator time budget tracking
    ├── interrupt/
    │   ├── mod.rs
    │   └── handler.rs
    ├── syscall/
    │   ├── mod.rs
    │   └── table.rs     # Syscall dispatch table
    └── config.rs        # Runtime configuration parser
```

## Dependencies

The kernel core has **zero external crate dependencies** at runtime.
Build-time only:

- `rustc` nightly (for `#![no_std]`, `#![no_main]`, inline assembly)
- `llvm` (via rustc, for code generation)
- `cc` (build script, for any assembly files)

## Testing Strategy

- Unit tests run in hosted mode (`#[cfg(test)]`) with mock HAL
- Integration tests run in QEMU with the real kernel binary
- Fuzzing of syscall interface with `cargo-fuzz`
- Memory safety verified with Miri where applicable
