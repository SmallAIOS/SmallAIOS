# Scheduling and Execution Model

SmallAIOS uses a cooperative, priority-based scheduling model designed for deterministic inference execution and safety-critical certification (DO-178C DAL A). This document describes how the model maps to POSIX and RTOS standards, the multi-core strategy, and design rules for contributors.

## POSIX Scheduling Alignment

The scheduler implements a 3-tier priority model that maps directly to POSIX `sched.h` scheduling policies:

| SmallAIOS Class | POSIX Equivalent | Constraint | Task Types |
|-----------------|------------------|------------|------------|
| `System` (priority 0) | `SCHED_FIFO` priority 99 | Hard RT < 1 ms | Watchdog, Health, Syslog, Timer |
| `Ipc` (priority 1) | `SCHED_FIFO` priority 50 | Soft RT < 10 ms | IpcRouter, Tcp, ModelReload |
| `Inference` (priority 2) | `SCHED_OTHER` / `SCHED_BATCH` | Best-effort with budgets | Inference, Gpu |

Priority ordering is strict: `System < Ipc < Inference` (lower numeric value = higher priority). The `RunQueue` always dequeues from the highest-priority non-empty queue first.

Cooperative yield at operator boundaries provides `sched_yield()` semantics -- the running task explicitly relinquishes the CPU at known points. This is `SCHED_FIFO` with explicit yield points rather than time-slice preemption. The `posix` crate exposes the corresponding syscall interface.

## RTOS Design Patterns

### 1. Run-to-Completion with Cooperative Yield

Each ONNX operator (Conv, MatMul, Relu, etc.) runs to completion, then the executor invokes a yield callback before starting the next operator. This is the standard pattern in FreeRTOS, Zephyr, and ThreadX.

The key property: preemption points are known at compile time. There is no non-deterministic interruption of compute. The ONNX executor (`onnx-rt/src/executor.rs`) accepts an optional `yield_fn: Option<fn()>` and calls it between operators.

### 2. Per-Operator Time Budgets

`OperatorBudget` enforces time bounds on individual operators:

- **Soft budget** (`budget_ns`): log a warning if exceeded. Default: 100 ms.
- **Hard limit** (`hard_limit_multiplier`): abort the operator if elapsed time exceeds `budget_ns * multiplier`. Default: 10x (1 second).

This maps to:
- **ARINC 653 time partitioning** -- each operator gets a bounded time window.
- **DO-178C WCET analysis** -- budgets provide the framework for worst-case execution time verification.

### 3. Priority Preemption at Yield Points Only

Higher-priority tasks preempt lower-priority tasks, but only at yield points (operator boundaries). Between yield points, the running operator has exclusive use of the core. This means:

- No mid-computation register save/restore.
- No cache thrashing during GEMM or convolution kernels.
- WCET is analyzable because yield points are static and enumerable.
- Compatible with DO-178C DAL A structural coverage requirements.

The `should_continue_inference()` function checks whether higher-priority work is pending. If a System or IPC task arrives, the inference task will yield at the next operator boundary.

### 4. Work-Stealing for Inference Only

Idle cores steal tasks from busy cores' `Inference` queues via `steal_task()`. System and IPC tasks are never stolen -- they remain pinned to their assigned core. This matches the QNX adaptive partitioning model where critical partitions are non-migratable.

## Multi-Core Strategy: AMP over SMP

SmallAIOS uses Asymmetric Multi-Processing (AMP) style core decomposition:

- **Core 0:** System + IPC tasks. Always responsive for health checks, watchdog, networking.
- **Cores 1-N:** Inference compute. Parallel GEMM tiles, convolution output channels.

This is operator-level data parallelism, not task-level parallelism. Each core processes a chunk of the same operator's work (e.g., a tile of a matrix multiply). The scheduler supports up to 128 cores (`RunQueue` is instantiated per core).

This model preserves deterministic scheduling as used in:

- **AUTOSAR** -- automotive RTOS with static core-to-function mapping.
- **ARINC 653 multi-core extensions** -- avionics partitioned scheduling across cores.
- **DO-178C certified systems** -- where WCET must be independently analyzable per core.

SMP (any task on any core) is explicitly avoided. SMP requires complex locking, cache coherency protocols (MESI/MOESI), and makes WCET analysis intractable for safety-critical certification. The AMP model trades scheduling flexibility for analyzability.

## Design Differences from Traditional RTOS

| Traditional RTOS | SmallAIOS | Rationale |
|------------------|-----------|-----------|
| Preemptive multitasking | Cooperative at operator boundaries | Deterministic WCET, no mid-GEMM preemption |
| Process isolation (MMU) | Single address space | Unikernel -- zero IPC overhead, smaller TCB |
| ~450 syscalls (Linux) | ~46 syscalls | Minimal attack surface for inference workload |
| Generic scheduling | Inference-aware `OperatorBudget` | Operators have known compute profiles |
| Time-slice round-robin | Priority queues + explicit yield | No timer interrupt overhead, predictable latency |
| Thread-level parallelism | Operator-level data parallelism | Preserves single-task-per-core determinism |

## Implementation References

| Component | Path | Key Types |
|-----------|------|-----------|
| Per-core run queues | `kernel/src/sched/executor.rs` | `RunQueue`, `OperatorBudget` |
| Task model | `kernel/src/sched/task.rs` | `SchedulingClass`, `TaskType`, `TaskState`, `Task` |
| Scheduler module | `kernel/src/sched/mod.rs` | Re-exports, `CpuAffinity` |
| Timer infrastructure | `kernel/src/sched/timer.rs` | Timer wheel for budget enforcement |
| ONNX yield integration | `onnx-rt/src/executor.rs` | `yield_fn` callback between operators |
| POSIX syscall layer | `posix/` | `sched_yield()` and scheduling syscalls |

## Design Guidelines for Contributors

These rules follow from the scheduling model and are not optional:

1. **Never add preemptive scheduling.** The cooperative model is a safety requirement, not a simplification. Preemption would invalidate WCET analysis and break DO-178C compliance.

2. **All new operators must be bounded in execution time.** Provide a WCET estimate in the operator's documentation. Unbounded operators (e.g., dynamic loops without iteration caps) are not acceptable.

3. **System-class tasks must complete within 1 ms.** No blocking I/O, no heap allocation, no unbounded loops. These tasks must be O(1) or O(small constant).

4. **GPU dispatch must yield between kernel launches, not during.** A GPU kernel launch is atomic from the scheduler's perspective. Yield after the launch returns, before the next launch.

5. **Multi-core work must use data parallelism.** Split operator work across cores (GEMM tiles, conv channels). Do not move entire tasks between cores. Work-stealing applies only to the Inference queue and is the sole exception.

6. **New syscalls require justification.** The ~46 syscall surface is intentional for security and verification. Every new syscall increases the attack surface and the verification burden. Document why an existing syscall cannot serve the need.

7. **Do not introduce blocking synchronization in System or IPC paths.** Mutexes, semaphores, and condition variables can cause priority inversion. Use lock-free structures or disable interrupts for short critical sections.

8. **Budget parameters are tuning knobs, not safety margins.** The default `OperatorBudget` (100 ms soft, 1 s hard) is a starting point. Production deployments must profile actual operator execution times and set budgets accordingly.
