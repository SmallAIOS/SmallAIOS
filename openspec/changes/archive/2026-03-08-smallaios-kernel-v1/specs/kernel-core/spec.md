# Delta for Kernel Core

## ADDED Requirements

### Requirement: Buddy Allocator
The kernel SHALL implement a buddy allocator for physical page-granularity memory allocation supporting 4 KiB, 2 MiB, and 1 GiB page sizes.

#### Scenario: Allocate and free 4 KiB pages
- WHEN a component requests a 4 KiB page allocation
- THEN the buddy allocator MUST return a page-aligned physical address
- AND the allocated page MUST NOT overlap with any other active allocation

#### Scenario: Coalesce freed buddy pages
- WHEN a 4 KiB page is freed
- AND its buddy page is also free
- THEN the allocator MUST coalesce them into a single 8 KiB block
- AND coalescing MUST continue recursively up to the maximum order

### Requirement: Slab Allocator
The kernel SHALL implement a slab allocator layered on the buddy allocator for sub-page kernel object allocations.

#### Scenario: Allocate fixed-size kernel objects
- WHEN a kernel subsystem requests allocation of a fixed-size object (e.g., TaskControlBlock)
- THEN the slab allocator MUST return a properly aligned pointer from a pre-allocated slab
- AND allocation MUST complete in O(1) amortized time

### Requirement: Tensor Memory Pool
The kernel SHALL provide a dedicated tensor memory pool with pre-allocated aligned buffers for ONNX inference workloads.

#### Scenario: Allocate a tensor buffer
- WHEN the ONNX runtime requests a tensor allocation with a given shape and dtype
- THEN the pool MUST return a buffer with alignment suitable for SIMD operations (at least 64-byte aligned)
- AND the buffer MUST be DMA-capable for GPU transfer eligibility

#### Scenario: Zero-copy tensor sharing
- WHEN two operators reference the same tensor buffer
- THEN the pool MUST use reference counting to track lifetime
- AND the buffer MUST NOT be freed until all references are released

### Requirement: Soft Real-Time Cooperative Scheduler
The kernel SHALL implement a soft real-time cooperative scheduler with three priority classes (SYSTEM, IPC, INFERENCE) and mandatory yield points between ONNX operators.

#### Scenario: Task yielding at operator boundaries
- WHEN an inference task completes an ONNX operator execution
- THEN the task MUST yield control to the scheduler
- AND the scheduler MUST check for pending SYSTEM and IPC tasks before resuming inference

#### Scenario: SYSTEM class preempts INFERENCE
- WHEN a SYSTEM class task (watchdog, health check) becomes runnable
- AND an INFERENCE class task has just yielded at an operator boundary
- THEN the scheduler MUST execute the SYSTEM task before resuming inference
- AND the SYSTEM task MUST complete within 1 ms

#### Scenario: IPC class preempts INFERENCE
- WHEN an IPC class task (message routing, TCP handling) becomes runnable
- AND an INFERENCE class task has just yielded at an operator boundary
- THEN the scheduler MUST execute the IPC task before resuming inference

#### Scenario: Idle core power management
- WHEN a CPU core has no runnable tasks in its local queue and no tasks to steal
- THEN the core MUST enter a low-power state (HLT on x86-64, WFI on ARM64)
- AND the core MUST wake on the next interrupt

### Requirement: Hardware Watchdog Timer
The kernel SHALL initialize and service a hardware watchdog timer to detect and recover from system hangs.

#### Scenario: Watchdog initialization
- WHEN the kernel boots
- THEN it MUST initialize the hardware watchdog with a configurable timeout (default: 30 seconds)
- AND the watchdog MUST be serviced by a SYSTEM class task at regular intervals

#### Scenario: Watchdog timeout triggers reset
- WHEN the watchdog timer is not serviced within its timeout period
- THEN the hardware MUST trigger a system reset
- AND the reset indicates a hang or deadlock condition

#### Scenario: Watchdog servicing at yield points
- WHEN an inference task yields at an operator boundary
- AND more than half the watchdog timeout has elapsed since the last service
- THEN the scheduler MUST service the watchdog before resuming any task

### Requirement: Per-Operator Time Budgets
The scheduler SHALL track per-operator execution time and enforce configurable time budgets.

#### Scenario: Operator within budget
- WHEN an ONNX operator completes within its configured time budget
- THEN the scheduler MUST continue execution normally with no diagnostic action

#### Scenario: Operator exceeds soft budget
- WHEN an ONNX operator execution time exceeds its soft budget (1x configured budget)
- THEN the scheduler MUST log a warning to syslog with operator name, actual time, and budget
- AND inference MUST continue normally

#### Scenario: Operator exceeds hard limit
- WHEN an ONNX operator execution time exceeds its hard limit (10x budget or configurable)
- THEN the scheduler MUST abort the inference session with an OperatorTimeout error
- AND MUST log the timeout event to syslog

#### Scenario: Whole-inference timeout
- WHEN the total wall-clock time for an onnx_run call exceeds the configured inference_timeout_ms
- THEN the scheduler MUST abort the inference at the next yield point with a TimedOut error

### Requirement: Work-Stealing Executor
The kernel SHALL implement a work-stealing executor with one worker per CPU core and lock-free per-core run queues.

#### Scenario: Steal work from busy core
- WHEN a worker core's local run queue is empty
- AND another core's run queue has two or more pending tasks
- THEN the idle worker MUST steal one task from the busy core's queue
- AND the steal operation MUST be lock-free

### Requirement: Syscall Interface
The kernel SHALL expose a minimal syscall interface of approximately 46 syscalls across Memory, Task, IPC, ONNX, Device, and System categories.

#### Scenario: Syscall dispatch in unikernel mode
- WHEN a component invokes a syscall in unikernel (container) mode
- THEN the kernel MUST dispatch via direct function call with no ring transition
- AND the syscall MUST validate all capability tokens before execution

#### Scenario: Syscall dispatch in VM mode
- WHEN a component invokes a syscall in VM mode
- THEN the kernel MUST handle the syscall/svc instruction trap
- AND dispatch to the correct handler via the syscall table

#### Scenario: Invalid syscall number
- WHEN a syscall is invoked with an unrecognized syscall number
- THEN the kernel MUST return an ENOSYS error
- AND MUST NOT panic or corrupt kernel state

### Requirement: Interrupt Handling
The kernel SHALL implement a top-half/bottom-half interrupt handling model.

#### Scenario: Timer interrupt processing
- WHEN a timer interrupt fires (APIC on x86-64, GIC on ARM64)
- THEN the top half MUST acknowledge the interrupt and enqueue a work item
- AND the bottom half MUST process scheduler ticks and pending timeouts asynchronously

#### Scenario: IPI for cross-core task migration
- WHEN the scheduler decides to migrate a task to another core
- THEN it MUST send an inter-processor interrupt to the target core
- AND the target core MUST wake from idle and process the migrated task

#### Scenario: MSI-X GPU completion interrupt
- WHEN the GPU signals command completion via MSI-X
- THEN the top half MUST acknowledge the interrupt
- AND the bottom half MUST wake the GPUTask awaiting that completion

### Requirement: Boot Sequence
The kernel SHALL support two boot modes: container mode (library OS) and VM mode (full kernel boot).

#### Scenario: Container mode boot
- WHEN SmallAIOS starts as a container process
- THEN the kernel MUST initialize allocators, scheduler, ONNX runtime, and IPC router
- AND MUST begin accepting inference requests within 500 ms of process start
- AND MUST NOT require elevated container privileges

#### Scenario: VM mode boot
- WHEN SmallAIOS boots as a virtual machine kernel
- THEN the kernel MUST set up page tables, GDT/IDT (x86-64) or exception vectors (ARM64), and interrupt controller
- AND MUST enumerate PCIe devices and initialize GPU if present
- AND MUST complete full boot and reach ready state within 2 seconds
