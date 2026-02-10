# Phase 2: Core Kernel — Memory, Scheduler, Syscalls

## Objective

Build the fundamental kernel services: physical and virtual memory management,
task scheduler with async executor, and the syscall dispatch table.

## Dependencies

- Phase 1 complete (boot to serial output on both architectures)

## Tasks

### 2.1 Physical Memory Allocator
- [ ] Parse memory map from bootloader (Multiboot2 / UEFI / DTB)
- [ ] Implement buddy allocator (orders 0-21, 4 KiB to 8 GiB)
- [ ] Support huge pages (2 MiB) and giant pages (1 GiB)
- [ ] NUMA-aware allocation (detect NUMA topology from ACPI SRAT / DTB)
- [ ] Unit tests: alloc, free, split, merge, exhaustion behavior

### 2.2 Virtual Memory Manager
- [ ] x86-64: 4-level page table management (PML4)
- [ ] ARM64: 4-level page table management (4 KiB granule)
- [ ] Map/unmap/protect operations
- [ ] Kernel virtual memory layout per spec
- [ ] Guard pages for stack overflow detection
- [ ] Unit tests: mapping, unmapping, protection changes

### 2.3 Kernel Heap
- [ ] Slab allocator with size classes (16B - 2048B)
- [ ] Per-CPU slab caches (lock-free fast path)
- [ ] Implement `#[global_allocator]` trait
- [ ] Fallback to buddy allocator for large allocations
- [ ] Unit tests: alloc/free patterns, stress test, fragmentation

### 2.4 Tensor Memory Pool
- [ ] Dedicated memory region for tensor buffers
- [ ] Arena allocator with bump pointer + free list
- [ ] Aligned allocations (64B, 2 MiB, as requested)
- [ ] Reference-counted tensor buffers
- [ ] Pool reset between inference runs
- [ ] GPU-capable allocations (pinned pages for DMA)
- [ ] Unit tests: allocation patterns matching real inference workloads

### 2.5 Interrupt Handling
- [ ] x86-64: Full IDT setup with proper exception handlers
- [ ] x86-64: APIC timer interrupt for scheduler ticks
- [ ] x86-64: IPI for cross-core notifications
- [ ] ARM64: GICv3 initialization and interrupt routing
- [ ] ARM64: ARM Generic Timer for scheduler ticks
- [ ] ARM64: SGI (Software Generated Interrupts) for IPI
- [ ] Top-half / bottom-half split architecture
- [ ] Unit tests: timer fires, IPI delivery

### 2.6 Task Scheduler
- [ ] Task struct: state, stack, future, priority, affinity
- [ ] Per-CPU run queues (lock-free LIFO)
- [ ] Work-stealing (lock-free FIFO from other queues)
- [ ] Async executor: poll futures, wake on events
- [ ] Timer-based scheduling (cooperative with timeout fallback)
- [ ] CPU idle (HLT/WFI) when no tasks ready
- [ ] SMP: boot secondary CPUs, each runs executor loop
- [ ] Unit tests: spawn, yield, join, work stealing

### 2.7 Syscall Interface
- [ ] Syscall dispatch table (function pointer array)
- [ ] x86-64: `syscall`/`sysret` handler (for VM mode)
- [ ] ARM64: `svc` handler (for VM mode)
- [ ] Direct function call path (for unikernel mode)
- [ ] Error code translation
- [ ] All memory management syscalls implemented
- [ ] All task management syscalls implemented
- [ ] Unit tests: each syscall with valid and invalid inputs

### 2.8 Early Console Upgrade
- [ ] Formatted logging (`log` crate interface, no dependency)
- [ ] Log levels: error, warn, info, debug, trace
- [ ] Timestamp prefix (from timer)
- [ ] Ring buffer for log storage (for later IPC access)

## Exit Criteria

- Buddy allocator manages all physical memory correctly
- Page tables set up for kernel virtual memory layout
- `Box::new()`, `Vec::new()` work (global allocator functional)
- Tensor pool allocates/frees aligned buffers correctly
- Timer interrupts fire on schedule on both architectures
- Can spawn 1000 tasks, they all complete via work-stealing executor
- All syscalls dispatch correctly with proper error handling
- Kernel logs with timestamps to serial console
- All unit tests pass in hosted mode
- Integration test: boot in QEMU, spawn tasks, verify completion
