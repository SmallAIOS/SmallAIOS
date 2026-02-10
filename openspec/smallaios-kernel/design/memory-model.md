# Memory Model Design

## Overview

Memory management in SmallAIOS is optimized for AI inference workloads, which have
a distinctive pattern: large, aligned, mostly-read tensor buffers combined with
small, frequent kernel object allocations.

## Memory Hierarchy

```
┌───────────────────────────────────────────────────┐
│ Layer 4: Tensor API                                │
│   tensor_alloc(shape, dtype) → TensorHandle       │
│   Zero-copy, reference-counted, GPU-mappable      │
├───────────────────────────────────────────────────┤
│ Layer 3: Kernel Heap                               │
│   #[global_allocator] — Box, Vec, String, etc.    │
│   Slab allocator for small objects                │
├───────────────────────────────────────────────────┤
│ Layer 2: Page Allocator                            │
│   Buddy allocator for 4K/2M/1G pages             │
│   NUMA-aware, huge-page-aware                     │
├───────────────────────────────────────────────────┤
│ Layer 1: Physical Memory Map                       │
│   Boot-time enumeration of available RAM          │
│   Platform-specific (UEFI memmap, DTB, aux vec)   │
└───────────────────────────────────────────────────┘
```

## Layer 1: Physical Memory Map

At boot, the platform provides a physical memory map:

```rust
pub struct MemoryRegion {
    pub base: PhysAddr,
    pub size: usize,
    pub kind: MemoryKind,
}

pub enum MemoryKind {
    Usable,          // Available for allocation
    Reserved,        // Firmware/hardware reserved
    AcpiReclaimable, // Can be reclaimed after ACPI init
    Mmio,            // Memory-mapped I/O
    KernelCode,      // Our own code/data
}
```

In container mode, the "physical memory" is the process's virtual address space,
and we use `mmap(MAP_ANONYMOUS)` to allocate pages from the host.

## Layer 2: Buddy Allocator

The buddy allocator manages physical pages in power-of-two blocks:

```
Order 0:  4 KiB    (single page)
Order 1:  8 KiB    (2 pages)
...
Order 9:  2 MiB    (512 pages, huge page)
...
Order 18: 1 GiB    (262144 pages, giant page)
Order 21: 8 GiB    (max allocation)
```

### Algorithm

Standard binary buddy with free lists per order:

```
Allocate(order):
    if free_list[order] not empty:
        return free_list[order].pop()
    else:
        block = Allocate(order + 1)  // Get larger block
        split block into two buddies
        free_list[order].push(buddy_1)
        return buddy_0

Free(block, order):
    buddy = block XOR (1 << (order + PAGE_SHIFT))
    if buddy is free at same order:
        remove buddy from free_list[order]
        Free(merged_block, order + 1)  // Merge and recurse
    else:
        free_list[order].push(block)
```

### NUMA Awareness

On multi-socket systems, maintain separate buddy allocators per NUMA node:

```rust
pub struct PhysAllocator {
    nodes: Vec<BuddyAllocator>,  // One per NUMA node
}

impl PhysAllocator {
    pub fn alloc(&self, order: usize, node: NumaNode) -> Option<PhysAddr> {
        // Try local node first, then fallback to others
        self.nodes[node.0].alloc(order)
            .or_else(|| self.nodes.iter().find_map(|n| n.alloc(order)))
    }
}
```

## Layer 3: Slab Allocator

For small kernel objects (task structs, capability tokens, IPC messages), a slab
allocator reduces fragmentation and improves cache locality.

### Size Classes

| Size Class | Object Size | Objects per Page |
|---|---|---|
| 0 | 16 bytes | 256 |
| 1 | 32 bytes | 128 |
| 2 | 64 bytes | 64 |
| 3 | 128 bytes | 32 |
| 4 | 256 bytes | 16 |
| 5 | 512 bytes | 8 |
| 6 | 1024 bytes | 4 |
| 7 | 2048 bytes | 2 |

Objects larger than 2048 bytes go directly to the buddy allocator (page-aligned).

### Per-CPU Caches

Each CPU core has a local free list per size class (8-16 objects). This eliminates
lock contention for the common case:

```
Allocate:
    1. Check per-CPU cache → return if available (no lock)
    2. Refill per-CPU cache from global slab (takes lock briefly)
    3. If global slab empty, allocate new page from buddy allocator

Free:
    1. Return to per-CPU cache (no lock)
    2. If per-CPU cache full, drain half to global slab
```

## Layer 4: Tensor Memory Pool

The most performance-critical allocator. Tensor buffers have specific requirements:

### Requirements

1. **Alignment**: 64-byte aligned (cache line) for CPU SIMD;
   256-byte or 512-byte for some AVX-512 operations;
   page-aligned for GPU DMA
2. **Size**: Typically 4 KB to 1 GB; common sizes for given model are predictable
3. **Lifetime**: Short (per-inference intermediate) or long (model weights)
4. **Sharing**: Zero-copy between operators, potentially GPU-mapped

### Design: Arena with Size-Class Pools

```
Tensor Pool
├── Small pool:  [64B - 4KB]    — Slab allocator
├── Medium pool: [4KB - 2MB]    — Page-granularity buddy allocator
├── Large pool:  [2MB - 1GB]    — Huge page allocator
└── Giant pool:  [1GB+]         — Direct mmap (rare)
```

### Pre-allocation

At session creation, the memory planner computes tensor lifetimes and pre-allocates
a buffer pool sized exactly for the model's peak memory usage:

```rust
pub struct TensorPool {
    /// Pre-allocated arena for this session
    arena: *mut u8,
    arena_size: usize,
    /// Bump pointer for sequential allocation
    cursor: AtomicUsize,
    /// Free list for reuse within the arena
    free_list: Vec<(usize, usize)>,  // (offset, size)
}
```

Between inference runs, the pool is reset (cursor back to 0) rather than
individually freeing tensors. This makes per-inference allocation essentially free.

### Reference Counting

Tensors shared between operators use atomic reference counting:

```rust
pub struct TensorBuffer {
    pub data: *mut u8,
    pub size: usize,
    pub refcount: AtomicU32,
    pub pool: *mut TensorPool,
    pub gpu_mapped: AtomicBool,
}
```

When refcount hits 0, the buffer is returned to the pool (not freed to the OS).

## GPU Memory Management

### GPU VRAM Layout

```
┌─────────────────────────────────────┐  ← VRAM top
│ Reserved (firmware, display)        │
├─────────────────────────────────────┤
│ Model weights (loaded once)         │
├─────────────────────────────────────┤
│ Inference workspace (per-run)       │
│  ├─ Input tensors                   │
│  ├─ Intermediate tensors            │
│  └─ Output tensors                  │
├─────────────────────────────────────┤
│ Kernel code (PTX/SASS)              │
├─────────────────────────────────────┤
│ Command buffers                     │
└─────────────────────────────────────┘  ← VRAM base
```

### GPU Memory Allocator

Simple bump allocator with regions:

```rust
pub struct GpuAllocator {
    /// Static region: weights, kernels (allocated once, never freed)
    static_cursor: usize,
    static_end: usize,
    /// Dynamic region: inference workspace (reset between runs)
    dynamic_base: usize,
    dynamic_cursor: AtomicUsize,
    dynamic_end: usize,
}
```

The dynamic region is reset between inference runs, same as the CPU tensor pool.

### CPU ↔ GPU Transfer

```
Pinned CPU Memory ←──DMA──→ GPU VRAM
     ↕ (zero-copy)
Tensor Pool
```

- Input tensors allocated in pinned (page-locked) CPU memory
- GPU DMA engine copies to VRAM asynchronously
- Output tensors copied back to pinned memory via DMA
- Pinned memory region is pre-allocated at boot

## Memory Safety Invariants

1. All allocator state is protected by Rust's type system
2. Physical-to-virtual mappings are tracked; double-map is impossible
3. GPU memory is not directly accessible from CPU (no wild pointers)
4. Tensor buffers are bounds-checked on creation; operators trust the bounds
5. No memory is ever used uninitialized (Rust guarantees + explicit zeroing for security)
6. Stack overflow protection via guard pages (one unmapped page below each stack)

## Memory Diagnostics

Available via IPC at `smallaios/v1/metrics/memory`:

```
Physical pages: total=524288, used=131072, free=393216
Kernel heap: allocated=2.5MB, peak=3.1MB
Tensor pool: capacity=1024MB, used=412MB, peak=890MB
GPU VRAM: total=16384MB, static=2048MB, dynamic_used=1024MB
Slab caches: [16B: 45/256] [32B: 128/128] [64B: 12/64] ...
```
