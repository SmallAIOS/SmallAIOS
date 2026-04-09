## Context

The ONNX runtime (after `onnx-cpu-runtime-v1`) executes operators sequentially on one core. The kernel scheduler has per-core run queues (up to 128 cores), work-stealing for INFERENCE tasks, and CPU affinity — but this infrastructure has never been used for operator-level parallelism.

The existing `gemm_f32` in `gemm.rs` uses cache-blocked tiling (64x64 tiles with 8x8 micro-kernels). The tile loops are independent — each tile reads from A and B and writes to a disjoint region of C. This is the textbook case for data-parallel decomposition.

Similarly, convolution (output channels are independent), element-wise ops (each element is independent), and reductions (partial sums can be computed independently then merged) are all parallelizable at the data level.

The challenge: SmallAIOS is `#![no_std]` in kernel mode. There's no `std::thread::spawn`. Parallelism must go through the kernel scheduler's run queues. In container mode, `std::thread` is available.

## Goals / Non-Goals

**Goals:**
- Split compute-heavy operators across available CPU cores
- Dual-mode thread pool: kernel scheduler integration (kernel mode) + `std::thread` (container mode)
- Auto-tuning: only parallelize when the work is large enough to amortize fork/join overhead
- Preserve cooperative scheduling: parallel sub-tasks yield at tile boundaries
- Linear or near-linear speedup for GEMM and Conv on 2-8 cores

**Non-Goals:**
- NUMA-aware scheduling (future — requires memory topology discovery)
- Vectorized SIMD intrinsics (orthogonal — compiler auto-vectorization handles this)
- GPU offload (handled by `compute-abstraction-v1`)
- Parallel graph execution (running independent operators simultaneously — this change parallelizes *within* a single operator)
- Dynamic core count adjustment during inference

## Decisions

### D1: Dual-Mode Thread Pool — `CorePool` Abstraction

A `CorePool` abstraction provides `parallel_for(range, closure)` semantics. Two implementations behind `#[cfg]`:

```rust
pub struct CorePool {
    num_threads: usize,
}

impl CorePool {
    pub fn new(num_threads: usize) -> Self;
    
    /// Split `range` into `num_threads` chunks, execute `f` on each chunk in parallel,
    /// collect results. Blocks until all chunks complete.
    pub fn parallel_for<F, R>(&self, range: Range<usize>, f: F) -> Vec<R>
    where
        F: Fn(Range<usize>) -> R + Send + Sync,
        R: Send;
}
```

**Container mode** (`cfg(feature = "std")` or `cfg(target_os)`): uses `std::thread::scope` (Rust 1.63+) for zero-cost scoped threads. No heap allocation for thread handles — the scope ensures all threads join before returning.

**Kernel mode** (`cfg(not(feature = "std"))` via `no_std`): posts sub-tasks to per-core `RunQueue` entries as INFERENCE-class tasks, then busy-waits (with yield) until all sub-tasks mark completion via an atomic counter. This reuses the existing scheduler infrastructure.

**Why not `rayon`:** Not `no_std` compatible. Also pulls in a large dependency tree.

### D2: Parallel GEMM — Tile Row Decomposition

Split the outer tile loop (rows of C) across cores. Each core computes a horizontal band of the output matrix:

```
Core 0: C[0..M/4,    :]  ← tiles from A[0..M/4, :] × B
Core 1: C[M/4..M/2,  :]  ← tiles from A[M/4..M/2, :] × B  
Core 2: C[M/2..3M/4, :]  ← tiles from A[M/2..3M/4, :] × B
Core 3: C[3M/4..M,   :]  ← tiles from A[3M/4..M, :] × B
```

Each core runs the existing `gemm_tile` + `micro_kernel_8x8` on its assigned rows. No synchronization needed — output regions are disjoint. B is read-shared (cache-friendly since each core reads the same B tiles).

**Why row decomposition over 2D tiling:** Simpler, and row decomposition keeps B accesses cache-coherent across cores (they all read the same columns of B). 2D tiling would require more complex partitioning for marginal gain.

### D3: Parallel Conv — Output Channel Decomposition

Split output channels across cores. Each core computes a subset of output feature maps:

```
Core 0: output[:, 0..C/4, :, :]
Core 1: output[:, C/4..C/2, :, :]
...
```

Each output channel depends only on all input channels and its own filter weights — no cross-channel dependency.

### D4: Parallel Element-wise and Reduction

**Element-wise** (Add, Mul, Relu, etc.): split the flat data array into equal chunks:
```
Core 0: elements[0..N/4]
Core 1: elements[N/4..N/2]
...
```

**Reduction** (ReduceMean, ReduceSum, Softmax): two-phase parallel reduction:
1. Each core computes partial sum/max over its chunk
2. Main thread merges partial results (single-threaded — the merge is O(num_cores), negligible)

Softmax is: parallel max → subtract → parallel exp+sum → parallel divide.

### D5: Auto-Tuning Threshold

Parallelism has overhead: thread spawn/post, cache coherency, synchronization. For small tensors, the overhead exceeds the speedup. Heuristic:

| Operator | Parallelize when |
|----------|-----------------|
| GEMM | M × K × N > 65,536 (e.g., 64×64×16 or larger) |
| Conv | output_channels × H × W > 16,384 |
| Element-wise | num_elements > 32,768 |
| Reduction | num_elements > 65,536 |

These thresholds are configurable via `Session` parameters and can be tuned per-platform. The defaults are conservative — better to under-parallelize than to waste cycles on fork/join overhead.

### D6: Integration with OperatorBudget

Parallel execution changes the timing profile. When profiling is enabled:
- Measure wall-clock time (should decrease with more cores)
- The `OperatorBudget` thresholds remain wall-clock based — parallel execution helps meet tighter budgets
- Add a `parallel_efficiency` metric: `serial_time / (wall_time × num_cores_used)` to detect cases where parallelism isn't helping

## Risks / Trade-offs

**[Risk] Kernel-mode thread pool complexity** — Posting to `RunQueue` and waiting for completion requires careful atomic coordination. Mitigation: Start with container-mode (`std::thread::scope`) implementation and comprehensive tests. Add kernel-mode implementation as a second phase.

**[Risk] Cache thrashing on small core counts** — If cores share L2/L3 cache, parallel GEMM may cause cache eviction. Mitigation: Tile size (64×64 = 48 KB per tile set) is chosen to fit in L1. Each core works on its own tiles. L2/L3 pressure comes from shared B matrix reads, which are sequential and cache-friendly.

**[Risk] Diminishing returns beyond 4-8 cores** — Memory bandwidth becomes the bottleneck for GEMM, not compute. Mitigation: Auto-tuning threshold caps effective parallelism. On memory-bound workloads, fewer cores may be optimal. Future: add bandwidth-aware scheduling.

**[Trade-off] Container vs. kernel implementation gap** — `std::thread::scope` is simple and correct. Kernel-mode `RunQueue` integration is significantly more complex. We accept this gap and prioritize container mode first.

## Open Questions

- **Q1:** Should `CorePool` in kernel mode use the existing work-stealing infrastructure, or a simpler barrier-based approach? Work-stealing adds complexity but handles load imbalance better.
- **Q2:** For RISC-V targets with many small cores (e.g., 64 cores), should the threshold heuristic adapt to core count? More cores = lower per-core overhead = lower threshold.
- **Q3:** Should we support pinning parallel sub-tasks to specific cores (NUMA locality) in the initial implementation, or defer?
