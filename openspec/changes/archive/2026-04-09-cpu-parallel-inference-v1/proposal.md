## Why

SmallAIOS targets CPU-only deployments (RISC-V boards, ARM64 edge devices, x86 servers without GPUs) where inference compute is entirely CPU-bound. The current ONNX runtime executes operators sequentially on a single core. On an 8-core machine, 7 cores sit idle during inference. For compute-heavy operators like MatMul/GEMM and Conv, the work is naturally parallelizable — the existing `gemm_f32` micro-kernel already tiles the computation into independent 64x64 blocks that can run on separate cores with no synchronization.

This change adds operator-level parallelism: splitting individual compute-heavy operators across available CPU cores while preserving the cooperative scheduling model. The kernel scheduler already has per-core run queues, work-stealing, and CPU affinity — this change uses that infrastructure to distribute GEMM tiles, convolution windows, and reduction passes across cores.

This is critical for scenarios where: (1) no GPU is available, (2) the model is small enough that GPU transfer overhead exceeds compute savings, or (3) INT8/quantized inference on AVX-512 is faster on CPU than GPU.

## What Changes

- Add a thread pool / core pool abstraction that maps to the kernel's per-core run queues (kernel mode) or `std::thread` (container mode)
- Implement parallel GEMM: split tile loops across cores, join results
- Implement parallel Conv: split output channels across cores
- Implement parallel reduction ops (ReduceMean, ReduceSum, Softmax): split reduction domain, merge partial results
- Implement parallel element-wise ops (Add, Mul, Relu, etc.): split tensor data range across cores
- Add auto-tuning heuristic: parallelize only when tensor size exceeds threshold (avoid overhead for small tensors)
- Integrate with existing `OperatorBudget` — parallel execution should reduce per-operator wall-clock time

## Capabilities

### New Capabilities
- `cpu-parallel-compute`: Operator-level parallelism for CPU inference — thread pool, parallel GEMM, parallel Conv, parallel element-wise, auto-tuning heuristic

### Modified Capabilities
- `onnx-cpu-execution`: Add requirements for parallel operator dispatch and core utilization
- `onnx-runtime`: Add requirements for configurable parallelism (core count, threshold)

## Impact

- **Code:** New `onnx-rt/src/parallel.rs` (thread pool + parallel dispatch), modifications to `gemm.rs`, `operators.rs` (parallel variants of compute-heavy ops)
- **Kernel:** Uses existing per-core `RunQueue` and work-stealing in kernel mode; `std::thread` in container mode
- **Performance:** Significant speedup on multi-core CPU-only targets; no change on single-core
- **Config:** New session parameter `max_threads` (default: number of available cores)
- **Dependencies:** None in kernel mode. Container mode uses `std::thread` (already available)
