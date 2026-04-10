## 1. CorePool Abstraction

- [ ] 1.1 Create `onnx-rt/src/parallel.rs` with `CorePool` struct: `num_threads: usize`, constructor `new(num_threads)`
- [ ] 1.2 Implement container-mode `parallel_for`: use `std::thread::scope` to split range into `num_threads` chunks, execute closure on each, collect results
- [ ] 1.3 Implement `parallel_for` threshold guard: if range length is below a minimum chunk size, execute sequentially on calling thread
- [ ] 1.4 Add `CorePool` to `Session` configuration: `max_threads` parameter (default: detect available cores via `std::thread::available_parallelism()` in container mode, `1` in kernel mode)
- [ ] 1.5 Unit tests: `parallel_for` with 1 thread (sequential), 2 threads, 4 threads — verify results match sequential execution
- [ ] 1.6 Unit test: `parallel_for` with range smaller than num_threads — verify no panic, correct results

## 2. Parallel GEMM

- [ ] 2.1 Create `gemm_f32_parallel(pool, m, n, k, a, b, c)` in `gemm.rs`: split outer tile loop (rows of C) across `pool.num_threads` row bands
- [ ] 2.2 Each worker computes `gemm_f32` on its assigned row range using existing `gemm_tile` + `micro_kernel_8x8`
- [ ] 2.3 Add threshold check: if `m * k * n <= GEMM_PARALLEL_THRESHOLD` (default 65,536), call sequential `gemm_f32` directly
- [ ] 2.4 Wire `op_matmul` and `op_gemm` to use `gemm_f32_parallel` when `CorePool` has >1 thread
- [ ] 2.5 Unit tests: parallel GEMM correctness vs sequential for sizes 64×64, 128×128, 256×256, 512×512
- [ ] 2.6 Benchmark test: compare parallel vs sequential GEMM GFLOPS at 256×256 and 512×512

## 3. Parallel Convolution

- [ ] 3.1 Create `op_conv_parallel(pool, input, weight, bias)` in `operators.rs`: split output channels across workers
- [ ] 3.2 Each worker computes convolution for its assigned output channel range
- [ ] 3.3 Add threshold check: if `output_channels * H * W <= CONV_PARALLEL_THRESHOLD` (default 16,384), use sequential `op_conv`
- [ ] 3.4 Unit tests: parallel conv correctness vs sequential for various kernel sizes and channel counts

## 4. Parallel Element-wise Operations

- [ ] 4.1 Create generic `parallel_elementwise(pool, data, f)` helper: splits flat data array into chunks, applies closure to each chunk
- [ ] 4.2 Wire into `op_add`, `op_sub`, `op_mul`, `op_div`: use parallel variant when `num_elements > ELEMENTWISE_PARALLEL_THRESHOLD` (default 32,768)
- [ ] 4.3 Wire into `op_relu`, `op_sigmoid`, `op_tanh`, `op_clip`: same threshold check
- [ ] 4.4 Unit tests: parallel element-wise correctness for Add, Relu, Sigmoid with large tensors (100K elements)

## 5. Parallel Reduction Operations

- [ ] 5.1 Create `parallel_reduce(pool, data, identity, reduce_fn, merge_fn)` helper: each worker reduces its chunk, main thread merges partial results
- [ ] 5.2 Wire into `op_reduce_sum` and `op_reduce_mean`: partial sums per chunk, final sum/mean on main thread
- [ ] 5.3 Implement parallel Softmax: parallel max → broadcast subtract → parallel exp+sum → parallel divide
- [ ] 5.4 Add threshold check: if `num_elements <= REDUCTION_PARALLEL_THRESHOLD` (default 65,536), use sequential
- [ ] 5.5 Unit tests: parallel reduction correctness — verify sum and mean match sequential within f32 tolerance

## 6. Auto-Tuning Configuration

- [ ] 6.1 Define `ParallelConfig` struct in `parallel.rs`: `max_threads`, `gemm_threshold`, `conv_threshold`, `elementwise_threshold`, `reduction_threshold` with defaults
- [ ] 6.2 Add `ParallelConfig` to `Session` — passed through to executor and operator dispatch
- [ ] 6.3 Implement `ParallelConfig::sequential()` factory: returns config with all thresholds set to `usize::MAX` (disables parallelism)
- [ ] 6.4 Implement `ParallelConfig::default_for_cores(n)` factory: adjusts thresholds based on core count (lower thresholds for many cores)

## 7. Profiling Integration

- [ ] 7.1 Add `cores_used` and `parallel_efficiency` fields to operator timing report
- [ ] 7.2 When profiling enabled: record wall-clock time and estimated serial time, compute efficiency ratio
- [ ] 7.3 Log warning when parallel efficiency drops below 50% (indicates overhead exceeds benefit)

## 8. Kernel-Mode CorePool (Phase 2)

- [ ] 8.1 Implement `no_std` `CorePool` variant: post sub-tasks to per-core `RunQueue` as INFERENCE-class tasks
- [ ] 8.2 Implement atomic completion counter: each worker decrements on completion, main task busy-waits with yield
- [ ] 8.3 Implement `parallel_for` using `RunQueue::push` + completion barrier
- [ ] 8.4 Wire CPU core count detection from DTB/ACPI/SBI (use `cpu_count` from arch crate's `SystemInfo`)
- [ ] 8.5 Integration test: verify parallel GEMM produces correct results in kernel-mode scheduler context

## 9. End-to-End Testing

- [ ] 9.1 Integration test: run a multi-layer model (MatMul → Add → Relu → MatMul → Softmax) with `max_threads=4`, verify output matches `max_threads=1`
- [ ] 9.2 Benchmark test: compare inference latency for a medium model (128×128 MatMul chain) at 1, 2, 4, 8 threads
- [ ] 9.3 Verify `just test` passes with parallel code; run `just clippy` and `just fmt-check`
