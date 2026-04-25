## Why

The hybrid GPU executor delivers excellent **single-request** latency
(33 ms / 145 ms / 565 ms / 0.16 ms for ResNet-50 / SqueezeNet /
MobileNetV2 / MLP at batch=1). For serving workloads, throughput
matters more than latency. Today every inference services exactly one
input — Blackwell tensor cores are radically underutilized at batch=1,
where activations are tiny (~600 KB on ResNet-50) and the GPU spends
most of its time on dispatch/launch overhead rather than compute.
Batching N requests through one inference would scale GPU utilization
near-linearly until the kernel-bound regime, unlocking 10×–50×
throughput for serving deployments.

## What Changes

- Add a `Session::run_batched(&[InferenceInput])` API that accepts
  `N` named-input arrays of identical shape (the only-changing
  dimension is the leading batch dim) and produces `N` outputs.
- Implement at the executor level by stacking N inputs along the
  batch axis before dispatching, then unstacking outputs at the
  end. The hybrid + cuDNN paths already accept arbitrary batch
  size — the change is in the host-side stacking/unstacking logic.
- Add a `BatchPolicy` to `SessionConfig`:
  - `Static(N)` — every `run_batched` call must supply exactly N
    inputs.
  - `Dynamic { max: N, pad: bool }` — accept anything from 1 to N;
    if `pad`, repeat the last input to reach N (and discard the
    padded outputs).
  - `None` — default; `run_batched` is rejected.
- Make the existing `run(&[InferenceInput])` a single-element path
  through the same machinery (degenerate case of batch=1).
- Validate that all batched inputs share rank and dim sizes except
  for the batch axis; return `BatchShapeMismatch` otherwise.
- Add concurrency-friendly throughput benchmarks:
  `bench_resnet50_throughput_batched(B)` for `B ∈ {1, 4, 16, 64}`.
- Document batching semantics in `docs/architecture.md` and the
  benchmark doc.

## Capabilities

### New Capabilities

- `inference-batching`: covers the new `run_batched` API, batch
  validation rules, padding behavior, and shape contracts.

### Modified Capabilities

- `onnx-runtime`: extend the CUDA Execution Provider scenarios with
  batched-inference contracts (the existing scenarios cover batch=1
  implicitly; this change makes batch>1 explicit).
- `onnx-cpu-execution`: extend graph executor scenarios with the
  batched-input contract (the executor must accept arbitrary leading
  dim).

## Impact

- **Code:** new `onnx-rt/src/session.rs::Session::run_batched`,
  changes to `executor_hybrid.rs` to stack/unstack along the batch
  axis, new `BatchPolicy` enum + field on `SessionConfig`, possibly
  new `onnx-rt/src/batch.rs` for the stacking helpers.
- **Tests:** unit tests for stacking/unstacking, batched-output
  validation, padded batches, shape-mismatch errors. New
  throughput benchmarks in `bench_vision_models.rs`.
- **Downstream:** unblocks production-style serving (multi-request
  load). Single-request latency is unaffected (batch=1 path is the
  same as today).
- **Dependencies:** none new. cuDNN/cuBLAS already accept batched
  tensors.
- **Out of scope (flagged):** request-level scheduling / batch
  formation across HTTP requests (belongs in the container/server
  layer, not the runtime). Padding strategies beyond "repeat last
  input" (e.g., zero-pad, dynamic-shape support per ONNX). KV-cache
  batching for LLM workloads (separate change). Concurrent
  multi-Session inference / queue management.
