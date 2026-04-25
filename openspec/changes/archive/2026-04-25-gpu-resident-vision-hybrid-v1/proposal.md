## Why

Hybrid GPU inference shows a tight correlation between GPU-eligible-op
fraction and measured speedup on DGX Spark: MLP at 100% eligibility
gets 7.47×; ResNet-50 v2 at 31% gets only 1.17×. Every non-GPU op
between two GPU-eligible ops forces an activation to memcpy
device→host, run scalar math on CPU, then memcpy host→device. For
ResNet-50, that's 117 CPU excursions per inference (51 BatchNorm + 50
Relu + 16 Add ops between 53 Convs), driving the CPU leg to ~16 s per
image. cuDNN already provides GPU kernels for all of these, and
`onnx-rt/src/cuda/gpu_executor.rs` already has a GPU-resident executor
used by the safetensors/LLM path — the vision ONNX path just isn't
wired to use it yet.

## What Changes

- Add a device-resident hybrid execution mode to the ONNX
  `execute_graph` path. Tensors track whether their canonical copy
  lives on host or device; ops routed to GPU consume and produce
  device tensors, ops routed to CPU consume and produce host tensors,
  and the runtime inserts memcpys only at boundaries between modes
  (including graph outputs).
- Add cuDNN-backed GPU implementations of the three op families that
  sandwich the Convs in vision models: `BatchNormalization`, `Relu`
  (reused for `Clip`/`LeakyRelu` via the same cuDNN activation call),
  and pooling (`MaxPool`/`AveragePool`/`GlobalAveragePool`).
- Add `gpu_add` for the residual-connection Add pattern used by
  ResNet.
- Share attribute parsing between CPU and GPU paths for BatchNorm,
  Pool, and Activation — same pattern as the `ConvAttrs` work in
  `conv-attribute-coverage-v1`.
- Extend `SessionConfig` with an opt-in `gpu_residency` mode so users
  can choose the hybrid path vs the current op-by-op dispatch
  (which stays the default until this lands on all covered models).
- Extend the hardware-gated benchmark harness to hard-assert output
  shape + diff for ResNet-50, SqueezeNet, and MobileNetV2 once both
  paths use the hybrid executor. Update
  `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` with the new numbers and
  a short architecture note.

## Capabilities

### New Capabilities

- _None_ — all changes extend existing capabilities.

### Modified Capabilities

- `onnx-runtime`: add requirements for the device-resident hybrid
  execution mode (tensor-location tracking, CPU/GPU boundary memcpys,
  per-op dispatch decisions), plus new requirements for the
  BatchNorm/Activation/Pool/Add cuDNN-backed operators under the CUDA
  execution provider.
- `onnx-cpu-execution`: extend the executor attribute-propagation
  requirement so BatchNorm, Activation, and Pool operators get their
  attributes parsed once and shared with both CPU and GPU dispatch —
  mirroring the existing Conv attribute pattern.

## Impact

- **Code:** primarily `onnx-rt/src/cuda/` (new kernel wrappers for
  BatchNorm / Activation / Pool / Add), `onnx-rt/src/cuda/ffi.rs`
  (new cuDNN FFI bindings for BN forward inference, activation
  forward, pooling forward), `onnx-rt/src/executor.rs` (device-tensor
  tracking in `execute_graph`), `onnx-rt/src/session.rs`
  (`SessionConfig::gpu_residency` field), and
  `onnx-rt/src/operators.rs` (shared attribute parsers for BN / Pool).
- **Tests:** per-op CPU-vs-GPU numerical tests in
  `onnx-rt/tests/test_cuda.rs` for BN / Relu / MaxPool / AveragePool
  / GlobalAveragePool / Add. End-to-end benchmark harness
  (`bench_vision_models.rs`) flips its soft-report-on-failure logic to
  hard assertions once the hybrid path is available.
- **Downstream:** vision benchmarks move from 1.17–2.54× to a target
  3–5×+ on the same hardware. No user-facing API breakage —
  `SessionConfig::gpu_residency` defaults to the current op-by-op
  behavior.
- **Dependencies:** CUDA 13.0 already installed on the DGX Spark
  workstation. cuDNN 9.20 ships with it. No new crate dependencies.
- **Out of scope (flagged so readers don't expect them):** batching
  (different change — throughput vs latency), multi-stream DMA
  overlap, Conv-BN-Relu fusion, Float16/BF16/INT8 precision switches,
  and adding GPU kernels for pure shape-path ops (Gather, Reshape,
  Concat, Shape, Cast, Unsqueeze) — those produce tiny tensors and
  the memcpy overhead is negligible.
