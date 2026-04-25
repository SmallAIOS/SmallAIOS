## Why

Blackwell GPUs (DGX Spark's GB10) deliver up to 2× FP32-equivalent
throughput when using FP8 (E4M3 / E5M2) tensor cores instead of TF32.
The CUDA stack on this machine already exposes FP8 GEMM via cuBLASLt
(verified in `arm64-gpu-container-v1` task 12.5: `gpu_gemm_fp8` works
end-to-end on `[16, 16] × [16, 16]` correctness tests). What's missing
is an end-to-end FP8 inference path — Conv via cuDNN's FP8 fused-op
API, runtime conversion of f32 weights to E4M3, and a benchmark proving
the speedup on a real model. Today every Conv on the hybrid path runs
TF32; flipping the dominant Conv load to FP8 should compress
ResNet-50's 33 ms hybrid latency further (target: <20 ms).

## What Changes

- Add cuDNN FP8 Conv FFI + a `gpu_conv2d_device_fp8` wrapper that
  uses `cudnnBackend*` with the `CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`
  + FP8 input tensor descriptors.
- Add a runtime weight conversion helper:
  `fp32_weights_to_fp8_e4m3(&Tensor) -> Tensor` and
  `fp32_weights_to_fp8_e5m2(&Tensor) -> Tensor`. Quantization is
  per-tensor scaled (max-abs) for v1; per-channel scaling can come
  later.
- Extend `SessionConfig::gpu_precision` (or add a new
  `compute_precision: ComputePrecision { Tf32, Fp16, Bf16, Fp8E4M3, Fp8E5M2 }`
  enum) so the user can opt into FP8 inference. Default
  remains `Tf32`.
- When precision is `Fp8E4M3` (or `Fp8E5M2`) and the hybrid executor
  encounters a Conv/Gemm/MatMul op, convert weights at session
  initialization time (one-shot, cached in
  `device_initializer_cache`), convert inputs at op boundary, run
  the FP8 kernel, dequantize outputs back to f32 for downstream ops.
- Add cuDNN `cudnnBackendCreateDescriptor` + descriptor-attribute
  FFI bindings (the new cuDNN backend API; v8 frontend interface).
- Wire FP8 dispatch into `gpu_conv2d_device` and `gpu_gemm_device`
  via a dtype-checking branch.
- Validate end-to-end: `bench_resnet50_cpu_vs_gpu_hybrid_fp8`
  shows ≥1.5× speedup over TF32 with `max_abs_diff < 5e-2`
  (FP8 quantization is lossier than TF32; the looser tolerance
  reflects realistic FP8-quantized model accuracy).
- Update `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` with the FP8
  results.

## Capabilities

### New Capabilities

- _None_ — FP8 is a precision mode of existing GPU inference.

### Modified Capabilities

- `onnx-runtime`: extend the CUDA Execution Provider with FP8
  inference scenarios (E4M3 and E5M2), per-tensor weight
  quantization at session initialization, and the `compute_precision`
  configuration knob.

## Impact

- **Code:** new `onnx-rt/src/cuda/fp8.rs` module wrapping the cuDNN
  backend API for FP8 Conv, plus `gpu_conv2d_device_fp8` /
  `gpu_gemm_device_fp8`. Extensions to `onnx-rt/src/cuda/ffi.rs`
  for `cudnnBackend*` bindings. New `onnx-rt/src/quantize.rs`
  module for f32→fp8 conversion (host-side). Changes to
  `executor_hybrid.rs::try_gpu_dispatch` to route Conv/Gemm/MatMul
  through the FP8 path when precision is `Fp8E4M3`/`Fp8E5M2`. New
  `compute_precision` field on `SessionConfig`.
- **Tests:** per-op CPU-vs-GPU FP8 correctness tests (loosened
  tolerance ~5e-2 for E4M3, ~1e-1 for E5M2). New
  `bench_resnet50_cpu_vs_gpu_hybrid_fp8e4m3` benchmark.
- **Downstream:** opens a 1.5–2× perf win on FP8-friendly models
  (Conv-heavy vision graphs). Default behavior unchanged when
  `compute_precision = Tf32`.
- **Dependencies:** cuDNN 9 backend API for FP8 Conv. Available in
  the cuDNN 9.20 shipped with CUDA 13.0 on DGX Spark.
- **Out of scope (flagged):** per-channel weight quantization
  (only per-tensor for v1), calibration-based scale selection
  (uses per-tensor max-abs which is the simplest correct
  default), FP8 inference for non-Conv/Gemm/MatMul ops (BN/Relu
  stay at f32 because their compute is already cheap), a dedicated
  ONNX FP8 model loader (the change quantizes f32 models on-the-
  fly; loading already-FP8 models is `arm64-gpu-container-v1`'s
  task 12.6 territory), NVFP4 / MXFP4 block-scaled paths
  (separate change).
