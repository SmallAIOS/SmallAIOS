## 1. cuDNN backend API FFI

- [ ] 1.1 Add to `onnx-rt/src/cuda/ffi.rs`: `cudnnBackendDescriptor_t`, `cudnnBackendDescriptorType_t` enum (TENSOR_DESCRIPTOR, OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, OPERATIONGRAPH_DESCRIPTOR, EXECUTION_PLAN_DESCRIPTOR, VARIANT_PACK_DESCRIPTOR, ENGINE_DESCRIPTOR, ENGINECFG_DESCRIPTOR, HEUR_MODE_DESCRIPTOR)
- [ ] 1.2 Add `cudnnBackendAttributeName_t` enum with the attributes we need (TENSOR_DATA_TYPE, TENSOR_DIMENSIONS, TENSOR_STRIDES, TENSOR_UNIQUE_ID, TENSOR_BYTE_ALIGNMENT, OPERATION_CONVOLUTION_X / Y / W / FORWARD_DESC, OPERATIONGRAPH_OPS / HANDLE, ENGINEHEUR_OPERATION_GRAPH / MODE, ENGINEHEUR_RESULTS, ENGINECFG_ENGINE, EXECUTION_PLAN_ENGINE_CONFIG / HANDLE, VARIANT_PACK_UNIQUE_IDS / DATA_POINTERS / WORKSPACE)
- [ ] 1.3 Add `cudnnBackendAttributeType_t` enum (HANDLE, INT64, FLOAT, DATA_TYPE, BACKEND_DESCRIPTOR, etc.)
- [ ] 1.4 Add CUDNN data-type variants: `CUDNN_DATA_FP8_E4M3`, `CUDNN_DATA_FP8_E5M2`
- [ ] 1.5 Add extern declarations: `cudnnBackendCreateDescriptor`, `cudnnBackendDestroyDescriptor`, `cudnnBackendSetAttribute`, `cudnnBackendGetAttribute`, `cudnnBackendFinalize`, `cudnnBackendExecute`

## 2. cuDNN backend RAII wrappers

- [ ] 2.1 Create `onnx-rt/src/cuda/backend.rs` module
- [ ] 2.2 Add `pub struct BackendDesc { desc: cudnnBackendDescriptor_t }` with `new(type) -> Result<Self, CudaError>` (calls Create+Set+Finalize), `Drop` calls Destroy
- [ ] 2.3 Add a builder pattern for setting attributes before finalize: `BackendDescBuilder::new(type).set_int64(...).set_data_type(...).finalize()`
- [ ] 2.4 Wire `pub mod backend;` in `onnx-rt/src/cuda/mod.rs`

## 3. Quantization helpers

- [ ] 3.1 Create `onnx-rt/src/quantize.rs` module
- [ ] 3.2 Implement `pub fn fp32_to_fp8_e4m3(x: f32) -> u8` per the OCP FP8 (E4M3FN) spec — clamp to ±448.0, round to nearest, encode sign+exp+mant
- [ ] 3.3 Implement `pub fn fp32_to_fp8_e5m2(x: f32) -> u8` — clamp to ±57344.0, round to nearest
- [ ] 3.4 Implement `pub fn fp8_e4m3_to_fp32(b: u8) -> f32` (for round-trip tests)
- [ ] 3.5 Implement `pub fn fp8_e5m2_to_fp32(b: u8) -> f32`
- [ ] 3.6 Implement `pub fn quantize_tensor_per_tensor_e4m3(t: &Tensor) -> (Tensor, f32)` — returns FP8 tensor + per-tensor scale; tensor dtype is `DataType::Float8E4M3`
- [ ] 3.7 Implement `quantize_tensor_per_tensor_e5m2`
- [ ] 3.8 Wire `pub mod quantize;` in `onnx-rt/src/lib.rs`
- [ ] 3.9 Unit tests: round-trip error bounds (E4M3 < 5%, E5M2 < 12%); zero handling; subnormal handling for E4M3; bit-pattern correctness against the OCP reference

## 4. ComputePrecision SessionConfig field

- [ ] 4.1 Add `pub enum ComputePrecision { Tf32 (default), Fp16, Bf16, Fp8E4M3, Fp8E5M2 }` in `onnx-rt/src/session.rs`
- [ ] 4.2 Add `pub compute_precision: ComputePrecision` field to `SessionConfig`
- [ ] 4.3 Update `Default for SessionConfig`
- [ ] 4.4 Add internal mapping `ComputePrecision -> GpuPrecision` (Tf32→Tf32, Fp16→F16, Bf16→F32 (BF16 not yet a runtime-init mode), Fp8*→F32 (FP8 routes through dedicated path))
- [ ] 4.5 Update SessionConfig literals in `tests/integration_inference.rs` etc.

## 5. FP8-aware device-initializer cache

- [ ] 5.1 Extend the per-Session device cache type to also store an `Option<f32>` per name representing the per-tensor scale (None for non-quantized tensors)
- [ ] 5.2 In the lazy-init code path in `Session::run`, branch on `compute_precision`: for FP8 modes, run weight tensors through `quantize_tensor_per_tensor_e4m3` / `_e5m2` before uploading to device; cache the (FP8 tensor, scale) tuple
- [ ] 5.3 Skip quantization for non-Conv/Gemm/MatMul initializers (BN params, shape tensors stay f32 / int64)
- [ ] 5.4 Add a helper `is_quantizable_initializer(name: &str, graph: &ExecutionGraph) -> bool` that checks whether the initializer feeds into a Conv/Gemm/MatMul node

## 6. FP8 Conv kernel

- [ ] 6.1 Create `onnx-rt/src/cuda/fp8.rs` module (or extend `gpu_executor.rs`)
- [ ] 6.2 Implement `pub fn gpu_conv2d_device_fp8(rt, x: &DeviceTensor, w_fp8: &DeviceTensor, w_scale: f32, bias, pads, strides, dilations, group, fp8_mode: Fp8Mode) -> Result<DeviceTensor, CudaError>` using the cuDNN backend API
- [ ] 6.3 Build a 5-descriptor graph: input tensor (FP8), filter tensor (FP8), output tensor (FP32), conv operation, operation graph; finalize → execution plan → variant pack with device pointers; execute
- [ ] 6.4 Apply weight scale via the operation descriptor's `CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA` (set to `w_scale`)
- [ ] 6.5 Add a similar `gpu_gemm_device_fp8` for Gemm/MatMul (cuBLASLt FP8 GEMM already exists from arm64-gpu-container-v1; just wire it into the per-tensor scale flow)

## 7. Hybrid dispatch routing

- [ ] 7.1 In `executor_hybrid.rs::try_gpu_dispatch`, add a `compute_precision` parameter (sourced from session config; pass through `execute_graph_hybrid`)
- [ ] 7.2 In the "Conv" branch, when `compute_precision` is `Fp8E4M3` or `Fp8E5M2`, look up the FP8 weight + scale from the cache, call `gpu_conv2d_device_fp8`
- [ ] 7.3 If FP8 dispatch returns an error, fall back to `gpu_conv2d_device` (TF32 path) within the same op call; log a single warning per Session
- [ ] 7.4 Same routing for "Gemm" / "MatMul" branches

## 8. Per-op FP8 correctness tests

- [ ] 8.1 `test_gpu_conv2d_fp8e4m3_matches_tf32_within_5e2` in `onnx-rt/tests/test_cuda.rs`: ResNet-style `[1, 64, 56, 56]` Conv, FP8 vs TF32, assert `max_abs_diff < 5e-2`
- [ ] 8.2 `test_gpu_conv2d_fp8e5m2_matches_tf32_within_1e1`
- [ ] 8.3 `test_quantize_tensor_e4m3_round_trip_error_bounds`
- [ ] 8.4 `test_quantize_tensor_e5m2_round_trip_error_bounds`
- [ ] 8.5 `test_gpu_conv2d_fp8_falls_back_on_unsupported_shape`: deliberately pass a shape cuDNN's FP8 backend rejects (e.g., extreme aspect ratio); assert the fallback to TF32 succeeds and output is correct

## 9. End-to-end FP8 vision benchmarks

- [ ] 9.1 Add `BenchMode::HybridFp8E4M3` variant to `bench_vision_models.rs`
- [ ] 9.2 Add `bench_resnet50_cpu_vs_gpu_hybrid_fp8e4m3`
- [ ] 9.3 Add `bench_squeezenet_cpu_vs_gpu_hybrid_fp8e4m3` and `bench_mobilenet_v2_cpu_vs_gpu_hybrid_fp8e4m3`
- [ ] 9.4 Each bench reports: latency, speedup vs TF32 hybrid baseline, `max_abs_diff` vs CPU reference
- [ ] 9.5 Run all on DGX Spark; record results in `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md`
- [ ] 9.6 Verify ResNet-50 ≥1.5× speedup vs TF32 hybrid; `max_abs_diff < 5e-2`

## 10. Cumulative-error chain test

- [ ] 10.1 Add `test_fp8_five_conv_chain_max_abs_diff_under_quarter` — runs a synthetic 5-Conv chain (no BN/Relu in between to maximize error accumulation) in both TF32 and FP8 E4M3 modes
- [ ] 10.2 Assert `max_abs_diff < 0.25` (5× the per-op tolerance)

## 11. Documentation

- [ ] 11.1 Add an "FP8 Inference" section to `docs/architecture.md` covering ComputePrecision, weight quantization, activation boundary behavior
- [ ] 11.2 Update `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` with FP8 results
- [ ] 11.3 Add `compute_precision` to the SessionConfig doc-comments
- [ ] 11.4 Note in the doc that FP8 accuracy is per-tensor; per-channel is a follow-up

## 12. Final verification

- [ ] 12.1 `cargo fmt -p smallaios-onnx-rt`
- [ ] 12.2 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cpu -- -D warnings`
- [ ] 12.3 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cuda -- -D warnings`
- [ ] 12.4 `cargo test -p smallaios-onnx-rt --no-default-features --features cpu` — full suite green
- [ ] 12.5 `cargo test -p smallaios-onnx-rt --lib --no-default-features --features cuda` — full suite green
- [ ] 12.6 `cargo test -p smallaios-onnx-rt --release --test bench_vision_models --features cuda -- --ignored --nocapture` — all FP8 benches meet target
- [ ] 12.7 `openspec validate fp8-vision-inference-v1 --strict` passes
