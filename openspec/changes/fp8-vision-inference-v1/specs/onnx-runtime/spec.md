## ADDED Requirements

### Requirement: ComputePrecision Configuration
The runtime SHALL expose a `SessionConfig::compute_precision: ComputePrecision` field with variants `Tf32` (default), `Fp16`, `Bf16`, `Fp8E4M3`, and `Fp8E5M2`. The selected mode SHALL determine which numerical precision is used for Conv / Gemm / MatMul operations dispatched on the CUDA execution provider. Default behavior MUST be byte-for-byte identical to the pre-`fp8-vision-inference-v1` execution path.

#### Scenario: Default precision matches existing TF32 behavior
- **WHEN** `SessionConfig::compute_precision` is not set (defaults to `ComputePrecision::Tf32`) and a hybrid inference is run
- **THEN** the GPU dispatch behavior MUST match the pre-change `gpu-resident-vision-hybrid-v1` execution byte-for-byte (same kernels, same algos, same numerical outputs)

#### Scenario: Fp8E4M3 selects the FP8 dispatch path for Conv/Gemm/MatMul
- **WHEN** `compute_precision = ComputePrecision::Fp8E4M3` is set on a session and a Conv operator is dispatched in hybrid mode
- **THEN** the runtime MUST route the operator to `gpu_conv2d_device_fp8` instead of `gpu_conv2d_device`
- **AND** MUST use the FP8 E4M3-quantized version of the Conv's weight tensor from the device-initializer cache
- **AND** MUST configure the cuDNN backend descriptor with `CUDNN_DATA_FP8_E4M3` for input + weight tensors and `CUDNN_DATA_FLOAT` for output

#### Scenario: Fp8E5M2 selects the higher-range FP8 mode
- **WHEN** `compute_precision = ComputePrecision::Fp8E5M2` is set
- **THEN** the runtime MUST use `CUDNN_DATA_FP8_E5M2` for input + weight tensors
- **AND** the corresponding weight quantization helper `quantize_tensor_per_tensor_e5m2` MUST be used at session initialization

### Requirement: One-Shot FP8 Weight Quantization
When the session is configured for an FP8 `compute_precision`, the runtime SHALL quantize every Conv / Gemm / MatMul weight tensor from f32 to FP8 exactly once at session initialization (or first hybrid `run()` call). The quantized FP8 tensors and per-tensor scales SHALL be cached in the existing `device_initializer_cache` and reused across all subsequent inferences.

#### Scenario: Per-tensor max-abs scaling computes a scale factor
- **WHEN** quantizing an f32 weight tensor `W` to FP8 E4M3
- **THEN** the runtime MUST compute `scale = max_abs(W) / FP8_E4M3_MAX` (where `FP8_E4M3_MAX = 448.0`)
- **AND** MUST encode each weight as `f32_to_fp8_e4m3(w / scale)` (clamped to the representable range)
- **AND** MUST persist `(quantized_tensor, scale)` in the device cache

#### Scenario: Quantized weights are reused across inferences
- **WHEN** a session has been initialized with FP8 precision and `Session::run` is called twice in succession with different inputs
- **THEN** the runtime MUST NOT re-quantize the weight tensors on the second call
- **AND** MUST issue a single device-resident FP8 weight buffer reused for both inferences

#### Scenario: Quantization is rejected for unsupported types
- **WHEN** an initializer tensor has a dtype other than `Float` or `BFloat16`
- **THEN** the FP8 quantization helper MUST skip the tensor (no quantization needed for int/shape tensors)
- **AND** the cache MUST hold the tensor in its original dtype for use by non-quantized ops (Reshape, Gather, etc.)

### Requirement: FP8 Inference Speedup Target
On DGX Spark with `GpuResidency::Hybrid` and `compute_precision = Fp8E4M3`, the runtime SHALL deliver a measurable inference latency reduction relative to the TF32 hybrid baseline.

#### Scenario: ResNet-50 FP8 hybrid hits speedup target
- **WHEN** the `bench_resnet50_cpu_vs_gpu_hybrid_fp8e4m3` benchmark is run on DGX Spark
- **THEN** the GPU mean latency MUST be at least 1.5× lower than the TF32 hybrid baseline
- **AND** the output `max_abs_diff` against the CPU reference MUST remain below `5e-2`

#### Scenario: FP8 dispatch falls back gracefully on unsupported shape
- **WHEN** the cuDNN backend descriptor finalization or execution returns a non-success status for a particular Conv shape under FP8
- **THEN** the runtime MUST fall back to the TF32 dispatch path for that op
- **AND** MUST log a single warning per Session indicating that FP8 dispatch fell back, and naming the offending op
- **AND** MUST NOT propagate the FP8 failure as an error from `Session::run` — the inference completes via TF32

### Requirement: FP8 Activation Boundary Behavior
The runtime SHALL keep activations in f32 (or bf16 / TF32) between ops in FP8 mode. Only the Conv / Gemm / MatMul kernel itself SHALL operate on FP8 inputs internally; outputs SHALL be returned in `Float` dtype for the next CPU or GPU operator to consume.

#### Scenario: BatchNorm following an FP8 Conv reads f32 activations
- **WHEN** a graph has the pattern `Conv(FP8) → BatchNormalization` and runs in hybrid mode with `compute_precision = Fp8E4M3`
- **THEN** the Conv MUST produce a `DeviceTensor` with `dtype = DataType::Float`
- **AND** the BatchNormalization MUST consume that f32 tensor unchanged (no explicit dequantize op)

#### Scenario: Quantization errors do not accumulate beyond per-Conv tolerance
- **WHEN** a five-Conv chain runs in FP8 mode
- **THEN** the cumulative `max_abs_diff` between the FP8 chain output and the TF32 reference MUST NOT exceed 5× the per-op FP8 tolerance (i.e. `max_abs_diff < 0.25` for an E4M3 chain of length 5)
- **AND** the test for this scenario MUST exercise an actual five-Conv subgraph from one of the vision benchmarks
