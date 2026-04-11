## ADDED Requirements

### Requirement: Real Int8 GEMM Kernel
The ONNX runtime SHALL implement `op_qlinear_matmul` and `op_qlinear_conv` using a tiled int8 kernel that accumulates in `i32`, folds zero-points at the edges, and saturates on store, rather than the dequantize-compute-requantize approximation shipped in the Tier 2 quantized operators.

#### Scenario: QLinearMatMul output matches ORT reference within 1 ULP
- **WHEN** a `QLinearMatMul` node is dispatched on two i8 tensors with known scales and zero-points
- **THEN** the output MUST be bit-equivalent to the reference Python `onnxruntime` implementation within ±1 ULP
- **AND** the result MUST NOT round-trip through `f32` during the inner product reduction
- **AND** accumulation MUST use `i32` storage that cannot overflow for `K ≤ 131071`

#### Scenario: Zero-point correction is folded out of the hot loop
- **WHEN** the real int8 kernel computes an inner product
- **THEN** the per-element zero-point subtraction MUST NOT appear inside the `k` reduction loop
- **AND** zero-point corrections MUST be precomputed as row-sum and column-sum terms applied at the edges

#### Scenario: Output saturates to i8 range
- **WHEN** a `QLinearMatMul` inner product produces an `i32` value outside `[i8::MIN, i8::MAX]` after the final scale-multiply
- **THEN** the kernel MUST clamp the result to the `i8` range before writing the output buffer
- **AND** MUST NOT wrap around

#### Scenario: QLinearConv reuses the real int8 GEMM
- **WHEN** a `QLinearConv` node is dispatched
- **THEN** its im2col-plus-GEMM path MUST invoke the real int8 GEMM kernel
- **AND** MUST NOT fall back to the dequant-compute-requant shim
- **AND** output correctness MUST match an `onnxruntime` reference within ±1 ULP

#### Scenario: Quantized LLM end-to-end within 1% relative error
- **WHEN** a quantized LLM (INT8 weights) is loaded and run with the real int8 kernel
- **THEN** the generated logits MUST be within 1% relative error of the same model run in f32
- **AND** the measured inference latency MUST be faster than the equivalent f32 run on the same hardware
