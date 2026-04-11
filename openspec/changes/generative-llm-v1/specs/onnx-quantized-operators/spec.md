## ADDED Requirements

### Requirement: Real Int8 GEMM Kernel
The ONNX runtime SHALL implement `op_qlinear_matmul` and `op_qlinear_conv` using a tiled int8 kernel that accumulates in `i32`, folds zero-points at the edges, and saturates on store, rather than the dequantize-compute-requantize approximation shipped in the Tier 2 quantized operators.

#### Scenario: QLinearMatMul output matches ORT reference within 1 quantized step
- **WHEN** a `QLinearMatMul` node is dispatched on two input tensors whose element types are any combination of `i8` and `u8` (per the ONNX QLinearMatMul spec), with known scales and zero-points, producing either an `i8` or `u8` output as declared by the node
- **THEN** the output MUST be bit-equivalent to the reference Python `onnxruntime` implementation within ±1 in the quantized integer domain (i.e. every output element differs by at most 1 quantized step from the reference)
- **AND** the result MUST NOT round-trip through `f32` during the inner product reduction
- **AND** accumulation MUST use signed `i32` storage whose capacity is dimensioned against the worst-case per-element product for the actual input dtype (signed `i8` worst case is `i8::MIN * i8::MIN = 16384`, supporting `K ≤ 131_071`; unsigned `u8` after zero-point folding operates on centered `i8`-equivalent values and thus shares the same bound)

#### Scenario: Zero-point correction is folded out of the hot loop
- **WHEN** the real int8 kernel computes an inner product
- **THEN** the per-element zero-point subtraction MUST NOT appear inside the `k` reduction loop
- **AND** zero-point corrections MUST be precomputed as row-sum and column-sum terms applied at the edges

#### Scenario: Output saturates to the declared output dtype range
- **WHEN** a `QLinearMatMul` inner product produces an `i32` value outside the declared output dtype's representable range after the final scale-multiply
- **THEN** the kernel MUST clamp the result to that output dtype's range before writing the output buffer — `[-128, 127]` for an `i8` output and `[0, 255]` for a `u8` output
- **AND** MUST NOT wrap around

#### Scenario: QLinearConv reuses the real int8 GEMM
- **WHEN** a `QLinearConv` node is dispatched
- **THEN** its im2col-plus-GEMM path MUST invoke the real int8 GEMM kernel
- **AND** MUST NOT fall back to the dequant-compute-requant shim
- **AND** output correctness MUST match an `onnxruntime` reference within ±1 in the quantized integer domain for both `i8` and `u8` output types

#### Scenario: Quantized LLM end-to-end within 1% relative error
- **WHEN** a quantized LLM (INT8 weights) is loaded and run with the real int8 kernel
- **THEN** the generated logits MUST be within 1% relative error of the same model run in f32
- **AND** the measured inference latency MUST be faster than the equivalent f32 run on the same hardware
