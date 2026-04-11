## Context

Current operator coverage in `onnx-rt/src/operators.rs`:

| Tier | Status | Operators |
|------|--------|-----------|
| 1 | **Done (29)** | Add, Sub, Mul, Div, MatMul, Gemm, Relu, Sigmoid, Tanh, Softmax, Conv, BatchNorm, LayerNorm, MaxPool, AvgPool, GlobalAvgPool, ReduceSum, ReduceMean, Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Cast, Gather, Slice, Pad, Clip |
| 2 | This change | LSTM, GRU, GELU, Pow, Sqrt, Exp, Log, Erf, Equal, Where, Einsum, Split, Expand, Tile, OneHot, QuantizeLinear, DequantizeLinear, QLinearMatMul, QLinearConv, etc. |
| 3 | Future | DFT, STFT, NMS, RoiAlign, Loop, If, Scan |

The hot path for adding operators is mechanical: implement `pub fn op_X(...) -> Result<Tensor, OpError>`, add to `OpKind` enum, add to dispatch helper match. The `byte_io::read_f32` / `write_f32` helpers handle data conversion. Every operator follows the same pattern.

The interesting design questions are:
1. **LSTM/GRU state management** — RNNs maintain hidden state across timesteps
2. **Quantized integer math** — different from f32 operators
3. **Einsum** — needs a parser for the equation string

## Goals / Non-Goals

**Goals:**
- Cover the operators needed by BERT, GPT-2-small, Whisper-tiny, quantized MobileNet/ResNet
- INT8 quantized inference path (smaller models, edge deployment)
- All operators behave correctly per ONNX spec
- Tests: at least 2 per operator (basic + edge case)
- All operators work in both sequential and (where applicable) parallel modes

**Non-Goals:**
- Performance optimization beyond correctness (future work)
- INT4 quantization (not in standard ONNX yet)
- Sparse tensors
- Control flow ops (Loop, If, Scan) — separate change
- Custom ops or external operator libraries

## Decisions

### D1: One Module Per Operator Group

Rather than appending 30 functions to the already-large `operators.rs`, split into logical groups:

```
onnx-rt/src/
├── operators.rs                  (existing — Tier 1)
├── ops/
│   ├── mod.rs                    (re-exports)
│   ├── math.rs                   (Pow, Sqrt, Exp, Log, Erf, Neg, Abs, Floor, Ceil, Round)
│   ├── compare.rs                (Equal, Less, Greater, Where, Min, Max, Not)
│   ├── activations.rs            (Gelu, LeakyRelu, Elu, Swish)
│   ├── recurrent.rs              (LSTM, GRU, RNN)
│   ├── transformer.rs            (Einsum, Split, Expand, Tile, OneHot, ScaledDotProductAttention)
│   └── quantized.rs              (QuantizeLinear, DequantizeLinear, QLinearMatMul, QLinearConv)
```

This keeps the codebase navigable. The existing `operators.rs` stays as-is — new ops go in `ops/`.

### D2: LSTM Hidden State as Output Tensor

ONNX LSTM returns `(Y, Y_h, Y_c)` where Y is the output sequence, Y_h is the final hidden state, Y_c is the final cell state. Our `op_lstm` returns a `Vec<Tensor>` with all three outputs. The session executor already supports multi-output operators (Y, Y_h, Y_c → 3 names in `node.outputs`).

### D3: INT8 Quantization Math

Quantized ops use the formula:
```
quantized = round(float / scale) + zero_point  // clipped to [0, 255] for u8 or [-128, 127] for i8
float = (quantized - zero_point) * scale
```

The `QLinearMatMul` operator does:
```
output = (a - a_zero) * a_scale @ (b - b_zero) * b_scale / output_scale + output_zero
```
This is implemented as integer math (i32 accumulator) for speed, then dequantized at the boundary. For the initial implementation, we can use a simpler "dequantize-compute-requantize" approach that just calls existing `op_matmul` after dequantization, then quantizes the result. This is correct but slower; real INT8 acceleration is a future change.

### D4: Einsum Parser

Einsum uses a string equation like `"bij,bjk->bik"` (batched matmul). For the initial implementation:
1. Parse the equation into input subscripts and output subscript
2. Identify the contraction axes (subscripts in inputs but not output)
3. Implement common cases (matmul, dot product, batched matmul) directly
4. Fall back to a generic loop-based contraction for other cases

We don't need to support every possible einsum expression — just the ones used by transformer models (typically `bhij,bhjk->bhik` for attention).

### D5: GELU via Polynomial Approximation

GELU is `x * Phi(x)` where Phi is the standard normal CDF. The exact form uses `erf`:
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

We need an `erf` implementation. The Abramowitz & Stegun polynomial approximation is `~1.5e-7` accurate, which is enough for inference:
```
erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
where t = 1 / (1 + p*x), p = 0.3275911
a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429
```

This is `no_std` compatible and uses our existing `expf_approx`.

## Risks / Trade-offs

**[Risk] LSTM correctness on long sequences** — RNNs accumulate numerical error. Mitigation: Test against known reference outputs from PyTorch/ONNX Runtime for short sequences (length 5-10). Long-sequence accuracy can be validated later with real model tests.

**[Risk] Quantization precision loss** — INT8 inference is inherently lossy. Mitigation: Test accuracy on a quantized MobileNet against the f32 reference; expect 1-2% top-1 accuracy drop, which is acceptable.

**[Risk] Code volume** — ~2,000 lines is significant. Mitigation: Split into modules (D1), each module gets its own agent for parallel implementation. Each operator is independent.

**[Trade-off] Initial INT8 is slow (dequantize-compute-requantize)** — Real INT8 needs i8 GEMM kernels. Acceptable for v0.3 — quantized models load and run, even if not at full speed. Optimization is a future change.

## Open Questions

- **Q1:** Should `op_lstm` support `forget_bias` parameter (some old models)? *Leaning toward: yes, default 1.0*
- **Q2:** Should we add `op_einsum` or just the specific cases needed by attention? *Leaning toward: einsum with common-case fast paths, fallback for others*
