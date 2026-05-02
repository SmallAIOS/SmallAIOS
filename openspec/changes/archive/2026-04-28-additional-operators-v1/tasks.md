## 1. Module Structure

- [x] 1.1 Create `onnx-rt/src/ops/` directory
- [x] 1.2 Create `onnx-rt/src/ops/mod.rs` with re-exports
- [x] 1.3 Register the new module in `onnx-rt/src/lib.rs`
- [x] 1.4 Decide whether to keep existing `operators.rs` as-is or move Tier 1 ops into `ops/tier1/` (initial answer: keep as-is for diff hygiene)

## 2. Math Primitives (ops/math.rs)

- [x] 2.1 Implement `op_pow(base, exponent)` — element-wise with broadcasting
- [x] 2.2 Implement `op_sqrt(input)` — element-wise sqrt (use sqrt_approx)
- [x] 2.3 Implement `op_exp(input)` — element-wise exp (use expf_approx)
- [x] 2.4 Implement `op_log(input)` — element-wise natural log
- [x] 2.5 Implement `op_erf(input)` — error function via Abramowitz & Stegun polynomial
- [x] 2.6 Implement `op_neg(input)`, `op_abs(input)`, `op_floor`, `op_ceil`, `op_round`
- [x] 2.7 Unit tests for each math op (basic, edge cases, special values)

## 3. Comparison and Selection (ops/compare.rs)

- [x] 3.1 Implement `op_equal`, `op_not_equal`, `op_less`, `op_less_or_equal`, `op_greater`, `op_greater_or_equal`
- [x] 3.2 Implement `op_where(cond, x, y)` — element-wise select
- [x] 3.3 Implement `op_min`, `op_max` — element-wise with broadcasting
- [x] 3.4 Implement `op_not(bool_input)`
- [x] 3.5 Unit tests for each comparison op

## 4. Composite Activations (ops/activations.rs)

- [x] 4.1 Implement `op_gelu(input)` using `op_erf` and the formula `0.5 * x * (1 + erf(x/sqrt(2)))`
- [x] 4.2 Implement `op_leaky_relu(input, alpha)` 
- [x] 4.3 Implement `op_elu(input, alpha)`
- [x] 4.4 Implement `op_swish(input)` (also called silu): `x * sigmoid(x)`
- [x] 4.5 Unit tests with reference values from PyTorch

## 5. Recurrent (ops/recurrent.rs)

- [x] 5.1 Implement `op_rnn(x, w, r, b, h0)` — basic RNN with tanh
- [x] 5.2 Implement `op_lstm(x, w, r, b, h0, c0)` — full LSTM with i/f/g/o gates
- [x] 5.3 Implement `op_gru(x, w, r, b, h0)` — GRU with reset/update gates
- [x] 5.4 Bidirectional support: forward + reverse pass, concat outputs
- [x] 5.5 Unit tests against known reference outputs (small sequences, hand-computed)

## 6. Transformer Building Blocks (ops/transformer.rs)

- [x] 6.1 Implement `op_split(input, axis, num_outputs_or_sizes)` returning `Vec<Tensor>`
- [x] 6.2 Implement `op_expand(input, target_shape)` with broadcasting
- [x] 6.3 Implement `op_tile(input, repeats)`
- [x] 6.4 Implement `op_one_hot(indices, depth, on_value, off_value)`
- [x] 6.5 Implement `op_einsum(equation, inputs)` — parse equation, dispatch to matmul/dot/contraction
- [x] 6.6 Unit tests, especially for einsum with `bij,bjk->bik` and attention patterns

## 7. Quantized (ops/quantized.rs)

- [x] 7.1 Implement `op_quantize_linear(input, scale, zero_point)` — float → int8/uint8
- [x] 7.2 Implement `op_dequantize_linear(input, scale, zero_point)` — int8/uint8 → float
- [x] 7.3 Implement `op_qlinear_matmul` — initial implementation: dequantize, matmul, requantize
- [x] 7.4 Implement `op_qlinear_conv` — same approach
- [x] 7.5 Unit tests: round-trip quantize/dequantize, qlinear matmul accuracy vs f32

## 8. OpKind and Registry Updates

- [x] 8.1 Add new variants to `OpKind` enum (~30 entries)
- [x] 8.2 Update `OpKind::parse_str` to handle new operator names
- [x] 8.3 Update `OperatorRegistry` to include the new ops
- [x] 8.4 Update `classify_op` in profile.rs to assign new ops to budget categories

## 9. Executor Dispatch

- [x] 9.1 Add new ops to the appropriate `dispatch_*` helper in `executor.rs`
- [x] 9.2 Math/comparison/activations → `dispatch_arithmetic` or new `dispatch_math` helper
- [x] 9.3 LSTM/GRU/RNN → new `dispatch_recurrent` helper (multi-output)
- [x] 9.4 Transformer ops → new `dispatch_transformer` helper
- [x] 9.5 Quantized ops → new `dispatch_quantized` helper

## 10. Validation

- [x] 10.1 `just fmt` clean
- [x] 10.2 `just clippy --all-targets` clean
- [x] 10.3 `just test` all passing
- [x] 10.4 New test count: at least 60 unit tests across all new operators
- [x] 10.5 Update `docs/scheduling-model.md` operator class table if needed
