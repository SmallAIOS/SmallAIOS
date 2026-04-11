## 1. Tensor Byte I/O Helpers

- [ ] 1.1 Create `onnx-rt/src/byte_io.rs` with `F32_SIZE`, `I32_SIZE`, `I64_SIZE`, `F64_SIZE` constants
- [ ] 1.2 Add `read_f32`, `write_f32`, `read_i32`, `write_i32`, `read_i64`, `write_i64`, `read_f64`, `write_f64` helpers
- [ ] 1.3 Add `allocate_tensor_data(elements, dtype)` helper
- [ ] 1.4 Register module in `onnx-rt/src/lib.rs`
- [ ] 1.5 Replace 50+ inline `f32::from_le_bytes` calls in `operators.rs` with `byte_io::read_f32`
- [ ] 1.6 Replace 6+ inline calls in `executor.rs`
- [ ] 1.7 Replace `vec![0u8; total * 4]` patterns with `byte_io::allocate_tensor_data`
- [ ] 1.8 Unit tests for all byte_io helpers

## 2. Refactor dispatch_node()

- [ ] 2.1 Extract `dispatch_arithmetic(kind, inputs, attrs)` for Add/Sub/Mul/Div/MatMul/Gemm
- [ ] 2.2 Extract `dispatch_activation(kind, inputs, attrs)` for Relu/Sigmoid/Tanh/Softmax
- [ ] 2.3 Extract `dispatch_shape(kind, inputs, attrs)` for Reshape/Transpose/Flatten/Squeeze/Unsqueeze/Concat/Gather/Slice/Pad/Cast/Clip
- [ ] 2.4 Extract `dispatch_convolution(kind, inputs, attrs)` for Conv
- [ ] 2.5 Extract `dispatch_normalization(kind, inputs, attrs)` for BatchNormalization/LayerNormalization
- [ ] 2.6 Extract `dispatch_pooling(kind, inputs, attrs)` for MaxPool/AveragePool/GlobalAveragePool
- [ ] 2.7 Extract `dispatch_reduction(kind, inputs, attrs)` for ReduceMean/ReduceSum
- [ ] 2.8 Top-level `dispatch_node()` becomes a 7-arm match delegating to category helpers
- [ ] 2.9 Verify all 246 onnx-rt tests still pass

## 3. Refactor op_cast()

- [ ] 3.1 Extract `cast_f32_to_i32`, `cast_i32_to_f32`, `cast_f32_to_i64`, `cast_i64_to_f32` private functions
- [ ] 3.2 Top-level `op_cast()` becomes a small match on `(input.data_type, target)` 
- [ ] 3.3 Reuse `byte_io::read_*` and `byte_io::write_*` helpers
- [ ] 3.4 Verify cast tests still pass

## 4. ConvParams Struct

- [ ] 4.1 Define `ConvParams` struct in `onnx-rt/src/operators.rs`
- [ ] 4.2 Refactor `conv_compute()` to take `ConvParams` (4 args instead of 9)
- [ ] 4.3 Update all call sites of `conv_compute`
- [ ] 4.4 Remove `#[allow(clippy::too_many_arguments)]` annotation
- [ ] 4.5 Verify Conv tests still pass

## 5. Helpers in executor.rs

- [ ] 5.1 Add `read_first_f32(tensor: Option<&Tensor>) -> Option<f32>` helper
- [ ] 5.2 Replace duplicated min_val/max_val/constant_value extraction in dispatch_node
- [ ] 5.3 Verify executor tests still pass

## 6. Named Constants in expf_approx

- [ ] 6.1 Define `EXP_CLAMP_MAX`, `EXP_CLAMP_MIN` constants
- [ ] 6.2 Define `EXP_POLY_C2`, `EXP_POLY_C3`, `EXP_POLY_C4` polynomial coefficients
- [ ] 6.3 Define `F32_EXPONENT_BIAS`, `F32_MANTISSA_BITS` constants
- [ ] 6.4 Update `expf_approx()` body to use the constants
- [ ] 6.5 Add module-level doc comment explaining the polynomial approximation

## 7. Reduce Complexity in op_concat, op_squeeze, op_transpose

- [ ] 7.1 Extract inner copy loop in `op_concat()` into `copy_tensor_to_concat_output()` helper
- [ ] 7.2 Extract axis-matching logic in `op_squeeze()` into `should_squeeze_axis()` helper
- [ ] 7.3 Extract coordinate transformation in `op_transpose()` into helper
- [ ] 7.4 Verify all shape op tests still pass

## 8. DHCP Option Parsing Deduplication

- [ ] 8.1 Add `iter_dhcp_options(data: &[u8]) -> impl Iterator<Item = (u8, &[u8])>` helper
- [ ] 8.2 Refactor `parse_options()` to use the iterator
- [ ] 8.3 Refactor `get_option_value()` to use the iterator
- [ ] 8.4 Verify DHCP tests still pass

## 9. Validation

- [ ] 9.1 Run `just fmt` — clean
- [ ] 9.2 Run `just clippy` — clean
- [ ] 9.3 Run `just test` — all tests pass (1300+ workspace tests)
- [ ] 9.4 Verify cognitive complexity reductions:
  - dispatch_node() < 15
  - op_cast() < 15
  - op_concat() < 12
  - op_squeeze() < 12
- [ ] 9.5 Confirm no behavior changes via integration tests
