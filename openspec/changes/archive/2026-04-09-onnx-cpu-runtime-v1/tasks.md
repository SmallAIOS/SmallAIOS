## 1. Graph Executor Infrastructure

- [x] 1.1 Create `onnx-rt/src/executor.rs` with `execute_graph()` function signature: takes `&ExecutionGraph`, input tensors, initializers, optional yield callback, profiling flag; returns `Result<Vec<Tensor>, SessionError>`
- [x] 1.2 Implement tensor value map (`BTreeMap<String, Tensor>`) initialization: load graph inputs and initializer tensors by name
- [x] 1.3 Implement topological-order iteration loop: for each `NodeIndex` in `execution_order()`, resolve input tensors from map, dispatch operator, store outputs in map
- [x] 1.4 Implement output extraction: after graph execution, collect output tensors by name from the value map and return in model output order
- [x] 1.5 Wire `Session::run()` in `session.rs` to call `executor::execute_graph()` instead of returning `NotImplemented`

## 2. Operator Dispatch

- [x] 2.1 Implement `dispatch_node()` function: match `OpKind::parse_str(&node.op_type)` to the corresponding `op_*` function, passing input tensors and attributes
- [x] 2.2 Extend `ExecutionNode` with `attributes: Vec<AttributeProto>` field (reuses existing ONNX AttributeProto type)
- [x] 2.3 Propagate attributes from `NodeProto` to `ExecutionNode` during graph construction in `graph.rs`
- [x] 2.4 Wire attribute access into existing operators that need them: `op_conv` (pads, strides, kernel_shape), `op_softmax` (axis)

## 3. Tier 1 Operators — Core Math

- [x] 3.1 Implement `op_sub`: element-wise subtraction with broadcasting (mirror `op_add` pattern)
- [x] 3.2 Implement `op_mul`: element-wise multiplication with broadcasting
- [x] 3.3 Implement `op_div`: element-wise division with broadcasting (handle division by zero → f32::INFINITY)
- [x] 3.4 Implement `op_gemm`: General Matrix Multiply wrapping `gemm_f32` with `alpha`, `beta`, `transA`, `transB` attributes and optional bias C
- [x] 3.5 Unit tests for Sub, Mul, Div, Gemm with known inputs/outputs including broadcast cases

## 4. Tier 2 Operators — Activations

- [x] 4.1 Implement `op_sigmoid`: element-wise `1 / (1 + expf_approx(-x))` for f32 tensors
- [x] 4.2 Implement `op_tanh`: element-wise `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` using `expf_approx`
- [x] 4.3 Unit tests for Sigmoid and Tanh with known values (0, 1, -1, large positive/negative)

## 5. Tier 3 Operators — Shape and Data Movement

- [x] 5.1 Implement `op_transpose`: permute tensor dimensions according to `perm` attribute
- [x] 5.2 Implement `op_concat`: concatenate tensors along specified `axis`
- [x] 5.3 Implement `op_flatten`: reshape tensor to 2D at given `axis`
- [x] 5.4 Implement `op_squeeze`: remove dimensions of size 1 at specified axes
- [x] 5.5 Implement `op_unsqueeze`: insert dimensions of size 1 at specified axes
- [x] 5.6 Implement `op_cast`: type conversion between f32, int32, int64, int8 (reinterpret tensor data buffer with new DataType)
- [x] 5.7 Implement `op_gather`: gather elements along axis using index tensor
- [x] 5.8 Implement `op_slice`: extract sub-tensor with starts, ends, axes, steps
- [x] 5.9 Implement `op_pad`: pad tensor with constant, reflect, or edge modes
- [x] 5.10 Implement `op_clip`: clamp tensor values to [min, max] range
- [x] 5.11 Unit tests for all Tier 3 operators with edge cases (empty slices, negative axes, identity transposes)

## 6. Tier 4 Operators — Normalization and Pooling

- [x] 6.1 Implement `op_batch_normalization`: normalize using mean/variance/scale/bias with `epsilon` attribute
- [x] 6.2 Implement `op_layer_normalization`: normalize along last N dimensions with scale and optional bias
- [x] 6.3 Implement `op_maxpool`: sliding window max over NCHW input with `kernel_shape`, `strides`, `pads`
- [x] 6.4 Implement `op_averagepool`: sliding window mean over NCHW input with same attributes as MaxPool
- [x] 6.5 Implement `op_global_average_pool`: reduce spatial dims (H, W) to 1x1 via mean
- [x] 6.6 Implement `op_reduce_mean`: reduce along specified axes with `keepdims` attribute
- [x] 6.7 Implement `op_reduce_sum`: reduce along specified axes with `keepdims` attribute
- [x] 6.8 Unit tests for all Tier 4 operators with known-good reference values

## 7. Scheduler Integration

- [x] 7.1 Add `yield_fn: Option<fn()>` parameter to `execute_graph()` and call it after each operator dispatch
- [x] 7.2 Add `enable_profiling: bool` to `Session` configuration (already exists in SessionConfig)
- [ ] 7.3 Implement per-operator timing: measure ticks before/after dispatch, compare against `OperatorBudget` thresholds (soft warning, hard abort at 10x)
- [ ] 7.4 Wire timing behind `#[cfg]` gate so timing overhead is zero when profiling is disabled

## 8. End-to-End Testing

- [x] 8.1 Create integration test: build a small ONNX-like graph programmatically (MatMul → Add → Relu), run through executor, verify output values
- [ ] 8.2 Create integration test: multi-branch graph (input → [MatMul, Add] → Concat → Softmax), verify correct routing and output
- [x] 8.3 Create integration test: graph with initializers (model weights as constants), verify weights are loaded and used correctly
- [ ] 8.4 Verify `just test` passes with all new code; run `just clippy` and `just fmt-check`
