## 1. Sub-Graph Executor Infrastructure

- [x] 1.1 Create `onnx-rt/src/sub_executor.rs` with the module skeleton, doc comments, and public API: `run_sub_graph(compiled: &ExecutionGraph, outer_refs: &[(String, Tensor)], initializers: &[TensorProto], carried: Vec<Tensor>, ...) -> Result<Vec<Tensor>, SessionError>`
- [x] 1.2 Extend `ExecutionNode` with an optional `sub_graphs: Vec<ExecutionGraph>` field to carry compiled inner graphs (If uses 2 slots, Loop uses 1, Scan uses 1)
- [x] 1.3 Extend the graph builder in `graph.rs` to walk `AttributeProto::graph` for `If` / `Loop` / `Scan` nodes at load time, recursively compile each inner `GraphProto` into an `ExecutionGraph`, and attach the compiled graph to the parent `ExecutionNode.sub_graphs`
- [x] 1.4 Implement the sub-graph value-map construction helper that seeds a fresh `BTreeMap<String, Tensor>` from (a) a caller-supplied carried-value list and (b) a precomputed outer-refs list, while passing initializers through by reference
- [x] 1.5 Implement the recursion boundary: `run_sub_graph` internally calls `execute_graph` on the compiled inner graph and extracts outputs by declared name
- [x] 1.6 Implement the value-map reuse optimization (Open Question Q1, option c): reuse a single inner value map across iterations by clearing it between iterations
- [x] 1.7 Hook sub-graph budget accounting into `profile.rs`: `Loop` / `If` / `Scan` are single-unit entries in `InferenceProfile.operators`; inner operators land in the same vector with a parent-path annotation
- [x] 1.8 Bubble inner hard-limit errors up to the parent operator unchanged
- [x] 1.9 Safety net: compile-time `MAX_LOOP_ITERATIONS = 1_000_000` hard cap independent of per-operator budget
- [x] 1.10 Unit test: compile a simple inner graph at load time and verify it is cached on `ExecutionNode.sub_graphs`
- [x] 1.11 Unit test: run_sub_graph with a trivial `Identity` body correctly returns its input
- [x] 1.12 Unit test: nested `If` inside `Loop` body runs to completion with correct outputs
- [x] 1.13 Unit test: inner hard-limit aborts the parent `Loop` cleanly
- [x] 1.14 Unit test: value-map reuse across iterations does not leak state between iterations

## 2. Control-Flow Operators

- [x] 2.1 Create `onnx-rt/src/ops/control_flow.rs` module skeleton
- [x] 2.2 Implement `op_if`: inspect condition tensor, select then/else sub-graph, invoke `sub_executor::run_sub_graph`, return branch outputs
- [x] 2.3 Implement `op_loop`: parse `M`, `cond`, `v_initial` from node inputs; run iteration loop per D4; dispatch inner body via `sub_executor::run_sub_graph` per iteration; collect `v_final`
- [x] 2.4 Implement `op_scan`: iterate the scan-input axis; invoke body per element; accumulate outputs into a new sequence tensor
- [x] 2.5 Add `OpKind::If`, `OpKind::Loop`, `OpKind::Scan` variants and dispatch cases in `executor::dispatch_node`
- [x] 2.6 Unit tests for `If`: both branches, different output shapes per branch
- [x] 2.7 Unit tests for `Loop`: M-only termination, cond_out early termination, cond=false zero iterations, carried values threaded correctly
- [x] 2.8 Unit tests for `Scan`: constant-add body, multi-element sequence
- [x] 2.9 Integration test: a control-flow node graph runs end-to-end through `Session::run()`

## 3. Generative and Normalization Operators

- [x] 3.1 Create `onnx-rt/src/ops/generative.rs` module skeleton
- [x] 3.2 Implement `op_rms_normalization`: compute `x / sqrt(mean(x^2) + eps) * gamma`
- [x] 3.3 Implement `op_matmul_integer`: `i8 × i8 → i32` matmul with zero-point folding, calls into the real int8 kernel from task 4
- [x] 3.4 Implement `op_dynamic_quantize_linear`: compute per-tensor scale and zero-point, produce u8 output
- [x] 3.5 Implement `op_random_normal` using a deterministic xoshiro256++ PRNG seeded from the `seed` attribute
- [x] 3.6 Implement `op_random_normal_like` delegating shape to the input tensor
- [x] 3.7 Implement `op_random_uniform` with the same PRNG
- [x] 3.8 Implement `op_random_uniform_like` delegating shape to the input tensor
- [x] 3.9 Implement `op_multinomial`: inverse-CDF sampling from a probability vector, seeded
- [x] 3.10 Implement `op_bernoulli`: per-element biased coin flip, seeded
- [x] 3.11 Implement `op_dropout`: inference mode is identity; training mode is out of scope (document and return input unchanged when `training_mode = false`)
- [x] 3.12 Implement `op_eye_like`: identity matrix with the input's shape
- [x] 3.13 Implement `op_reduce_l1`: sum of absolute values along axes
- [x] 3.14 Implement `op_reduce_l2`: square root of sum of squares along axes
- [x] 3.15 Implement `op_reduce_log_sum`: log of sum along axes
- [x] 3.16 Implement `op_reduce_log_sum_exp`: log of sum of exp along axes (with the standard max-shift trick for numerical stability)
- [x] 3.17 Implement `op_reduce_sum_square`: sum of squares along axes
- [x] 3.18 Implement `op_lp_normalization`: normalize by Lp norm along specified axis
- [x] 3.19 Implement `op_mean_variance_normalization`: subtract mean and divide by sqrt(variance)
- [x] 3.20 Implement `op_softplus`: `log(1 + exp(x))` with overflow-safe form
- [x] 3.21 Add `OpKind` variants and dispatch cases for all 19 generative/norm ops
- [x] 3.22 Unit tests for each op with known reference values (one test per op minimum)
- [x] 3.23 Unit test: RMSNormalization matches PyTorch `nn.RMSNorm` within 1e-5
- [x] 3.24 Unit test: RandomUniform is reproducible across two invocations with the same seed

## 4. Real Int8 GEMM Kernel

- [x] 4.1 Design the tiled inner kernel: 8×8 register-blocked MAC into `i32` accumulators
- [x] 4.2 Implement the outer 64×64 cache-blocked loop following the existing `gemm_f32` structure
- [x] 4.3 Implement row-sum and column-sum precomputation for zero-point folding
- [x] 4.4 Implement the final scale-multiply-and-saturate store stage
- [x] 4.5 Replace the existing `op_qlinear_matmul` body (dequant-compute-requant shim) with the real kernel
- [x] 4.6 Update `op_qlinear_conv` to call the real kernel through its im2col path
- [x] 4.7 Unit test: GEMM of two small i8 matrices matches a hand-computed reference exactly
- [x] 4.8 Unit test: zero-point = 0 case matches plain i32 accumulation
- [x] 4.9 Unit test: non-zero zero-point case matches full formula
- [x] 4.10 Unit test: saturation on overflow produces `i8::MAX` / `i8::MIN`, not wrap
- [x] 4.11 Unit test: output within ±1 quantized step of a reference-ORT vector for both i8 and u8 outputs (checked-in test vectors)

## 5. Inventory Updates

- [x] 5.1 Flip `(OpKind::Loop, Planned(Phase::P2))` → `Implemented` in `SUPPORTED_OPS_INVENTORY`
- [x] 5.2 Flip `(OpKind::If, Planned(Phase::P2))` → `Implemented`
- [x] 5.3 Flip `(OpKind::Scan, Planned(Phase::P2))` → `Implemented`
- [x] 5.4 Flip all 19 generative/norm ops from `Planned(Phase::P2)` → `Implemented`
- [x] 5.5 Verify no Phase 2 op remains as `Planned(Phase::P2)` in the inventory

## 6. Validation

- [x] 6.1 `just fmt-check` passes
- [x] 6.2 `just clippy` passes with `-D warnings`
- [x] 6.3 `just test` passes with all new unit tests
- [x] 6.4 End-to-end integration test: load a small GPT-2 (or GPT-2-style) ONNX export, run generation inside a single `Session::run()` call, verify output tokens match a reference Python ORT run
- [x] 6.5 End-to-end integration test: load an INT8 quantized LLM, compare generated logits to an f32 run of the same model, assert within 1% relative error
- [x] 6.6 Latency regression test: quantized int8 LLM inference latency is strictly less than the f32 equivalent on the same hardware
- [x] 6.7 Architecture test: `just arch-check` passes; no new cycles introduced
- [x] 6.8 Coverage gate: `cargo-llvm-cov --fail-under-lines 80` still passes; new runtime code does not regress overall coverage
- [x] 6.9 `openspec validate generative-llm-v1 --strict` passes
