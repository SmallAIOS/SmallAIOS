## Why

Phase 1 of the ONNX coverage roadmap (`transformer-models-v1` + `vision-transformers-v1`, PRs #76 and #77) adds 44 operators that unlock encoder-only workloads: BERT, DistilBERT, ViT, DeiT, CLIP image towers. Those models run start-to-finish in a single `Session::run()` call because their dataflow is a fixed DAG.

Generative LLMs are different. GPT-2, LLaMA, Mistral, and every decoder-only transformer in the wild generate tokens in a *loop*: run the graph, sample, feed the sampled token back, repeat until EOS or `max_new_tokens`. There are two ways to support this:

1. **External loop** — host code runs `Session::run()` N times, managing the KV cache and sampling between calls. Works, but (a) every iteration pays full graph-compilation and value-map allocation overhead, (b) the cooperative scheduler sees N independent inferences instead of one WCET-budgeted generation, and (c) quantized graphs pay f32 latency because the current INT8 shim dequantizes-computes-requantizes on every call.
2. **In-graph loop** — the ONNX model embeds its own generation loop via the `Loop` operator. The entire generation becomes a *single* `Session::run()` call. This is how `onnxruntime-genai` and `llama.cpp`'s ONNX exporter ship decoder models today.

Phase 2 commits to option (2). This requires control-flow operators (`If`, `Loop`, `Scan`), a sub-graph executor that can recursively evaluate the bodies of those operators, a handful of generative-model-specific ops (RMSNorm, MatMulInteger, DynamicQuantizeLinear, samplers), and — critically — a *real* INT8 GEMM kernel that replaces the Tier 2 dequantize-compute-requantize shim. Without the real kernel, quantized LLMs decode at f32 speed, defeating the point of quantization for edge deployment.

All three pieces (control-flow, generative ops, int8 kernels) are bundled into a single tier because partial delivery produces a non-functional state: shipping `Loop` without real int8 kernels means quantized LLMs don't actually go faster; shipping int8 kernels without `Loop` means LLMs still need an external loop; shipping both without `RMSNormalization` / `MatMulInteger` / samplers means LLaMA-class models still can't load.

## What Changes

- **Sub-graph executor (~1500 LOC of new runtime code).** A new `onnx-rt/src/sub_executor.rs` module extending `execute_graph` to recursively evaluate compiled inner graphs with isolated value scopes, shared initializer scope, and budget accounting that bubbles up to the parent operator. This is the largest single piece of new runtime code added to the project since the initial commit.

- **3 control-flow operators:** `If`, `Loop`, `Scan`. Each wraps an inner `GraphProto` that the sub-graph executor compiles once (at `Session` load time) and dispatches N times during inference. `Loop` implements the full ONNX termination semantics (`M` / `cond` / `cond_out`).

- **19 generative / normalization operators:** `RMSNormalization`, `MatMulInteger`, `DynamicQuantizeLinear`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `Multinomial`, `Bernoulli`, `Dropout`, `EyeLike`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceSumSquare`, `LpNormalization`, `MeanVarianceNormalization`, `Softplus`.

- **Real INT8 GEMM kernel.** Replace the dequant-compute-requant body of `op_qlinear_matmul` (and `op_qlinear_conv`'s im2col->gemm path) with a tiled kernel that accumulates in `i32`, folds zero-points at the edges, saturates on store, and produces output within ±1 in the quantized integer domain of a reference Python ORT implementation.

- **Inventory updates.** Flip the 22 new operators from `OperatorStatus::Planned(Phase::P2)` to `OperatorStatus::Implemented` in the `SUPPORTED_OPS_INVENTORY` table that Phase 1 adds. Phase 2 does not invent the inventory machinery — it consumes it.

## Capabilities

### Modified Capabilities
- `onnx-cpu-execution`: Add requirements for the sub-graph executor and the three control-flow operators (`If`, `Loop`, `Scan`).
- `onnx-quantized-operators`: Add a requirement for the real int8 GEMM kernel.

## Impact

- **Code:**
  - `onnx-rt/src/sub_executor.rs` — new file, ~1500 LOC (the largest single addition). Recursive graph executor, sub-graph compilation cache, scope management, budget plumbing.
  - `onnx-rt/src/ops/control_flow.rs` — new file, `op_if`, `op_loop`, `op_scan`.
  - `onnx-rt/src/ops/generative.rs` — new file, 19 generative/norm ops.
  - `onnx-rt/src/ops/quantized.rs` — modified, real i8 GEMM body.
  - `onnx-rt/src/executor.rs` — modified, `dispatch_node` gains three new cases (If/Loop/Scan) that hand off to `sub_executor`.
  - `onnx-rt/src/graph.rs` — modified, `ExecutionGraph` extended to carry compiled sub-graphs.
  - `onnx-rt/src/operators.rs` — modified, 22 new `OpKind` variants; inventory flips.

- **APIs:** No breaking changes to `Session` public API. The addition is internal: `ExecutionGraph` now carries sub-graph state, and the executor recurses.

- **Risk:** The sub-graph executor is the largest single piece of new runtime code since the project's initial commit and touches the hot dispatch path. Mitigation: full test suite of bounded loops, nested `If`, three-level-deep `Scan`, plus an end-to-end GPT-2-small single-call generation test. The real int8 kernel carries bit-equivalence risk versus reference ORT; mitigation is a ±1 quantized-step test against hand-computed reference vectors.

- **WCET:** `Loop` as a single accounting unit in the budget table, not per-iteration. Hard-limit aborts the whole loop. Sub-graph operators that exceed their own inner budget bubble up a hard-limit error to the parent `Loop`.

- **Testing:** ~60 new unit tests, plus an end-to-end integration test running GPT-2-small generation inside a single `Session::run()` call.

- **Dependencies:** None new. Everything stays `#![no_std]` with `alloc`.

## Out of Scope

Phase 2 is deliberately narrow. The following are explicitly **not** in this change and will land in Phase 3, Phase 4, or are deferred entirely:

- **Audio ops** (MelSpectrogram, STFT, DFT, HannWindow, HammingWindow, BlackmanWindow) — Phase 3.
- **Additional detection-specific ops beyond Phase 1** (e.g., NonMaxSuppression, MaxRoiPool) — Phase 3. Note: `RoiAlign` and `TopK` already land in Phase 1 (`vision-transformers-v1`, PR #76).
- **ONNX-ML classical ML ops** (TreeEnsemble, LinearClassifier, SVMClassifier, Imputer, Scaler) — Phase 4 or deferred.
- **Sequence and Optional types** (SequenceConstruct, SequenceAt, OptionalGetElement) — deferred; modern LLMs don't use them.
- **String tensors** (Tokenizer, StringSplit, RegexFullMatch) — deferred; tokenization stays host-side.
- **Training-mode operators** (all `*Grad` variants, SGD, Adam, Momentum) — out of scope for inference runtime.
- **GPU dispatch of sub-graph interiors.** The sub-graph executor is CPU-only in Phase 2. GPU crates remain stubs; offloading `Loop` bodies to GPU is tracked as open work in the design doc's Future Work section.
- **JIT compilation of inner loop bodies.** The compiled inner graph is an interpreted `ExecutionGraph`, not machine code. JIT is Future Work.
- **Graph optimizer integration.** The inner graph does not yet participate in constant folding / fusion passes across the `Loop` boundary. Phase 2 runs the inner body as-is; cross-boundary optimization is Phase 3.
