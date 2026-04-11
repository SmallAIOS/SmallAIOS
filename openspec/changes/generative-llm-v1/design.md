## Context

After Phase 1 lands (`transformer-models-v1` + `vision-transformers-v1`, PRs #76 and #77), the SmallAIOS ONNX runtime supports 44 new operators and the `OperatorStatus` inventory machinery, unlocking every encoder-only workload we care about: BERT, DistilBERT, ViT, DeiT, CLIP image towers. But the runtime still cannot execute a *decoder-only* autoregressive model — GPT-2, LLaMA, Mistral, Phi — because these models require three things the runtime does not yet have:

1. **Control flow inside the graph.** A decoder model generates tokens in a loop: run the transformer, sample a token, feed it back, stop on EOS or `max_new_tokens`. ONNX exports model this as an in-graph `Loop` operator whose body is an inner `GraphProto`. No `Loop` means the caller must run generation from the outside, paying full graph compilation and value-map allocation overhead on every token, and defeating cooperative WCET budgeting (the scheduler sees N independent inferences instead of one bounded generation).

2. **Quantized-friendly generative ops.** Modern LLMs use `RMSNormalization` (LLaMA family), `MatMulInteger` (int-weight-only quantization), and `DynamicQuantizeLinear` (per-token activation quantization). They sample with `Multinomial` or top-k temperature sampling seeded by `RandomUniform`. None of these exist today.

3. **A real int8 GEMM kernel.** The Tier 2 quantized ops (`op_qlinear_matmul`, `op_qlinear_conv`) land in Phase 1 as a *dequantize-compute-requantize* shim: read `i8` input, convert to `f32`, run the existing f32 GEMM, quantize back to `i8`. That compiles and passes correctness tests but is *slower* than the f32 path because of the round-trip. Without a real int8 kernel, "quantized LLaMA" is a marketing sticker, not an optimization.

These three pieces have to ship together. Shipping `Loop` without real int8 kernels means quantized LLMs don't actually get faster. Shipping int8 kernels without `Loop` means LLaMA still needs an external host loop. Shipping both without `RMSNormalization` means LLaMA simply fails to load (unknown operator). Phase 2 therefore bundles all three into one change.

## Goals / Non-Goals

**Goals:**
- Add 21 new operators (3 control-flow, 18 generative/norm) to the CPU runtime.
- Add a sub-graph executor that recursively evaluates inner graphs from `If`, `Loop`, `Scan` bodies.
- Replace the dequant-compute-requant body of `op_qlinear_matmul` / `op_qlinear_conv` with a real tiled i32-accumulator kernel.
- Integrate the sub-graph executor with the existing WCET budget enforcement in `profile.rs` and `sched-types`.
- Validate end-to-end: GPT-2-small generation in a single `Session::run()` call; int8-quantized LLM within 1% relative error vs f32.
- Maintain `#![no_std]` compatibility, zero new external dependencies.

**Non-Goals:**
- GPU dispatch of sub-graph interiors — CPU only in Phase 2. Future Work.
- JIT compilation of inner loop bodies — interpreted only. Future Work.
- Graph-optimizer integration across the `Loop` boundary (constant folding, fusion) — Phase 3.
- Audio / detection / ONNX-ML / sequence / string operators — not on the LLM critical path.
- Training-mode operators (all `*Grad`, SGD, Adam) — out of scope for an inference runtime.

## Decisions

### D1: Bundle control-flow + generative ops + real int8 kernels into one tier

**Decision.** Ship all three pieces in a single OpenSpec change (`generative-llm-v1`) rather than split into three smaller ones.

**Rationale.** The three are co-dependent for end-user value:

- `Loop` without real int8 kernels → quantized LLMs decode at f32 speed.
- Real int8 kernels without `Loop` → still need an external host generation loop.
- Both without `RMSNormalization`/`MatMulInteger` → LLaMA and Phi fail to load with "unknown operator".

Any two-of-three delivery produces a non-functional state where a user can technically run a model but gets no benefit. The phase boundary is defined by user value, not implementation convenience.

**Alternative.** Split into `control-flow-v1`, `generative-ops-v1`, `int8-kernels-v1`. Rejected: three PRs merge out of order and produce transient non-functional states on `develop`.

### D2: Sub-graph executor architecture — compile-once, dispatch-many

**Decision.** At `Session` load time, when graph construction encounters an `If`, `Loop`, or `Scan` node, the inner `GraphProto` embedded in that node's attributes is compiled into its own `ExecutionGraph` (topological sort, attribute propagation, the whole pipeline) and stored in a new field on the parent `ExecutionNode`. At dispatch time, `dispatch_node` matches on `OpKind::If` / `OpKind::Loop` / `OpKind::Scan` and hands control to `sub_executor::run_sub_graph(compiled_inner, outer_value_map, carried_values, budget, …)`.

The sub-executor recursively calls `execute_graph` on the compiled inner graph with a *fresh* `value_map` that is seeded with carried values and allowed to read outer initializers by name.

```
Session::run()
  └── execute_graph(outer_graph)
        ├── node[0]  Add       (dispatch_node → op_add)
        ├── node[1]  Loop      (dispatch_node → op_loop)
        │     └── for iter in 0..M:
        │           sub_executor::run_sub_graph(inner_graph)
        │             └── execute_graph(inner_graph)    ← recursion
        │                   ├── node[0]  RMSNorm
        │                   ├── node[1]  MatMul
        │                   ├── node[2]  If               ← recursion again
        │                   │     └── execute_graph(then_branch)
        │                   └── node[3]  Softmax
        └── node[2]  Identity
```

**Rationale.** The alternative — interpret the `GraphProto` node list directly each iteration, without pre-compiling into `ExecutionGraph` — would mean re-running topological sort, attribute decoding, and type inference on every `Loop` iteration. A 512-token generation on a 32-layer model would re-sort the graph 512 times. Compile-once puts all the setup cost into `Session::new()`, where it belongs.

Compilation also lets the optimizer (once it grows cross-body awareness in Phase 3) treat the inner graph as first-class.

**Alternative considered.** Interpret the `AttributeProto::graph` directly each iteration. Rejected for the reason above.

### D3: Scope and value passing

**Decision.** Each sub-graph invocation gets a *fresh* `BTreeMap<String, Tensor>` value map. The outer value map is *not* shared. Instead:

- **Initializers are shared.** The `&[TensorProto]` initializer list is threaded through to the sub-executor unchanged. Initializers are read-only model weights; there is no reason to copy them into each inner scope.
- **Outer values referenced by the body** (per ONNX Loop spec, the body can reference outer-graph tensors by name) are copied into the inner value map at the start of each iteration via a pre-computed `outer_refs: Vec<String>` list. The copy is shallow (the `Tensor` struct holds a `Vec<u8>` that can be cloned cheaply via `Arc` in a future optimization; Phase 2 uses `Tensor::clone()` and accepts the cost).
- **Loop-carried values** (the `v_initial` slots and the `v_final` outputs of ONNX Loop) are threaded iteration-to-iteration: iteration N's `v_final` becomes iteration N+1's `v_initial`. The sub-executor owns this rotation; the inner graph executor does not see it.
- **Outer-graph values are read-only inside the body.** If the body writes a tensor that shadows an outer name, the write lives only in the inner value map; the outer map is untouched. On sub-graph exit, the inner map is dropped.

```
Outer value_map:           Inner value_map (Loop iter 3):
  "input"   → T0             "input"        → T0       (copied outer ref)
  "weights" → W              "iter_num"     → I3       (loop var)
  "output"  → (pending)      "cond_in"      → B_true   (carried)
                             "hidden_state" → H3       (carried, from iter 2)
                             "logits"       → L3       (computed)
                             "next_hidden"  → H4       (computed, becomes iter 4 input)
                             "cond_out"     → B_true   (computed)

  initializers &[TensorProto]  ← shared reference, not copied into either map
```

**Rationale.** ONNX Loop semantics require that body writes do not escape the body unless via the declared output list. A fresh inner scope enforces this structurally. Sharing initializers avoids O(iterations × num_weights) clones, which for a 32-layer transformer is the difference between 10ms and 10s per generation.

### D4: Loop termination conditions

**Decision.** The `Loop` operator implements all three ONNX termination signals per the [ONNX Loop spec](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop):

1. **`M` (maximum trip count)** — optional i64 input. If provided, the loop runs at most `M` iterations.
2. **`cond` (external termination)** — optional bool input. If provided and `false` on entry, the loop runs zero iterations.
3. **`cond_out` (body-emitted termination)** — the body's second output (after `iter_num`) is a bool; if `false` at the end of iteration N, the loop stops and iteration N's carried outputs become the loop's final outputs.

Combination semantics: the loop continues if and only if *all* of: (`M` is absent OR current_iter < M) AND (`cond` was absent OR last `cond_out` was true). Hitting any stop condition ends the loop cleanly.

Pseudocode:

```
op_loop(M, cond_initial, v_initial..., inner_graph):
    if M is Some(0): return v_initial
    if cond_initial is Some(false): return v_initial

    v_current = v_initial
    cond_current = cond_initial.unwrap_or(true)
    iter = 0

    loop:
        if M.is_some() and iter >= M.unwrap(): break
        if !cond_current: break

        inner_inputs = [iter_tensor(iter), bool_tensor(cond_current)] ++ v_current
        (cond_out, v_new...) = run_sub_graph(inner_graph, inner_inputs, budget)

        v_current = v_new
        cond_current = cond_out
        iter += 1

    return v_current
```

**Rationale.** All three termination paths are real-world: `M` is how `max_new_tokens` is encoded, `cond` is how "skip generation entirely if already EOS" is encoded, `cond_out` is how "stop at EOS token" is encoded. A spec-compliant `Loop` needs all three.

### D5: WCET budget integration for sub-graphs

**Decision.** The `Loop` operator is a single accounting unit in the `OperatorBudget` table, not one unit per iteration. The full time spent inside the loop (summed across all iterations, including sub-dispatch overhead) is the "actual" time compared against the `Loop` budget.

Inner-body operators *also* check their own budgets during execution. If an inner operator exceeds its own hard limit, the sub-executor returns `Err(HardLimit)`, which bubbles up to the parent `Loop` and immediately terminates the loop with `SessionError::ExecutionFailed`.

The inner operators' measurements land in the same `InferenceProfile.operators` vector as outer ones, annotated with a parent path (e.g. `"Loop[0] > MatMul[3]"`) for post-hoc analysis.

**Rationale.** Per-iteration accounting would blow up the profile table for a 512-token generation. Whole-loop accounting matches the way an end user reasons about latency: "this whole generation took 2.3s, budget was 3s, OK".

Parent-path annotation preserves the inner detail without exploding the table size (the profile records one row per *operator instance*, not per *iteration*).

### D6: Real int8 GEMM kernel

**Decision.** Replace the `op_qlinear_matmul` body (Tier 2 dequant-compute-requant shim) with a tiled kernel that:

1. **Accumulates in `i32`** rather than `f32`. For an `M×K × K×N` matmul with `i8` inputs, each accumulator slot needs to hold up to `K * i8::MAX * i8::MAX = K * 16129`, which for `K ≤ 131071` fits in `i32` without overflow. LLM K values are well under that.
2. **Folds zero-points at the edges.** The math is `output[i,j] = a_scale * b_scale / y_scale * sum_k((a[i,k] - a_zp) * (b[k,j] - b_zp)) + y_zp`. Expanding the inner product and precomputing the three zero-point correction sums lets the hot loop stay `acc += a[i,k] * b[k,j]` with no per-element subtractions.
3. **Saturates on store.** The final `i32` result is clamped to `[i8::MIN, i8::MAX]` before writing to output.
4. **Uses the same cache-blocked tile sizes** as `gemm_f32` (8×8 inner, 64×64 outer) so the micro-architecture story stays consistent.

**Rationale.** The dequant-compute-requant shim accumulates `f32` rounding error across the K reduction. For long-sequence LLM matmuls (K=4096 for LLaMA-7B attention projections) this error compounds. An `i32` accumulator is exact up to the saturation point, so the output is bit-equivalent (within ±1 ULP after the final scale-multiply, which is all that's achievable) to the reference Python `onnxruntime` implementation.

The shim also does twice the memory bandwidth of a real kernel (read i8, write f32, read f32, write i8) and gives up the 4× memory density advantage of int8. The real kernel reads and writes i8 only, and keeps the f32 scales in registers.

**Alternative considered.** Use `f32` accumulators with Kahan summation for the correction term. Rejected: slower than `i32` accumulation, and still not bit-equivalent to reference ORT.

### D7: Op grouping and file layout

**Decision.**

- `onnx-rt/src/ops/control_flow.rs` — `op_if`, `op_loop`, `op_scan`. New file.
- `onnx-rt/src/ops/generative.rs` — the 18 generative/norm ops. New file.
- `onnx-rt/src/ops/quantized.rs` — modified; real int8 kernel replaces the shim.
- `onnx-rt/src/sub_executor.rs` — the sub-graph executor itself. **New top-level file in `onnx-rt/src/`, not under `ops/`.**

The sub-executor is not an operator; it is infrastructure that operators (`If`/`Loop`/`Scan`) call into. It has the same conceptual weight as `executor.rs` itself (it is the recursive half of `execute_graph`). Putting it under `ops/` would mislead contributors.

**Rationale.** The `ops/` directory holds functions of the form `fn op_foo(&[Tensor], &[AttributeProto]) -> Result<Vec<Tensor>, _>`. The sub-executor has a different signature (it takes a compiled `ExecutionGraph`, a value map, and a budget handle) and different lifecycle semantics (it can recurse, it owns scope management). File layout should match responsibility.

### D8: Phase 2 does NOT re-add the operator inventory

**Decision.** Phase 1 (`transformer-models-v1`) introduces the `SUPPORTED_OPS_INVENTORY: &[(OpKind, OperatorStatus)]` table and the `OperatorStatus::{Implemented, Planned(Phase::P1|P2|P3|P4)}` enum. Phase 2 only *flips* the relevant entries from `Planned(Phase::P2)` to `Implemented`.

```diff
-    (OpKind::Loop, OperatorStatus::Planned(Phase::P2)),
+    (OpKind::Loop, OperatorStatus::Implemented),
```

Phase 2 does not touch the inventory machinery itself (struct, table shape, reporting API).

**Rationale.** Coordination. Two concurrent PRs that both edit the inventory struct would conflict. By drawing a clean boundary — "Phase 1 owns inventory machinery, Phase 2 owns the Phase-2 entries" — we avoid rebase churn.

## Alternatives Considered

### A1: Interpret `AttributeProto::graph` per iteration (no compilation cache)

Instead of compiling the inner graph into an `ExecutionGraph` at load time, walk the raw `GraphProto` each iteration, building up a fresh node list and tensor map. Simpler control flow in the sub-executor, no new state on `ExecutionNode`.

**Rejected** because topological sort and attribute decoding are *not* free. A 512-token generation on a 32-layer model would do this 512 times. Typical graph-build time for a single transformer block is a few milliseconds; 512 × 32 × a few ms is several minutes of pure graph-building overhead before any inference happens. Compile-once moves that cost to `Session::new()`.

### A2: Support only `Loop` and skip `If` and `Scan`

`Loop` alone is enough for autoregressive generation. `If` and `Scan` don't appear in common decoder exports.

**Rejected** because (a) the sub-graph executor is the same infrastructure for all three — skipping `If` and `Scan` saves about 150 LOC of operator glue but zero LOC of the hard part (scope management, compilation cache, budget integration); (b) `If` *does* appear in quantized export paths (branches for dequant vs int path), and (c) `Scan` is used by ONNX exports of RNNs and some attention implementations; if we claim "control flow" we should deliver the full set.

### A3: Use an external Python loop and skip Phase 2 entirely

Tell users to run generation from host code: call `Session::run()` N times, manage the KV cache in Python, do sampling in Python.

**Rejected** because (a) it defeats the WCET story — the scheduler sees N independent inferences instead of one bounded generation, so there is no way to budget "total time to generate 512 tokens"; (b) every iteration pays graph compilation and value-map allocation overhead; (c) it leaks the autoregressive loop out of the runtime boundary, so any future optimization (KV-cache pooling, speculative decoding, flash-attention) has to be duplicated by every caller; (d) it is not what any production ONNX LLM runtime ships (`onnxruntime-genai`, `onnxruntime-web`, `llama.cpp` all support in-graph `Loop`).

### A4: Build a full graph optimizer / compiler IR before Phase 2

Rather than add `Loop` / `If` / `Scan` to the existing interpreter, write a proper compiler IR first — SSA, scheduling passes, register-style tensor allocation — and run *all* operators (not just the Phase 2 ones) through it.

**Rejected** because it is a six-month project and this change is a two-week project. We can build a compiler IR *later*, at which point the sub-graph executor's `ExecutionGraph` representation becomes the natural input to the IR builder (it already has topological order, attribute propagation, and type info). The compile-once cache from D2 is in fact a stepping stone toward a compiler IR.

## Open Questions

### Q1: Sub-graph value-map memory cost

Each `Loop` iteration allocates a fresh inner `BTreeMap<String, Tensor>`. For a 32-layer transformer with ~150 named intermediate tensors per layer, that is ~5000 BTreeMap entries per iteration. At 32 bytes of overhead per entry (BTreeMap node), that is ~160 KB of map overhead per iteration, before the actual tensor data. Over 512 iterations this is *reallocated* each time.

**Options:**
- (a) Accept the cost. Typical transformer inference is dominated by matmul, not map operations.
- (b) Pre-allocate an arena of tensor slots at compile time, keyed by index not name.
- (c) Reuse a single inner value map across iterations by clearing it between iterations.

**Leaning:** (c) — reuse + clear. The map structure is the same across iterations (same set of tensor names), so clearing instead of reallocating avoids the allocator churn. Phase 2 ships (c); (b) is a Phase 3 optimization.

### Q2: How does `Scan` interact with the cooperative scheduler's yield points?

The existing `execute_graph` calls `yield_fn` after each operator completes. Inside a `Scan` body, operators also call `yield_fn`. For a `Scan` with 512 sequence elements and a 10-op body, that is ~5120 yield points — potentially more than the scheduler budget between System/IPC tasks.

**Options:**
- (a) Yield after every inner op (current behavior, unchanged).
- (b) Yield only at the outer `Scan` boundary, skipping inner yields.
- (c) Yield every N inner ops, where N is chosen to meet a target yield frequency (~1ms of wall time).

**Leaning:** (c) with N derived from the operator-budget class — cheap ops coalesce, expensive ops yield individually. Phase 2 ships (a) as the safe default; (c) is a tunable in a follow-up.

### Q3: How do we test `Loop` termination without hanging tests?

A buggy termination check could cause `op_loop` to run forever, wedging CI. Unit tests need a hard safety net.

**Options:**
- (a) Every test passes an explicit `M` with a small upper bound (e.g. M=100) and asserts early exit at the expected iteration.
- (b) The executor enforces a compile-time-hard-coded absolute max iteration count (e.g. 1,000,000) that is independent of the per-operator budget.
- (c) CI runs tests under a wall-clock timeout.

**Leaning:** all three. (a) is the unit-test convention, (b) is runtime belt-and-suspenders, (c) is CI-level belt-and-suspenders. Phase 2 ships (a) and (b); (c) is already in CI.
