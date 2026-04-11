# Sub-Graph Executor Design

> Status: Design document for Phase 2 of the ONNX coverage roadmap (`generative-llm-v1`).
> Implementation lives in `onnx-rt/src/sub_executor.rs` (to be created).
> This document is the authoritative reference for that module and will be referenced
> from the OpenSpec change and by future implementers.

## 1. Overview and Motivation

The SmallAIOS ONNX runtime executes models via a flat, topologically-sorted node
list (`ExecutionGraph` in `onnx-rt/src/graph.rs`). At inference time,
`execute_graph` in `onnx-rt/src/executor.rs` walks that list in order, resolving
each node's input tensors from a named `BTreeMap<String, Tensor>` value map,
dispatching to a CPU operator, and writing outputs back to the map.

This flat-walk model handles every encoder-style model we care about (BERT,
ViT, CLIP image tower). It does **not** handle autoregressive decoders —
GPT-2, LLaMA, Mistral, Phi — because those models contain **control-flow
operators** whose bodies are themselves graphs. In particular:

- `Loop` — runs an inner graph N times, threading carried values between
  iterations. This is how ONNX encodes autoregressive generation.
- `If` — selects one of two inner graphs based on a boolean condition.
  This appears in quantized export paths (choose between dequant and int
  branches) and in safety-gated model variants.
- `Scan` — runs an inner graph once per element of a sequence-axis
  input, building up an output sequence. Used by ONNX exports of RNNs
  and some attention implementations.

None of these can be implemented as ordinary operators because their
"parameters" are not tensors — they are entire sub-graphs. Dispatching them
requires the runtime to recursively evaluate those sub-graphs with a proper
value scope, budget accounting, and termination semantics.

The **sub-graph executor** is the module that provides that recursive
evaluation. It is the single largest piece of new runtime code added in
Phase 2 (approximately 1500 lines of Rust), and it is the mechanism that
turns SmallAIOS from an encoder-only runtime into a general-purpose ONNX
runtime capable of running generative LLMs.

### Why this matters for SmallAIOS

Generative LLM inference can be driven one of two ways:

1. **External generation loop.** The host calls `Session::run()` once per
   output token, manages the KV cache in host code, samples the next token
   in host code, feeds the sampled token back, and iterates until EOS or
   `max_new_tokens`. This works, but:
   - Every call pays full value-map allocation overhead.
   - The cooperative scheduler sees N independent inferences instead of
     one bounded generation. WCET budgeting is impossible.
   - The autoregressive loop has leaked out of the runtime boundary, so
     every caller has to re-implement KV-cache management, sampling, and
     stop-condition checking.
   - Production ONNX LLM runtimes (`onnxruntime-genai`, `llama.cpp`'s ONNX
     exporter) do not ship this mode.

2. **In-graph generation loop.** The entire generation loop lives inside
   the ONNX model as a `Loop` operator. A single `Session::run()` call
   produces the full generated sequence. The scheduler sees one atomic
   inference. The WCET budget can be set as "total time to generate 512
   tokens ≤ 3000 ms", which is the unit a user actually cares about. The
   KV cache, sampling, and termination logic live inside the model graph
   where the exporter (not the runtime caller) is responsible for them.

Phase 2 commits to option 2. The sub-graph executor is the piece that
makes option 2 possible.

## 2. ONNX Control-Flow Semantics Primer

The full spec lives at <https://github.com/onnx/onnx/blob/main/docs/Operators.md>.
What follows is the minimum understanding needed to implement the three
operators correctly.

### 2.1 `Loop`

Signature (ONNX opset 21):

```
Inputs:
  M        : optional i64 scalar — maximum trip count
  cond     : optional bool scalar — initial external condition
  v_initial: variadic — initial values for loop-carried dependencies

Outputs:
  v_final  : variadic — final values of loop-carried dependencies

Attribute:
  body     : GraphProto — the loop body graph
```

The body graph has a fixed input signature:

```
Body inputs:
  iter_num   : i64 scalar — current iteration number (starts at 0)
  cond_in    : bool scalar — current condition value
  v_in_1..N  : variadic — current values of loop-carried dependencies

Body outputs:
  cond_out   : bool scalar — whether to continue to the next iteration
  v_out_1..N : variadic — next values of loop-carried dependencies
```

Termination rule (pseudocode):

```
iter = 0
cond = cond_input.unwrap_or(true)
v = v_initial

while (M is None or iter < M) and cond:
    (cond, v) = body(iter_num = iter, cond_in = cond, v_in = v)
    iter += 1

return v
```

Three independent stop signals: `M`, the input `cond`, and the body-emitted
`cond_out`. A spec-compliant `Loop` must support all three and their
combinations.

### 2.2 `If`

```
Inputs:
  cond : bool scalar

Outputs:
  outputs : variadic — matches the arity of both branches

Attributes:
  then_branch : GraphProto
  else_branch : GraphProto
```

Both branches must have the same output arity (though not necessarily the
same output shapes — a then-branch can produce `[1, 768]` and an else-branch
can produce `[1, 1024]`). Exactly one branch is evaluated per dispatch.

### 2.3 `Scan`

```
Inputs:
  initial_state_and_scan_inputs : variadic

Outputs:
  final_state_and_scan_outputs : variadic

Attributes:
  body               : GraphProto
  num_scan_inputs    : i64
  scan_input_axes    : optional list of axes
  scan_input_directions : optional list of forward/reverse
  scan_output_axes   : optional list
  scan_output_directions: optional list
```

`Scan` is `Loop` with implicit iteration count (= length of the scan axis
of the scan inputs) and automatic slicing/stacking of the per-element
inputs and outputs. Conceptually:

```
for i in 0..scan_length:
    body_input = slice(scan_input, axis=scan_axis, index=i)
    (state, scan_output_slice) = body(state, body_input)
    output[i] = scan_output_slice
```

Phase 2 implements the simple case only: `num_scan_inputs = 1`, default
axes, forward direction. Future work generalizes.

## 3. Architecture

### 3.1 Compile-Once, Dispatch-Many

Sub-graphs are compiled **once** at `Session::new()` time, not per
dispatch. When the graph builder (`graph.rs`) walks the outer
`GraphProto.node` list and encounters an `If`, `Loop`, or `Scan` node, it:

1. Reads the inner `GraphProto` from the node's `body`, `then_branch`, or
   `else_branch` attribute.
2. Recursively runs the existing graph-build pipeline on that inner proto
   (topological sort, attribute propagation, type inference).
3. Stores the resulting `ExecutionGraph` on the parent `ExecutionNode`'s
   new `sub_graphs: Vec<ExecutionGraph>` field.
4. `If` produces 2 compiled inner graphs (then + else), `Loop` and `Scan`
   produce 1 each.

At dispatch time, `executor::dispatch_node` matches on `OpKind::If`,
`OpKind::Loop`, `OpKind::Scan` and hands control to the sub-executor
along with a reference to the already-compiled inner graphs.

```
// At Session::new()
ExecutionGraph {
    nodes: [
        Node { op: "MatMul", sub_graphs: [] },
        Node { op: "Loop",   sub_graphs: [body_graph] },  ← compiled once
        Node { op: "Add",    sub_graphs: [] },
    ]
}

// At Session::run()
execute_graph(outer)
  ├── dispatch_node(MatMul)    → op_matmul(...)
  ├── dispatch_node(Loop)      → op_loop(..., node.sub_graphs[0])
  │       └── for iter in 0..M:
  │             sub_executor::run_sub_graph(node.sub_graphs[0], ...)
  │               └── execute_graph(node.sub_graphs[0])   ← recursion
  └── dispatch_node(Add)       → op_add(...)
```

### 3.2 Dispatcher Glue

`executor::dispatch_node` grows three new match arms:

```rust
OpKind::If => op_if(inputs, attributes, &node.sub_graphs, outer_map, ctx),
OpKind::Loop => op_loop(inputs, attributes, &node.sub_graphs, outer_map, ctx),
OpKind::Scan => op_scan(inputs, attributes, &node.sub_graphs, outer_map, ctx),
```

Note that these three operators, unlike every other operator, take a
reference to the outer value map and to the execution context (budget,
time source, yield callback, profile). This is the one place in the
runtime where operator glue is allowed to know about the executor's
internals. The break in abstraction is contained to these three ops.

## 4. Scope Rules

Sub-graph execution uses **isolated value scope with shared initializer
scope**. Diagrams first, text second.

### 4.1 Diagram: Loop Iteration 3 of 8

```
                   Outer value_map (owned by execute_graph)
                  ┌────────────────────────────────────────┐
                  │ "input_ids" → T_in                     │
                  │ "hidden"    → H_current  (updated each │
                  │                          Loop iter)    │
                  │ "output"    → (pending)                │
                  └────────────────────────────────────────┘
                                   │
                                   │ passes OUTER_REFS + CARRIED
                                   ▼
     Inner value_map (fresh per iter, or cleared-and-reused per Q1)
    ┌──────────────────────────────────────────────────────────────┐
    │ [Seeded from caller]                                         │
    │   "iter_num"    → scalar i64 = 3                             │
    │   "cond_in"     → scalar bool = true                         │
    │   "v_in_state"  → H3       ← loop-carried from iter 2        │
    │   "input_ids"   → T_in     ← outer ref, read-only here       │
    │                                                              │
    │ [Filled by the inner graph's nodes]                          │
    │   "q"           → Q3       (from MatMul)                     │
    │   "k"           → K3                                         │
    │   "v"           → V3                                         │
    │   "attn"        → A3                                         │
    │   "v_out_state" → H4       ← becomes iter-4 v_in_state       │
    │   "cond_out"    → true     ← decides whether iter 4 runs     │
    └──────────────────────────────────────────────────────────────┘

                    Initializers &[TensorProto]
                    ┌─────────────────────────────────┐
                    │ W_q, W_k, W_v, W_o, gamma, beta │
                    │ (read by BOTH outer and inner)  │
                    └─────────────────────────────────┘
```

### 4.2 Rules

1. **Fresh inner map per sub-graph invocation.** A new
   `BTreeMap<String, Tensor>` is created (or, per the Phase 2 performance
   optimization, an existing map is cleared and reused).

2. **Initializers are shared.** The `&[TensorProto]` initializer slice is
   threaded through to the recursive `execute_graph` call unchanged. This
   is critical: for a 32-layer transformer, cloning initializers into the
   inner scope per iteration would cost O(iterations × num_weights) Tensor
   clones — tens of megabytes per iteration.

3. **Outer references are eagerly copied.** The ONNX `Loop` spec allows the
   body to reference outer tensors by name (not just carried values). Those
   names are precomputed at compile time by scanning the inner graph's
   input list against the outer graph's tensor names. The list is stored
   on the parent node; per iteration, the sub-executor copies each referenced
   tensor from the outer map into the inner map by shallow clone (the
   `Tensor` struct holds a `Vec<u8>`, which is a deep clone in Phase 2; a
   future optimization switches to `Arc<Vec<u8>>` for free clones).

4. **Outer values are read-only from inside the body.** The inner map is
   seeded with outer refs, but any writes from the inner body only mutate
   the inner map. On exit the inner map is dropped.

5. **Body-declared outputs are the only values that escape.** The body
   `GraphProto` has an explicit output name list. On sub-graph exit, only
   those names are extracted and returned to the caller, which then routes
   them (via the Loop/Scan op) into the parent node's outputs.

6. **Carried values rotate across iterations.** Iteration N's `v_out_*`
   outputs become iteration N+1's `v_in_*` inputs. This rotation is owned
   by the `op_loop` / `op_scan` implementation; the recursive
   `execute_graph` sees them as ordinary inputs/outputs.

## 5. Loop Termination Handling

The three termination signals combine as follows. In pseudocode:

```rust
fn op_loop(
    m: Option<i64>,
    cond_initial: Option<bool>,
    v_initial: Vec<Tensor>,
    body: &ExecutionGraph,
    ctx: &mut ExecCtx,
) -> Result<Vec<Tensor>, SessionError> {
    // Trivial skip cases
    if m == Some(0) {
        return Ok(v_initial);
    }
    if cond_initial == Some(false) {
        return Ok(v_initial);
    }

    let mut v_current = v_initial;
    let mut cond_current = cond_initial.unwrap_or(true);
    let mut iter: i64 = 0;

    // Hard safety net (D9 in design.md)
    const MAX_LOOP_ITERATIONS: i64 = 1_000_000;

    loop {
        if iter >= MAX_LOOP_ITERATIONS {
            return Err(SessionError::ExecutionFailed(
                "Loop exceeded compile-time safety limit".into()
            ));
        }
        if let Some(max) = m {
            if iter >= max { break; }
        }
        if !cond_current { break; }

        // Seed inner inputs: iter_num, cond_in, v_in_*
        let inner_inputs = build_inner_inputs(iter, cond_current, &v_current);

        // Recurse
        let inner_outputs = sub_executor::run_sub_graph(
            body,
            &inner_inputs,
            ctx.initializers,
            ctx,  // budget, profile, time_source, yield
        )?;

        // Split: outputs[0] = cond_out, outputs[1..] = v_out_*
        cond_current = inner_outputs[0].as_bool_scalar()?;
        v_current = inner_outputs[1..].to_vec();

        iter += 1;
    }

    Ok(v_current)
}
```

Stop-condition test order matters: we check `M` before checking
`cond_current` so that a zero-trip `Loop` with `M = 0` exits without
evaluating `cond_current` (which is fine because we've already handled
`cond_initial = false` above).

Unit tests for termination exhaustively cover:
- `M = 0` → zero iterations, returns `v_initial`.
- `M = 1` → exactly one iteration.
- `M = 64`, body always `cond_out = true` → exactly 64 iterations.
- `M = 64`, body emits `cond_out = false` at iter 32 → stops at 32.
- `M = None`, `cond_initial = Some(false)` → zero iterations.
- `M = None`, body eventually emits `cond_out = false` at iter 10 → stops at 10.
- `cond_initial = Some(true)`, `M = None`, body always true → hits
  `MAX_LOOP_ITERATIONS` safety net and errors.

## 6. WCET Budget Integration

The existing `profile.rs` tracks per-operator timing against thresholds in
`OperatorBudget`. Each operator dispatch measures wall time before and
after, compares against its budget class, and records an
`OperatorMeasurement` row in `InferenceProfile.operators`.

Phase 2 extends this with the following rules:

1. **`Loop` / `If` / `Scan` are single units in the parent profile.** One
   `OperatorMeasurement` row per instance, not one per iteration. The
   `actual_us` field is the wall-clock sum across *all* iterations,
   including sub-dispatch overhead.

2. **Inner operators still measure their own time.** When an inner
   operator runs inside a sub-graph, it produces an `OperatorMeasurement`
   the same way as an outer operator — but the row is tagged with a
   parent-path annotation like `"Loop[5] > MatMul[12]"` so post-hoc
   analysis can reconstruct the nested structure.

3. **Hard-limit aborts bubble up.** If an inner operator exceeds its
   hard limit, the sub-executor returns
   `Err(SessionError::ExecutionFailed(...))` immediately. The parent
   `op_loop` does not catch this — it propagates it upward. The
   `InferenceProfile.hard_limit_aborted` flag is set.

4. **Soft-limit warnings accumulate at the innermost level.** A warning
   inside a `Loop` body counts as one warning per iteration (this is the
   one place where per-iteration accounting leaks through, because
   suppressing warnings inside a loop body would hide real performance
   issues).

5. **The parent `Loop`'s own hard limit is measured against its total
   time.** A `Loop` with a 3000 ms hard limit and 512 iterations each
   averaging 10 ms will exceed the hard limit at iteration 300 and abort
   from the *outer* check, even if no single inner operator exceeded its
   own limit.

### 6.1 Diagram: Profile Row Structure for a Nested Dispatch

```
InferenceProfile.operators: Vec<OperatorMeasurement>
  ┌────────────────────────────────────────────────────────────┐
  │ [0] "Embed"       Add         12 us   Ok                   │
  │ [1] "Block1"      Loop     3450 us   SoftLimit             │
  │ [2]   │   "Attn"     MatMul    18 us   Ok  (iter=0)        │
  │ [3]   │   "Attn"     MatMul    19 us   Ok  (iter=0)        │
  │ [4]   │   "Softmax"  Softmax    3 us   Ok  (iter=0)        │
  │ [5]   │   "Attn"     MatMul    18 us   Ok  (iter=1)        │
  │ ...                                                        │
  │ [N] "LMHead"      MatMul    41 us   Ok                     │
  └────────────────────────────────────────────────────────────┘

  Indentation / hierarchy:
    Row [1] is the aggregate for the whole Loop (used for budget check).
    Rows [2..N-1] are per-iteration inner measurements, annotated with
    iter=K and parent_index=[1]. They contribute to profile detail but
    NOT to the row [1] budget check (that uses its own wall-time sum).
```

## 7. Memory Layout

### 7.1 Per-Iteration Map Lifetime

Naïve implementation allocates a fresh `BTreeMap<String, Tensor>` per
iteration. For a 32-layer transformer with ~150 intermediate tensors per
layer, that's ~5000 BTreeMap entries; at ~32 bytes of per-entry overhead
that's ~160 KB of map overhead per iteration, reallocated 512 times per
generation. This is measurable in microbenchmarks.

Phase 2 ships the **clear-and-reuse** optimization (Open Question Q1,
option c): the sub-executor owns a single `BTreeMap` across all iterations
of a given `Loop`/`Scan` invocation, calls `.clear()` between iterations
to drop tensor references without dropping the map's allocated buckets,
and re-inserts the new iteration's seeded values.

Benchmarks post-implementation will tell us whether an index-based (no-map)
arena is worth the complexity. Phase 3 may replace the map entirely with
an index-addressed slot table derived from the memory planner.

### 7.2 Tensor Cloning

The inner value-map insertions use `Tensor::clone()`, which today does a
full `Vec<u8>::clone()` of the raw data buffer. For outer-ref tensors that
the body reads but never writes, this is pure overhead.

Phase 2 accepts this cost and documents it as a Phase 3 optimization: switch
`Tensor.raw_data` from `Vec<u8>` to `Arc<Vec<u8>>` so clones are refcount
bumps. The switch touches every operator signature and is out of scope here.

### 7.3 Initializer Scope

Initializers (`&[TensorProto]`) are **never copied**. They are threaded
through to the recursive `execute_graph` call as a reference. Each inner
invocation's seeding loop inserts initializer *Tensors* (materialized on
first use) into the inner value map, but the source `TensorProto` slice
is shared.

Materializing initializers on every iteration would be wasteful — they
don't change. A future optimization can cache the materialized `Tensor`s
at `Session::new()` time and avoid the per-iteration `tensor_from_proto`
work.

## 8. Performance Considerations

- **Graph compilation is not free but happens once.** Session load time
  increases proportionally to the number of sub-graphs in the model. For
  a single-`Loop` decoder this is +1 graph compilation.

- **Compilation cache hit rate is 100%.** The compiled inner graph is
  stored by value on the parent `ExecutionNode`; every iteration hits
  the same compiled object. There is no "cache miss" path.

- **Value-map reuse saves ~5% of per-iteration wall time** for a
  32-layer transformer at 512-token generation. Measured on a
  representative benchmark in the `bench` crate (to be added as part of
  task 6.6).

- **Dispatch overhead per inner op is unchanged.** The recursive
  `execute_graph` call has the same per-node overhead as the outer call
  — one BTreeMap lookup per input name, one BTreeMap insert per output
  name, one match arm for dispatch. No new overhead.

- **Budget checks happen at every level.** Inner operators check their
  own budgets, and the parent `Loop` checks its own aggregate. This is
  two checks per iteration (outer + innermost), not quadratic.

- **WCET determinism is preserved.** Because the `Loop` is a single
  budgeted unit at the parent level, the worst-case time for the whole
  generation is bounded by the parent's hard limit. This is what makes
  "generate 512 tokens ≤ 3 seconds" a meaningful WCET statement.

## 9. Testing Strategy

### 9.1 Unit Tests (in `onnx-rt/src/sub_executor.rs`)

- **Bounded Loop with fixed M.** Hand-written 3-node inner body that
  increments a carried counter. M = 10. Assert output counter = 10.

- **Loop with body-emitted cond_out.** Inner body emits false at a
  fixed iteration. Assert loop stops there.

- **If with both branches different shapes.** Then-branch produces
  `[2, 3]`, else-branch produces `[2, 5]`. Dispatch both conditions.

- **Nested If inside Loop body.** Loop runs 10 iterations, inner body
  has an If that alternates on `iter_num & 1`. Assert the carried
  output reflects the alternation.

- **Scan with constant-add body.** Sequence input `[0, 1, 2, 3, 4]`,
  body adds 1. Output `[1, 2, 3, 4, 5]`. Body invocation count = 5.

- **Deep nesting.** Loop inside If inside Loop, three levels deep,
  total 100 innermost operator invocations. Assert correct final
  output and that no map allocation count exceeds the reused-map
  invariant.

- **Safety net trip.** Loop with `M = None`, `cond_initial = true`,
  body that always emits `cond_out = true`. Assert `MAX_LOOP_ITERATIONS`
  safety net triggers and returns an error, not a hang.

- **Inner hard-limit bubbles up.** Inner body contains an operator
  with a hand-adjusted budget small enough to always trip. Assert the
  parent `Loop` returns `Err(ExecutionFailed)` and that
  `InferenceProfile.hard_limit_aborted == true`.

- **Value-map reuse does not leak across iterations.** Loop body reads
  a tensor name it never writes. Assert it sees the fresh per-iteration
  value, not a stale one from the previous iteration.

### 9.2 Integration Tests (in `onnx-rt/tests/`)

- **Single-call GPT-2-small generation.** Load a small GPT-2-style ONNX
  export that contains an in-graph `Loop`. Call `Session::run()` once.
  Compare generated token IDs against a reference Python ORT run.

- **Int8 quantized decoder.** Same model, int8 weights. Assert output
  logits within 1% relative error of the f32 run and that latency is
  lower.

- **Scan-based RNN.** Small GRU or LSTM ONNX export that uses `Scan`
  rather than the dedicated `LSTM`/`GRU` operators. Assert output
  matches reference.

### 9.3 Hang Prevention

Every unit test that involves `Loop` passes an explicit `M` bound. The
compile-time `MAX_LOOP_ITERATIONS` constant catches bugs in the termination
logic itself. CI tests run under a wall-clock timeout. Three layers of
protection.

## 10. Future Work

The following items are **explicitly not** in Phase 2 but are natural
follow-ups. Each is tracked here to inform future design conversations.

### 10.1 Graph Optimizer Integration Across Loop Boundaries

Today the optimizer (constant folding, fusion, memory planning) treats
the outer graph only. The inner graph is optimized independently at
compile time, so *within* the inner graph, normal optimization applies.
But cross-boundary optimization — e.g. hoisting a loop-invariant MatMul
out of the loop body — does not happen.

Phase 3 can add this. The current data structure (`ExecutionNode.sub_graphs`)
is compatible: the optimizer can walk into sub-graphs and rewrite them.

### 10.2 JIT Compilation of Inner Bodies

The recursive `execute_graph` call is still an interpreter. For a
generation loop that runs 512 times, the dispatch overhead per inner
node is paid 512 times. A JIT pass that lowers the inner graph to machine
code (via `cranelift` in container mode, via hand-written codegen in
kernel mode) would amortize that overhead.

Phase 2 does not do this. The compile-once cache from §3.1 is the
natural place for a JIT'd artifact to live — replace `ExecutionGraph`
with `CompiledInnerBody { interpreted: ExecutionGraph, jit: Option<...> }`
and dispatch to the JIT path when available.

### 10.3 GPU Dispatch from Inside Loop Bodies

The sub-graph executor today is CPU-only. Operators inside the `Loop`
body run on CPU regardless of whether outer operators run on GPU. A
future change lifts this restriction: the `GpuBackend` gets threaded
through the sub-executor, and GPU-supported ops inside the body run on
GPU, with CPU fallbacks for the rest.

This is trickier than it looks because each iteration would incur
GPU→CPU→GPU boundary crossings unless the whole body runs on GPU. The
right answer is probably a "GPU body" fast path that only kicks in when
every inner op is GPU-supported.

### 10.4 Arena-Allocated Value Map

The BTreeMap value map is general but slow. Phase 3 can replace it with
an index-addressed slot table derived from the memory planner: every
tensor name is resolved to a slot index at compile time, and the runtime
touches a `Vec<Option<Tensor>>` instead of a `BTreeMap<String, Tensor>`.

The sub-graph executor is exactly the place where this would pay off
most — per-iteration allocation overhead disappears.

### 10.5 Speculative Decoding and KV-Cache Pooling

Once in-graph generation works, higher-level optimizations become
tractable: draft-model speculative decoding, KV-cache pooling across
concurrent inferences, continuous batching. All of them live at or
above the `Loop` level and benefit from the WCET-budget-as-single-unit
property of §6.

These are research-grade features. Not Phase 2, not Phase 3. Noted
here for completeness.

---

## Appendix A: Relationship to Existing Runtime Code

- `onnx-rt/src/executor.rs` — extended with 3 new dispatch cases. `execute_graph`
  is called recursively by `sub_executor::run_sub_graph`.
- `onnx-rt/src/graph.rs` — `ExecutionGraph`/`ExecutionNode` extended with
  `sub_graphs` field; graph builder gains a recursive-compile step.
- `onnx-rt/src/sub_executor.rs` — new module, ~1500 LOC. Owns all the logic
  in §§3–7 above.
- `onnx-rt/src/ops/control_flow.rs` — new module. `op_if`, `op_loop`, `op_scan`
  are thin wrappers that unpack inputs, compute termination, and call into
  the sub-executor.
- `onnx-rt/src/profile.rs` — extended with parent-path annotation on
  `OperatorMeasurement`.
- `onnx-rt/src/operators.rs` — 3 new `OpKind` variants (`If`, `Loop`, `Scan`)
  plus the 18 generative/norm variants.

## Appendix B: Glossary

- **Compiled inner graph.** The `ExecutionGraph` produced by recursively
  running the graph builder on a `GraphProto` extracted from an If/Loop/Scan
  attribute. Cached on the parent node at load time.
- **Carried value.** A tensor that threads across iterations of a `Loop` or
  `Scan`: iteration N's output slot `k` becomes iteration N+1's input slot
  `k`.
- **Outer reference.** A tensor name used inside a sub-graph body that is
  not a carried value and not an initializer — it refers to a tensor in the
  outer graph's value map. Seeded into the inner map at iteration start.
- **Sub-executor.** The module (`onnx-rt/src/sub_executor.rs`) that owns
  recursive execution of inner graphs.
- **Inner body.** The `GraphProto` embedded in an `If`/`Loop`/`Scan`
  attribute; after compilation it becomes a compiled inner graph.
- **Termination signal.** One of the three ways a `Loop` can stop: `M` max
  trip count, `cond` external condition, or body-emitted `cond_out`.
- **Aggregate budget unit.** A measurement row in the inference profile
  that represents the total time spent inside a control-flow operator,
  covering all iterations and all inner dispatch overhead.
