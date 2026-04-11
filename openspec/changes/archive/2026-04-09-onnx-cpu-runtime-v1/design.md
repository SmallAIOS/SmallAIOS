## Context

The ONNX runtime (`onnx-rt/`) has a complete frontend pipeline: protobuf parsing, model validation, execution graph construction with topological sort (Kahn's algorithm), and graph optimization (fusion, constant folding, memory planning). It also has 6 working CPU operators (Add, MatMul/GEMM, Relu, Softmax, Reshape, Conv) with broadcast support, shape inference, and a cache-blocked 8x8 GEMM micro-kernel.

The gap: `Session::run()` at `session.rs:399-403` returns `NotImplemented`. The execution graph's `topological_order` is computed but never iterated. Operator functions exist but are never called from the graph traversal path. There is no tensor value map to route intermediate results between nodes.

The `ExecutionNode` struct already has `inputs: Vec<String>` and `outputs: Vec<String>` (tensor names), plus `op_type: String` for dispatch. The `ExecutionGraph` has `topological_order: Vec<NodeIndex>` ready to iterate.

## Goals / Non-Goals

**Goals:**
- Wire `Session::run()` to execute inference end-to-end on CPU
- Complete all 29 Tier 1 CPU operators
- Integrate with the kernel scheduler via yield points at operator boundaries
- Measure and check per-operator execution time against `OperatorBudget`
- Maintain `#![no_std]` compatibility — no external dependencies

**Non-Goals:**
- GPU execution (separate `compute-abstraction-v1` change)
- Container HTTP server (separate `container-runtime-v1` change)
- AVX-512/SVE kernel selection (future optimization; current GEMM auto-vectorizes)
- WCET calibration (spec exists but deferred to safety-critical work)
- INT8 quantized operators (future optimization pass)
- Multi-threaded parallel operator execution (single-core cooperative model)

## Decisions

### D1: Graph Executor with HashMap Tensor Value Map

The executor will iterate `ExecutionGraph.topological_order`, maintaining a `BTreeMap<String, Tensor>` that maps tensor names to computed values. Before execution, graph inputs and initializer tensors are loaded into the map. Each node reads its inputs from the map by name, calls the corresponding operator function, and writes outputs back.

**Why BTreeMap over linear search:** The graph can have hundreds of tensors. Name-based lookup via `BTreeMap` is O(log n) per access. A `Vec` with linear scan would be O(n). BTreeMap is available in `alloc` (no `std::collections::HashMap` needed in `no_std`).

**Alternative considered:** Index-based addressing (replace string names with integer indices during graph build). More efficient but requires a refactor of `ExecutionNode` and the graph builder. Save for a future optimization pass.

### D2: Operator Dispatch via Match on OpKind

A single `dispatch_node()` function will match on `OpKind::parse_str(&node.op_type)` and call the corresponding `op_*` function. This is a flat match statement — no trait objects, no vtable, no dynamic dispatch.

**Why not trait-based dispatch:** The operator set is closed (29 ops, compile-time known). A match statement is simpler, generates better code, and avoids the complexity of trait objects in `no_std`. When GPU backends are added later, the dispatch function will gain a device-selection layer, but the CPU path stays as a match.

### D3: Operator Implementation Strategy — 4 Priority Tiers

Operators are implemented in dependency order matching common model patterns:

| Tier | Operators | Rationale |
|------|-----------|-----------|
| 1 (core math) | Sub, Mul, Div, Gemm | Same broadcast pattern as existing Add; Gemm wraps existing `gemm_f32` |
| 2 (activations) | Sigmoid, Tanh | Element-wise using existing `expf_approx` |
| 3 (shape/data) | Transpose, Concat, Flatten, Squeeze, Unsqueeze, Cast, Gather, Slice, Pad, Clip | No computation, just data movement/reinterpretation |
| 4 (reduction/norm) | BatchNorm, LayerNorm, MaxPool, AvgPool, GlobalAvgPool, ReduceMean, ReduceSum | Most complex; require windowed iteration or reduction loops |

This order means each tier unblocks progressively more model architectures: Tier 1+2 handles MLPs, adding Tier 3 handles data pipelines, adding Tier 4 handles CNNs and transformers.

### D4: Scheduler Integration via Yield Callback

The executor accepts an optional `yield_fn: Option<fn()>` called after each operator completes. In kernel mode, this calls into the cooperative scheduler to allow higher-priority System/IPC tasks to preempt. In container mode or tests, it's `None` (no yield).

**Why callback, not async:** The ONNX runtime is `#![no_std]` and doesn't use `async/await`. The kernel scheduler is cooperative — yield is a simple function call, not an await point. Adding async would require an executor runtime in the ONNX crate, which is unnecessary complexity.

### D5: Per-Operator Timing via Inline Measurement

Each operator dispatch measures wall-clock time (via `kernel::hal::timer_ticks()` in kernel mode, or skipped in test mode behind `#[cfg]`). The measured time is compared against `OperatorBudget` thresholds from the scheduler. Soft limit logs a warning; hard limit (10x budget) aborts with `SessionError::ExecutionFailed`.

Timing is opt-in via a `Session` configuration flag (`enable_profiling: bool`) to avoid overhead in production inference where budgets aren't needed.

### D6: New File `executor.rs` for Graph Traversal

The graph execution loop lives in a new `onnx-rt/src/executor.rs` file rather than inline in `session.rs`. This keeps `session.rs` focused on the public API (load, validate, configure) and puts execution logic (tensor routing, dispatch, timing) in its own module.

`Session::run()` calls `executor::execute_graph(&self.graph, inputs, &self.initializers, yield_fn, profiling)`.

## Risks / Trade-offs

**[Risk] BTreeMap allocation per inference call** — Creating a new BTreeMap for each `run()` call allocates. Mitigation: The map is populated once with graph inputs + initializers, then grows as intermediates are computed. For typical models (50-200 tensors), this is a few KB. A future optimization can pre-allocate a reusable arena keyed by the memory planner's buffer assignments.

**[Risk] Missing ONNX attribute handling** — Some operators (Conv, Softmax, Gemm) have attributes (padding, axis, transA/transB). The current `ExecutionNode` doesn't carry attributes. Mitigation: Extend `ExecutionNode` with an `attributes: Vec<(String, AttributeValue)>` field during graph construction. The `NodeProto` already has attributes; they just need to be propagated.

**[Risk] f32-only operators** — All current implementations are f32. Models using f16/bf16/int8 will fail at dispatch. Mitigation: Cast operator handles type conversion, and the Tensor type already supports multiple DataTypes. Real mixed-precision support is a future change.

**[Trade-off] Flat dispatch vs. extensibility** — The flat match approach means adding a new operator requires editing the match. This is acceptable for a closed operator set and avoids the overhead of a registration system.

## Open Questions

- **Q1:** Should the executor reuse the memory planner's buffer assignments for the tensor value map, or is that a separate optimization? *Leaning toward: separate optimization, post-MVP.*
- **Q2:** How should initializer tensors (model weights) be stored? Currently the protobuf parser extracts them but they're not persisted in the Session. Need to wire `GraphProto.initializer` into the executor's initial tensor map.
