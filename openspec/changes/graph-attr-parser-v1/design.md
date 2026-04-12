# Design: graph-attr-parser-v1

## Context

Phase 2 (`generative-llm-v1`, PR #89) shipped the sub-graph executor (`onnx-rt/src/sub_executor.rs`) and the three control-flow operators (`If`/`Loop`/`Scan` in `onnx-rt/src/ops/control_flow.rs`) with full unit-test coverage. Both use synthetic, in-memory `ExecutionGraph` bodies constructed by test-only `_with_body` helpers. The protobuf parser side of that story was explicitly deferred.

Today:

```rust
// onnx-rt/src/protobuf.rs, inside decode_attribute()
6 => {
    // g (GraphProto) — skip for now
    decoder.skip_field(header.wire_type)?;
}
```

and:

```rust
// onnx-rt/src/onnx_types.rs
pub struct AttributeProto {
    pub name: String,
    pub attr_type: AttributeType,
    pub f: f32,
    pub i: i64,
    pub s: Vec<u8>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<Vec<u8>>,
    pub t: Option<Box<TensorProto>>,
    // no `g` field
}
```

This change closes that gap with minimum surface area.

## Goals

- Decode `AttributeProto.g` (field 6) on-disk so `Session::new_from_file()` can load models with embedded `If`/`Loop`/`Scan` bodies.
- Wire the compiled inner graph into the existing sub-graph executor without changing its public behaviour.
- Delete the test-only `_with_body` shims — after this change, the production parser path is the only path, and the existing Phase 2 tests still cover the executor logic because they can now build an `AttributeProto { g: Some(...) }` directly instead of calling a special constructor.

## Non-Goals

- Decoding other unimplemented `AttributeType` variants (`SparseTensor`, `TypeProto`, plural `Graphs`, etc.).
- Adding operator-set overrides at the sub-graph level (no consumer uses them).
- Redesigning the sub-graph executor's value-scope semantics — those stay as Phase 2 shipped them.

## Decisions

### D1: `AttributeProto` field layout

ONNX `AttributeProto` field 6 is `g: GraphProto` (a single nested message, not repeated). Add a corresponding Rust field:

```rust
pub struct AttributeProto {
    // ... existing fields ...
    pub t: Option<Box<TensorProto>>,
    pub g: Option<Box<GraphProto>>,  // NEW
}
```

Boxed because `GraphProto` is significantly larger than the other scalar fields and the common case (`attr_type` in `{Float, Int, Ints, Floats, Tensor}`) should not pay for the inline memory. Matches the existing treatment of `t: Option<Box<TensorProto>>`. Update `Default for AttributeProto` to set `g: None`.

### D2: Recursive parser entry point

The protobuf decoder already has a `decode_graph` function used by `decode_model` for the top-level `ModelProto.graph` (field 2). Reuse it in `decode_attribute` field 6:

```rust
6 => {
    // g (GraphProto) — recursive parse of nested body graph.
    let graph_data = decoder.read_length_delimited()?;
    let nested = decode_graph_with_depth(graph_data, depth + 1)?;
    attr.g = Some(alloc::boxed::Box::new(nested));
    if attr.attr_type == AttributeType::Undefined {
        attr.attr_type = AttributeType::Graph;
    }
}
```

Recursion is bounded by the graph nesting depth in the ONNX file — typically 1 (body of a `Loop` containing standard ops), occasionally 2 (`Loop` body containing an `If`). We add an explicit guard:

```rust
const MAX_GRAPH_NESTING_DEPTH: usize = 16;
```

returning `ProtoError::NestingTooDeep` if exceeded. The public `decode_graph` / `decode_attribute` entry points keep their existing signatures; the depth counter threads through private helpers `decode_graph_with_depth` / `decode_attribute_with_depth`. The public entrypoints just start at depth 0.

### D3: When does the inner `GraphProto` become an `ExecutionGraph`?

Two options:

- **(a) At parse time** — `decode_attribute` returns an already-compiled `ExecutionGraph`, never a raw `GraphProto`. Pro: errors caught at load. Con: the protobuf layer would need to depend on the graph-builder layer, inverting the current parser → graph → executor flow. Con: every parsed model pays compilation cost even for attributes it ignores.
- **(b) At graph-build time** — parser stores the raw `GraphProto`; the graph builder compiles it during `build_execution_graph`. Pro: compilation is one explicit step the graph builder is already responsible for. Pro: preserves layering (protobuf layer stays pure data, graph layer owns "now executable").

**Decision: (b).** The graph builder is the single source of truth for "this is now executable", and the compilation cost should be visible at the call site that traverses nodes. The parser stays a dumb deserializer.

### D4: Storage on `ExecutionNode`

Add a new field to `ExecutionNode`:

```rust
pub struct ExecutionNode {
    // ... existing fields ...
    pub attributes: Vec<AttributeProto>,
    pub inner_graphs: BTreeMap<String, ExecutionGraph>,  // NEW
}
```

Keyed by attribute name because `If` uses *two* inner graphs (`then_branch` and `else_branch`) and `Loop`/`Scan` use one (`body`). A `BTreeMap<String, _>` is deterministic-order (important for debug output) and small (≤ 2 entries in practice). `BTreeMap` rather than `Vec<(String, _)>` for readable lookup at dispatch time.

The graph builder populates this map inside `create_execution_nodes`:

```rust
fn create_execution_nodes(graph: &GraphProto) -> Result<Vec<ExecutionNode>, GraphError> {
    let mut out = Vec::with_capacity(graph.node.len());
    for (i, node_proto) in graph.node.iter().enumerate() {
        let mut inner_graphs = BTreeMap::new();
        for attr in &node_proto.attribute {
            if let Some(inner_proto) = &attr.g {
                let inner_exec = build_execution_graph_with_depth(inner_proto, depth + 1)?;
                inner_graphs.insert(attr.name.clone(), inner_exec);
            }
        }
        out.push(ExecutionNode {
            node_index: NodeIndex::new(i),
            op_type: node_proto.op_type.clone(),
            name: node_proto.name.clone(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            dependencies: Vec::new(),
            attributes: node_proto.attribute.clone(),
            inner_graphs,
        });
    }
    Ok(out)
}
```

`create_execution_nodes` changes from infallible to `Result<..., GraphError>` (it can now propagate inner-graph compilation failures). `build_execution_graph` itself threads depth through a new private helper `build_execution_graph_with_depth`, matching the parser's depth-threading pattern. `GraphError::NestingTooDeep` is added alongside `CyclicGraph`/`MissingInput`.

**Outer-referenced names.** Inside an `If`/`Loop` body, a node may reference a tensor defined in the *outer* graph (captured values and loop-carried inputs). The current `resolve_dependencies` raises `GraphError::MissingInput` for any unresolved name. To stay layering-clean, the inner `build_execution_graph` call must *not* treat outer names as errors. We handle this by:

1. Passing the outer graph's known-names set as an allow-list into the recursive call, OR
2. Deferring the check: treat any unresolved name in an inner graph as an implicit outer reference (the sub-graph executor already seeds these at runtime from its parent scope).

**Decision: option 2.** The sub-graph executor already owns the value-resolution semantics (it explicitly merges outer scope at entry). The graph builder's only job for inner graphs is to produce a topologically ordered list of nodes; unresolved names are left to the runtime to satisfy. We add a new `build_execution_graph_inner` variant that skips the `MissingInput` check for names absent from *both* input_names and the producer map. Cycles are still errors; missing inputs become runtime `SubExecutorError::UnresolvedOuterRef` if they are never supplied.

### D5: Sub-graph executor integration

Phase 2 shipped the executor with three test-only constructors:

```rust
// onnx-rt/src/sub_executor.rs (Phase 2)
#[doc(hidden)]
pub fn op_if_with_body(
    cond: &Tensor,
    then_body: &ExecutionGraph,
    else_body: &ExecutionGraph,
    ctx: &mut SubExecCtx,
) -> Result<Vec<Tensor>, SubExecError> { ... }

#[doc(hidden)]
pub fn op_loop_with_body(...) -> Result<_, _> { ... }

#[doc(hidden)]
pub fn op_scan_with_body(...) -> Result<_, _> { ... }
```

After this change, `executor.rs::dispatch_node` calls `sub_executor::run_sub_graph` directly, reading the body from `node.inner_graphs`:

```rust
OpKind::If => {
    let cond = values.get_required(&node.inputs[0])?;
    let then_body = node.inner_graphs.get("then_branch")
        .ok_or(ExecutionError::MissingInnerGraph("then_branch"))?;
    let else_body = node.inner_graphs.get("else_branch")
        .ok_or(ExecutionError::MissingInnerGraph("else_branch"))?;
    let selected = if cond_is_true(cond)? { then_body } else { else_body };
    sub_executor::run_sub_graph(selected, /* captured outer values */, ctx)?
}
OpKind::Loop => {
    let body = node.inner_graphs.get("body")
        .ok_or(ExecutionError::MissingInnerGraph("body"))?;
    sub_executor::run_loop(body, /* M, cond, v_initial */, ctx)?
}
OpKind::Scan => {
    let body = node.inner_graphs.get("body")
        .ok_or(ExecutionError::MissingInnerGraph("body"))?;
    sub_executor::run_scan(body, /* inputs, axes */, ctx)?
}
```

The three `_with_body` helpers are deleted. Phase 2's unit tests that currently call them are updated (straightforward mechanical change: build an `AttributeProto { name: "body".into(), attr_type: AttributeType::Graph, g: Some(Box::new(inner_proto)), ..Default::default() }`, push it into a `NodeProto.attribute`, let the graph builder compile it).

### D6: Validation

Three layers of validation:

1. **Protobuf round-trip unit tests** (`protobuf.rs` tests module): encode a minimal `AttributeProto` whose `g` field contains a hand-built `GraphProto` with one `Add` node; decode; assert the round-tripped `AttributeProto.g` has the expected node count, inputs, outputs. Plus a negative test: decode a maliciously-nested 17-deep graph and expect `ProtoError::NestingTooDeep`.

2. **Graph builder unit tests** (`graph.rs` tests module): construct an in-memory `GraphProto` whose top-level contains a `Loop` node with an `AttributeProto { g: Some(inner), .. }`. Call `build_execution_graph`. Assert the returned `ExecutionGraph.nodes[0].inner_graphs["body"]` has the expected node count and topological order. Plus negative tests for nesting-depth overflow and inner-graph cycles.

3. **End-to-end fixture test** (feature-gated, in `onnx-rt/tests/` or the existing `tests/fixtures/` mechanism): a small synthetic ONNX file — a 2-input `Loop` wrapping a `MatMul + Add` body — exported from a scripted PyTorch builder (`tests/fixtures/scripts/build_loop_fixture.py`, not checked into the main test shard). The test loads the file via `Session::new_from_file()`, runs it, and compares against a reference value. Gated behind an `onnx-fixtures` cargo feature so it only runs when the fixture is present, matching the pattern used by the existing `bert-tiny` / `vit-tiny` fixture tests.

## Alternatives Considered

### Alt 1: Explicit `Graph` IR layer between parser and executor

Introduce a new `ir::Graph` type that sits between `GraphProto` and `ExecutionGraph`, with its own recursion/validation logic, and have the graph builder consume `ir::Graph` instead of `GraphProto`.

**Rejected** because (a) it duplicates work `ExecutionGraph` already does (topological sort, dependency resolution, name tables), (b) it adds a layer that no other part of the runtime needs, and (c) the recursion bound is cheap enough that threading a depth counter through two existing functions is strictly less code than adding a new IR.

### Alt 2: Compile inner graphs lazily at first dispatch

Store raw `GraphProto` on `ExecutionNode` and compile it the first time `dispatch_node` hits the parent op. Pro: cheaper load for models that have `If` branches they never take.

**Rejected** because (a) the savings are negligible — inner graphs are small by construction, and (b) we prefer "load-time errors" so a malformed model fails fast at `Session::new()` rather than mid-inference during a deadline-critical call. Consistent with the project's DO-178C-oriented design stance.

### Alt 3: Flatten all inner graphs into the outer graph at parse time

Inline `If`/`Loop`/`Scan` bodies into the parent graph with gated execution, eliminating the need for a sub-graph executor entirely.

**Rejected** because (a) Phase 2 already shipped the sub-graph executor and extensively tested it — throwing it away now is a pure loss, (b) `Loop` has dynamic iteration counts that cannot be flattened at compile time, and (c) flattening `If` duplicates the dead branch's work because `dispatch_node` would still visit both paths in topological order.

## Open Questions

None at planning time. The sub-graph executor's scope-management semantics (Phase 2) are already settled, and the parser extension is mechanically simple once field 6 is wired up.

## Success Criteria

- `decode_attribute` round-trips `AttributeProto.g` losslessly for up to 16 levels of nesting; depth 17 returns `ProtoError::NestingTooDeep`.
- `build_execution_graph` populates `inner_graphs` for every node whose `NodeProto` has graph-valued attributes, with correct topological order on each inner graph.
- `executor::dispatch_node` handles `OpKind::If`/`Loop`/`Scan` entirely from `ExecutionNode.inner_graphs` — `grep -n '_with_body' onnx-rt/src/` returns zero hits outside `#[cfg(test)]` blocks (or zero hits total if the helpers are fully deleted).
- End-to-end: a synthetic on-disk `loop.onnx` file loads via `Session::new_from_file()` and runs to completion, producing the expected output tensor, with zero test-only helper calls in the runtime path.
- `cargo fmt`, `cargo clippy -D warnings`, and `cargo test --workspace` all pass.
- Estimated ~300 LOC of production code, ~400 LOC of tests.
