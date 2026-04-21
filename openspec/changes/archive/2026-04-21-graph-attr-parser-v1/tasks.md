# Tasks: graph-attr-parser-v1

> Implementation gated on `generative-llm-v1` (PR #89) landing in develop.
> These tasks assume `onnx-rt/src/sub_executor.rs` and `onnx-rt/src/ops/control_flow.rs` already exist.

## 1. AttributeProto extension

- [x] 1.1 Add `pub g: Option<alloc::boxed::Box<GraphProto>>` field to `AttributeProto` in `onnx-rt/src/onnx_types.rs`.
- [x] 1.2 Update `impl Default for AttributeProto` to initialize `g: None`.
- [x] 1.3 Update the existing `AttributeProto` unit tests (`onnx_types.rs` tests module) to include a case asserting `g == None` on default, and one case constructing a graph-valued attribute by hand.

## 2. Protobuf decoder field 6

- [x] 2.1 Add `ProtoError::NestingTooDeep` variant and `const MAX_GRAPH_NESTING_DEPTH: usize = 16` in `onnx-rt/src/protobuf.rs`.
- [x] 2.2 Introduce private helpers `decode_graph_with_depth(bytes, depth)` and `decode_attribute_with_depth(bytes, depth)`; keep the public `decode_graph` / `decode_attribute` as thin wrappers calling at `depth = 0`.
- [x] 2.3 Replace the `6 => decoder.skip_field(...)` stub in `decode_attribute_with_depth` with a length-delimited read + recursive `decode_graph_with_depth(..., depth + 1)`. Set `attr.g = Some(Box::new(...))` and flip `attr_type` to `AttributeType::Graph` when currently `Undefined`.
- [x] 2.4 Add `protobuf.rs` unit tests: (a) round-trip a `Loop` body with one `Add` node through `decode_attribute`; (b) reject a 17-level nested graph with `ProtoError::NestingTooDeep`; (c) assert that a scalar attribute still round-trips with `g = None`; (d) assert `attr_type` auto-flips to `Graph` when field 20 is omitted.

## 3. Graph builder recursive compilation

- [x] 3.1 Add `GraphError::NestingTooDeep` variant in `onnx-rt/src/graph.rs`.
- [x] 3.2 Introduce private helper `build_execution_graph_with_depth(graph, depth)` and `build_execution_graph_inner(graph, depth)` (the latter relaxes `MissingInput` for names unknown in the inner scope). Public `build_execution_graph` becomes a thin wrapper at `depth = 0`.
- [x] 3.3 Change `create_execution_nodes` signature to `fn create_execution_nodes(graph: &GraphProto, depth: usize) -> Result<Vec<ExecutionNode>, GraphError>`; for each attribute with `g.is_some()`, recursively call `build_execution_graph_inner(inner, depth + 1)` and insert into the node's `inner_graphs`.
- [x] 3.4 Thread the new `Result` return through `build_execution_graph` (currently calls `create_execution_nodes` infallibly — now it must `?`).
- [x] 3.5 Add graph builder unit tests: (a) top-level Loop with inner MatMul+Add compiles and populates `inner_graphs["body"]`; (b) If with both branches populates two entries; (c) nesting depth 17 returns `GraphError::NestingTooDeep`; (d) inner graph referencing an outer-defined tensor name does NOT return `MissingInput`.

## 4. ExecutionNode.inner_graphs field

- [x] 4.1 Add `pub inner_graphs: alloc::collections::BTreeMap<String, ExecutionGraph>` field to `ExecutionNode` in `onnx-rt/src/graph.rs`.
- [x] 4.2 Initialize to empty in any `ExecutionNode` construction site outside `create_execution_nodes` (search with `grep -rn 'ExecutionNode {' onnx-rt/`); typically only test helpers need updating.
- [x] 4.3 Make sure `ExecutionNode` still derives `Debug, Clone`; verify `BTreeMap<String, ExecutionGraph>` satisfies both.

## 5. Dispatcher integration + delete test-only helpers

- [x] 5.1 In `onnx-rt/src/executor.rs::dispatch_node`, handle `OpKind::If` by looking up `node.inner_graphs["then_branch"]` and `node.inner_graphs["else_branch"]`, then calling `sub_executor::run_sub_graph` directly with the selected branch. Surface `ExecutionError::MissingInnerGraph(&'static str)` on a `None` lookup.
- [x] 5.2 Handle `OpKind::Loop` by looking up `node.inner_graphs["body"]` and calling `sub_executor::run_loop(...)`. Likewise for `OpKind::Scan` → `sub_executor::run_scan(...)`.
- [x] 5.3 Delete `op_if_with_body`, `op_loop_with_body`, `op_scan_with_body` from `onnx-rt/src/sub_executor.rs` (or gate them behind `#[cfg(test)] #[doc(hidden)]` if a handful of Phase 2 tests still need them as a transition step — prefer deletion, update the tests to go through the parser/builder path).
- [x] 5.4 Update any Phase 2 unit tests that currently call `_with_body` helpers: rewrite them to construct an `AttributeProto { name: "body".into(), g: Some(Box::new(inner_proto)), attr_type: AttributeType::Graph, .. }` and let the graph builder compile it. Grep for `_with_body` in `onnx-rt/` and confirm zero hits post-change.

## 6. End-to-end test with synthetic Loop ONNX file

- [x] 6.1 Add a Python fixture builder script `onnx-rt/tests/fixtures/scripts/build_loop_fixture.py` that uses the `onnx` library (not torch — keep it minimal) to construct a `ModelProto` with a `Loop` wrapping `MatMul + Add`, running for a fixed `M = 4` iterations. Write the result to `onnx-rt/tests/fixtures/loop_matmul_add.onnx`. Document prerequisites (`pip install onnx`) in a top-of-file comment.
- [x] 6.2 Add an integration test in `onnx-rt/tests/` (or the existing fixture-test module) gated behind the `onnx-fixtures` cargo feature. The test loads `loop_matmul_add.onnx` via `Session::new_from_file()`, runs it with known inputs, and asserts the output matches a hand-computed reference within `f32::EPSILON * 16.0`.
- [x] 6.3 Verify `grep -n '_with_body' onnx-rt/src/` returns zero hits outside `#[cfg(test)]` (or zero hits total).

## 7. Validation: fmt / clippy / test

- [x] 7.1 `just fmt-check` (or `cargo fmt -- --check`) passes.
- [x] 7.2 `just clippy` (or `cargo clippy --workspace -- -D warnings`) passes.
- [x] 7.3 `just test` passes (all existing 4,143+ tests plus the new unit + fixture tests). Confirm `cargo-semver-checks` flags the change as `minor` (new public fields on `AttributeProto` and `ExecutionNode`) and that the PR title is `feat(onnx-rt): decode AttributeProto.g for on-disk If/Loop/Scan`.
