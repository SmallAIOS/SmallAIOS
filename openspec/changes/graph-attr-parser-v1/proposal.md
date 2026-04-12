## Why

Phase 2 of the ONNX coverage roadmap (`generative-llm-v1`, PR #89) ships the sub-graph executor (`onnx-rt/src/sub_executor.rs`) and the three control-flow operators (`If`, `Loop`, `Scan` in `onnx-rt/src/ops/control_flow.rs`). Both landed with full unit-test coverage using synthetic, in-memory `ExecutionGraph` bodies constructed by test-only `_with_body` helpers (`op_loop_with_body`, `op_if_with_body`, `op_scan_with_body`).

What Phase 2 **deferred** — and what this change closes — is the protobuf parser side of that story. Today `onnx-rt/src/protobuf.rs::decode_attribute` explicitly skips field 6 of `AttributeProto`:

```rust
6 => {
    // g (GraphProto) — skip for now
    decoder.skip_field(header.wire_type)?;
}
```

That means when a *real* on-disk ONNX file contains an `If`/`Loop`/`Scan` node, the body `GraphProto` is silently dropped during parsing. The dispatcher then has no inner graph to execute and must bail out. The control-flow operators are effectively only reachable through unit tests, not through `Session::new_from_file()`.

Closing this gap unlocks **single-call in-graph autoregressive generation** for any decoder-only model exported via `optimum --use-cache`-style tooling. Those are the models the entire Phase 2 sub-graph executor was built to run. Without this parser change, they still require an external host-side token loop.

The deferral was intentional: Agent L, who implemented Phase 2, scoped the parser extension out as a "focused follow-up" because (a) the sub-graph executor itself is the large, risky piece and deserved its own change, and (b) graph-attribute parsing is mechanically simple once the executor is in place. See the `generative-llm-v1` archived change and the PR #89 description for the original rationale.

## What Changes

- **Extend `AttributeProto`** (`onnx-rt/src/onnx_types.rs`) with a new field `pub g: Option<Box<GraphProto>>`. Boxed because `GraphProto` is large and the common case (scalar attributes) should not pay for it.
- **Decode field 6 of `AttributeProto`** in `protobuf.rs::decode_attribute`. Replace the `skip_field` stub with a recursive call to `decode_graph` (which already exists for parsing top-level `ModelProto.graph`). Flip `attr_type` to `AttributeType::Graph` when the field is encountered without an explicit field 20 type tag.
- **Extend `ExecutionNode`** (`onnx-rt/src/graph.rs`) with a new field `pub inner_graphs: BTreeMap<String, ExecutionGraph>`, keyed by attribute name (`body`, `then_branch`, `else_branch`).
- **Recursive graph compilation**. Extend `build_execution_graph` to walk each created `ExecutionNode`'s attributes, and for every attribute with a non-`None` `g` field, compile the inner `GraphProto` into its own `ExecutionGraph` via a recursive call, storing the result on the parent node's `inner_graphs` map.
- **Dispatcher integration**. Update `executor.rs::dispatch_node` so that `OpKind::If`/`Loop`/`Scan` read the body graph from `node.inner_graphs` and hand it to `sub_executor::run_sub_graph` directly. Delete (or `#[cfg(test)]`-gate and `#[doc(hidden)]`) the test-only `op_loop_with_body` / `op_if_with_body` / `op_scan_with_body` shim helpers.
- **End-to-end test**. Add a fixture-based integration test that loads a small synthetic ONNX file containing an embedded `Loop` body and runs it to completion via the public `Session` API, proving the path works without any test-only helpers. Feature-gated like the existing fixture tests so it does not need to run in every CI shard.

### Modified Capabilities

- `onnx-cpu-execution`: Add three requirements covering graph-attribute parsing, inner-graph compilation, and dispatcher wiring.

## Impact

- **Code:**
  - `onnx-rt/src/onnx_types.rs` — add `AttributeProto.g` field and update `Default`.
  - `onnx-rt/src/protobuf.rs` — decode field 6 of `AttributeProto` via recursive `decode_graph`; add `MAX_GRAPH_NESTING_DEPTH` safety net and `ProtoError::NestingTooDeep` variant.
  - `onnx-rt/src/graph.rs` — add `ExecutionNode.inner_graphs` field; extend `build_execution_graph` to compile inner graphs recursively.
  - `onnx-rt/src/executor.rs` — route `OpKind::If`/`Loop`/`Scan` dispatch through `node.inner_graphs`.
  - `onnx-rt/src/sub_executor.rs` — drop the three `_with_body` test shims (or gate them behind `#[cfg(test)]` + `#[doc(hidden)]`).
  - Tests: ~8 new unit tests covering field-6 decoding round-trips, nested-graph depth limiting, inner-graph compilation, and recursive topological sort; 1 end-to-end fixture test with a real on-disk ONNX `Loop`.
- **APIs:** No breaking changes to the public `Session` API. `AttributeProto` gains a new public field (additive); `ExecutionNode` gains a new public field (additive). `cargo-semver-checks` will flag these as minor bumps; conventional commit is `feat(onnx-rt)`.
- **Estimated size:** ~300 LOC of production code plus ~400 LOC of tests. Small and focused.
- **Risk:** The graph builder has so far only been called on the top-level `GraphProto`. Recursive compilation must not rely on any module-level state (it currently does not — `build_execution_graph` is a pure function over its `GraphProto` argument), and must not accidentally treat inner graph inputs as missing when they reference outer-graph values. The design note D4 addresses this: outer-referenced names are intentionally left as `MissingInput`-safe (handled by the sub-graph executor at runtime, not at build time) because the same mechanism already exists for loop-carried values.

## Out of Scope

- **Full `Scan` axis handling.** `Scan` uses the same field-6 machinery to carry its body, so the parser change automatically unblocks it. However, `Scan` has additional `scan_input_axes` / `scan_output_axes` / `scan_input_directions` / `scan_output_directions` attributes that control iteration order and output stacking. Those are already decoded by Phase 2 as integer-list attributes; this change does not touch them. If the Phase 2 sub-graph executor fully consumes those attributes (it should — see `op_scan_with_body` in `sub_executor.rs`), `Scan` lights up for free. If not, a small follow-up will be needed.
- **Other deferred `AttributeType` variants** (`SparseTensor`, `TypeProto`, `Graphs` plural, `SparseTensors`, `TypeProtos`). These are not used by any operator in the Phase 1 or Phase 2 inventory. This change is deliberately narrow to field 6 only.
- **Graph-attribute support for operator-set-dependent opset version checks.** Inner graphs inherit the outer graph's opset; per-subgraph opset overrides are an ONNX feature no one in our inventory uses.

## Risks

- **Recursive `build_execution_graph` stack depth.** The ONNX schema permits arbitrary nesting, but in practice all models we care about have nesting depth ≤ 3 (typical: a `Loop` containing a body with a few MatMul/Add ops). We add a `MAX_GRAPH_NESTING_DEPTH = 16` safety net returning `GraphError::NestingTooDeep` on overflow. This is cheaper and more predictable than letting the host stack blow up.
- **Dead test helpers becoming stale.** The sub-graph executor currently exposes three test-only constructors that build inner graphs by hand. Once the real parser path lands, those helpers should be deleted, not left around, so Phase 2's tests prove the same code path the production parser exercises. The dispatcher wiring step in this change includes that deletion (or at minimum `#[cfg(test)]`-gating).
- **Ordering with `generative-llm-v1` merge.** This change is logically a follow-up to PR #89 and cannot be implemented until #89 merges. The OpenSpec artifacts in this folder can be reviewed independently; implementation is gated on #89 landing in develop.
