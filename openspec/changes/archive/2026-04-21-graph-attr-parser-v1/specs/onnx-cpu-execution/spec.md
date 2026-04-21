## ADDED Requirements

### Requirement: Graph Attribute Parsing
The ONNX runtime SHALL decode `AttributeProto.g` (field 6, a single nested `GraphProto`) into an owned `Option<Box<GraphProto>>` value when parsing model files. The parser SHALL enforce a maximum graph nesting depth of 16 and return `ProtoError::NestingTooDeep` for deeper inputs.

#### Scenario: Loop body GraphProto round-trips losslessly
- **WHEN** a model file contains a `Loop` node whose `body` attribute (`AttributeProto`, field 6) wraps a `GraphProto` with two nodes (`MatMul`, `Add`)
- **THEN** `decode_attribute` MUST return an `AttributeProto` with `g = Some(Box::new(...))`
- **AND** the returned inner `GraphProto` MUST contain exactly two `NodeProto` entries with `op_type = "MatMul"` and `op_type = "Add"` in that order
- **AND** `attr_type` MUST be set to `AttributeType::Graph` even when the original bytes omit field 20

#### Scenario: Nested graph depth limit
- **WHEN** a maliciously constructed model nests `AttributeProto.g` values 17 levels deep
- **THEN** `decode_attribute` MUST return `ProtoError::NestingTooDeep`
- **AND** MUST NOT recurse into the 17th level
- **AND** MUST NOT blow the host stack

#### Scenario: Non-graph attribute unchanged
- **WHEN** a model file contains a scalar `Float` attribute with no `g` field set
- **THEN** `decode_attribute` MUST return `g = None`
- **AND** the parser behavior for all other `AttributeType` variants MUST be unchanged from the pre-change baseline

### Requirement: Inner Graph Compilation
The graph builder SHALL compile every `AttributeProto.g` value reached during `build_execution_graph` into its own standalone `ExecutionGraph` and store it on the parent `ExecutionNode.inner_graphs`, keyed by the attribute's name.

#### Scenario: Loop body compiled and cached on parent node
- **WHEN** `build_execution_graph` is called on a `GraphProto` whose top-level contains a `Loop` node with a `body` graph attribute
- **THEN** the returned `ExecutionGraph.nodes[loop_index].inner_graphs` MUST contain a single entry with key `"body"`
- **AND** the inner `ExecutionGraph` MUST have its own populated `topological_order` with `node_count > 0`
- **AND** the parent `ExecutionNode.attributes` MUST still contain the original `AttributeProto` (attribute cloning is not replaced by the inner-graph map)

#### Scenario: If node compiles both branches
- **WHEN** `build_execution_graph` is called on a graph containing an `If` node with both `then_branch` and `else_branch` attributes
- **THEN** the parent `ExecutionNode.inner_graphs` MUST contain two entries with keys `"then_branch"` and `"else_branch"`
- **AND** each inner `ExecutionGraph` MUST have its own topological order

#### Scenario: Inner graph outer-referenced names are not rejected
- **WHEN** an inner graph references a tensor name defined by the outer graph (captured value or loop-carried input) rather than produced by a sibling node inside the inner graph
- **THEN** the recursive `build_execution_graph_inner` call MUST NOT return `GraphError::MissingInput` for that name
- **AND** MUST leave resolution to the sub-graph executor at runtime

#### Scenario: Inner graph nesting depth overflow
- **WHEN** `build_execution_graph` encounters a chain of nested inner graphs deeper than 16 levels
- **THEN** the builder MUST return `GraphError::NestingTooDeep`

### Requirement: Sub-Graph Dispatch From Inner Graphs
The dispatcher SHALL execute `If`, `Loop`, and `Scan` operators by reading the compiled inner graph from `ExecutionNode.inner_graphs`, without relying on any test-only constructors or externally passed body parameters.

#### Scenario: On-disk Loop model runs end-to-end
- **WHEN** a `Session` is created from a model file whose top-level graph contains a `Loop` wrapping a body of `MatMul + Add` and `Session::run()` is called with valid inputs
- **THEN** the dispatcher MUST retrieve `node.inner_graphs["body"]` at dispatch time
- **AND** MUST pass it directly to `sub_executor::run_loop`
- **AND** MUST NOT invoke any function whose name contains `_with_body`
- **AND** the final output tensors MUST match a hand-computed reference within `f32::EPSILON * 16.0`

#### Scenario: On-disk If model selects a branch
- **WHEN** a model file contains an `If` node and the runtime condition evaluates to `true`
- **THEN** the dispatcher MUST execute `node.inner_graphs["then_branch"]` via `sub_executor::run_sub_graph`
- **AND** MUST NOT execute the `else_branch`

#### Scenario: Missing inner graph surfaces a clear error
- **WHEN** a malformed model presents an `If` node whose `then_branch` attribute has `g = None`
- **THEN** the dispatcher MUST return `ExecutionError::MissingInnerGraph("then_branch")`
- **AND** MUST NOT panic
