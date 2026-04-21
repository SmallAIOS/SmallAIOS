# onnx-cpu-execution Specification

## Purpose

CPU operator dispatch, graph traversal, tensor value routing, and
scheduler integration for end-to-end ONNX inference. Companion
capability to `metal-gpu-inference` and `onnx-runtime` — this spec
focuses on the executor's dispatch path, how GPU and CPU
implementations interleave within a single graph, the tier-1 CPU
operator surface, scheduler/yield integration, and per-op timing
budgets.
## Requirements
### Requirement: Graph Executor Tensor Routing
The ONNX runtime SHALL execute inference by iterating the topologically-sorted execution graph, routing tensor values between operators via a named tensor map.

#### Scenario: Execute a linear graph (MatMul → Add → Relu)
- **WHEN** a session is loaded with a 3-node graph (MatMul → Add → Relu) and `run()` is called with valid input tensors
- **THEN** the executor MUST iterate nodes in topological order
- **AND** each node MUST read its inputs from the tensor value map by name
- **AND** each node MUST write its outputs to the tensor value map
- **AND** the final output tensor MUST match the expected mathematical result

#### Scenario: Execute a graph with branching (shared input to multiple operators)
- **WHEN** a graph has one input feeding both a MatMul and an Add node
- **THEN** the executor MUST make the input tensor available to both nodes
- **AND** both branches MUST execute and produce correct outputs

#### Scenario: Initializer tensors loaded before execution
- **WHEN** a model contains initializer tensors (weights, biases)
- **THEN** the executor MUST load all initializers into the tensor value map before executing any node
- **AND** initializer values MUST be accessible by name during operator dispatch

### Requirement: Operator Dispatch via OpKind
The ONNX runtime SHALL dispatch each graph node to the corresponding CPU operator function based on the node's `op_type` string.

#### Scenario: Dispatch a supported operator
- **WHEN** a graph node has `op_type` matching a registered `OpKind` variant
- **THEN** the dispatcher MUST call the corresponding `op_*` function with the node's input tensors
- **AND** MUST return the operator's output tensor(s)

#### Scenario: Reject an unsupported operator at runtime
- **WHEN** a graph node has an `op_type` not in the `OpKind` enum
- **THEN** the dispatcher MUST return `SessionError::ExecutionFailed` with the unsupported operator name
- **AND** MUST NOT execute any subsequent nodes

### Requirement: Operator Dispatch Path
The executor's `dispatch_node` function SHALL check for an available GPU backend before falling through to CPU execution. When a GPU backend is present and the operator is in its supported set, execution SHALL occur on the GPU. When no GPU backend is present or the operator is not supported, execution SHALL fall through to the existing CPU implementation with zero behavioral change.

#### Scenario: Mixed GPU/CPU graph execution
- **WHEN** a model graph contains both GPU-supported and unsupported operators
- **THEN** the executor MUST interleave GPU and CPU dispatch within the same graph traversal
- **AND** tensor values MUST be transferred between host and device as needed
- **AND** the final graph outputs MUST be identical (within floating-point tolerance) to a pure-CPU execution of the same graph

### Requirement: Node Attribute Propagation
The ONNX runtime SHALL propagate operator attributes from the ONNX protobuf `NodeProto` through to operator dispatch.

#### Scenario: Conv operator receives padding and stride attributes
- **WHEN** a Conv node specifies `pads`, `strides`, and `kernel_shape` attributes
- **THEN** the executor MUST pass these attributes to the `op_conv` function
- **AND** the convolution MUST use the specified padding and stride values

#### Scenario: Softmax operator receives axis attribute
- **WHEN** a Softmax node specifies an `axis` attribute
- **THEN** the executor MUST pass the axis value to `op_softmax`
- **AND** softmax normalization MUST be computed along the specified axis

### Requirement: Tier 1 CPU Operator Completeness
The ONNX runtime SHALL implement all 29 Tier 1 operators for CPU execution with f32 tensors, with parallel variants for compute-heavy operators.

#### Scenario: Element-wise binary operators (Sub, Mul, Div)
- **WHEN** two f32 tensors are provided as inputs to Sub, Mul, or Div
- **THEN** the operator MUST compute the element-wise result with NumPy-style broadcasting
- **AND** the output shape MUST match the broadcast output shape
- **AND** if num_elements exceeds the parallel threshold, computation MUST be distributed across available cores

#### Scenario: Gemm operator wraps GEMM micro-kernel
- **WHEN** a Gemm node is dispatched with matrices A, B, and optional bias C
- **THEN** the operator MUST compute `alpha * A @ B + beta * C` using the existing `gemm_f32` micro-kernel
- **AND** MUST support `transA` and `transB` attributes
- **AND** if M × K × N exceeds the parallel threshold, tile rows MUST be distributed across available cores

#### Scenario: Activation operators (Sigmoid, Tanh)
- **WHEN** an f32 tensor is provided to Sigmoid or Tanh
- **THEN** the operator MUST compute the element-wise activation function
- **AND** Sigmoid MUST use `1 / (1 + exp(-x))` with the `no_std` `expf_approx`
- **AND** Tanh MUST use `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- **AND** if num_elements exceeds the parallel threshold, computation MUST be distributed across available cores

#### Scenario: Shape manipulation operators (Transpose, Flatten, Squeeze, Unsqueeze)
- **WHEN** a tensor and shape parameters are provided
- **THEN** the operator MUST return a tensor with the requested shape
- **AND** the data MUST be reordered (Transpose) or reinterpreted (Flatten, Squeeze, Unsqueeze) correctly

#### Scenario: Data movement operators (Concat, Gather, Slice, Pad, Cast, Clip)
- **WHEN** tensors and operator-specific parameters are provided
- **THEN** each operator MUST produce the correct output per the ONNX specification
- **AND** Concat MUST support concatenation along any axis
- **AND** Cast MUST support conversion between f32, f16, int32, int64, and int8

#### Scenario: Normalization operators (BatchNormalization, LayerNormalization)
- **WHEN** an input tensor, scale, bias, mean, and variance are provided
- **THEN** the operator MUST normalize the input using the provided statistics
- **AND** the output MUST match the ONNX specification for the operator

#### Scenario: Pooling operators (MaxPool, AveragePool, GlobalAveragePool)
- **WHEN** a 4D tensor (NCHW) is provided with kernel size and stride
- **THEN** MaxPool MUST compute the maximum value in each pooling window
- **AND** AveragePool MUST compute the mean value in each pooling window
- **AND** GlobalAveragePool MUST reduce spatial dimensions to 1x1

#### Scenario: Reduction operators (ReduceMean, ReduceSum)
- **WHEN** a tensor and reduction axes are provided
- **THEN** the operator MUST reduce along the specified axes
- **AND** MUST support `keepdims` attribute (default true)

### Requirement: Scheduler Yield Between Operators
The ONNX runtime SHALL yield to the kernel scheduler after each operator completes to allow higher-priority tasks to execute.

#### Scenario: Yield after each operator in kernel mode
- **WHEN** the executor is configured with a yield callback and executes a multi-node graph
- **THEN** the yield callback MUST be invoked after each operator completes
- **AND** execution MUST resume at the next operator after the scheduler returns

#### Scenario: No yield in container/test mode
- **WHEN** the executor is configured without a yield callback
- **THEN** operators MUST execute sequentially without any yield points
- **AND** execution performance MUST NOT be affected by yield infrastructure

### Requirement: Per-Operator Timing and Budget Enforcement
The ONNX runtime SHALL measure operator execution time and enforce configurable time budgets.

#### Scenario: Soft budget warning
- **WHEN** an operator's execution time exceeds its soft budget threshold
- **THEN** the runtime MUST log a warning with the operator name, measured time, and budget

#### Scenario: Hard budget abort
- **WHEN** an operator's execution time exceeds 10x its configured budget
- **THEN** the runtime MUST abort the current inference
- **AND** MUST return `SessionError::ExecutionFailed` with timing details

#### Scenario: Profiling disabled by default
- **WHEN** a session is created without enabling profiling
- **THEN** no timing measurement overhead SHALL be incurred during inference

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

### Requirement: Sub-Graph Executor
The ONNX runtime SHALL support recursive execution of inner graphs embedded inside `If`, `Loop`, and `Scan` operators via a sub-graph executor with isolated value scope and shared initializer scope.

#### Scenario: Inner graph compiled once at session load time
- **WHEN** a model containing a `Loop` operator is loaded
- **THEN** the graph builder MUST compile the inner `GraphProto` into a standalone `ExecutionGraph` during `Session::new()`
- **AND** the compiled inner graph MUST be cached on the parent `ExecutionNode` and reused across all iterations at dispatch time
- **AND** the inner graph MUST NOT be rebuilt on each iteration

#### Scenario: Isolated value scope with shared initializers
- **WHEN** a sub-graph is executed from inside a `Loop` or `Scan` body
- **THEN** the sub-executor MUST create a fresh `value_map` seeded with loop-carried values and outer-referenced names
- **AND** model initializer tensors MUST be visible by name inside the sub-graph without being copied per iteration
- **AND** writes made inside the body MUST NOT mutate the outer `value_map`
- **AND** on sub-graph exit only the body's declared output tensors MUST be propagated back to the parent

#### Scenario: Nested If inside a Loop body
- **WHEN** a model contains a `Loop` whose body contains an `If` node that itself contains MatMul and Softmax nodes
- **THEN** the sub-graph executor MUST recursively execute the `If` branch selected by the runtime condition for each iteration
- **AND** the inner results MUST be routed correctly back through the `Loop`'s carried-value slots
- **AND** final output values MUST match a hand-computed reference

### Requirement: Loop Operator
The ONNX runtime SHALL implement the `Loop` operator with full ONNX termination semantics, supporting all three stop signals (`M`, `cond`, `cond_out`) in combination.

#### Scenario: Max trip count `M` bounds iterations
- **WHEN** a `Loop` node is dispatched with `M = 64` and a body that always emits `cond_out = true`
- **THEN** the loop MUST execute exactly 64 iterations
- **AND** the outputs MUST be the carried values from iteration 63

#### Scenario: Body-emitted `cond_out` stops early
- **WHEN** a `Loop` node is dispatched with `M = 64` and a body whose `cond_out` returns `false` at iteration 32
- **THEN** the loop MUST stop at the end of iteration 32
- **AND** the outputs MUST be iteration 32's carried values
- **AND** iterations 33..63 MUST NOT execute (no further iterations execute)

#### Scenario: External `cond = false` skips the loop entirely
- **WHEN** a `Loop` node is dispatched with `cond = false`
- **THEN** the loop MUST execute zero iterations
- **AND** the outputs MUST equal the initial carried values (`v_initial`)

#### Scenario: Loop-carried values thread through iterations
- **WHEN** a `Loop` body emits a new hidden-state tensor at each iteration as a carried output
- **THEN** iteration N+1 MUST receive iteration N's emitted tensor as its input for that slot
- **AND** the final output MUST be iteration last's emitted tensor

### Requirement: If Operator
The ONNX runtime SHALL implement the `If` operator with both `then` and `else` branches compiled at graph build time.

#### Scenario: Select the then-branch on true
- **WHEN** an `If` node receives a condition tensor containing `true`
- **THEN** the sub-graph executor MUST evaluate only the `then` branch
- **AND** MUST return the `then` branch outputs
- **AND** MUST NOT evaluate the `else` branch

#### Scenario: Branches with different output shapes
- **WHEN** an `If` node's then-branch produces a shape `[1, 768]` and its else-branch produces `[1, 1024]`
- **THEN** the dispatcher MUST return the shape matching the selected branch
- **AND** downstream operators MUST see the branch-specific shape

### Requirement: Scan Operator
The ONNX runtime SHALL implement the `Scan` operator for the simple sequence case where the body is applied element-by-element to a sequence-dimensional input.

#### Scenario: Scan applies a constant-add body to each element
- **WHEN** a `Scan` node is configured with a body `body(x_in) = x_in + 1` and given a sequence input `[0, 1, 2, 3, 4]`
- **THEN** the output sequence MUST be `[1, 2, 3, 4, 5]`
- **AND** the body MUST be invoked exactly 5 times
- **AND** each invocation MUST receive the corresponding sequence element as `x_in`

### Requirement: Sub-Graph WCET Budget Integration
The sub-graph executor SHALL participate in the existing operator budget enforcement system so that `Loop`, `If`, and `Scan` are accounted for as single atomic units at the parent level, and inner hard-limit failures bubble up as parent hard-limit failures.

#### Scenario: Loop is a single budget accounting unit
- **WHEN** a `Loop` with 100 iterations executes inside a profiled session
- **THEN** the `InferenceProfile.operators` list MUST contain exactly one entry for the `Loop` op
- **AND** its `actual_us` MUST equal the wall-clock sum across all iterations plus sub-dispatch overhead
- **AND** the entry MUST be compared against the `OperatorBudget` for the `Loop` class, not against 100 separate per-iteration budgets

#### Scenario: Inner hard-limit aborts the whole loop
- **WHEN** an inner operator inside a `Loop` body exceeds its own hard budget limit
- **THEN** the sub-executor MUST return `SessionError::ExecutionFailed` from inside the sub-graph
- **AND** the parent `Loop` MUST stop iterating immediately
- **AND** the error MUST bubble up to `Session::run()` unchanged

### Requirement: Generative and Normalization Operator Completeness
The ONNX runtime SHALL implement the following generative, normalization, and reduction operators for f32 CPU execution: `RMSNormalization`, `MatMulInteger`, `DynamicQuantizeLinear`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `Multinomial`, `Bernoulli`, `Dropout`, `EyeLike`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceSumSquare`, `LpNormalization`, `MeanVarianceNormalization`, `Softplus`.

#### Scenario: RMSNormalization matches PyTorch reference
- **WHEN** an `RMSNormalization` node is dispatched on an input tensor with `epsilon = 1e-6`
- **THEN** the output MUST match a PyTorch `nn.RMSNorm` reference within 1e-5 relative tolerance

#### Scenario: DynamicQuantizeLinear produces valid per-tensor scale
- **WHEN** a `DynamicQuantizeLinear` node is dispatched on an f32 tensor
- **THEN** the output MUST be a quantized `u8` tensor, an f32 scale, and a `u8` zero-point
- **AND** dequantizing the output MUST reconstruct the input within quantization error

#### Scenario: RandomUniform is reproducible given a seed
- **WHEN** two `RandomUniform` nodes are dispatched with identical `seed`, `shape`, `low`, and `high` attributes
- **THEN** both outputs MUST be bit-identical

#### Scenario: Sampling via Multinomial produces index in distribution support
- **WHEN** a `Multinomial` node is dispatched on a probability vector and a seed
- **THEN** the output MUST be a tensor of indices each in `[0, vocab_size)`

### Requirement: Phase 2 Inventory Flip
The `SUPPORTED_OPS_INVENTORY` table added by Phase 1 SHALL be updated such that the 22 Phase 2 operators (3 control-flow plus 19 generative/norm) are marked `OperatorStatus::Implemented` upon completion of this change.

#### Scenario: Inventory reflects Phase 2 completion
- **WHEN** the Phase 2 change is fully implemented
- **THEN** the inventory MUST contain `(OpKind::Loop, OperatorStatus::Implemented)`
- **AND** MUST contain `(OpKind::If, OperatorStatus::Implemented)`
- **AND** MUST contain `(OpKind::Scan, OperatorStatus::Implemented)`
- **AND** MUST contain `OperatorStatus::Implemented` entries for each of the 19 generative/norm operators
- **AND** no Phase 2 operator SHALL remain as `Planned(Phase::P2)`

