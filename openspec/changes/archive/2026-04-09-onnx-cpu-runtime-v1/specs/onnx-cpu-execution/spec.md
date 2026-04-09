## ADDED Requirements

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
The ONNX runtime SHALL implement all 29 Tier 1 operators for CPU execution with f32 tensors.

#### Scenario: Element-wise binary operators (Sub, Mul, Div)
- **WHEN** two f32 tensors are provided as inputs to Sub, Mul, or Div
- **THEN** the operator MUST compute the element-wise result with NumPy-style broadcasting
- **AND** the output shape MUST match the broadcast output shape

#### Scenario: Gemm operator wraps GEMM micro-kernel
- **WHEN** a Gemm node is dispatched with matrices A, B, and optional bias C
- **THEN** the operator MUST compute `alpha * A @ B + beta * C` using the existing `gemm_f32` micro-kernel
- **AND** MUST support `transA` and `transB` attributes

#### Scenario: Activation operators (Sigmoid, Tanh)
- **WHEN** an f32 tensor is provided to Sigmoid or Tanh
- **THEN** the operator MUST compute the element-wise activation function
- **AND** Sigmoid MUST use `1 / (1 + exp(-x))` with the `no_std` `expf_approx`
- **AND** Tanh MUST use `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`

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
- **AND** inference MUST continue normally

#### Scenario: Hard budget abort
- **WHEN** an operator's execution time exceeds 10x its configured budget
- **THEN** the runtime MUST abort the current inference
- **AND** MUST return `SessionError::ExecutionFailed` with timing details

#### Scenario: Profiling disabled by default
- **WHEN** a session is created without enabling profiling
- **THEN** no timing measurement overhead SHALL be incurred during inference
