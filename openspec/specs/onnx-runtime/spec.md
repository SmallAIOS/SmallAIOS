# onnx-runtime Specification

## Purpose
TBD - created by archiving change smallaios-kernel-v1. Update Purpose after archive.
## Requirements
### Requirement: Model Loading via Protobuf Parsing
The ONNX runtime SHALL parse ONNX model files using a minimal protobuf parser code-generated from the onnx.proto3 schema.

#### Scenario: Load a valid ONNX model
- WHEN the runtime receives a valid .onnx protobuf-serialized model file
- THEN it MUST parse the ModelProto structure including graph, nodes, initializers, and metadata
- AND MUST support ONNX IR version 10 (ONNX 1.16+) and opset version 21

#### Scenario: Reject a malformed protobuf
- WHEN the runtime receives a corrupted or truncated protobuf file
- THEN it MUST return a descriptive OnnxError without panicking
- AND MUST NOT allocate unbounded memory during parsing

### Requirement: Model Validation
The runtime SHALL validate loaded models before execution to ensure all operators, shapes, and data types are supported.

#### Scenario: Validate operator support
- WHEN a model contains only Tier 1 operators (MatMul, Conv, Relu, Softmax, etc.)
- THEN validation MUST succeed
- AND the runtime MUST report the list of operators used

#### Scenario: Reject unsupported opset version
- WHEN a model specifies an opset version greater than 21
- THEN validation MUST fail with an UnsupportedOpset error
- AND MUST report the requested vs. supported opset versions

### Requirement: Graph Optimization — Operator Fusion
The runtime SHALL implement operator fusion passes to combine sequences of operators into single fused kernels.

#### Scenario: Fuse Conv-BatchNorm-Relu
- WHEN the graph contains a Conv node followed by BatchNormalization followed by Relu
- THEN the optimizer MUST fuse them into a single FusedConvBNRelu node
- AND the fused node MUST produce numerically equivalent output (within f32 epsilon)

#### Scenario: Fuse MatMul-Add into FusedLinear
- WHEN the graph contains a MatMul node followed by an Add with a constant bias
- THEN the optimizer MUST fuse them into a single FusedLinear (GEMM) node

### Requirement: Graph Optimization — Constant Folding
The runtime SHALL pre-compute subgraphs composed entirely of constant inputs at model load time.

#### Scenario: Fold static reshape
- WHEN a Reshape operator has all-constant inputs (data and shape)
- THEN the optimizer MUST evaluate the operator at load time
- AND replace it with the computed constant tensor in the graph

### Requirement: Graph Optimization — Memory Planning
The runtime SHALL analyze tensor lifetimes and plan buffer reuse to minimize peak memory consumption.

#### Scenario: Reuse dead tensor buffers
- WHEN tensor A's lifetime ends at operator 3 and tensor C begins at operator 4
- THEN the memory planner MUST assign tensor C to the same buffer as tensor A
- AND peak memory MUST be reduced compared to naive allocation

### Requirement: CPU Execution Provider — x86-64 SIMD
The CPU execution provider SHALL use runtime CPUID detection to select optimal SIMD kernels for x86-64.

#### Scenario: Select AVX-512 GEMM on capable hardware
- WHEN the CPU supports AVX-512F and AVX-512VL
- THEN the execution provider MUST select the AVX-512 GEMM kernel for MatMul operations
- AND MUST fall back to AVX2 kernels if AVX-512 is unavailable

#### Scenario: Execute quantized inference with VNNI
- WHEN the CPU supports AVX-512 VNNI and the model uses INT8 quantized operators
- THEN the execution provider MUST use VNNI dot-product instructions for INT8 MatMul

### Requirement: CPU Execution Provider — ARM64 SIMD
The CPU execution provider SHALL use runtime feature detection to select optimal SIMD kernels for ARM64.

#### Scenario: Select SVE kernels on capable hardware
- WHEN the CPU supports SVE/SVE2 extensions
- THEN the execution provider MUST select SVE GEMM kernels for MatMul operations
- AND MUST fall back to NEON kernels if SVE is unavailable

### Requirement: CUDA Execution Provider
The runtime SHALL support GPU inference via a CUDA execution provider that launches custom PTX kernels on NVIDIA GPUs.

#### Scenario: GPU kernel launch for MatMul
- WHEN an inference session is configured with the CUDA execution provider
- AND the graph contains a MatMul operator
- THEN the runtime MUST launch a GPU kernel using tensor core HMMA instructions for fp16/bf16
- AND MUST use the GPU memory pool for intermediate tensor allocation

#### Scenario: Async DMA transfers
- WHEN input tensors reside in host memory and the session uses GPU execution
- THEN the runtime MUST transfer inputs to GPU via async DMA
- AND MUST overlap DMA transfers with computation where possible

### Requirement: Operator-Level Scheduler Integration
The runtime SHALL insert mandatory scheduler yield points between every operator in the execution graph and support per-operator time budgets.

#### Scenario: Yield between operators
- WHEN the runtime executes an inference graph
- THEN it MUST yield to the scheduler after each operator completes
- AND the yield MUST allow higher-priority tasks (SYSTEM, IPC) to execute before inference resumes

#### Scenario: Per-operator timing
- WHEN an operator executes during inference
- THEN the runtime MUST measure its wall-clock execution time
- AND MUST compare the measured time against the operator's configured budget

#### Scenario: Operator budget exceeded
- WHEN an operator's execution time exceeds its soft budget
- THEN the runtime MUST log a warning with operator name, actual time, and budget to syslog
- AND MUST continue inference normally

#### Scenario: Operator hard timeout
- WHEN an operator's execution time exceeds its hard limit (default: 10x budget)
- THEN the runtime MUST abort the inference and return OnnxError::OperatorTimeout

### Requirement: WCET Calibration
The runtime SHALL support optional worst-case execution time calibration during session creation for edge deployment targets.

#### Scenario: Calibration run during session creation
- WHEN a session is created with calibrate_wcet enabled
- THEN the runtime MUST execute each operator once with representative input data
- AND MUST compute WCET estimates as measured_time × wcet_safety_factor
- AND MUST assign calibrated estimates as operator budgets for the session

#### Scenario: Calibration on constrained hardware
- WHEN calibration runs on a constrained target (Jetson Nano, RPi)
- THEN the recommended wcet_safety_factor MUST be >= 3.0 to account for thermal throttling and clock variability

### Requirement: Session API
The runtime SHALL expose a Session API with load, create_session, run, and metadata operations.

#### Scenario: Create and run an inference session
- WHEN a client calls load_model with valid ONNX bytes followed by create_session
- THEN the runtime MUST return a ready Session handle
- AND calling run with correctly shaped input tensors MUST return output tensors matching the model's output specification

#### Scenario: Query model metadata
- WHEN a client calls get_metadata on a loaded model
- THEN the runtime MUST return the model's input names/shapes, output names/shapes, opset version, and producer name

