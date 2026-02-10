# Phase 3: ONNX Runtime — Parse, Optimize, Execute

## Objective

Implement the ONNX model loading, graph optimization, and CPU execution provider
so that SmallAIOS can load and run inference on a real ONNX model (target: MobileNetV2
image classification as the first end-to-end model).

## Dependencies

- Phase 2 complete (memory management, scheduler, syscalls)

## Tasks

### 3.1 Protobuf Parser
- [ ] Minimal protobuf decoder (varint, length-delimited, fixed32/64)
- [ ] Code-generate Rust structs from `onnx.proto3` (build script)
- [ ] Parse `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`
- [ ] Parse `AttributeProto` (all attribute types)
- [ ] Handle external data references (tensors stored outside protobuf)
- [ ] Fuzz: random bytes → parser must not panic or OOM

### 3.2 Model Validation
- [ ] Validate ONNX IR version (require >= 7)
- [ ] Validate opset version (require >= 13, target 21)
- [ ] Check all operators are in the supported set (or return clear error)
- [ ] Validate tensor shapes (where statically known)
- [ ] Validate data types are supported
- [ ] Check for cycles in the graph (must be DAG)

### 3.3 Execution Graph
- [ ] Convert ModelProto → internal Graph representation
- [ ] Topological sort of operators
- [ ] Resolve operator inputs/outputs (name → tensor reference)
- [ ] Handle initializers (constant tensors / model weights)
- [ ] Handle subgraphs (If, Loop, Scan operators — Phase 3.x stretch)

### 3.4 Graph Optimizer
- [ ] **Constant folding**: Evaluate subgraphs with all-constant inputs at load time
- [ ] **Operator fusion**: Conv+BN+Relu, MatMul+Add, LayerNorm patterns
- [ ] **Dead code elimination**: Remove unused operators and tensors
- [ ] **Layout optimization**: NCHW → NHWC conversion for CPU (better cache behavior)
- [ ] **Common subexpression elimination**: Share identical computations
- [ ] Optimization levels: None, Basic (fold+DCE), Extended (+fusion), Full (+layout)

### 3.5 Memory Planner
- [ ] Compute tensor lifetimes (first use → last use in topological order)
- [ ] Identify tensors that can share buffers (non-overlapping lifetimes)
- [ ] Compute peak memory usage
- [ ] Pre-allocate tensor pool for session
- [ ] Identify in-place operations (output overwrites input)

### 3.6 CPU Execution Provider — Core Operators

**Tier 1 operators (implement in this order):**

Math/elementwise:
- [ ] `Add`, `Sub`, `Mul`, `Div` (broadcast semantics)
- [ ] `Relu`, `Sigmoid`, `Tanh`, `Softmax`
- [ ] `Clip`, `Abs`, `Neg`, `Sqrt`, `Exp`, `Log`

Tensor manipulation:
- [ ] `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`
- [ ] `Concat`, `Split`, `Slice`, `Gather`
- [ ] `Flatten`, `Expand`

Linear algebra:
- [ ] `MatMul` (GEMM kernel — this is the most critical operator)
- [ ] `Gemm` (generalized matrix multiply with alpha/beta/transA/transB)

Convolution:
- [ ] `Conv` (2D convolution with im2col + GEMM strategy)
- [ ] `DepthToSpace`, `SpaceToDepth`

Normalization:
- [ ] `BatchNormalization` (inference mode — no running stats update)
- [ ] `LayerNormalization`

Pooling:
- [ ] `AveragePool`, `MaxPool`
- [ ] `GlobalAveragePool`, `GlobalMaxPool`

Reduction:
- [ ] `ReduceMean`, `ReduceSum`, `ReduceMax`

### 3.7 SIMD Kernels

x86-64:
- [ ] AVX2 GEMM micro-kernel (8x8 f32 register tile)
- [ ] AVX2 elementwise operations (vectorized add, mul, relu, etc.)
- [ ] AVX-512 GEMM micro-kernel (16x16 f32) — optional, for supported CPUs
- [ ] Runtime CPU feature detection → kernel dispatch

ARM64:
- [ ] NEON GEMM micro-kernel (8x8 f32)
- [ ] NEON elementwise operations
- [ ] SVE GEMM kernel (scalable vector length) — optional
- [ ] Runtime feature detection → kernel dispatch

### 3.8 GEMM Implementation
- [ ] Naive reference implementation (for testing)
- [ ] Cache-blocked GEMM (L1/L2/L3 aware tiling)
- [ ] GOTO-style panel-based algorithm for large matrices
- [ ] Packing routines (A panel pack, B panel pack)
- [ ] Multi-threaded GEMM (partition across cores)
- [ ] Benchmark against OpenBLAS/MKL for performance parity check

### 3.9 Session API
- [ ] `load_model(bytes) → Model`
- [ ] `create_session(model, options) → Session`
- [ ] `run(session, inputs) → outputs`
- [ ] Session options: execution providers, optimization level, thread count
- [ ] Session caches optimized graph (subsequent runs skip optimization)

### 3.10 End-to-End Test
- [ ] Download MobileNetV2 ONNX model (from ONNX model zoo)
- [ ] Preprocess test image (resize, normalize — done offline)
- [ ] Run inference in SmallAIOS
- [ ] Compare output to reference (onnxruntime on Linux) — must match within tolerance
- [ ] Measure latency and throughput

## Exit Criteria

- Can load and parse any valid ONNX model (opset 13-21)
- Graph optimizer reduces MobileNetV2 operator count by ≥ 20%
- Memory planner reduces peak memory by ≥ 30% vs. naive allocation
- All Tier 1 operators pass correctness tests (vs. numpy reference)
- GEMM performance within 2x of OpenBLAS on same hardware
- MobileNetV2 inference produces correct classification on test image
- End-to-end latency < 50ms on modern x86-64 (single core)
- Runs on both x86-64 and ARM64 in QEMU
