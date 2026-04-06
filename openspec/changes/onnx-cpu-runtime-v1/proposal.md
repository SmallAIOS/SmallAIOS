## Why

The ONNX runtime has 6 real CPU operator implementations (Add, MatMul, Relu, Softmax, Reshape, Conv), a working GEMM micro-kernel with cache blocking, a topologically-sorted execution graph builder, and a session API — but `Session::run()` returns `NotImplemented`. The operators exist, the graph exists, but nothing connects them. This change wires the execution path end-to-end so that ONNX models can actually run inference on CPU, and completes the remaining Tier 1 operators needed to support common model architectures (MLP, CNN, basic transformers).

## What Changes

- Wire `Session::run()` to iterate the execution graph node-by-node, dispatching each node to the corresponding CPU operator function with tensor I/O plumbing
- Implement a tensor value map to track intermediate results through graph execution
- Complete the remaining 23 CPU operator implementations (of 29 registered in `OpKind`):
  - Priority 1 (core math): Sub, Mul, Div, Gemm (wrapping existing `gemm_f32`)
  - Priority 2 (activations): Sigmoid, Tanh
  - Priority 3 (shape/data): Transpose, Concat, Flatten, Squeeze, Unsqueeze, Cast, Gather, Slice, Pad, Clip
  - Priority 4 (normalization/pooling): BatchNormalization, LayerNormalization, MaxPool, AveragePool, GlobalAveragePool, ReduceMean, ReduceSum
- Add scheduler yield points between operator executions (cooperative async integration)
- Add per-operator timing measurement and budget checking against `OperatorBudget`

## Capabilities

### New Capabilities
- `onnx-cpu-execution`: CPU operator dispatch, graph traversal, tensor value routing, and scheduler integration for end-to-end ONNX inference

### Modified Capabilities
- `onnx-runtime`: Add requirements for complete Tier 1 operator coverage and Session::run() execution semantics

## Impact

- **Code:** `onnx-rt/src/session.rs` (graph execution loop), `onnx-rt/src/operators.rs` (23 new operator functions), new `onnx-rt/src/executor.rs` (graph traversal + tensor value map)
- **APIs:** `Session::run()` changes from returning `NotImplemented` to producing actual output tensors
- **Testing:** Each new operator needs unit tests with known inputs/outputs; end-to-end test with a small model graph
- **Dependencies:** None new — all `#![no_std]` with `alloc`
