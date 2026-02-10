# Spec 03: ONNX Runtime Integration

## Overview

SmallAIOS includes a **native ONNX runtime** as a first-class kernel service. Unlike
traditional OS designs where the ML runtime is an application, SmallAIOS treats ONNX
inference as the primary workload the entire system is built around.

The runtime parses ONNX models, optimizes their execution graphs, and dispatches
operators to hardware-specific execution providers (CPU, GPU).

## ONNX Format Support

### Target Specification

- **ONNX IR version**: 10 (corresponds to ONNX 1.16+)
- **Opset version**: 21 (latest stable)
- **Model format**: Protobuf-serialized `.onnx` files
- **External data**: Support for models with external tensor data files

### Supported Data Types

| ONNX Type | Rust Type | Notes |
|---|---|---|
| FLOAT | f32 | Primary inference type |
| FLOAT16 | half::f16 | GPU-optimized inference |
| BFLOAT16 | half::bf16 | Training compat, GPU inference |
| DOUBLE | f64 | Rare, full support |
| INT8 | i8 | Quantized inference |
| UINT8 | u8 | Quantized inference |
| INT16 | i16 | Limited use |
| INT32 | i32 | Shape/index operations |
| INT64 | i64 | Shape/index operations |
| BOOL | bool | Control flow |
| STRING | &[u8] | NLP tokenizer compat |

### Operator Coverage (Priority Tiers)

**Tier 1 — Must have for launch (covers ~90% of production models):**

Tensor ops: `Reshape`, `Transpose`, `Concat`, `Split`, `Slice`, `Gather`,
`Squeeze`, `Unsqueeze`, `Flatten`, `Expand`, `Tile`, `Pad`

Math ops: `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `Exp`, `Log`,
`Pow`, `Abs`, `Neg`, `Reciprocal`, `Clip`, `Floor`, `Ceil`, `Round`

Activation ops: `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `LeakyRelu`,
`Elu`, `Selu`, `Gelu`, `HardSigmoid`, `HardSwish`

Normalization ops: `BatchNormalization`, `LayerNormalization`, `InstanceNormalization`,
`GroupNormalization`

Pooling ops: `AveragePool`, `MaxPool`, `GlobalAveragePool`, `GlobalMaxPool`

Conv ops: `Conv`, `ConvTranspose`, `DepthToSpace`, `SpaceToDepth`

Reduction ops: `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProd`

Attention ops: `Attention`, `MultiHeadAttention` (contrib ops for transformers)

**Tier 2 — Post-launch:**

Quantization: `QuantizeLinear`, `DequantizeLinear`, `QLinearConv`, `QLinearMatMul`

RNN: `LSTM`, `GRU`, `RNN`

Detection: `NonMaxSuppression`, `RoiAlign`

**Tier 3 — As needed:**

Remaining ONNX ops added based on model requirements.

## Architecture

```
┌─────────────────────────────────────────────┐
│              ONNX Session API               │
│  (load, create_session, run, get_metadata)  │
├─────────────────────────────────────────────┤
│            Graph Optimizer                   │
│  (fusion, constant folding, layout opt)     │
├─────────────────────────────────────────────┤
│           Execution Planner                  │
│  (memory planning, operator scheduling,     │
│   device placement)                          │
├──────────┬──────────────────────────────────┤
│ CPU EP   │         CUDA EP                  │
│ (x86/ARM)│    (NVIDIA GPU)                  │
├──────────┴──────────────────────────────────┤
│         Tensor Memory Manager               │
│  (allocation, pooling, DMA, zero-copy)      │
└─────────────────────────────────────────────┘
```

## Operator-Level Scheduling Integration

The ONNX runtime is tightly integrated with the kernel's soft real-time scheduler.
Between every operator in the execution graph, the runtime inserts a **mandatory
scheduler yield point**. This enables:

- **Priority preemption**: SYSTEM/IPC tasks preempt inference at operator boundaries
- **Time budgets**: Each operator is timed; overruns are logged and optionally aborted
- **Watchdog servicing**: Long inference chains don't starve the hardware watchdog
- **Observability**: Per-operator timing metrics for profiling and capacity planning

### Execution Loop (Pseudocode)

```rust
for operator in execution_plan.operators() {
    let start = sys_time();

    // Execute the operator
    operator.execute(&mut tensors)?;

    let elapsed = sys_time() - start;

    // Check time budget
    if elapsed > operator.budget() {
        sys_log(WARN, &format!("{}: {}ms > {}ms budget",
            operator.name(), elapsed.as_millis(), operator.budget().as_millis()));
    }
    if elapsed > operator.hard_limit() {
        return Err(OnnxError::OperatorTimeout(operator.name()));
    }

    // Mandatory yield — scheduler checks for higher-priority work
    task_yield().await;
}
```

### WCET Calibration

For edge targets (Jetson Nano, Raspberry Pi), the runtime supports a calibration
phase during model load:

1. Execute each operator once with representative input tensors
2. Compute WCET estimate: measured time × configurable safety factor (default: 3x)
3. Assign WCET estimates as operator budgets for the session
4. Track actual vs estimated at runtime; auto-adjust safety factors

### Session Options (Extended)

```rust
pub struct SessionOptions {
    pub execution_providers: Vec<ExecutionProvider>,
    pub optimization_level: OptLevel,
    pub num_threads: usize,
    pub enable_profiling: bool,
    /// Per-operator time budget multiplier (1.0 = use defaults)
    pub operator_budget_scale: f32,
    /// Hard timeout for entire inference call (0 = no limit)
    pub inference_timeout_ms: u64,
    /// Enable WCET calibration run during session creation
    pub calibrate_wcet: bool,
    /// WCET safety factor (default: 3.0)
    pub wcet_safety_factor: f32,
}
```

## Model Loading Pipeline

```
1. Parse protobuf → ModelProto
2. Validate model (opset version, required ops, shapes)
3. Build execution graph (topological sort of operators)
4. Run graph optimizations:
   a. Constant folding (pre-compute static subgraphs)
   b. Operator fusion (Conv+BN+Relu → FusedConvBNRelu)
   c. Layout optimization (NCHW → NHWC for CPU, NCHW for GPU)
   d. Dead code elimination
   e. Common subexpression elimination
5. Plan memory (tensor lifetimes, in-place operations, buffer reuse)
6. Assign operators to execution providers
7. Compile/optimize per-EP operator implementations
8. Insert scheduler yield points between operators
9. (Optional) Run WCET calibration and assign operator budgets
10. Return ready-to-execute Session
```

## Graph Optimizations

### Operator Fusion Rules

| Pattern | Fused Op | Benefit |
|---|---|---|
| Conv → BatchNorm → Relu | FusedConvBNRelu | Eliminate BN, fuse activation |
| MatMul → Add | FusedLinear | Single GEMM call |
| LayerNorm → attention pattern | FusedAttention | Transformer block fusion |
| Mul → Add (with constants) | FusedAffine | Single pass |
| Multiple Concat | FusedConcat | Single memory copy |
| Softmax → Log | LogSoftmax | Numerically stable single-pass |

### Memory Planning

The memory planner analyzes tensor lifetimes and reuses buffers:

```
Tensor A: [op1 ... op3]        → buffer 0
Tensor B: [op2 ... op5]        → buffer 1
Tensor C: [op4 ... op7]        → buffer 0 (reused after A dies at op3)
```

This minimizes peak memory usage. For GPU inference, the planner also schedules
DMA transfers to overlap with computation.

## CPU Execution Provider

### x86-64 Optimizations

- **AVX2**: 256-bit SIMD for f32 tensor ops (baseline)
- **AVX-512**: 512-bit SIMD where available (Xeon, EPYC)
- **AVX-512 VNNI**: INT8 dot products for quantized inference
- **AMX**: Tile-based matrix multiply (Sapphire Rapids+)
- **FMA**: Fused multiply-add for GEMM kernels

Runtime CPU feature detection via `CPUID` selects optimal kernels at session creation.

### ARM64 Optimizations

- **NEON**: 128-bit SIMD (baseline, always available)
- **SVE/SVE2**: Scalable vector extensions (Graviton3+, Neoverse V2+)
- **SME**: Scalable matrix extensions (future, Neoverse V3+)
- **dotprod**: INT8 dot products for quantized inference
- **fp16**: Native half-precision arithmetic

Runtime CPU feature detection via `MRS` system registers.

### GEMM Strategy

Matrix multiplication is the dominant cost in inference. Strategy:

1. **Small matrices** (M,N,K < 32): Direct SIMD kernel
2. **Medium matrices** (< 512): Panel-based micro-kernel with register tiling
3. **Large matrices** (>= 512): GOTO-style GEMM with L1/L2/L3 cache blocking
4. **Quantized**: Separate INT8 GEMM kernels with accumulation in INT32

All GEMM kernels are hand-written in Rust with inline assembly for the hot loops.
No dependency on BLAS libraries.

## CUDA Execution Provider

See [Spec 05: Device HAL — NVIDIA section](05-device-hal.md#nvidia-gpu) for
low-level GPU access.

### GPU Operator Implementation

- Custom CUDA-equivalent kernels written in NVIDIA PTX assembly
- Compiled to device code at build time using `ptxas`
- Key fused kernels: FusedAttention, FusedConvBNRelu, FusedLinear
- Use tensor cores (HMMA instructions) for fp16/bf16 matrix multiply

### GPU Memory Management

- Unified tensor memory manager handles CPU↔GPU transfers
- Pinned host memory for DMA transfers
- GPU memory pool with sub-allocation to avoid per-tensor allocation overhead
- Async DMA engine overlaps transfers with computation

### Execution Strategy

```
1. Build GPU execution graph (subset of full graph)
2. Allocate GPU memory for all intermediate tensors
3. Transfer input tensors to GPU (async DMA)
4. Launch kernel sequence (CUDA streams equivalent)
5. Transfer output tensors to CPU (async DMA)
6. Synchronize and return results
```

## Session API

```rust
/// Load an ONNX model from bytes
pub fn load_model(data: &[u8]) -> Result<Model, OnnxError>;

/// Create an inference session with specific providers
pub fn create_session(
    model: &Model,
    opts: &SessionOptions,
) -> Result<Session, OnnxError>;

/// Run inference
pub fn run(
    session: &Session,
    inputs: &[NamedTensor],
) -> Result<Vec<NamedTensor>, OnnxError>;

/// Session options
pub struct SessionOptions {
    pub execution_providers: Vec<ExecutionProvider>,
    pub optimization_level: OptLevel,  // None, Basic, Extended, Full
    pub num_threads: usize,            // 0 = auto
    pub enable_profiling: bool,
}

pub enum ExecutionProvider {
    Cpu(CpuOptions),
    Cuda(CudaOptions),
}
```

## Protobuf Parsing

SmallAIOS includes a **minimal protobuf parser** that handles only the ONNX protobuf
schema. This is not a general-purpose protobuf library — it is code-generated from
`onnx.proto3` at build time and handles only the message types defined there.

This avoids depending on full `prost` or `protobuf` crates in the kernel.

## Crate Structure

```
onnx-rt/
├── Cargo.toml
├── build.rs                    # Protobuf code generation
├── proto/
│   └── onnx.proto3             # ONNX protobuf schema (reference)
└── src/
    ├── lib.rs
    ├── proto/
    │   └── generated.rs        # Generated protobuf types
    ├── model.rs                # Model loading and validation
    ├── graph.rs                # Execution graph representation
    ├── optimizer/
    │   ├── mod.rs
    │   ├── fusion.rs           # Operator fusion passes
    │   ├── constant_fold.rs    # Constant folding
    │   ├── layout.rs           # Data layout optimization
    │   └── memory_plan.rs      # Tensor memory planning
    ├── session.rs              # Session management
    ├── ops/
    │   ├── mod.rs              # Operator registry
    │   ├── math.rs             # Elementwise math operators
    │   ├── tensor.rs           # Tensor manipulation operators
    │   ├── nn.rs               # Neural network operators (conv, pool, norm)
    │   ├── activation.rs       # Activation functions
    │   ├── reduce.rs           # Reduction operators
    │   └── attention.rs        # Attention/transformer operators
    ├── ep/
    │   ├── mod.rs              # Execution provider trait
    │   ├── cpu/
    │   │   ├── mod.rs
    │   │   ├── gemm.rs         # Matrix multiply kernels
    │   │   ├── conv.rs         # Convolution kernels
    │   │   └── simd.rs         # SIMD abstraction layer
    │   └── cuda/
    │       ├── mod.rs
    │       ├── launch.rs       # Kernel launch
    │       └── kernels/        # PTX kernel sources
    └── tensor.rs               # Tensor type and operations
```

## Testing Strategy

- **Operator correctness**: Test each operator against reference numpy/onnxruntime output
- **Model accuracy**: Run standard ONNX model zoo models, compare output tensors
- **Performance**: Benchmark against onnxruntime on same hardware
- **Fuzz**: Fuzz the protobuf parser and operator inputs
- **Edge cases**: NaN, Inf, empty tensors, scalar tensors, zero-dim tensors
