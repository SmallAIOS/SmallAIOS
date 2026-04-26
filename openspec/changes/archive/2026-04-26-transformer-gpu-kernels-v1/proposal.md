## Why

The `safetensors-model-loader-v1` change lands the full model loading pipeline for Gemma 4 31B on DGX Spark — safetensors parsing, HuggingFace config parsing, programmatic graph construction, BF16 end-to-end, GPU-resident executor, and KV cache. When we wire it all together and run `Session::run()` on a Gemma model, the pipeline successfully builds the GPU-resident Session and then **fails on the very first forward-pass operator**:

```
Gather_0: no GPU implementation for Gather
```

The `execute_graph_gpu` dispatcher in `onnx-rt/src/cuda/gpu_executor.rs` only supports **MatMul, Gemm, MatMulInteger, and Conv** — the ops we added for CNN-style inference in `arm64-gpu-container-v1`. Every other operator in a modern transformer — `Gather`, `Add`, `Mul`, `RMSNormalization`, `RotaryEmbedding`, `GroupQueryAttention`, `Silu` — still falls through to the fail-fast "no GPU implementation" branch, which by design does NOT fall back to CPU inside a GPU-resident forward pass.

Without these ops on the GPU, `safetensors-model-loader-v1` is functionally inert — it can build the Session but cannot actually generate a token. This change fills the gap: it adds GPU implementations for the 7 operator kinds every modern transformer needs, so Gemma (and Llama 3, Qwen, Mistral, DeepSeek — all sharing the same architecture backbone) can run end-to-end on the GB10.

The target is NVIDIA Blackwell (compute capability 12.1) via CUDA 13.0 / cuBLAS 13.x / cuDNN 9.20 on the DGX Spark. The implementation strategy is cuBLAS / cuDNN API calls where possible, falling back to hand-written `__global__` CUDA kernels only where the standard libraries don't offer a suitable primitive.

## What Changes

### Phase 1 — Element-wise ops (simplest)

- **`Add` on GPU** — element-wise addition with broadcasting. For Gemma, used in residual connections (attention output + input) and the embedding scaling step. Implement via a thin hand-written CUDA kernel (`add_bf16_kernel`, `add_f32_kernel`) OR via `cublasAxpy` for simple cases. Support BF16 and F32 inputs.
- **`Mul` on GPU** — element-wise multiplication with broadcasting. Used in SwiGLU (`silu(gate) * up`) and embedding scaling (`embed * sqrt(hidden_size)`). Same strategy as Add.
- **`Silu` / `Swish` on GPU** — element-wise `x * sigmoid(x)` activation. Used in the Gemma MLP gate. Hand-written kernel — no standard library primitive.

### Phase 2 — Embedding lookup

- **`Gather` on GPU** — indexed lookup: given an embedding table and a list of token IDs, produce the corresponding embedding rows. The first operator in every transformer graph. Implement as a hand-written CUDA kernel that does one thread-block per output row. Input: Int64 token IDs; weight: BF16 embedding table; output: BF16 rows.

### Phase 3 — Normalization

- **`RMSNormalization` on GPU** — root-mean-square normalization along the last axis with a learned weight. `y = x * rsqrt(mean(x^2) + eps) * weight`. Hand-written kernel using warp-level reductions for the mean-of-squares. Support the Gemma `(1 + weight)` convention via a feature/attribute (load-time weight adjustment already handles this in `safetensors-model-loader-v1`, so the GPU kernel just does the standard formula).

### Phase 4 — Rotary position embedding

- **`RotaryEmbedding` on GPU** — applies rotary position embeddings to query and key tensors. Hand-written kernel. Takes Q or K tensor plus precomputed cos/sin tables (computed at graph build time in `safetensors-model-loader-v1::gemma`). Output is the rotated Q or K. Support standard RoPE; document p-RoPE as a follow-up attribute the kernel can honor when wired (the graph builder already flags p-RoPE advisorily).

### Phase 5 — Attention

- **`GroupQueryAttention` on GPU** — the largest piece. Fused SDPA (scaled dot-product attention) with grouped-query support, causal masking, and optional sliding window. Options for implementation:
  - **Option A: Flash Attention-style kernel** — memory-efficient, fused softmax, minimal intermediate allocation. Highest performance, highest implementation cost.
  - **Option B: Decomposed via cuBLAS** — compute QK^T via `cublasGemmStridedBatched`, apply mask + softmax via a custom kernel, then apply softmax output to V via another `cublasGemmStridedBatched`. Moderate perf, moderate cost, reuses cuBLAS for the matmuls.
  - **Option C: cuDNN Multi-Head Attention API** — cuDNN 9.x has a fused MHA API. Highest perf, least code. Unclear if it supports GQA with sliding window for arbitrary shapes.
  
  Initial implementation: **Option B** (decomposed via cuBLAS). Measure, then replace with Option C (cuDNN fused) or Option A (Flash Attention) in follow-ups as needed. Include KV cache wiring: the operator reads cached K/V from `GpuKvCache` (via `KvView`) for previous positions and uses only the new query position.

### Phase 6 — Executor wiring

- Extend `execute_graph_gpu`'s `dispatch_gpu_node` function to route each new op kind to its GPU implementation.
- Extend `GpuKvCache` integration: `GroupQueryAttention` reads/writes to the cache via `KvView`, so its dispatch path takes a mutable reference to the cache. Thread `Option<&mut GpuKvCache>` through `execute_graph_gpu_with_weights`.
- **Maintain the fail-fast contract** for any operator still missing from GPU — do not fall back to CPU inside a forward pass.

### Out of Scope (deferred)

- Flash Attention v2 / v3 style fully-fused kernels. Option B (decomposed) first, fused kernels in a later change if profiling shows attention is the bottleneck.
- cuDNN MHA API integration. We'll try it as an optimization after Option B works, but we don't depend on it.
- FP8 attention — stick to BF16.
- Quantized operators (INT8 attention, INT4 KV cache). Future work.
- Training / backward pass. Inference-only.
- Non-transformer operators (Pooling, Reduce, Concat, Split, element-wise comparisons). Add if a future model needs them.

## Capabilities

### New Capabilities

- `gpu-elementwise-ops`: GPU implementations of `Add`, `Mul`, and `Silu` with broadcasting support for BF16 and F32 tensors, runnable inside the `execute_graph_gpu` dispatcher.
- `gpu-embedding-lookup`: GPU implementation of `Gather` for indexed embedding lookup (token IDs → embedding vectors), the first operator in every transformer forward pass.
- `gpu-rms-normalization`: GPU implementation of `RMSNormalization` using warp-level reductions for the mean-of-squares and element-wise fused scaling.
- `gpu-rotary-embedding`: GPU implementation of `RotaryEmbedding` that rotates Q and K tensors in-place using precomputed cos/sin tables. Honors the standard RoPE formula; leaves hooks for p-RoPE.
- `gpu-grouped-query-attention`: GPU implementation of `GroupQueryAttention` via decomposed cuBLAS primitives (Q·K^T → mask → softmax → ·V) with causal and sliding-window masks. Reads and updates the GPU KV cache via `KvView`.

### Modified Capabilities

- `gpu-resident-execution` (from `safetensors-model-loader-v1`): The `execute_graph_gpu` dispatcher gains support for 7 new operator kinds (`Gather`, `Add`, `Mul`, `Silu`, `RMSNormalization`, `RotaryEmbedding`, `GroupQueryAttention`) and threads the GPU KV cache mutable reference through the forward pass.

## Impact

- **New module**: `onnx-rt/src/cuda/kernels/` with one file per operator group:
  - `elementwise.rs` (Add, Mul, Silu)
  - `gather.rs`
  - `rms_norm.rs`
  - `rotary.rs`
  - `attention.rs` (GroupQueryAttention)
  - Each file contains a safe Rust wrapper and the raw kernel launch. Hand-written CUDA kernels are compiled separately — see "CUDA kernel compilation" below.
- **`onnx-rt/src/cuda/gpu_executor.rs`**: `dispatch_gpu_node` gets new match arms for each operator. The function takes `Option<&mut GpuKvCache>` in the GQA path.
- **`onnx-rt/src/cuda/kv_cache.rs`**: `KvView` may need a mutable counterpart for GQA append. Likely no API change — append is already a method on `GpuKvCache` itself.
- **`onnx-rt/src/session.rs`**: `run_safetensors` threads the locked KV cache guard through to `execute_graph_gpu_with_weights`. The TODO from `safetensors-model-loader-v1` §9 gets resolved here.
- **CUDA kernel compilation**: hand-written `__global__` kernels need to be compiled with `nvcc` and linked into the final binary. Two options:
  - **Option I**: Compile kernels to PTX at build time via `build.rs`, load at runtime via cuBLAS / CUDA driver API. Lets the library keep a single Rust build. Requires `nvcc` in the build environment.
  - **Option II**: Write kernels in inline PTX using Rust's `asm!` macro. Rust + inline PTX is supported via the `nvptx64-nvidia-cuda` target. Hermetic builds.
  - **Option III**: Precompile kernels to PTX/SASS at dev time, check the `.ptx` files into the repo, load at runtime. Zero build-time CUDA dependency, but adds binary artifacts to the repo.
  
  Initial approach: **Option I** — `build.rs` runs `nvcc` to compile `.cu` files to PTX, embeds the PTX as a string constant. Requires `nvcc` available at build time (which we have for GPU-enabled builds — the `Dockerfile.cuda` already installs it). Container CPU-only builds skip the compile entirely since the `cuda` feature is disabled.

- **Testing**: Each operator gets numerical-correctness tests on real GB10 hardware (matches the pattern of the existing 33 CUDA integration tests). Compare GPU output against a scalar Rust reference implementation for small tensor sizes; compare against cuBLAS decomposition for large sizes.
- **Performance**: This change focuses on correctness first. Benchmarking and fused-kernel optimizations are follow-ups. The success criterion is "Gemma 4 31B end-to-end inference runs and produces semantically sensible output", not "X tokens/sec".
- **Dependencies**: No new Rust crates. Adds a build-time dependency on `nvcc` when the `cuda` feature is enabled — `nvcc` ships with the CUDA toolkit so any machine that can link against cuBLAS already has it.
- **Unblocks**: `safetensors-model-loader-v1` Section 12 (end-to-end Gemma 4 31B validation). Once this change lands, that section can be completed. Also unblocks Llama 3, Qwen, Mistral, DeepSeek as future model families since they all use the same operator set.
- **Follow-up changes enabled**: Flash Attention v2/v3 for attention performance; cuDNN MHA API as an alternative; FP8 attention; INT4 KV cache.
