## Context

SmallAIOS gained a working CUDA dispatch stack in `arm64-gpu-container-v1` (PR #106) — hand-rolled FFI for `cudaMalloc`/`cublasGemmEx`/`cudnnConvolutionForward`, a `CudaRuntime` that owns `CublasHandle` / `CublasLtHandle` / `CudnnHandle`, safe `DeviceBuffer`/`DevicePtr` wrappers, and four precision modes (F32 / TF32 / FP16 / BF16) plus INT8 via `cublasGemmEx` and FP8 via `cublasLtMatmul`. The supported operator set after that change is **four ops**: `MatMul`, `Gemm`, `MatMulInteger`, and `Conv`.

On top of that, `safetensors-model-loader-v1` (PR #107) added:

- A clean-room safetensors parser and HuggingFace config reader
- A programmatic `GraphBuilder` that emits `ExecutionGraph`s for model families (Gemma first)
- BF16 end-to-end with direct mmap-to-VRAM weight transfer (no host `Tensor` copy)
- A GPU-resident executor `execute_graph_gpu` in `onnx-rt/src/cuda/gpu_executor.rs` that threads `DeviceTensor`s between operators with no per-op host↔device transfer
- A persistent GPU KV cache (`GpuKvCache`) in `onnx-rt/src/cuda/kv_cache.rs` with `append` / `view` / `reset` / sliding-window pruning
- `Session::from_safetensors()` + `Session::reset_kv_cache()` and the `Arc<Mutex<GpuKvCache>>` field on `Session`

When that pipeline is wired up and `Session::run()` is invoked on a Gemma model, the forward pass reaches `dispatch_gpu_node` and fails on the first operator with:

```
Gather_0: no GPU implementation for Gather
```

This is by design. `dispatch_gpu_node` deliberately refuses to transfer `DeviceTensor`s back to the host to run a CPU operator mid-forward-pass — silent fallback would make LLM inference unusably slow and would defeat the GPU-residency contract documented in `safetensors-model-loader-v1` decision 10. The fix is to add GPU implementations of the missing operators, not to relax the contract.

The Gemma graph emitted by `build_gemma_graph` (see `safetensors-model-loader-v1/specs/gemma-architecture`) requires exactly **seven** operator kinds on GPU in addition to `MatMul`:

1. `Gather` — token-ID → embedding lookup (first op in every transformer)
2. `Add` — residual connections, embedding scale
3. `Mul` — SwiGLU `silu(gate) * up`, embedding scale by `sqrt(hidden_size)`
4. `Silu` — MLP gate activation
5. `RMSNormalization` — pre-/post-attention and pre-MLP norms
6. `RotaryEmbedding` — Q/K rotation at each attention layer
7. `GroupQueryAttention` — fused SDPA with GQA, causal mask, sliding window

The same seven ops also appear in Llama 3, Qwen 2/3, Mistral, and DeepSeek graphs — filling this gap unblocks the whole modern-transformer family, not just Gemma.

**Current file touchpoints:**
- `onnx-rt/src/cuda/gpu_executor.rs` — `execute_graph_gpu`, `DeviceTensor`, `dispatch_gpu_node` (the dispatcher this change extends)
- `onnx-rt/src/cuda/kv_cache.rs` — `GpuKvCache`, `KvView`, `LayerKind`
- `onnx-rt/src/cuda/dispatch.rs` — existing `gpu_gemm`, `gpu_gemm_int8`, `gpu_gemm_fp8` entry points
- `onnx-rt/src/cuda/ffi.rs` — raw CUDA runtime/cuBLAS/cuDNN FFI (we extend this with the driver API + NVRTC)
- `onnx-rt/src/cuda/mod.rs` — `CudaRuntime`, handle wrappers, `GpuPrecision`
- `onnx-rt/src/ops/microsoft.rs` — reference CPU implementations of RMSNormalization, RotaryEmbedding, GroupQueryAttention, SkipSimplifiedLayerNormalization (from `microsoft-fused-ops-v1`)
- `onnx-rt/src/ops/common.rs` — CPU `unary()` helper for activation ops
- `onnx-rt/src/session.rs` — `Session::run_safetensors` threads `Arc<Mutex<GpuKvCache>>` into the dispatcher

## Goals / Non-Goals

**Goals:**
- Seven new GPU operator implementations — `Gather`, `Add`, `Mul`, `Silu`, `RMSNormalization`, `RotaryEmbedding`, `GroupQueryAttention` — callable from `dispatch_gpu_node`
- BF16 and F32 first-class for every operator (matching Gemma's BF16 weights and the cuBLAS `CUBLAS_COMPUTE_32F_FAST_16BF` path already validated)
- KV cache wiring: `GroupQueryAttention` takes `&mut GpuKvCache`, appends the new token's K/V, and reads the full history via `KvView`
- Correctness first — numerical parity with the CPU reference implementations in `ops/microsoft.rs`, documented tolerance bands
- Fail-fast on unsupported shapes / dtypes (no silent host fallback inside the forward pass, matching `gpu-resident-execution` requirements)
- All existing CUDA integration tests (24 + 4 + 2 + 4 = 34 on GB10) continue to pass
- End-to-end Gemma forward pass completes without the `"no GPU implementation for Gather"` error and produces finite logits

**Non-Goals:**
- Flash Attention v2 / v3 style fully-fused attention kernels (tracked as a follow-up change once correctness is proven)
- cuDNN 9.x fused Multi-Head Attention API (noted as an optimization option, not required for v1)
- FP8 attention or FP8 activations (BF16 + F32 only — INT8/FP8/INT4/FP4 are out of scope)
- Quantized attention paths (INT8 Q/K/V, INT4 KV cache)
- Training / backward pass
- Non-transformer operators (Pool, Reduce, Concat, Split, comparison ops) — add when a future model needs them
- Dynamic shapes within a single forward pass — shapes are fixed per-session and determined at graph build time
- Portability to Hopper (CC 9.x) as a hard requirement — the code should be portable in principle, but v1 only validates on Blackwell (CC 12.1) on DGX Spark

## Decisions

### 1. CUDA kernel compilation via NVRTC at runtime, not `build.rs` + nvcc

**Decision:** Hand-written `__global__` kernels live as Rust `const &str` constants (or `include_str!`-loaded `.cu` files) inside the `onnx-rt/src/cuda/kernels/` module. At the first `Session::run()` call on a GPU-resident session, each kernel source is compiled to PTX via **NVRTC** (`nvrtcCreateProgram` / `nvrtcCompileProgram` / `nvrtcGetPTX`), loaded into a CUDA module via the driver API (`cuModuleLoadData`), and the `CUfunction` handle is cached on the `CudaRuntime`. Subsequent launches reuse the cached handle.

**Rationale:**
- No build.rs, no nvcc dependency at compile time, no embedded build-time CUDA toolchain requirement for `cargo build -p smallaios-onnx-rt --features cuda` — NVRTC ships inside `libnvrtc.so` alongside `libcudart.so` on every machine that has the CUDA runtime, which is the same set of machines where the `cuda` feature works at all.
- Keeps the crate buildable on machines without nvcc in `PATH` (CI Docker builds, non-GPU developer machines running `cargo check --features cuda`).
- Kernel source can be expressed inline next to the safe Rust wrapper — a reviewer sees the CUDA source and the Rust launch code in the same file, which is easier to audit than opaque compiled PTX.
- NVRTC compiles for the runtime GPU's compute capability automatically (`--gpu-architecture=sm_121` on GB10), which means the same binary runs on Hopper or Ada without rebuild.

**Trade-offs accepted:**
- First-call latency: ~50-200 ms per kernel for NVRTC compile. Amortized across the entire session lifetime (a single Gemma forward pass is already ~seconds), and mitigated by compiling all known kernels eagerly during `CudaRuntime::init_kernels()` rather than on first dispatch.
- No way to catch kernel syntax errors at `cargo build` time — syntax errors surface at session creation. Mitigation: a `#[test]` that exercises the `init_kernels()` path against every registered kernel, run in the CUDA integration test suite on GB10 hardware.
- PTX produced by NVRTC cannot be statically analyzed by `cargo audit` / `cargo-vet`. Mitigation: kernel source is inline Rust-visible code, gets reviewed in normal PR review.

**Alternatives considered:**
- *build.rs + nvcc → embed PTX* — the approach sketched in the proposal as "Option I". Rejected because it forces every `cuda` feature build to have `nvcc` on `PATH`, which breaks `cargo check --features cuda` on CPU-only workstations. The `cuda` feature already implies `std`; we do not want it to also imply "nvcc toolchain installed system-wide".
- *Precompiled `.ptx` files checked into the repo (Option III)* — rejected because it adds binary artifacts to the git tree, couples the repo to a specific compute capability, and does not handle kernel source edits cleanly in review.
- *Inline PTX via Rust's `asm!` macro on the `nvptx64-nvidia-cuda` target (Option II)* — rejected because it would require adding a second Rust target to the workspace and cross-compiling a parallel kernel crate; massive complexity for no observable runtime benefit over NVRTC.

### 2. Driver API + NVRTC FFI additions

**Decision:** Extend `onnx-rt/src/cuda/ffi.rs` with the minimal set of CUDA driver API and NVRTC bindings needed to compile, load, and launch kernels:

- Driver API: `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuModuleUnload`, `cuLaunchKernel` plus the `CUmodule` / `CUfunction` / `CUresult` opaque types.
- NVRTC: `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTXSize`, `nvrtcGetPTX`, `nvrtcGetProgramLogSize`, `nvrtcGetProgramLog`, `nvrtcDestroyProgram`, `nvrtcResult`.

All FFI stays behind `#[cfg(feature = "cuda")]`. The safe wrapper layer in `cuda/kernels/mod.rs` exposes:

```
pub(crate) fn compile_kernel(name: &str, source: &str, options: &[&str]) -> Result<Kernel, CudaError>
pub(crate) fn launch_kernel(k: &Kernel, grid: (u32, u32, u32), block: (u32, u32, u32), args: &[*mut c_void]) -> Result<(), CudaError>
```

`Kernel` owns the `CUmodule` and caches the `CUfunction`. Dropping a `Kernel` unloads the module.

**Rationale:** The driver API is the only way to launch JIT-compiled PTX; the runtime API (`<<<>>>`) only works for kernels known at C++ compile time. NVRTC is CUDA's in-process JIT compiler and is explicitly supported across every CUDA version SmallAIOS targets.

### 3. Per-operator file organization under `onnx-rt/src/cuda/kernels/`

**Decision:** One file per op family, each containing (a) the kernel source as a `const &str`, (b) a safe Rust wrapper `pub(crate) fn op_name_gpu(exec: &mut GpuExecutor, inputs: &[&DeviceTensor], attrs: &OpAttrs) -> Result<DeviceTensor, CudaError>`, and (c) any internal helpers.

```
onnx-rt/src/cuda/kernels/
    mod.rs          — Kernel type, compile_kernel / launch_kernel helpers, KernelRegistry on CudaRuntime
    elementwise.rs  — add, mul, silu (one kernel per (op, dtype))
    gather.rs       — gather_bf16, gather_f32 (embedding lookup)
    rms_norm.rs     — rms_norm_bf16, rms_norm_f32
    rotary.rs       — rotary_bf16, rotary_f32
    attention.rs    — gqa_softmax_mask, gqa_kv_expand, gqa_merge_heads + top-level gpu_gqa wrapper
```

Every file is gated behind `#[cfg(feature = "cuda")]`. The kernels module is mounted from `cuda/mod.rs` only when the feature is enabled.

**Rationale:** Matches the shape of `onnx-rt/src/cuda/dispatch.rs` and `cuda/conv.rs` from `arm64-gpu-container-v1`. Keeps the blast radius of each operator small and makes it trivial to add a new kernel: copy an existing file, edit the source, register it. Reviewers see a focused diff per op.

### 4. Element-wise kernels: grid-stride loop, 256 threads, broadcast via precomputed strides

**Decision:** Each element-wise kernel (`add`, `mul`, `silu`) is a 1D grid-stride loop with 256 threads per block and `ceil_div(numel, 256)` blocks (capped at 65535, grid-stride loop handles the remainder). Broadcasting is supported by precomputing the output stride array on the host (`[s0, s1, ..., sN-1]`) and the per-input stride-to-output-stride mapping, then passing them as `__constant__` args. Each thread computes its linear output index, decomposes to per-dim indices, recomposes to the input offsets, and reads/writes.

BF16 path uses `__nv_bfloat16` and `__nv_bfloat162` intrinsics from `<cuda_bf16.h>`. F32 path uses plain `float`. Silu uses `expf` for F32 and converts BF16 → F32 → sigmoid → F32 → BF16 per-element (tensor cores not useful for pointwise activations at this scale).

**Rationale:** Grid-stride loops are the standard idiom for element-wise kernels, handle arbitrary numel without branching on grid size, and are trivially correct. 256 threads is the sweet spot for Blackwell SM occupancy without register pressure. Precomputed strides on the host avoids per-thread divisions which are slow on GPU.

### 5. Gather kernel: one thread block per output row

**Decision:** The Gather kernel assumes the ONNX Gemma shape (`data`: `[vocab_size, hidden_size]`, `indices`: `[batch, seq_len]` Int64, `output`: `[batch, seq_len, hidden_size]`) with `axis=0`. Launch with `grid = (batch * seq_len, 1, 1)` and `block = (min(hidden_size, 1024), 1, 1)`. Each block loads its Int64 token ID, computes the source row offset `token * hidden_size * dtype_size`, and threads in the block cooperatively copy `hidden_size` elements from source row to destination row.

A separate kernel variant per input dtype (`gather_bf16`, `gather_f32`). Index tensor is always `i64` (Gemma emits Int64 token IDs).

**Rationale:** One thread block per output row is the standard embedding-lookup pattern, achieves coalesced global memory reads on the source row, and avoids atomics. Supporting only `axis=0` in v1 matches every transformer we care about (embedding tables are always `[vocab, hidden]` and looked up along axis 0). Other axes fail fast.

### 6. RMSNormalization kernel: warp-reduced mean-of-squares

**Decision:** One thread block per outer element (`batch * seq_len`), `min(hidden_size, 1024)` threads per block. Algorithm:

1. Each thread accumulates `sum += x[i] * x[i]` for its slice of the hidden dim (BF16 loaded and converted to F32 for accumulation).
2. Warp-level `__shfl_down_sync` reduction to produce per-warp partial sums.
3. Warp 0 reads per-warp partials from shared memory and reduces to `total_sum`.
4. Thread 0 computes `inv_rms = rsqrtf(total_sum / hidden_size + eps)` and writes to shared.
5. All threads read `inv_rms`, compute `y[i] = x[i] * inv_rms * weight[i]`, write back.

BF16 and F32 variants. Accumulation is always F32 regardless of input dtype — mandatory for numerical parity with the `cublasGemmEx` BF16 path which already accumulates in F32.

**Gemma `(1 + weight)` convention:** `safetensors-model-loader-v1` already pre-adds 1 to the RMSNorm weight at load time (see `gemma.rs`), so the GPU kernel executes the plain formula `y = x * rsqrt(mean + eps) * weight`. Documented as a load-time transform, not a kernel attribute.

**Reference:** CPU implementation in `onnx-rt/src/ops/microsoft.rs::rms_normalization`. Numerical parity validated in the op's test (see §10 below).

### 7. RotaryEmbedding kernel: one thread per rotation pair

**Decision:** RoPE is applied to Q (`[batch, num_heads, seq_len, head_dim]`) and K (`[batch, num_kv_heads, seq_len, head_dim]`) independently. `head_dim` is always even. Rotation pairs are `(x[2i], x[2i+1])`. The kernel takes precomputed `cos` and `sin` tables of shape `[max_seq_len, head_dim / 2]` as initializers — the graph builder from `safetensors-model-loader-v1/gemma.rs` already emits these as constant initializers at build time.

Launch: one thread per output element (grid-stride loop), each thread reads its `(x[2i], x[2i+1])` pair and its `(cos, sin)` values for the current position, computes:

```
out[2i]     = x[2i]     * cos - x[2i+1] * sin
out[2i+1]   = x[2i]     * sin + x[2i+1] * cos
```

BF16 and F32 variants. Both Q and K are supported by the same kernel, dispatched with different input tensors.

**p-RoPE (proportional RoPE):** Gemma 4 uses a proportional variant that scales the rotation frequencies. The `safetensors-model-loader-v1` graph builder bakes the proportional scaling into the precomputed `cos`/`sin` tables, so the GPU kernel executes the standard formula above. The advisory `p_rope` attribute is honored by the load-time table generation, not by the kernel.

### 8. GroupQueryAttention: decomposed via cuBLAS strided batched GEMM

**Decision:** Implement `gpu_gqa` as a composition of existing cuBLAS GEMM calls plus custom kernels for the parts cuBLAS does not cover. This is "Option B" from the proposal, and it is the v1 target. The path:

1. **KV expansion (if needed).** For GQA where `num_kv_heads < num_attention_heads`, call a small `gqa_kv_expand` kernel that replicates each KV head `num_attention_heads / num_kv_heads` times along the head axis. For standard MHA (`num_kv_heads == num_attention_heads`) this step is skipped.
2. **Q·K^T.** Call `cublasGemmStridedBatchedEx` with `batchCount = num_attention_heads`, M = `seq_len_q`, N = `seq_len_kv`, K = `head_dim`, inputs BF16/F32, compute type `CUBLAS_COMPUTE_32F` (FP32 accumulation), output F32 to the scratch buffer. This produces the `[num_heads, seq_len_q, seq_len_kv]` attention score matrix.
3. **Masked softmax.** Call a single custom `gqa_softmax_mask` kernel that, for each `(head, row)`:
    - Applies scale `1 / sqrt(head_dim)`
    - Applies causal mask (`j > i` → `-inf`)
    - Applies sliding-window mask if the layer is a local layer (`j < i - window` → `-inf`)
    - Warp-level reduction for row max → subtract max → `expf` → warp-level sum → divide
    - Writes the F32 softmax output in-place
4. **Softmax·V.** Call `cublasGemmStridedBatchedEx` again with `batchCount = num_attention_heads`, M = `seq_len_q`, N = `head_dim`, K = `seq_len_kv`, inputs F32 softmax + BF16/F32 V, output BF16/F32 to `[num_heads, seq_len_q, head_dim]`.
5. **Merge heads.** Call `gqa_merge_heads` kernel (pure transpose + reshape) to produce the final `[batch, seq_len_q, hidden_size]` tensor.

**KV cache wiring (load-bearing):** `dispatch_gpu_node` passes `Option<&mut GpuKvCache>` into `gpu_gqa`. Per invocation, in order:

1. The wrapper **first** appends the current step's K and V to the per-layer cache via `GpuKvCache::append(layer_idx, new_k, new_v, current_position)`.
2. The wrapper **then** reads the full cached K and V history via `GpuKvCache::view(layer_idx, current_position + 1)` and uses the returned `KvView` raw pointers as the K and V operands of step 2 above. The `seq_len_kv` dimension is `current_position + 1` for global layers, or `min(current_position + 1, window)` for sliding-window layers.
3. If `GpuKvCache` is `None` (the graph was built without a cache — e.g., a test), the wrapper computes attention using only the current step's K/V without caching. This path is used only in unit tests.

**Ordering contract:** Append before view. The attention output at position `t` depends on all positions `0..=t`, which requires the new `(K_t, V_t)` to be present in the cache before the view is constructed.

**Attention intermediate workspace.** The `[num_heads, seq_len_q, seq_len_kv]` score matrix grows quadratically with context length. For full Gemma contexts (256K tokens), even one head's scratch is ~256K² × 4 bytes = 256 GiB — unusable. Mitigation: `gpu_gqa` allocates a single reusable `DeviceBuffer` scratch cap on the `CudaRuntime` sized for the session's `max_seq_len` and fails fast if a request exceeds it. For sliding-window layers the scratch is sized by `window`, not `max_seq_len`, which Gemma's configuration enforces (the last layer is the only global layer and it still fits in practice). Document in the session creation path.

### 9. Executor dispatcher extension

**Decision:** Extend `dispatch_gpu_node` in `onnx-rt/src/cuda/gpu_executor.rs`:

1. Add a new match arm for each of the seven new op kinds (`Gather`, `Add`, `Mul`, `Silu`, `RMSNormalization`, `RotaryEmbedding`, `GroupQueryAttention`). Each arm calls the safe wrapper from `cuda/kernels/<op>.rs`, returning a new `DeviceTensor`.
2. The dispatcher signature gains a `kv_cache: Option<&mut GpuKvCache>` parameter. For attention nodes, the dispatcher unwraps it and passes `Some(&mut *cache)` to `gpu_gqa`. For non-attention nodes, it simply threads the borrow through (attention is the only op that touches the cache).
3. `execute_graph_gpu` and `execute_graph_gpu_with_weights` take the same `Option<&mut GpuKvCache>` parameter and pass it to every call site of `dispatch_gpu_node`.
4. `Session::run_safetensors` locks its `Arc<Mutex<GpuKvCache>>`, converts the guard into a `&mut GpuKvCache` for the duration of the forward pass, and passes `Some(cache)` down. This resolves the TODO left in `safetensors-model-loader-v1` §9.
5. The existing fail-fast "no GPU implementation for `<op>`" branch stays in place as the catch-all for any op not yet ported. After this change the branch is only reached for ops outside the Gemma / Llama / Qwen operator set.

**Rationale:** Threading the cache through the dispatcher instead of making it a field on the executor keeps the executor stateless — useful for tests that want to run a partial graph without a cache. Mutable borrow semantics make it explicit that two attention ops cannot run in parallel against the same cache (SmallAIOS is cooperative async, not SMP, so this is already the invariant).

### 10. Numerical correctness testing strategy

**Decision:** Every new GPU operator ships with two tiers of correctness tests, both gated behind `#[cfg(feature = "cuda")]` and marked `#[ignore]` so they run only when explicitly invoked (the existing convention for GB10 hardware tests):

1. **Scalar reference tier.** For small input sizes (e.g. `[2, 4, 16]` for RMSNorm, `[1, 2, 8, 16]` for GQA), a plain Rust loop computes the expected output and compares element-wise against the GPU output.
2. **CPU operator parity tier.** For medium input sizes (e.g. `[1, 32, 4096]`), the test runs the same inputs through the existing CPU operator in `ops/microsoft.rs` (or `ops/common.rs` for the element-wise ops) and compares element-wise.

**Tolerance bands:** F32 tests use `1e-3` absolute tolerance (matches the existing `arm64-gpu-container-v1` TF32 band). BF16 tests use `1e-2` absolute tolerance (matches the `cuda-container-runtime` BF16 scenarios in `safetensors-model-loader-v1`). Tolerances are documented in each test and in the spec files.

Run manually on GB10 via `just test-cuda` (or an equivalent invocation). CI runs the non-GPU crate-level tests that exercise the kernel compile path on a CPU-only runner via a mock that stubs out `nvrtcCompileProgram`.

### 11. Error handling

**Decision:** All kernel operations return `Result<_, CudaError>`. The `CudaError` enum in `cuda/mod.rs` gains three new variants:

- `KernelCompileFailed { name: String, log: String }` — emitted when `nvrtcCompileProgram` returns non-zero; the `log` comes from `nvrtcGetProgramLog`.
- `KernelLoadFailed { name: String, cuda_result: i32 }` — emitted when `cuModuleLoadData` fails.
- `KernelLaunchFailed { name: String, cuda_result: i32 }` — emitted when `cuLaunchKernel` fails, with the driver API result code.

Shape / dtype mismatches in the safe wrapper layer return existing `CudaError::InvalidShape` / `InvalidDtype` variants before any kernel launch. No panics on the GPU dispatch path.

### 12. Feature gating and `#![no_std]` boundary

**Decision:** The entire `onnx-rt/src/cuda/kernels/` module and its contents are gated behind `#[cfg(feature = "cuda")]`. The `cuda` feature already implies `std` (documented in `safetensors-model-loader-v1` decision 8), so kernel source strings, `String`-based error messages, and `HashMap`-backed `KernelRegistry` are fine.

The default-features `onnx-rt` build (`cargo build -p smallaios-onnx-rt`) stays `#![no_std]` and never sees a single line of this change. The bare-metal kernel targets (`x86_64-unknown-none`, `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`) are unaffected — they never enable `cuda`.

## Risks / Trade-offs

**[Decomposed attention performance vs Flash Attention]** The Option-B path does a full `[num_heads, seq_len, seq_len]` materialization of the QK^T matrix, which is exactly what Flash Attention was designed to avoid. This will be substantially slower than cuDNN's fused MHA or a hand-written FA2 kernel at large contexts. **Mitigation:** v1 explicitly prioritizes correctness. The spec allows a follow-up change (`transformer-gpu-fused-attention-v1` or similar) to swap in a fused path without touching the operator contract. Performance benchmarking is a v1 exit metric, not a gate.

**[NVRTC compilation overhead at first session load]** Compiling ~10-15 kernels at session creation adds a few hundred milliseconds to the first `Session::run()`. **Mitigation:** Compile all kernels eagerly during `CudaRuntime::init_kernels()` on runtime construction (not lazily on first dispatch), so the cost lands at boot, not on the hot path. Cache compiled `CUmodule`s on the `CudaRuntime` for reuse across sessions.

**[Numerical divergence between f32 CPU reference and BF16 GPU result]** BF16 has an 8-bit mantissa, so accumulated error over a full forward pass can drift ~1e-2 from the f32 reference. This is expected and matches the `cuda-container-runtime` BF16 tolerance documented in `safetensors-model-loader-v1`. **Mitigation:** Per-op tolerance bands documented in each spec scenario (see §10). Reviewers of future PRs should expect this and not flag it as a regression.

**[KvView raw pointer lifetime]** `GpuKvCache::view` returns a `KvView` containing raw device pointers into the cache's `DeviceBuffer`. These pointers are only valid as long as (a) the cache is not `reset()` and (b) the underlying `DeviceBuffer` has not been reallocated. **Mitigation:** The KV cache is pre-allocated at session creation (decision from `safetensors-model-loader-v1` §8) and never grows or moves during a forward pass. Document the invariant in a doc comment on `KvView` and on `gpu_gqa`'s safety preamble.

**[Choice of `cublasGemmStridedBatchedEx` vs `cublasGemmBatched`]** The Ex variant supports mixed precision (BF16 input, F32 accumulation, BF16 output), which we need. The basic `cublasGemmStridedBatched` only supports same-precision input/output. **Mitigation:** Use `cublasGemmStridedBatchedEx` uniformly and document the compute type matrix (`CUBLAS_COMPUTE_32F` for BF16 inputs, `CUBLAS_COMPUTE_32F_FAST_TF32` for F32 inputs).

**[Attention intermediate memory ballooning]** QK^T is `[num_heads, seq_len, seq_len]`. For `seq_len = 4096` and `num_heads = 32` on Gemma, that is 2 GiB in F32. At 256K context it's 8 TiB — infeasible. **Mitigation:** Cap the scratch buffer size on `CudaRuntime` at session creation time based on the model's `sliding_window` for local layers and a configurable `max_prefill_len` for global layers. Fail fast with a clear error when a request exceeds the cap. Document that sliding-window layers are the only reason Gemma inference fits in VRAM at all for long contexts, and that this is structural, not incidental.

**[NVRTC dependency on CUDA runtime headers]** Kernel source uses `<cuda_bf16.h>` intrinsics. NVRTC finds these via its bundled include path, but if the CUDA toolkit install is non-standard the compile fails with "cannot find `<cuda_bf16.h>`". **Mitigation:** Pass explicit include paths to NVRTC via `-I` options from the `CudaRuntime` init (probed at startup). If none found, emit a clear `KernelCompileFailed` error naming the missing header.

## Open Questions

1. **Precompiled PTX cache at build/package time.** Should we optionally cache compiled PTX to disk (e.g. in `${CUDA_CACHE_PATH}` or a SmallAIOS-specific cache) on first compile, and skip NVRTC on subsequent boots? Would shave ~200ms off container cold start. Default for v1: **no**, rely on runtime NVRTC. Revisit if cold-start latency is an issue in production.
2. **cuBLASLt Matmul with epilogue for fused attention.** `cublasLtMatmul` supports fused softmax + bias epilogues (`CUBLASLT_EPILOGUE_*`). A cuBLASLt-based attention path could fuse the scale + softmax step into the first GEMM call, eliminating one kernel launch per layer. This is Option C from the proposal and is a natural v2 optimization. Track as `transformer-gpu-fused-attention-v1`.
3. **Kernel launch latency telemetry.** Should `CudaRuntime` expose per-kernel launch counts and aggregate latency (via `cudaEventRecord` around each launch) for perf debugging? Would cost ~100ns per launch in event creation. Default for v1: **no** — keep the hot path as minimal as possible. Add a `cuda-profile` feature flag in a follow-up if the need arises.
4. **p-RoPE as a kernel attribute vs load-time bake.** v1 bakes p-RoPE into the precomputed `cos`/`sin` tables at graph build time. If a future model uses a p-RoPE variant that depends on the runtime token position in a way that can't be precomputed (e.g. YaRN), the kernel will need a `p_rope_scale` attribute. Track as future work.
