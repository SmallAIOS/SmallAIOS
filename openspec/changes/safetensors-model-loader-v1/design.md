## Context

SmallAIOS today is an ONNX-only inference runtime. The full pipeline is:

```
.onnx file → protobuf::decode_model → ModelProto
           → graph::build_execution_graph → ExecutionGraph
           → executor::execute_graph → outputs
```

`ExecutionGraph` is a topologically-sorted list of `ExecutionNode`s, each with an op type, input/output tensor names, and attribute list. The executor walks the graph node-by-node, dispatching each op to a CPU implementation (and now optionally GPU via the `arm64-gpu-container-v1` work).

The HuggingFace model ecosystem doesn't ship ONNX. Models like Gemma 4 31B-it are distributed as:

- **`config.json`** — JSON describing the architecture (layer count, hidden size, RoPE config, sliding window, etc.)
- **`model-00001-of-NNNNN.safetensors`** — sharded binary weight files containing tensors keyed by name (e.g. `model.layers.0.self_attn.q_proj.weight`)
- **`tokenizer.json` / `tokenizer.model`** — SentencePiece BPE vocab and merge rules
- **`generation_config.json`** — sampling defaults

The architecture is **implicit** in HuggingFace's PyTorch `modeling_*.py` files — there's no graph file. To run a HuggingFace model on SmallAIOS we have two options:

1. Convert via `optimum-cli export onnx` (often fails for 30B+, produces huge files, loses architectural info)
2. **Construct the ExecutionGraph programmatically** from the architecture config + weight bindings — bypassing ONNX entirely

Option 2 is what this change implements. We treat each model family (Gemma, Llama, Qwen) as a known architecture template parameterized by config values, and emit the same `ExecutionGraph` structure that the existing executor already runs.

**Current file touchpoints:**
- `onnx-rt/src/graph.rs` — `ExecutionGraph`, `ExecutionNode`, `build_execution_graph()` from protobuf
- `onnx-rt/src/tensor.rs` — `Tensor`, `DataType` (BFloat16 enum exists but operator paths are limited)
- `onnx-rt/src/operators.rs` — op registry and `OpKind` enum (already includes RMSNorm, RotaryEmbedding, GroupQueryAttention from microsoft-fused-ops-v1)
- `onnx-rt/src/ops/microsoft.rs` — Gemma-relevant fused ops
- `onnx-rt/src/session.rs` — `Session` with `cuda_runtime: Option<Arc<CudaRuntime>>`
- `onnx-rt/src/executor.rs` — `execute_graph()` runs the graph end-to-end
- `container/src/model_manager.rs` — discovers `.onnx` files at boot

## Goals / Non-Goals

**Goals:**
- Load Gemma 4 31B-it from safetensors files on DGX Spark
- Tokenize text input with the Gemma SentencePiece tokenizer
- Generate text completions via autoregressive decoding through the existing executor + GPU dispatch
- Reuse 90%+ of existing operators (RMSNorm, RotaryEmbedding, GQA, MatMul, Softmax, Add)
- BF16 native data flow (input weights, intermediate tensors, output) — not just BF16 compute with f32 I/O
- Architecture pluggability — adding Llama 3 or Qwen should be a new architecture template, not a runtime rewrite
- Stay clean-room: no external safetensors/tokenizer crates, hand-rolled minimal parsers

**Non-Goals:**
- Vision encoder (Gemma 4 multimodal) — text-only inference
- GGUF format
- Quantized weight loading (BF16 first; INT4/INT8 in follow-up)
- Streaming token output (return complete response)
- Multi-GPU pipeline parallelism
- Training, fine-tuning, LoRA
- Other model families in this change (architecture should generalize, but only Gemma is implemented)
- Prompt caching across requests
- Continuous batching (single inference at a time)

## Decisions

### 1. New `model-loader` module inside `onnx-rt`, not a separate crate

**Decision:** Add `onnx-rt/src/model_loader/` as a new module inside the existing `smallaios-onnx-rt` crate, not a separate workspace crate.

**Rationale:**
- The loader produces `ExecutionGraph` which lives in `onnx-rt::graph` — keeping them in the same crate avoids cross-crate type complexity
- Reuses `onnx-rt`'s existing `Tensor`, `DataType`, `AttributeProto`, `OpKind` types
- The container already depends on `smallaios-onnx-rt` — no Cargo.toml changes needed
- The crate name `smallaios-onnx-rt` becomes a slight misnomer (it's now "the inference runtime, with ONNX as one of two model formats") but renaming is more disruptive than the misnomer

**Alternative considered:** New `smallaios-model-loader` crate that depends on `onnx-rt`. Rejected because the loader is intrinsically tied to the executor's data structures, and the dependency would have to flow `model-loader → onnx-rt → compute`, adding a layer with no boundary value.

### 2. Safetensors parser: minimal, mmap-friendly, zero-copy

**Decision:** Hand-roll a `safetensors` format parser in `onnx-rt/src/model_loader/safetensors.rs`. The format is simple:
- 8 bytes: little-endian u64 = JSON header length
- N bytes: UTF-8 JSON header `{"tensor_name": {"dtype": "BF16", "shape": [4096, 4096], "data_offsets": [0, 33554432]}, ...}`
- Remaining bytes: raw tensor data, contiguous, no padding

**Rationale:**
- ~300 LOC, no external crate dependencies
- The official `safetensors` crate requires `std` and adds ~1 MB of binary weight — overkill
- Zero-copy: tensor data lives at file offsets, return slices into the mmap'd file
- mmap-friendly: open the file once, the OS pages in tensor data on demand (critical for 60+ GB Gemma files)

**API:**
```rust
pub struct SafetensorsFile {
    mmap: memmap2::Mmap,        // mmap the whole file
    tensors: BTreeMap<String, TensorEntry>,
}

pub struct TensorEntry {
    pub dtype: DataType,
    pub shape: Vec<i64>,
    pub data_offset_start: usize,
    pub data_offset_end: usize,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, LoaderError>;
    pub fn tensor_names(&self) -> impl Iterator<Item = &str>;
    pub fn get_tensor(&self, name: &str) -> Option<TensorView>;
}
```

`TensorView` is a borrowed slice — the actual tensor data is never copied into Rust ownership, the mmap region is the canonical storage.

**Sharded models:** Gemma 4 31B-it ships as `model-00001-of-00007.safetensors` ... `model-00007-of-00007.safetensors`. The `model.safetensors.index.json` file maps tensor names to shard files. `MultiShardSafetensors` wraps multiple `SafetensorsFile` instances.

**Note on `std` dependency:** `memmap2` requires `std`. The safetensors loader is gated behind a `safetensors` feature flag that implies `std`. Bare-metal kernel builds never enable it.

### 3. Programmatic graph construction via architecture templates

**Decision:** For each supported model family, define a Rust function that takes a config + weight store and emits an `ExecutionGraph`:

```rust
pub fn build_gemma_graph(
    config: &GemmaConfig,
    weights: &SafetensorsFile,
) -> Result<ExecutionGraph, LoaderError>;
```

The function constructs `ExecutionNode`s in the same format the ONNX path produces, so the existing `executor::execute_graph` runs them unchanged. Tensor data is loaded into `Tensor` structs with `raw_data` pointing to bytes copied from the safetensors mmap (or referenced — TBD based on tensor lifetime).

**Rationale:**
- The executor doesn't care where the graph came from — `ExecutionGraph` is the contract
- Each architecture is a few hundred lines of imperative graph-building code that mirrors PyTorch's `forward()` method
- Easy to add new families: copy `gemma.rs` to `llama.rs`, change the ops + naming conventions
- No "model description language" to design — Rust is the description language

**Alternative considered:** Generic config-driven builder where you describe layers as data (`Layer::Attention { num_heads: 32, ... }`). Rejected: too much abstraction for too few model families. Two architectures don't need a framework; we can refactor when we add the third.

**Layer template for Gemma 4:**
```rust
fn gemma_layer(graph: &mut GraphBuilder, layer_idx: usize, config: &GemmaConfig, weights: &SafetensorsFile) {
    let prefix = format!("model.layers.{}", layer_idx);
    
    // Pre-attention RMSNorm
    let normed_input = graph.rms_norm(&format!("{}.input_layernorm.weight", prefix), config.rms_norm_eps);
    
    // QKV projections (separate, not fused — Gemma uses separate q_proj/k_proj/v_proj)
    let q = graph.matmul(normed_input, &format!("{}.self_attn.q_proj.weight", prefix));
    let k = graph.matmul(normed_input, &format!("{}.self_attn.k_proj.weight", prefix));
    let v = graph.matmul(normed_input, &format!("{}.self_attn.v_proj.weight", prefix));
    
    // RoPE (proportional variant)
    let q_rot = graph.rotary_embedding(q, config.rope_theta, /* p_rope */ true);
    let k_rot = graph.rotary_embedding(k, config.rope_theta, true);
    
    // Sliding window OR global attention based on layer index
    let is_global = layer_idx == config.num_hidden_layers - 1 || layer_idx % config.sliding_window_pattern == 0;
    let attn_out = graph.attention(q_rot, k_rot, v, is_global, config.sliding_window);
    
    // Output projection
    let attn_proj = graph.matmul(attn_out, &format!("{}.self_attn.o_proj.weight", prefix));
    
    // Residual + post-attention RMSNorm
    let attn_residual = graph.add(input, attn_proj);
    let normed_attn = graph.rms_norm(&format!("{}.post_attention_layernorm.weight", prefix), config.rms_norm_eps);
    
    // MLP: gated SwiGLU
    let gate = graph.matmul(normed_attn, &format!("{}.mlp.gate_proj.weight", prefix));
    let up = graph.matmul(normed_attn, &format!("{}.mlp.up_proj.weight", prefix));
    let mlp = graph.matmul(graph.silu_mul(gate, up), &format!("{}.mlp.down_proj.weight", prefix));
    
    // Final residual
    graph.add(attn_residual, mlp)
}
```

**`GraphBuilder` helper:** A thin wrapper around `ExecutionGraph` that auto-generates output tensor names, tracks the latest tensor, and emits `ExecutionNode`s. ~200 LOC.

### 4. BF16 native tensor data flow

**Decision:** Tensors store BF16 as 2-byte values in `raw_data` (already supported in `DataType::BFloat16`). Extend operator implementations to handle BF16 inputs/outputs natively rather than converting to f32 at boundaries.

**Rationale:**
- Gemma weights are 62 GB in BF16. Converting to f32 doubles that to 124 GB, eating all of DGX Spark's unified memory.
- The cuBLAS BF16 compute path (validated in `arm64-gpu-container-v1`) accepts BF16 input pointers when `cudaDataType_t::CUDA_R_16BF` is passed. We just need to wire BF16 tensors through the GPU dispatch path.
- CPU operators need BF16 support too for ops that fall back from GPU (e.g., element-wise, normalization).

**Implementation approach:**
- New BF16 conversion helpers: `bf16_to_f32`, `f32_to_bf16` in `tensor.rs` (~20 LOC, just bit shifts)
- `gpu_gemm()` in `cuda/dispatch.rs` accepts a precision tag indicating tensor data type — if BF16, pass `CUDA_R_16BF` to cuBLAS instead of `CUDA_R_32F`
- CPU operators: for ops like Add/Mul that touch element data, add a BF16 variant that converts on read, computes in f32, converts on write
- Hot ops (MatMul, RMSNorm) get native BF16 paths to avoid the convert overhead

**Trade-off:** This is a chunky operator pass — every CPU op needs BF16 awareness. Mitigation: only the ops Gemma actually uses need BF16 (about 12 ops), not the whole catalog. The rest can keep their f32-only paths and we add BF16 lazily.

### 5. SentencePiece tokenizer: parse `tokenizer.json`, not `tokenizer.model`

**Decision:** Implement a tokenizer that parses HuggingFace's `tokenizer.json` file (the modern Tokenizers library format), not the legacy `tokenizer.model` SentencePiece protobuf.

**Rationale:**
- `tokenizer.json` is JSON — easy to parse with our existing minimal JSON parser
- Contains everything needed: vocab, merge rules, special tokens, normalization rules, pre-tokenizer config
- `tokenizer.model` is protobuf — would need protobuf decoding for an unrelated schema
- All modern HuggingFace models ship `tokenizer.json` alongside `tokenizer.model`
- The tokenizers crate uses the same format internally; we're matching the canonical representation

**Scope of tokenizer support:**
- BPE (Byte-Pair Encoding) — Gemma uses this
- Vocab as string→id and id→string maps
- Merge rules as ordered list
- Special tokens (BOS, EOS, PAD, UNK, system prompt markers)
- Whitespace + Unicode normalization (NFC)
- **Not implemented:** WordPiece, Unigram, SentencePiece sampling — Gemma doesn't need them

**Estimated size:** ~1000 LOC for the tokenizer + 500 LOC for tests.

### 6. Generation loop is provided by `llm-api-translation-v1`

**Decision:** This change does NOT implement the autoregressive generation loop, sampling strategies, or text-level API. Those are all provided by the parallel `llm-api-translation-v1` change (capabilities `llm-generation` and `llm-tokenizer`). This change makes safetensors models loadable via the same `Session::run()` interface that `llm-generation` already calls.

**Integration contract:** `Session::run()` for a safetensors model accepts the same inputs and returns the same outputs as `Session::run()` for an ONNX model — token IDs in, logits out. The generation loop in `llm-api-translation-v1` doesn't care which loader produced the session.

**KV cache contract:** The KV cache lives in the `Session` between calls (added in this change as `Session.kv_cache`). The generation loop in `llm-api-translation-v1` calls `Session::run()` repeatedly; this change ensures the cache persists across those calls and grows as new tokens are appended.

### 7. KV cache lives in the Session, GPU-resident

**Decision:** Each `Session` owns a per-layer KV cache that grows as tokens are generated. The cache is stored in GPU memory when GPU dispatch is active, host memory otherwise.

**Rationale:**
- Gemma 4 31B context can be 256K tokens × 60 layers — KV cache for full context is ~10s of GB
- Transferring this between host/device per token would dominate latency
- GPU residency means KV cache stays put, only the new query/key/value for the current token transfer
- Reuses `kv_compression` module for sliding window pruning (drop tokens beyond window)

**Trade-off:** This requires the Session to hold GPU buffers between calls — adds a `kv_cache: Option<KvCacheStore>` field. The `KvCacheStore` is GPU-aware (uses `DeviceBuffer` when GPU is available).

### 8. Phased rollout: parser → graph → tokenizer → generation → end-to-end

**Decision:** Implement and merge the change in 5 sequential phases (matching the proposal). Each phase has its own working tests and ships independently. Phase boundaries are merge points where the change can be paused.

**Rationale:**
- 5,000+ LOC change otherwise — too large for one PR
- Each phase is independently valuable: safetensors parser is useful even without graph building (debugging tool); tokenizer is useful before generation works
- Catches integration issues earlier
- Allows external review checkpoints

**Alternative considered:** All-or-nothing single PR. Rejected: review fatigue and integration risk too high.

## Risks / Trade-offs

**[Architectural drift from HuggingFace `modeling_gemma3.py`]** → If HuggingFace updates Gemma's reference implementation (e.g., changes how p-RoPE is computed, fixes a bug), our hand-coded graph builder will silently produce different results. **Mitigation:** Pin a specific version of the HuggingFace `transformers` library when validating numerical correctness, document the version, re-validate when bumping. Add a smoke test that compares our output against `transformers` reference for a fixed prompt.

**[BF16 numerical precision]** → 8-bit mantissa is enough for inference but accumulated error over 60 layers + 256K tokens could drift noticeably from f32 reference. **Mitigation:** Use BF16 for storage and tensor core compute, but accumulate in f32 (`CUBLAS_COMPUTE_32F_FAST_16BF` already does this). Validate that generated outputs are semantically correct, not bit-exact, against HuggingFace reference.

**[Sharded safetensors loading at scale]** → Gemma 4 31B has 7+ shard files totaling 62 GB. mmap'ing all of them at once may exceed file descriptor limits or VM region limits on some kernels. **Mitigation:** Open shards lazily as tensors are requested; close after the graph is built (the data is already memcpy'd to GPU VRAM at that point). Document the open file count for ops teams.

**[Hand-rolled tokenizer correctness]** → BPE is subtle — wrong tokenization produces nonsense output even if the model is correct. **Mitigation:** Test against the official HuggingFace Tokenizers Python library output for a corpus of strings, including edge cases (Unicode, whitespace, special tokens). Continuous validation against `transformers.AutoTokenizer.from_pretrained("google/gemma-4-31B-it")`.

**[KV cache memory blowup]** → Full 256K context KV cache for 31B model could exceed available VRAM if not pruned. **Mitigation:** Default `max_new_tokens` to 2048, document the relationship between context length and VRAM, integrate with sliding window pruning from `kv-compression`.

**[mmap'd safetensors and `#![no_std]` boundary]** → The `onnx-rt` crate is `#![no_std]` for kernel mode, but mmap is a `std` operation. **Mitigation:** Gate the safetensors loader behind a `safetensors` Cargo feature that implies `std`. Kernel/bare-metal builds never enable it. The runtime's `Tensor` type stays format-agnostic.

**[Architectural template explosion]** → If Llama, Qwen, Mistral, DeepSeek all need separate templates, we end up with N similar files. **Mitigation:** Accept that as the cost of correctness. Architecture-as-data is harder to maintain than architecture-as-code for fewer than ~5 model families. Refactor only when we have evidence that the repetition is hurting us.

**[Generation loop GPU↔CPU bottleneck]** → Our current per-op GPU dispatch transfers tensors back to CPU after each operator. For autoregressive generation with 60 layers × hundreds of ops × hundreds of tokens, this is fatal. **Mitigation:** This change must include graph-level GPU memory residency — tensors stay on GPU between ops. This is a known follow-up from `arm64-gpu-container-v1` and is a hard prerequisite for usable LLM inference.

## Decisions (Addendum)

### 9. Safetensors models are GPU-only — no CPU fallback path

**Decision:** Models loaded via the safetensors path REQUIRE a GPU. If `CudaRuntime::init()` fails or no GPU is available, safetensors model loading fails fast at boot with a clear error. Sessions backed by safetensors do not have a CPU execution path.

**Rationale:**
- Gemma 4 31B in BF16 is 62 GB. Running on CPU would be impractically slow (minutes per token at best) — no realistic deployment uses CPU for 30B+ models.
- CPU inference for these models is not a use case worth supporting; the engineering cost (BF16 on every CPU operator path) would be wasted.
- Forces tensors to live in GPU VRAM for the entire model lifetime — solves the per-op transfer problem by removing the CPU path entirely.
- ONNX models continue to support CPU execution unchanged — this constraint applies only to the safetensors loader and the model families it loads.

**Implementation:**
- `SafetensorsSession` (or a flag on `Session`) marks the session as GPU-required
- At session creation time, if `cuda_runtime.is_none()`, return `LoaderError::GpuRequired`
- The graph builder skips host tensor allocation entirely — weights load directly from safetensors mmap into `DeviceBuffer` via `cudaMemcpy`
- `Session::run()` for a GPU-required session takes input tensors that are already on GPU (or transfers them once at the start) and never leaves device memory

**Trade-off:** Loses the ability to debug LLM inference on CPU. Mitigation: smaller test models (e.g. Gemma 270M from the existing fixture suite) can still be loaded via the ONNX path for CPU debugging when needed. The safetensors path is the production LLM path; CPU-only debugging uses ONNX.

### 10. Graph-level GPU residency is included in this change (not deferred)

**Decision:** This change implements graph-level GPU memory residency as a hard requirement, not a follow-up. Tensors stay on GPU between operators for the entire forward pass and across generation steps.

**Rationale:**
- Per-op transfer makes LLM inference unusably slow (60 layers × hundreds of ops × hundreds of tokens = millions of round-trips)
- The "GPU-only" constraint above gives us permission to do this cleanly — no need to handle the CPU fallback case
- The design becomes simpler: tensors are `DeviceBuffer`s for the whole forward pass, never converted to host `Tensor`
- This is the natural shape of every other inference framework (TensorRT, vLLM, llama.cpp) — we're aligning with industry practice

**Implementation:**
- New `DeviceTensor` type alongside existing `Tensor`: lives in `DeviceBuffer`, knows its shape and dtype, no host data
- New executor path `execute_graph_gpu()` that operates on `DeviceTensor`s start to finish
- KV cache stored as `Vec<DeviceBuffer>` keyed by layer index, persists across `Session::run()` calls
- Input tokens converted host→device once at the start of generation; output logits converted device→host only after the final layer

**Scope impact:** Adds ~2000 LOC for the GPU executor path, but eliminates an entire class of perf bugs and is the only way LLM inference becomes useful.

## Resolved Questions

1. **Sharded loading**: Open all shards eagerly at load time, mmap each, build a unified tensor name → (file, offset) lookup. After weights are copied to GPU VRAM, drop the mmaps. File descriptor count is bounded (Gemma 4 31B = 7 shards, well under any limit).
2. **Tensor lifetime**: Weights are copied from safetensors mmap directly into `DeviceBuffer` (GPU VRAM) at graph build time. mmap regions are dropped after load. No host-side `Tensor` ownership for model weights — they live exclusively on GPU.
3. **GPU residency timing**: **Included in this change** (decision 10 above). Not deferred.
4. **Tokenizer validation corpus**: Capture HuggingFace Tokenizers Python output for ~1000 representative strings (English prose, code, Unicode edge cases, special tokens) into a JSON test fixture. Compare our tokenizer output against this fixture in CI.
5. **HTTP API shape**: Out of scope for this change. The OpenAI Chat Completions, Anthropic Messages, tokenizer, and generation loop are all provided by the parallel `llm-api-translation-v1` change. This change exposes safetensors-loaded models through the same `Session` interface that the existing API handlers and `llm-generation` loop already consume — no API surface changes are needed here.
