## Why

Production large language models (Gemma 4, Llama 3, Qwen, DeepSeek) are distributed as **safetensors files** on HuggingFace, not ONNX. The ONNX export pipeline (`optimum-cli`) frequently fails or produces multi-tens-of-GB files for 30B+ parameter models, and the resulting ONNX graphs have lost the architectural information needed for efficient execution. SmallAIOS today is an ONNX-only runtime, which locks us out of the entire HuggingFace model ecosystem and the LLMs that matter for inference workloads.

This change adds a second model loading path: parse safetensors files directly, read the HuggingFace `config.json` for architecture metadata, and construct an `ExecutionGraph` programmatically — bypassing ONNX entirely. The first concrete target is **Gemma 4 31B-it** running on the DGX Spark (124 GB unified memory comfortably holds the 62 GB BF16 weights), validating end-to-end LLM inference on SmallAIOS.

**Relationship to `llm-api-translation-v1`:** That change adds the OpenAI Chat Completions and Anthropic Messages API endpoints, the BPE tokenizer (`llm-tokenizer`), the autoregressive generation loop (`llm-generation`), and prompt template handling. It assumes the model is loadable and runnable. **This change is the prerequisite that makes Gemma 4 (and other safetensors-only models) loadable and runnable** — it is the model-loading + execution layer that `llm-api-translation-v1` calls into. The two changes ship as a pair: this one provides the model, that one provides the API surface.

## What Changes

### Phase 1 — Safetensors + Config Loading
- **`safetensors` parser**: read the binary header (JSON metadata + tensor offsets), mmap-friendly weight access, no full-file load
- **HuggingFace `config.json` parser**: extract architecture metadata (layer count, hidden size, attention heads, vocab size, RoPE config, sliding window, etc.)
- **BF16 native tensor support**: extend `DataType::BFloat16` end-to-end (raw bytes, conversion helpers, operator dispatch). The cuBLAS BF16 compute path is already validated.

### Phase 2 — Programmatic Graph Construction
- **Architecture-aware graph builder**: given a `GemmaConfig` + weight store, emit an `ExecutionGraph` with the right Gemma transformer layer pattern (RMSNorm → QKV projection → RoPE → sliding-window/global attention → output projection → RMSNorm → MLP gated SwiGLU → residual)
- **Reuse existing operators**: MatMul, Gemm, RMSNorm, RotaryEmbedding, GroupQueryAttention, Softmax, Add — most already exist from `microsoft-fused-ops-v1` and `additional-operators-v1`
- **New operators where needed**: sliding-window attention mask, proportional RoPE (p-RoPE) variant, gated SwiGLU activation
- **Weight binding**: link `safetensors` tensors to operator inputs by name (e.g. `model.layers.0.self_attn.q_proj.weight` → MatMul input)

### Phase 3 — Session API for safetensors models
- **`SafetensorsSession`** (or extension to `Session`) that the existing `llm-tokenizer` and `llm-generation` modules from `llm-api-translation-v1` can call into
- **`Session::run()` for GPU-resident models**: input token IDs as a tensor, returns logits as a tensor — same shape contract as ONNX models so the existing generation loop works unchanged
- **KV cache field on Session**: `Vec<DeviceBuffer>` keyed by layer index, persists across `Session::run()` calls. Reuses the existing `kv-compression` module's sliding window pruning where applicable

### Phase 4 — End-to-End Gemma 4 Inference
- **Load Gemma 4 31B-it from safetensors** on DGX Spark
- **Run via the existing `llm-generation` loop** (from `llm-api-translation-v1`) using the Session produced by this change
- **Validate** output matches HuggingFace Transformers reference output (semantically — exact bit-match isn't expected with TF32/BF16 tensor cores)
- **Verify** the `/v1/chat/completions` and `/v1/messages` endpoints (from `llm-api-translation-v1`) work end-to-end against the Gemma 4 model

### Out of Scope (deferred)
- **CPU execution path for safetensors models** — LLMs loaded via this path are GPU-only. ONNX models continue to support CPU execution unchanged. CPU inference for 30B+ models is not a viable use case.
- Vision encoder (Gemma 4 multimodal) — text-only inference first
- GGUF format support (use safetensors only)
- Quantized model loading (BF16 native first; INT4/INT8 in a follow-up)
- Other model families (Llama, Qwen, DeepSeek) — Gemma-first, but architecture should generalize
- Training, fine-tuning, LoRA adapters
- Streaming token generation over HTTP (return complete response only initially)
- Multi-GPU or pipeline parallelism

## Capabilities

### New Capabilities
- `safetensors-loader`: Parse `.safetensors` binary files (header JSON + tensor data sections), provide tensor lookup by name with zero-copy slice access where possible
- `huggingface-model-config`: Parse HuggingFace `config.json` to extract transformer architecture metadata (layers, hidden_size, num_attention_heads, num_key_value_heads, vocab_size, max_position_embeddings, rope_theta, sliding_window, etc.)
- `programmatic-graph-builder`: Construct an `ExecutionGraph` from a model architecture config + weight store, without ONNX as an intermediate format
- `gemma-architecture`: Encode Gemma 1/2/3/4 transformer layer structure (sliding window attention, p-RoPE, GQA, SwiGLU, RMSNorm) as a parameterized graph template
- `gpu-resident-execution`: Tensors stay on GPU between operators for the entire forward pass and across generation steps. Required for usable LLM inference (per-op host↔device transfer is too slow). KV cache also lives on GPU.

### Modified Capabilities
- `onnx-cpu-execution`: Add native BF16 tensor support — extend `DataType::BFloat16` handling in CPU operators where used by Gemma (RMSNorm, element-wise ops). Most operators stay f32-only.
- `cuda-container-runtime`: Add BF16 tensor I/O support to GPU dispatch (compute path already supports BF16 via `CUBLAS_COMPUTE_32F_FAST_16BF`, but tensor data flow needs BF16 raw byte handling). Add GPU-resident tensor lifetime management.

### Capabilities Reused from `llm-api-translation-v1`
This change does NOT reimplement the following — they are provided by the parallel `llm-api-translation-v1` change and we depend on them:
- `llm-tokenizer`: BPE tokenizer that loads HuggingFace `tokenizer.json` files
- `llm-generation`: Autoregressive token generation loop with sampling strategies and stop criteria
- `openai-chat-api`: OpenAI Chat Completions endpoint
- `anthropic-messages-api`: Anthropic Messages endpoint

## Impact

- **New module**: `onnx-rt/src/model_loader/` with safetensors parsing (`safetensors.rs`), config parsing (`config.rs`), programmatic graph construction (`graph_builder.rs`), and per-architecture templates (`gemma.rs`)
- **`onnx-rt/src/tensor.rs`**: BF16 raw byte conversion helpers (`bf16_to_f32`, `f32_to_bf16`)
- **`onnx-rt/src/cuda/dispatch.rs`**: BF16 input/output tensor support (currently only f32 I/O even when compute is BF16)
- **`onnx-rt/src/cuda/`**: New `gpu_executor.rs` (or refactor `dispatch.rs`) for graph-level GPU residency — tensors stay in `DeviceBuffer`s for the entire forward pass
- **`onnx-rt/src/session.rs`**: Add `kv_cache: Option<Vec<DeviceBuffer>>` field for persistent GPU KV cache; add `SessionKind::Safetensors` variant or feature flag to mark GPU-required sessions
- **`onnx-rt/src/ops/`**: New ops if needed — sliding-window attention mask, p-RoPE variant, gated SwiGLU. Most Gemma operators reuse existing implementations from `microsoft-fused-ops-v1` (RMSNorm, RotaryEmbedding, GroupQueryAttention).
- **`container/src/model_manager.rs`**: Detect HuggingFace model directories (presence of `config.json` + `*.safetensors`) and route to the safetensors loader
- **`container/src/main.rs`**: Wire safetensors-loaded sessions into the same `Session` map that handles HTTP requests; verify GPU is available at boot when safetensors models are present
- **Memory budget**: Gemma 4 31B-it needs ~62 GB VRAM in BF16, well within DGX Spark's 124 GB unified memory
- **Dependencies**: One new dev dependency: `memmap2` (for mmap'ing safetensors files in container mode, gated behind a `safetensors` feature). No new core dependencies — safetensors format is parsed by hand. Stays clean-room.
- **Test fixtures**: Download Gemma 4 31B-it weights (~62 GB) to `tests/fixtures/safetensors-models/` (gitignored)
- **Documentation**: New `docs/safetensors-loader.md` explaining the model loading pipeline and how to add new architectures
- **Coordination with `llm-api-translation-v1`**: This change provides the model loading and GPU execution that `llm-api-translation-v1` needs to actually run a model. The integration points are: `Session::run()` accepts token IDs and returns logits (same shape contract as ONNX models), and the container loads safetensors model directories alongside ONNX files.
- **Follow-up changes enabled**: This unlocks Llama 3, Qwen, DeepSeek, Mistral as future model families with minimal additional work — just new architecture templates against the same loader infrastructure
