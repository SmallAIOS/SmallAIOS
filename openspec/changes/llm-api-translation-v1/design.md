## Context

SmallAIOS's container HTTP server currently exposes `/v1/inference`
for raw tensor I/O and `/v1/models` for model listing. The LLM
ecosystem has converged on two API standards: OpenAI Chat Completions
and Anthropic Messages. Both are structurally similar — JSON-over-HTTP
with role-tagged message arrays — but differ in schema naming,
streaming format, and response structure. Supporting both from a
single internal representation is straightforward.

## Decisions

### D1: Shared internal `ChatRequest` / `ChatResponse` representation

**Decision.** Both API formats translate to a common internal struct
before touching the inference pipeline:

```rust
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,      // default 1.0
    pub top_p: f32,            // default 1.0
    pub top_k: Option<usize>,  // None = disabled
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

pub struct ChatMessage {
    pub role: ChatRole,       // System, User, Assistant
    pub content: String,
}

pub struct ChatResponse {
    pub content: String,
    pub finish_reason: FinishReason,  // Stop, MaxTokens, StopSequence
    pub usage: TokenUsage,
}

pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
```

**Rationale.** A single code path handles tokenization, generation,
and detokenization regardless of which API format the client used.
The translation to/from wire format is a thin ~50 LOC adapter per API.

### D2: API format mapping

| Feature | OpenAI | Anthropic | Internal |
|---|---|---|---|
| Endpoint | `/v1/chat/completions` | `/v1/messages` | — |
| System prompt | `messages[0].role="system"` | top-level `system` field | `system_prompt` |
| User message | `role: "user"` | `role: "user"` | `ChatRole::User` |
| Assistant message | `role: "assistant"` | `role: "assistant"` | `ChatRole::Assistant` |
| Max tokens | `max_tokens` (optional) | `max_tokens` (required) | `max_tokens` |
| Temperature | `temperature` | `temperature` | `temperature` |
| Top-p | `top_p` | `top_p` | `top_p` |
| Stop sequences | `stop: [...]` | `stop_sequences: [...]` | `stop_sequences` |
| Streaming | `stream: true` + SSE `data: {...}` | `stream: true` + SSE `event: ...` | `stream` |
| Response content | `choices[0].message.content` | `content[0].text` | `content` |
| Finish reason | `"stop"` / `"length"` | `"end_turn"` / `"max_tokens"` | `FinishReason` enum |
| Model ID in response | `model: "..."` | `model: "..."` | from request |
| Usage | `usage: {prompt_tokens, completion_tokens, total_tokens}` | `usage: {input_tokens, output_tokens}` | `TokenUsage` |

**Decision.** The translation adapters handle these mappings. The
internal pipeline never sees API-specific field names.

### D3: BPE tokenizer from scratch

**Decision.** Implement a minimal BPE tokenizer (~500 LOC) in
`container/src/tokenizer.rs` that loads HuggingFace `tokenizer.json`
files. The format is well-documented: a JSON object containing
`model.vocab` (token → ID mapping), `model.merges` (BPE merge rules),
`added_tokens` (special tokens like `<|begin_of_text|>`), and
`decoder` (byte-level fallback config).

**What we implement:**
- Load vocab + merges from `tokenizer.json`
- Encode: text → byte pairs → iterative merge → token IDs
- Decode: token IDs → bytes → UTF-8 text
- Special token handling: `<bos>`, `<eos>`, `<pad>`, model-specific
  tokens like `<|im_start|>` (Qwen/DeepSeek), `<start_of_turn>`
  (Gemma), `<|begin_of_text|>` (Llama)

**What we do NOT implement:**
- Pre-tokenization regex (HuggingFace's `pre_tokenizer` — we use a
  simpler whitespace split + byte-fallback that handles 95% of cases)
- Sentencepiece compatibility (only BPE, not Unigram)
- Training / vocab extension

**Rationale.** External tokenizer crates are `std`-only or pull in
50+ dependencies. The BPE algorithm is ~200 lines of core logic; the
rest is JSON parsing and special-token handling. SmallAIOS already has
a JSON parser in `container/src/json.rs`.

### D4: Prompt template system

**Decision.** Each model has a prompt template that converts a
`Vec<ChatMessage>` into the model's expected token sequence. Templates
are loaded from a `prompt_template.json` file alongside the model, or
from a built-in registry keyed by model architecture.

Built-in templates for launch:
- **Llama 3**: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n`
- **Gemma 3**: `<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n`
- **DeepSeek/Qwen**: `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`
- **GPT-2**: plain concatenation (no chat template — legacy)
- **ChatML**: generic fallback used by many fine-tunes

**Rationale.** Prompt templates are the #1 source of "it works but
generates garbage" bugs in local LLM servers. Making them explicit
and model-specific avoids that trap.

### D5: Generation loop

**Decision.** The generation loop is implemented in
`container/src/generation.rs` as a host-side loop:

```
1. Apply prompt template → token IDs
2. Create initial KV cache (empty)
3. For each step until max_tokens or stop:
   a. Run one forward pass: Session::run(token_ids, kv_cache)
   b. Extract logits from output
   c. Apply temperature + top-k + top-p sampling
   d. Check for stop token or stop sequence
   e. If streaming, emit SSE chunk
   f. Append sampled token, update KV cache
4. Detokenize output tokens → text
5. Return ChatResponse
```

This is a host-side loop (not using the ONNX `Loop` operator) because:
- The tokenizer and sampling logic are Rust, not ONNX ops
- The SSE streaming emission happens between iterations
- The stop-sequence check requires string-level comparison

The Phase 2 sub-graph executor's `Loop` would be used only if the
entire generation (including sampling) were expressed inside the
ONNX graph, which is rare in practice.

### D6: SSE streaming

**Decision.** Extend the container's HTTP server to support chunked
transfer encoding with `Content-Type: text/event-stream`.

OpenAI SSE format:
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n
data: [DONE]\n\n
```

Anthropic SSE format:
```
event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n
event: message_stop\ndata: {"type":"message_stop"}\n\n
```

Both formats are emitted from the same generation loop; the
translation adapter serializes each token into the appropriate
SSE event shape.

### D7: Model selection

**Decision.** The `model` field in the request maps to a loaded model
via the existing `ModelManager`. If the model name matches a loaded
model's name or file stem, that model is used. If no match, return
404. The `/v1/models` endpoint already lists available models.

### D8: File layout

```
container/src/
├── handlers.rs           (modified: add chat_completions + messages handlers)
├── chat_api.rs           (NEW: ChatRequest/ChatResponse + translation adapters)
├── tokenizer.rs          (NEW: BPE tokenizer loading tokenizer.json)
├── generation.rs         (NEW: autoregressive generation loop)
├── sampling.rs           (NEW: temperature, top-k, top-p sampling)
├── prompt_templates.rs   (NEW: model-specific prompt formatting)
├── sse.rs                (NEW: Server-Sent Events chunked response writer)
└── json.rs               (existing: may need extension for nested parsing)
```

## Alternatives considered

### A1: Support only OpenAI format
Many local inference servers (llama.cpp, vLLM, Ollama) support only
OpenAI format. **Rejected:** the user explicitly wants both, and
Anthropic's format is gaining adoption. The incremental cost of the
second adapter is ~50 LOC given the shared internal representation.

### A2: Use an external tokenizer crate (tokenizers, tiktoken-rs)
**Rejected:** both are `std`-only with large dependency trees.
SmallAIOS's `no_std` constraint requires a from-scratch BPE
implementation. The core algorithm is small; the JSON parsing
already exists in the container.

### A3: Express generation inside the ONNX graph using Loop
**Rejected for the default path:** the tokenizer and sampling
logic are Rust code, not ONNX operators. The host-side loop also
enables SSE streaming between iterations. ONNX `Loop`-based
generation remains available for models that export it, but the
chat API uses the host loop.

### D9: Model memory budget enforcement

**Decision.** Before loading a model, the `ModelManager` SHALL check
the model file size against the platform's available memory and reject
the load if it would exceed a configurable budget. The budget defaults
to 80% of available physical RAM (leaving headroom for the KV cache,
the runtime, and the OS).

```rust
pub struct MemoryBudget {
    pub max_model_bytes: usize,    // 0 = no limit
    pub max_kv_cache_bytes: usize, // 0 = no limit
    pub headroom_fraction: f32,    // default 0.2 (reserve 20%)
}
```

The check is:
1. Query available physical RAM (via `sysinfo` on macOS/Linux, or a
   compile-time constant for bare-metal). For `no_std` bare-metal,
   use the `large-memory` feature flag's page count (1 GiB default,
   64 GiB with the flag).
2. Compute `budget = total_ram * (1.0 - headroom_fraction)`
3. If `model_file_size > budget`, return an error:
   `ModelTooLarge { model_size, budget, available_ram }`
4. Additionally estimate KV cache requirements: for a decoder model
   with `num_layers * 2 * num_kv_heads * head_dim * max_seq_len * 4`
   bytes (f32). If the model + estimated KV cache exceeds budget,
   warn (don't hard-reject, since compressed KV is an option).

**Platform-specific RAM detection:**
- **macOS container:** `sysctl hw.memsize` via libc. Already available
  in `container/` which links libc.
- **Linux container:** `/proc/meminfo` MemAvailable. Fallback to
  `sysconf(_SC_PHYS_PAGES) * page_size`.
- **Bare-metal kernel:** the kernel's physical memory manager reports
  total pages. Exposed via a `platform::available_memory_bytes()` API.

**Rationale.** Without this guard, a user who tries to load Llama-70B
on a 16 GB Mac gets an OOM kill with no diagnostic. The budget check
gives a clear error before any allocation happens. The 80% default
leaves room for the KV cache, the runtime's value maps, and the OS.
The KV cache estimate is advisory (warns, doesn't reject) because
the `kv-cache-quantization-v1` compression can reduce it 6x.

## Open questions

**Q1:** Should we support function/tool calling in the first version?
Both APIs have it (OpenAI `tools[]`, Anthropic `tool_use`). Recommend
deferring — it adds schema complexity and the core value is chat
text generation.

**Q2:** Should the tokenizer support Sentencepiece (Unigram model)?
Some older models (T5, mBART) use it. Recommend deferring — BPE
covers Llama, Gemma, GPT-2, DeepSeek, Mistral, and Phi.

**Q3:** How do we handle multi-modal inputs (images in messages)?
Both APIs support it. Recommend deferring — text-only for v1.
