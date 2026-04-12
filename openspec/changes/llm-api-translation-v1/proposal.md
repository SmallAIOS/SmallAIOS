## Why

SmallAIOS can now load and build execution graphs for every major LLM family (Llama, Gemma, DeepSeek, GPT-2). The container exposes a raw `/v1/inference` endpoint that accepts tensor inputs and returns tensor outputs — functional but unusable by any existing LLM tooling. Every LLM client library, agent framework, and chat UI in the ecosystem speaks one of two APIs: **OpenAI Chat Completions** (`POST /v1/chat/completions`) or **Anthropic Messages** (`POST /v1/messages`). Without compatibility with at least one of these, SmallAIOS cannot be used as a drop-in local inference server.

The two APIs are structurally similar — both are request/response over HTTP with JSON bodies, both support streaming via Server-Sent Events, and both model conversations as sequences of role-tagged messages. The differences are in schema naming, system-prompt handling, stop-reason vocabulary, streaming event format, and token-counting conventions. A thin translation layer can support both by mapping to a shared internal representation and routing to the same ONNX execution pipeline.

## What Changes

- **Add `POST /v1/chat/completions`** — OpenAI-compatible chat completions endpoint. Accepts the standard OpenAI request schema (`model`, `messages[]`, `temperature`, `max_tokens`, `stream`, `stop`, `top_p`, `frequency_penalty`, `presence_penalty`). Returns the standard OpenAI response schema (`choices[]` with `message` and `finish_reason`). Supports both non-streaming and SSE streaming responses.

- **Add `POST /v1/messages`** — Anthropic-compatible messages endpoint. Accepts the Anthropic request schema (`model`, `messages[]`, `system`, `max_tokens`, `temperature`, `top_p`, `stream`, `stop_sequences`). Returns the Anthropic response schema (`content[]` with `type: "text"`, `stop_reason`). Supports both non-streaming and SSE streaming.

- **Add a shared `ChatRequest` internal representation** — both API formats translate to this common struct before hitting the inference pipeline. This avoids duplicating the tokenization → generation → detokenization logic.

- **Add a tokenizer abstraction** — LLM chat APIs require tokenization (converting text to token IDs) and detokenization (converting output token IDs back to text). SmallAIOS needs a minimal tokenizer that can load HuggingFace `tokenizer.json` files (the BPE tokenizer format used by Llama, Gemma, GPT-2, etc.). This is a new module — the ONNX runtime works with tensors, not text.

- **Add a generation loop** — the chat completions endpoint needs to run autoregressive token generation: repeatedly call the ONNX model with the current token sequence, sample the next token (temperature, top-k, top-p), append it, and repeat until a stop condition. This uses the Phase 2 `Loop` operator if the model exports it, or an external host loop otherwise.

- **Keep the existing `/v1/inference`** endpoint unchanged for raw tensor I/O.

## Capabilities

### New Capabilities
- `openai-chat-api`: OpenAI-compatible Chat Completions endpoint with streaming, stop sequences, and sampling parameters.
- `anthropic-messages-api`: Anthropic-compatible Messages endpoint with streaming and Anthropic-specific schema.
- `llm-tokenizer`: Minimal BPE tokenizer that loads HuggingFace `tokenizer.json` files for text↔token conversion.
- `llm-generation`: Autoregressive token generation loop with temperature, top-k, top-p sampling and stop-sequence detection.

### Modified Capabilities
- `onnx-cpu-execution`: The container's HTTP server gains two new routes. No changes to the inference engine itself — the new endpoints are a layer above it.

## Impact

**Affected code:**
- `container/src/handlers.rs` — new handler functions for the two endpoints
- `container/src/` — new modules: `chat_api.rs` (translation layer), `tokenizer.rs`, `generation.rs`, `sampling.rs`
- `container/src/http.rs` — SSE streaming support (chunked transfer encoding)

**Affected APIs:**
- Two new HTTP endpoints added to the container's API surface
- Existing endpoints (`/v1/inference`, `/v1/models`, `/healthz`, `/readyz`) unchanged

**Dependencies:**
- Tokenizer: needs a BPE decoder. Either implement from scratch (~500 LOC for the HuggingFace tokenizer.json format) or find a `no_std`-compatible tokenizer crate. Prefer from-scratch to maintain the zero-external-deps stance.

**Risks:**
- Tokenizer correctness: BPE has edge cases (byte-fallback, special tokens, unicode normalization). A from-scratch implementation needs careful testing against HuggingFace's reference output.
- Streaming latency: SSE requires chunked transfer encoding. The existing HTTP server may need extension to support streaming responses.
- Model-specific prompt templates: different LLMs expect different prompt formatting (Llama's `<|begin_of_text|>`, Gemma's `<start_of_turn>`, etc.). The translation layer needs a configurable prompt-template system.
