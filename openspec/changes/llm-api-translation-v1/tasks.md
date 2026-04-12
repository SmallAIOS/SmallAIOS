## 1. Shared Internal Representation

- [ ] 1.1 Create `container/src/chat_api.rs` with `ChatRequest`, `ChatResponse`, `ChatMessage`, `ChatRole`, `FinishReason`, `TokenUsage` structs
- [ ] 1.2 Implement `ChatRequest::from_openai_json(body: &str)` — parse OpenAI Chat Completions request
- [ ] 1.3 Implement `ChatRequest::from_anthropic_json(body: &str)` — parse Anthropic Messages request
- [ ] 1.4 Implement `ChatResponse::to_openai_json(&self)` — serialize to OpenAI response format
- [ ] 1.5 Implement `ChatResponse::to_anthropic_json(&self)` — serialize to Anthropic response format
- [ ] 1.6 Unit tests for both request parsers and response serializers with reference JSON fixtures

## 2. BPE Tokenizer

- [ ] 2.1 Create `container/src/tokenizer.rs` with `Tokenizer` struct
- [ ] 2.2 Implement `Tokenizer::from_json(data: &str)` — parse HuggingFace `tokenizer.json` format (vocab, merges, added_tokens)
- [ ] 2.3 Implement `Tokenizer::encode(text: &str) -> Vec<u32>` — BPE encoding with byte-fallback
- [ ] 2.4 Implement `Tokenizer::decode(ids: &[u32]) -> String` — token ID to UTF-8 text
- [ ] 2.5 Handle special tokens (BOS, EOS, PAD, model-specific) per `added_tokens` config
- [ ] 2.6 Unit tests: encode/decode round-trip, special tokens, empty input, unicode, byte-fallback
- [ ] 2.7 Reference comparison test: encode a known string with the Llama tokenizer and compare token IDs against a hand-verified expected output

## 3. Prompt Templates

- [ ] 3.1 Create `container/src/prompt_templates.rs` with `PromptTemplate` trait and built-in registry
- [ ] 3.2 Implement Llama 3 template (`<|begin_of_text|>` format)
- [ ] 3.3 Implement Gemma template (`<start_of_turn>` format)
- [ ] 3.4 Implement DeepSeek/Qwen ChatML template (`<|im_start|>` format)
- [ ] 3.5 Implement generic ChatML fallback template
- [ ] 3.6 Implement GPT-2 plain concatenation template
- [ ] 3.7 Implement `PromptTemplate::from_json(path: &str)` for custom templates alongside model files
- [ ] 3.8 Auto-detect template from model name or architecture metadata
- [ ] 3.9 Unit tests for each template with sample conversations

## 4. Sampling

- [ ] 4.1 Create `container/src/sampling.rs` with `Sampler` struct
- [ ] 4.2 Implement temperature scaling (logits / temperature)
- [ ] 4.3 Implement top-k filtering (keep only top-k logits)
- [ ] 4.4 Implement top-p (nucleus) filtering (keep smallest set exceeding cumulative probability p)
- [ ] 4.5 Implement greedy decoding (temperature = 0 → argmax)
- [ ] 4.6 Implement deterministic sampling with seed for reproducibility
- [ ] 4.7 Unit tests: greedy selects max, top-k filters correctly, top-p threshold, temperature scaling

## 5. Generation Loop

- [ ] 5.1 Create `container/src/generation.rs` with `GenerationLoop` struct
- [ ] 5.2 Implement the token-by-token generation loop: encode → run model → sample → check stop → append
- [ ] 5.3 Implement EOS token detection (stop on model's EOS token ID)
- [ ] 5.4 Implement stop-sequence detection (string-level comparison on decoded output)
- [ ] 5.5 Implement max_tokens enforcement
- [ ] 5.6 Integrate with `Session::run()` for ONNX model execution
- [ ] 5.7 Thread KV cache between iterations (pass present_k/present_v from output to next input)
- [ ] 5.8 Unit test: generation with a mock model that returns predictable logits
- [ ] 5.9 Integration test: generation with a real DistilGPT-2 model (uses test fixtures)

## 6. SSE Streaming

- [ ] 6.1 Create `container/src/sse.rs` with SSE writer supporting chunked transfer encoding
- [ ] 6.2 Implement OpenAI-format SSE emission (`data: {...}\n\n` + `data: [DONE]\n\n`)
- [ ] 6.3 Implement Anthropic-format SSE emission (`event: content_block_delta\ndata: {...}\n\n`)
- [ ] 6.4 Extend the HTTP server to support `Content-Type: text/event-stream` responses
- [ ] 6.5 Unit tests for SSE formatting

## 7. HTTP Endpoint Handlers

- [ ] 7.1 Add `handle_chat_completions(req, manager, tokenizer_registry)` in `handlers.rs`
- [ ] 7.2 Add `handle_messages(req, manager, tokenizer_registry)` in `handlers.rs`
- [ ] 7.3 Route `/v1/chat/completions` and `/v1/messages` in the HTTP server's dispatch
- [ ] 7.4 Integration test: POST to `/v1/chat/completions` with a mock model, verify OpenAI response schema
- [ ] 7.5 Integration test: POST to `/v1/messages` with a mock model, verify Anthropic response schema
- [ ] 7.6 Integration test: streaming response with SSE verification

## 8. Model Memory Budget

- [ ] 8.1 Add `MemoryBudget` struct to `container/src/config.rs`
- [ ] 8.2 Implement platform-specific available RAM detection (macOS `hw.memsize`, Linux `/proc/meminfo`, bare-metal page count)
- [ ] 8.3 Add pre-load size check in `ModelManager::load_model()` — reject with `ModelTooLarge` if model exceeds budget
- [ ] 8.4 Add KV cache size estimate warning (advisory, not hard-reject)
- [ ] 8.5 Configure budget via environment variable `SMALLAIOS_MAX_MODEL_MB` (default: 80% of RAM)
- [ ] 8.6 Unit tests: budget check rejects oversized model, allows within-budget model, warns on KV estimate

## 9. Validation

- [ ] 9.1 `just fmt` clean
- [ ] 9.2 `just clippy --all-targets` clean
- [ ] 9.3 `just test` all passing
- [ ] 9.4 At least 40 new unit tests across tokenizer, sampling, generation, SSE, and API handlers
- [ ] 9.5 Verify `/v1/inference` endpoint unchanged (regression test)
