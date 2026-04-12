## 1. Safetensors Parser

- [ ] 1.1 Add `memmap2` as a dev-dependency of `onnx-rt` (or `container`) gated behind a `safetensors` feature flag
- [ ] 1.2 Create `onnx-rt/src/model_loader/mod.rs` module behind `#[cfg(feature = "safetensors")]`
- [ ] 1.3 Create `onnx-rt/src/model_loader/safetensors.rs` with `SafetensorsFile` struct (mmap + parsed header)
- [ ] 1.4 Implement header parsing: read 8-byte LE u64 length, parse JSON header into `BTreeMap<String, TensorEntry>`
- [ ] 1.5 Implement `TensorEntry { dtype, shape, data_offset_start, data_offset_end }` and dtype string → `DataType` mapping (`BF16`, `F16`, `F32`, `I8`, `I32`, `I64`)
- [ ] 1.6 Implement `SafetensorsFile::tensor_view(name)` returning a borrowed slice into the mmap region
- [ ] 1.7 Create `onnx-rt/src/model_loader/sharded.rs` with `MultiShardSafetensors` — parses `model.safetensors.index.json` and opens all shards eagerly
- [ ] 1.8 Unit tests: parse a minimal safetensors file, lookup tensor by name, verify byte slice contents
- [ ] 1.9 Unit test: sharded model loading with a 2-shard synthetic fixture

## 2. HuggingFace Config Parser

- [ ] 2.1 Create `onnx-rt/src/model_loader/config.rs` with `ModelArchitecture` enum (`Gemma3`, `Gemma4`, `Llama`, `Qwen`, `Unknown`)
- [ ] 2.2 Implement `ModelArchitecture::detect(config_json: &str)` based on the `architectures` array in `config.json`
- [ ] 2.3 Define `GemmaConfig` struct with: `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings`, `rope_theta`, `sliding_window`, `sliding_window_pattern`, `rms_norm_eps`, `bos_token_id`, `eos_token_id`
- [ ] 2.4 Implement `GemmaConfig::from_json(json: &str)` using the existing `json.rs` parser in `container` (or a minimal parser in `onnx-rt`)
- [ ] 2.5 Config validation: reject on missing required fields, out-of-range values, or inconsistent combinations (e.g., `hidden_size % num_attention_heads != 0`)
- [ ] 2.6 Unit tests with real Gemma 4 31B-it `config.json` fixture

## 3. BF16 Tensor Support

- [ ] 3.1 Add `bf16_to_f32(bytes: &[u8]) -> Vec<f32>` helper in `onnx-rt/src/tensor.rs` (zero-extend mantissa)
- [ ] 3.2 Add `f32_to_bf16(values: &[f32]) -> Vec<u8>` helper (round-to-nearest-even truncation)
- [ ] 3.3 Verify `DataType::BFloat16::element_size() == 2` and `byte_size()` math is correct for BF16 tensors
- [ ] 3.4 Add BF16 path to `RMSNormalization` CPU operator (convert on read, compute in f32, convert on write)
- [ ] 3.5 Add BF16 path to `Add`, `Mul` CPU operators (element-wise convert-compute-convert)
- [ ] 3.6 Add BF16 path to `SiLU` and element-wise activation CPU operators
- [ ] 3.7 Reject BF16 tensors with a clear error in operators that don't support it (no silent coercion)
- [ ] 3.8 Unit tests: BF16 round-trip, RMSNorm with BF16 input, Add/Mul with BF16 inputs

## 4. BF16 GPU Dispatch

- [ ] 4.1 Update `cuda::dispatch::gpu_gemm()` to detect BF16 input tensors and pass `CUDA_R_16BF` to `cublasGemmEx`
- [ ] 4.2 Produce BF16 output tensors when inputs are BF16 (not silently promote to f32)
- [ ] 4.3 Update `gpu_conv2d()` to handle BF16 input (set cuDNN tensor descriptor to `CUDNN_DATA_BFLOAT16`)
- [ ] 4.4 Update `try_cuda_dispatch` in executor to route BF16 tensors through the GPU path
- [ ] 4.5 Unit test: BF16 GEMM numerical correctness on GB10 hardware
- [ ] 4.6 Unit test: BF16 Conv numerical correctness on GB10 hardware

## 5. GPU-Resident Executor

- [ ] 5.1 Create `onnx-rt/src/cuda/gpu_executor.rs` with `DeviceTensor { buffer: DeviceBuffer, shape, dtype }` struct
- [ ] 5.2 Implement `execute_graph_gpu(graph, input_device_tensors, runtime) -> Result<Vec<DeviceTensor>>` — runs the full forward pass entirely on GPU
- [ ] 5.3 Operator dispatch in `execute_graph_gpu` returns `DeviceTensor` outputs that feed into the next operator without host transfer
- [ ] 5.4 Add `DeviceTensor::to_host() -> Tensor` and `Tensor::to_device(&CudaRuntime) -> DeviceTensor` for boundary conversions
- [ ] 5.5 Fail fast when a GPU-resident graph encounters an operator without a GPU implementation (no silent CPU fallback inside the forward pass)
- [ ] 5.6 Unit test: simple graph (MatMul → Add → RMSNorm) runs end-to-end in `execute_graph_gpu` with zero host-side copies

## 6. Programmatic Graph Builder

- [ ] 6.1 Create `onnx-rt/src/model_loader/graph_builder.rs` with `GraphBuilder` struct wrapping an `ExecutionGraph` under construction
- [ ] 6.2 Implement automatic tensor name allocation (`tensor_0`, `tensor_1`, ...)
- [ ] 6.3 Implement `GraphBuilder::add_initializer(name, tensor)` for binding weight tensors
- [ ] 6.4 Implement operator helpers: `matmul`, `add`, `mul`, `rms_norm`, `rotary_embedding`, `attention`, `swiglu`, `embedding_lookup`
- [ ] 6.5 Implement `GraphBuilder::build() -> Result<ExecutionGraph>` with DAG validation and missing-weight detection
- [ ] 6.6 Add a GPU-resident variant: `GraphBuilder::load_weights_to_gpu(runtime, safetensors)` that transfers each initializer directly into `DeviceBuffer`s via `cudaMemcpy` from the mmap region
- [ ] 6.7 Unit test: build a 2-layer synthetic graph, verify node order and initializer bindings

## 7. Gemma Architecture Template

- [ ] 7.1 Create `onnx-rt/src/model_loader/gemma.rs` with `build_gemma_graph(config, safetensors) -> ExecutionGraph`
- [ ] 7.2 Implement `embedding_lookup` for `model.embed_tokens.weight`
- [ ] 7.3 Implement single-layer function: RMSNorm → Q/K/V projections → RoPE → attention → output projection → residual → post-attn RMSNorm → SwiGLU MLP → residual
- [ ] 7.4 Handle sliding-window vs global attention pattern (interleave based on `sliding_window_pattern`, final layer always global)
- [ ] 7.5 Handle GQA via `num_key_value_heads < num_attention_heads` — size K/V projections correctly
- [ ] 7.6 Apply p-RoPE attribute to RotaryEmbedding nodes (`p_rope: true` for Gemma 4)
- [ ] 7.7 Apply Gemma RMSNorm convention (`1 + weight`) — either via operator attribute or by pre-adding 1 at load time
- [ ] 7.8 Build final layer: RMSNorm → lm_head MatMul → vocab logits
- [ ] 7.9 Loop over `num_hidden_layers` to build the full model graph
- [ ] 7.10 Unit test: build Gemma 4 270M graph from fixture config (smaller sanity check before 31B)
- [ ] 7.11 Integration test: build Gemma 4 31B-it graph from real downloaded weights, verify node count matches expected architecture

## 8. KV Cache Management

- [ ] 8.1 Create `onnx-rt/src/cuda/kv_cache.rs` with `GpuKvCache` struct owning `Vec<(DeviceBuffer, DeviceBuffer)>` for (K, V) per layer
- [ ] 8.2 Implement `GpuKvCache::allocate(config, max_seq_len, runtime)` — pre-allocate per-layer buffers sized for max context
- [ ] 8.3 Implement `GpuKvCache::append(layer_idx, new_k, new_v, position)` — write new token's K/V into the cache at the current position
- [ ] 8.4 Implement `GpuKvCache::view(layer_idx, up_to_position)` — return a view into cached K/V up to current position for attention
- [ ] 8.5 Implement `GpuKvCache::reset()` — clear position counter, reuse allocated buffers
- [ ] 8.6 Implement sliding-window pruning for Gemma local attention layers (keep last 1024 tokens)
- [ ] 8.7 Add `kv_cache: Option<Arc<Mutex<GpuKvCache>>>` field to `Session` (interior mutability so `&Session` can update cache)
- [ ] 8.8 Unit test: KV cache append/view/reset lifecycle on GPU

## 9. Session Integration

- [ ] 9.1 Add `SessionKind::Safetensors` variant (or `is_gpu_resident: bool` field) to `Session`
- [ ] 9.2 Implement `Session::from_safetensors(dir, cuda_runtime)` — loads config, safetensors shards, builds Gemma graph, pre-loads weights to GPU, allocates KV cache
- [ ] 9.3 Update `Session::run()` for safetensors sessions: accept token ID tensor, use `execute_graph_gpu` with internal KV cache, return logits tensor
- [ ] 9.4 Implement `Session::reset_kv_cache()` public method for starting new generation sessions
- [ ] 9.5 Fail `Session::from_safetensors()` with a clear error if `cuda_runtime.is_none()`
- [ ] 9.6 Unit test: create a tiny safetensors fixture with 2 layers, build a Session, run a single forward pass, verify logits shape

## 10. Coordination with `llm-api-translation-v1`

- [ ] 10.1 Verify the `Session::run()` interface signature is compatible with what `llm-generation` expects (token IDs in, logits out)
- [ ] 10.2 For safetensors sessions, the Session manages KV cache internally. Update (or propose update to) `llm-generation` task 5.7 so the generation loop does NOT thread `past_k`/`past_v` through input tensors for GPU-resident sessions — instead, the Session's internal cache persists across `Session::run()` calls
- [ ] 10.3 Add a capability flag to `Session` indicating whether the caller must thread KV cache externally (ONNX) or whether the Session manages it internally (safetensors). `llm-generation` branches on this flag
- [ ] 10.4 Confirm `llm-api-translation-v1` Gemma prompt template (`<start_of_turn>` format) works with Gemma 4 (verify special tokens match the tokenizer.json for Gemma 4)
- [ ] 10.5 Integration test: run `llm-generation` loop end-to-end against a safetensors Gemma model with the existing `llm-tokenizer` — confirm the contract works
- [ ] 10.6 Document the integration contract in `docs/safetensors-integration.md`

## 11. Container Integration

- [ ] 11.1 Update `container/src/model_manager.rs` to detect safetensors model directories (presence of `config.json` + `*.safetensors` or `model.safetensors.index.json`)
- [ ] 11.2 Route detected safetensors directories to `Session::from_safetensors()` instead of the ONNX loader
- [ ] 11.3 Update `container/src/main.rs` `load_sessions()` to handle both ONNX and safetensors sessions through the same Session map
- [ ] 11.4 Add `container/Cargo.toml` `safetensors` feature that enables `smallaios-onnx-rt/safetensors`
- [ ] 11.5 Fail fast at boot if safetensors models are present but no `CudaRuntime` is available
- [ ] 11.6 Add download script helper for safetensors models (uses `huggingface-cli download` or curl with HF token)
- [ ] 11.7 Integration test: container boots with a small safetensors fixture (Gemma 2 270M or equivalent), serves `/v1/chat/completions`

## 12. End-to-End Gemma 4 31B Validation

- [ ] 12.1 Download Gemma 4 31B-it safetensors weights to `tests/fixtures/safetensors-models/gemma-4-31b-it/` (gitignored, ~62 GB)
- [ ] 12.2 Load Gemma 4 31B-it in the container with GPU backend enabled
- [ ] 12.3 Verify all weights load into GPU VRAM within expected memory budget
- [ ] 12.4 Send a chat completion request via `/v1/chat/completions` (using the `llm-api-translation-v1` handler) with a simple prompt
- [ ] 12.5 Verify the response contains coherent generated text
- [ ] 12.6 Compare output against HuggingFace `transformers.AutoModelForCausalLM` reference for the same prompt (semantic, not bit-exact)
- [ ] 12.7 Benchmark tokens/sec on DGX Spark with TF32 + BF16 compute modes
- [ ] 12.8 Send an Anthropic-format request via `/v1/messages` with the same prompt, verify equivalent output
- [ ] 12.9 Document results in `docs/benchmarks/gemma-4-31b-dgx-spark.md`

## 13. Validation

- [ ] 13.1 `cargo fmt --all -- --check` clean
- [ ] 13.2 `cargo clippy --workspace` clean with no new warnings
- [ ] 13.3 `cargo test -p smallaios-onnx-rt` passes (default and `cuda` feature)
- [ ] 13.4 `cargo test -p smallaios-container` passes
- [ ] 13.5 At least 30 new unit tests across safetensors parser, config parser, graph builder, Gemma template, BF16 ops, GPU executor, KV cache
- [ ] 13.6 `cargo test -p smallaios-onnx-rt --features cuda --test test_cuda` passes on DGX Spark (existing 24 GPU tests still pass)
- [ ] 13.7 Verify ONNX model path unchanged (regression test with existing model fixtures)
- [ ] 13.8 Verify `llm-api-translation-v1` tests still pass after Session interface coordination changes
