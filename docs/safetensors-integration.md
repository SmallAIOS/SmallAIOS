# Safetensors Session Integration Contract

This document describes how the safetensors-backed `Session` produced by
`openspec/changes/safetensors-model-loader-v1` integrates with the
`llm-api-translation-v1` change (tokenizer, generation loop, and
OpenAI/Anthropic HTTP translators). It exists so that when
`llm-api-translation-v1` is implemented, the generation loop can be
wired against the real `Session` API without archaeology.

## Session API for LLM clients

`Session` exposes two execution paths, discriminated by
`Session::kind() -> SessionKind`:

| `SessionKind` | Constructor               | Execution path                                | KV cache ownership |
|---------------|---------------------------|-----------------------------------------------|---------------------|
| `Onnx`        | `Session::initialize(...)`| `executor::execute_graph` (CPU or GPU ops)    | **External** — caller threads `past_k`/`past_v` through inputs |
| `Safetensors` | `Session::from_safetensors(dir, cuda_rt)` | `cuda::execute_graph_gpu_with_weights` on device-resident weights | **Internal** — persistent `GpuKvCache` held by the `Session` |

Public accessors the `llm-generation` loop should rely on:

- `Session::kind() -> SessionKind`
- `Session::manages_kv_cache_internally() -> bool` — convenience flag
- `Session::run(&[InferenceInput]) -> Result<Vec<InferenceOutput>, SessionError>`
- `Session::reset_kv_cache() -> Result<(), SessionError>` (no-op for
  `Onnx` sessions, resets position counter for `Safetensors` sessions)
- `Session::input_names()` / `Session::output_names()`

### Input / output tensor contract

Both kinds accept token IDs as a host-side `Tensor` and return logits as
a host-side `Tensor`. The safetensors path transparently moves the
input to device, executes on GPU, and copies the output logits back to
host memory.

Per the synthetic Gemma graph built by `build_gemma_graph`:

- **Input**: `"input_ids"` — `Tensor { dtype: Int64, shape: [batch, seq_len] }`
- **Output**: `"logits"` (first entry of `output_names`) —
  `Tensor { dtype: BFloat16, shape: [batch, seq_len, vocab] }`

The ONNX path is model-dependent: its input and output names come from
the protobuf graph itself and `llm-generation` must inspect
`Session::input_names()` to determine which tensors to populate.

## Two KV cache management modes

### Mode A — ONNX Session (external KV threading)

The generation loop explicitly threads `past_key_values.{layer}.{k,v}`
through `InferenceInput`s and receives updated `present_key_values...`
through `InferenceOutput`s. This is the mainstream ONNX LLM pattern and
is what `llm-api-translation-v1` design §D5 currently describes.

```text
for token in prompt + sampled:
    inputs  = [input_ids_tensor, past_k_0, past_v_0, ..., past_k_L, past_v_L]
    outputs = session.run(&inputs)?
    logits  = outputs[0]
    past_k_{i}, past_v_{i} = outputs[i*2+1..]
```

### Mode B — Safetensors Session (internal KV management)

The `Session` allocates a `GpuKvCache` at construction time (sized for
`min(config.max_position_embeddings, 2048)`) and mutates it across
successive `run()` calls via interior mutability on
`Arc<Mutex<GpuKvCache>>`. The generation loop MUST NOT attempt to
construct or thread KV tensors — they are not part of the graph's
input/output signature.

Recommended calling pattern for `llm-generation` when
`session.manages_kv_cache_internally()` is `true`:

```text
session.reset_kv_cache()?;          // start of a new completion
for token in prompt:                // prefill
    outputs = session.run(&[token_id_tensor_of(token)])?;
for _ in 0..max_new_tokens:         // decode
    outputs = session.run(&[token_id_tensor_of(last_sampled)])?;
    logits  = &outputs[0].tensor;
    next    = sample(logits);
    if next == eos { break; }
```

Prefill may also batch the whole prompt in a single `run()` call with
`shape = [1, prompt_len]`; the safetensors executor handles that the
same way.

The generation loop should branch based on
`Session::manages_kv_cache_internally()`:

```rust
let outputs = if session.manages_kv_cache_internally() {
    // Mode B: single input, Session owns KV state.
    session.run(&[input_ids_input])?
} else {
    // Mode A: thread past_k/past_v through inputs.
    session.run(&build_inputs_with_kv(&input_ids, &past_kv))?
};
```

## Gemma 4 prompt template

`llm-api-translation-v1` design §D4 lists the Gemma 3 template:

```text
<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n
```

This same template is used for Gemma 4 (the HuggingFace `chat_template`
field in `tokenizer_config.json` for `google/gemma-4-*-it` checkpoints
is identical to Gemma 2/3: turn-based with `<start_of_turn>` /
`<end_of_turn>` wrappers and a `model` turn marker). No per-family
override is needed for Gemma 4 on top of the Gemma 3 template.

A few caveats for `llm-generation` to keep in mind:

1. **System prompts.** Gemma does not have a dedicated `system` role in
   its chat template. When the OpenAI request contains a `system`
   message, `llm-api-translation-v1` should either prepend it to the
   first user turn or drop it with a warning. This is spec'd in
   `llm-api-translation-v1/specs/llm-tokenizer/spec.md` scenario
   "Gemma prompt format".
2. **BOS token.** Gemma tokenizers prepend `<bos>` automatically.
   Callers should pass `add_special_tokens = true` on the initial
   prefill and `false` on each per-token decode step so `<bos>` is not
   re-emitted.
3. **Multi-turn continuation.** For a multi-turn conversation, the full
   history must be re-tokenized each request unless the generation loop
   can reuse a persisted `Session` whose `GpuKvCache` still holds the
   previous turn. For the initial `llm-api-translation-v1` milestone we
   recommend the simple path: `reset_kv_cache()` at the start of each
   request, re-prefill the full history.

If `llm-api-translation-v1` ships with a different template, update
both that change and this document in lockstep. This file is the
single source of truth for the safetensors/LLM interface contract.

## Error contract

`Session::run()` and `Session::from_safetensors()` both return
`SessionError`. The `llm-generation` loop should surface these as
`InferenceError::ModelError(String)` to the HTTP layer.

For safetensors sessions specifically:

- `SessionError::InvalidModel(String)` — bad `config.json`, missing
  weight file, or unsupported architecture. Raised only from
  `from_safetensors`, never from `run`.
- `SessionError::ExecutionFailed(String)` — GPU dispatch error, missing
  GPU operator, or CUDA error during `run()`. The string message
  contains enough detail to pinpoint the failing operator (the Section
  5 executor includes `"no GPU implementation for {op}"` for
  unsupported ops).

## References

- `onnx-rt/src/session.rs` — `Session`, `SessionKind`,
  `Session::from_safetensors`, `Session::run`, `Session::reset_kv_cache`,
  `Session::kind`, `Session::manages_kv_cache_internally`
- `onnx-rt/src/cuda/gpu_executor.rs` — GPU-resident forward pass
- `onnx-rt/src/cuda/kv_cache.rs` — `GpuKvCache`
- `openspec/changes/safetensors-model-loader-v1/` — this change
- `openspec/changes/llm-api-translation-v1/` — sister change (unimplemented as of 2026-04-12)
