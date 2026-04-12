## 1. Domain-aware OpKind machinery

- [ ] 1.1 Add `Domain` enum (`StandardOnnx`, `MicrosoftFused`) to
  `onnx-rt/src/operators.rs`
- [ ] 1.2 Extend `OpKind` so each variant records its domain; add
  helper `fn domain(&self) -> Domain`
- [ ] 1.3 Add `OperatorRegistry::lookup_by_domain_and_name(&str, &str)
  -> Option<OpKind>` and route `"com.microsoft"` / `""` / `"ai.onnx"`
- [ ] 1.4 Update the graph builder in `onnx-rt/src/graph.rs` to read
  `NodeProto.domain` and pass it through `lookup_by_domain_and_name`

## 2. Shared `scaled_dot_product_attention` helper

- [ ] 2.1 Write the SDPA helper signature in `ops/microsoft.rs`:
  `fn scaled_dot_product_attention(q, k, v, scale, causal_mask_fn)`
- [ ] 2.2 Implement online-softmax (max-stabilized) SDPA with tiling
  in the `Sq` dimension at 64 rows per tile (see design Q3)
- [ ] 2.3 Add unit test: SDPA on `B=1, heads=2, Sq=3, Sk=4, hd=8`
  against a hand-computed reference
- [ ] 2.4 Add unit test: SDPA with causal mask, verify zero probability
  mass at masked positions
- [ ] 2.5 Add unit test: SDPA broadcasting a single KV head across
  multiple query heads (GQA group-size test)

## 3. Shared `apply_rope_in_place` helper

- [ ] 3.1 Write the helper signature in `ops/microsoft.rs`:
  `fn apply_rope_in_place(tensor, cos_cache, sin_cache, positions, interleaved)`
- [ ] 3.2 Implement non-interleaved rotation (HF / Llama layout)
- [ ] 3.3 Implement interleaved rotation (DeepSeek layout), branched
  on the `interleaved` flag in the hot loop
- [ ] 3.4 Add unit test: non-interleaved rotation against a Python
  reference on a small `(1, 2, 64)` tensor
- [ ] 3.5 Add unit test: interleaved rotation, confirming the result
  differs from non-interleaved on the same input

## 4. `SimplifiedLayerNormalization`

- [ ] 4.1 Add `op_simplified_layer_normalization` in
  `ops/microsoft.rs` as a ~20 LOC wrapper over
  `op_rms_normalization_raw`
- [ ] 4.2 Unit test: matches `op_rms_normalization` output on
  `(2, 4, 8)` input within 1e-6 absolute
- [ ] 4.3 Unit test: optional bias input is added after normalization

## 5. `SkipSimplifiedLayerNormalization`

- [ ] 5.1 Add `op_skip_simplified_layer_normalization` in
  `ops/microsoft.rs` (~50 LOC)
- [ ] 5.2 Unit test: three-input form (`x`, `skip`, `scale`) produces
  correct normalized output and pre-layernorm sum
- [ ] 5.3 Unit test: four-input form with bias
- [ ] 5.4 Unit test: second output (pre-layernorm sum) equals
  `x + skip + bias` exactly

## 6. Standalone `RotaryEmbedding`

- [ ] 6.1 Add `op_rotary_embedding` in `ops/microsoft.rs` (~150 LOC)
- [ ] 6.2 Handle both rank-3 (`(B, Sq, hidden)`) and rank-4
  (`(B, num_heads, Sq, head_dim)`) inputs via shape inference on
  `rank` + `num_heads` attribute
- [ ] 6.3 Unit test: rank-3 non-interleaved matches HF reference
- [ ] 6.4 Unit test: rank-4 interleaved matches DeepSeek reference
- [ ] 6.5 Unit test: `position_ids` broadcast from `(1, Sq)` to
  `(B, Sq)`

## 7. `MultiHeadAttention`

- [ ] 7.1 Add `op_multi_head_attention` in `ops/microsoft.rs` (~250 LOC)
- [ ] 7.2 Implement input parsing for unpacked Q/K/V form
- [ ] 7.3 Implement input parsing for packed QKV form (single input
  split along the last axis)
- [ ] 7.4 Wire in-place past KV-cache concat (same mechanism as GQA,
  without grouping)
- [ ] 7.5 Unit test: unpacked Q/K/V on `(1, 4, 64)` with `num_heads = 8`
- [ ] 7.6 Unit test: packed QKV form
- [ ] 7.7 Unit test: with past KV-cache (`past_key` and `past_value`
  supplied)
- [ ] 7.8 Unit test: no internal RoPE — confirm the op does not
  mutate Q or K along the rotary dimension

## 8. `GroupQueryAttention`

- [ ] 8.1 Add `op_group_query_attention` in `ops/microsoft.rs`
  (~600-800 LOC)
- [ ] 8.2 Input parsing: required 9-input form (Q, K, V, past_K,
  past_V, seqlens_k, total_sequence_length, cos_cache, sin_cache)
- [ ] 8.3 Attribute parsing: `num_heads`, `kv_num_heads`, `scale`,
  `local_window_size`, `do_rotary`, `rotary_interleaved`
- [ ] 8.4 Return `UnsupportedAttribute` if `local_window_size != -1`
- [ ] 8.5 Reshape Q, K, V to rank-4 `(B, heads, Sq, head_dim)`
- [ ] 8.6 Apply RoPE to Q and K in place using the shared helper
  (skip if `do_rotary == 0`)
- [ ] 8.7 Concatenate K, V with past-KV in place
- [ ] 8.8 Grouped attention dispatch: outer loop over the
  `kv_num_heads` groups, inner call to the shared SDPA helper
- [ ] 8.9 Reshape output back to `(B, Sq, hidden)` and return it
  alongside `present_key` / `present_value`
- [ ] 8.10 Unit test: hand-crafted 4-head / 2-kv-head / head_dim-8
  case against Python ORT reference within 1e-5 absolute
- [ ] 8.11 Unit test: KV-cache in-place mutation across two calls
- [ ] 8.12 Unit test: `do_rotary = 0` path (query is not rotated
  inside the op)

## 9. Dispatcher wiring + `classify_op`

- [ ] 9.1 Add `dispatch_microsoft_fused` in `onnx-rt/src/executor.rs`
  matching on the Microsoft subset of `OpKind`
- [ ] 9.2 Extend `classify_op` in `onnx-rt/src/profile.rs`: GQA and
  MHA → `OperatorClass::Attention`; the two norm variants and
  RotaryEmbedding → `OperatorClass::Elementwise`
- [ ] 9.3 Register the `microsoft` module in `onnx-rt/src/ops/mod.rs`

## 10. End-to-end model loading tests

- [ ] 10.1 Add `onnx-rt/tests/real_model_loading.rs` — load
  `Llama-3.2-1B` from `SMALLAIOS_MODEL_DIR/llama-3.2-1b.onnx`, run a
  single-token generation, assert top-1 token matches the golden
  reference in `onnx-rt/tests/fixtures/microsoft_fused_ops/llama.bin`
- [ ] 10.2 Extend with `Gemma 3 1b`, same fixture scheme
- [ ] 10.3 Extend with `DeepSeek-R1-Distill-Qwen-1.5B` f32 forward
  pass, assert output hidden states within 1e-3 absolute

## 11. Inventory updates + roadmap doc flip

- [ ] 11.1 Flip all 5 entries in
  `SUPPORTED_OPS_INVENTORY` from `Skipped` to `Implemented`
- [ ] 11.2 Flip the corresponding rows in
  `docs/onnx-coverage-roadmap.md` from `Skipped-vendor` to
  `Implemented (microsoft-fused-ops-v1)` in the same PR as the
  implementation

## 12. Validation

- [ ] 12.1 `just fmt`, `just clippy`, `just test` all pass
- [ ] 12.2 `openspec validate microsoft-fused-ops-v1 --strict` passes
- [ ] 12.3 Re-run `tools/coverage-probe` against `Llama-3.2-1B`,
  `Gemma 3 1b`, and `DeepSeek-R1-Distill-Qwen-1.5B`; REPORT.md MUST
  show zero unknown operators for all three model families after
  this change lands
