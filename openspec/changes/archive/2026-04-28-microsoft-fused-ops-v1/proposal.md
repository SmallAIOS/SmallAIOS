# Microsoft Fused Ops for Llama / Gemma / DeepSeek

## Why

Phase 1 (`transformer-models-v1`, `vision-transformers-v1`) and Phase 2
(`generative-llm-v1`) are complete. Together they took the CPU ONNX
runtime from 29 to 131 operators and added a sub-graph executor that
makes in-graph `Loop`-driven autoregressive generation feasible. The
empirical coverage probe that just landed (PR #88, see
`tools/coverage-probe/REPORT.md`) confirms that BERT, ViT, and GPT-2
now load end-to-end from their canonical HuggingFace ONNX exports.

The same probe also shows the uncomfortable truth about modern LLMs:
**the canonical HF ONNX exports of Llama 3.2, Gemma 3, and DeepSeek-R1
all fail to load today**, and in every case the failure is one or
more unknown operators from the `com.microsoft` domain. The union of
the ops required across the three model families is exactly five:

| `com.microsoft` op | Llama-3.2-1B | Gemma 3 1b | DeepSeek-R1-Distill-Qwen-1.5B |
|---|:---:|:---:|:---:|
| `SimplifiedLayerNormalization` | ✅ | ✅ | ✅ (1 use) |
| `SkipSimplifiedLayerNormalization` | ✅ | ❌ (plain Add) | ✅ |
| `GroupQueryAttention` | ✅ | ✅ | ❌ |
| `MultiHeadAttention` | ❌ | ❌ | ✅ |
| `RotaryEmbedding` | ❌ (fused in GQA) | ❌ (fused in GQA) | ✅ (standalone) |

No single op can be dropped from this union without breaking one of the
three families. Gemma uses a plain `Add` residual instead of the fused
`SkipSimplifiedLayerNormalization`; DeepSeek factors RoPE out as a
standalone op instead of fusing it into attention. The set is
irreducible.

The original `onnx-full-coverage-roadmap-v1` (now archived under
`openspec/changes/archive/`) classified every `com.microsoft` op as
**Skipped-vendor** on the assumption that they were optional
optimization tricks specific to Microsoft's ONNX Runtime. That decision
was wrong. These are not optional optimizations — they are how
HuggingFace's `optimum-cli export onnx --task text-generation`
pipeline writes every modern decoder-only LLM, because that pipeline
uses the ORT transformer optimizer by default. Without these five ops,
SmallAIOS cannot load **any** LLM from the canonical HF export path.

This change reverses that Skipped-vendor decision for these five
specific ops while leaving the long tail of other `com.microsoft` ops
(QLinear* fused activations, EmbedLayerNormalization variants,
QAttention, etc.) Skipped. We implement exactly what the empirical
probe says we need — no more, no less.

## What Changes

- **Add a `Domain` enum on `OpKind`** (`StandardOnnx | MicrosoftFused`)
  so the registry can differentiate standard ONNX operators from
  namespaced vendor fusions. The dispatcher checks the node's `domain`
  field (currently ignored) to route to the correct handler.
- **Add 5 new `OpKind` variants** for the Microsoft-domain fused ops:
  - `SimplifiedLayerNormalization` (all three families)
  - `SkipSimplifiedLayerNormalization` (Llama + DeepSeek; Gemma uses
    plain `Add` residual and does not need this op)
  - `GroupQueryAttention` (Llama + Gemma; the substantial one)
  - `MultiHeadAttention` (DeepSeek; older non-GQA fusion)
  - `RotaryEmbedding` (DeepSeek; standalone RoPE, interleaved and
    non-interleaved variants)
- **Extend `OperatorRegistry`** to list the 5 new variants with
  `Domain::MicrosoftFused`, and extend `classify_op` in
  `onnx-rt/src/profile.rs` so each gets a WCET budget class
  (`Attention` for GQA/MHA, `Elementwise` for the norm variants,
  `Elementwise` for `RotaryEmbedding`).
- **New file `onnx-rt/src/ops/microsoft.rs`** containing all 5 operator
  functions plus two shared internal helpers:
  - `scaled_dot_product_attention(q, k, v, mask, scale)` — shared by
    `GroupQueryAttention` and `MultiHeadAttention`.
  - `apply_rope_in_place(tensor, cos_cache, sin_cache, position_ids,
    interleaved)` — shared by `GroupQueryAttention` (RoPE is fused
    inside), and by the standalone `RotaryEmbedding` op.
- **New dispatcher hook** `dispatch_microsoft_fused` in
  `onnx-rt/src/executor.rs` that routes nodes with
  `domain == "com.microsoft"` to the Microsoft op table.
- **Inventory update** (`SUPPORTED_OPS_INVENTORY`): flip each of the 5
  ops from `Skipped` to `Implemented`. Flip the roadmap document at
  `docs/onnx-coverage-roadmap.md` in the same PR.
- **End-to-end validation tests**: load `Llama-3.2-1B`, `Gemma 3 1b`,
  and `DeepSeek-R1-Distill-Qwen-1.5B` from their canonical HF ONNX
  exports via `Session::new_from_file()` and run a 1-token generation
  (or for DeepSeek f32, an f32 forward pass).

## Impact

**Affected specs:**
- `onnx-cpu-execution` — adds 5 operator requirements and 2
  cross-cutting requirements (domain-aware OpKind, empirical model
  loading).

**Affected code:**
- New file: `onnx-rt/src/ops/microsoft.rs` (~1200-1500 LOC estimated).
- `onnx-rt/src/operators.rs` — 5 new `OpKind` variants, `Domain` enum,
  `OperatorRegistry` entries.
- `onnx-rt/src/executor.rs` — `dispatch_microsoft_fused` hook.
- `onnx-rt/src/profile.rs` — `classify_op` extension.
- `onnx-rt/src/ops/mod.rs` — register new module.
- `onnx-rt/tests/microsoft_fused_ops.rs` — per-op unit tests.
- `onnx-rt/tests/real_model_loading.rs` — end-to-end loading tests for
  the three LLM families.
- `docs/onnx-coverage-roadmap.md` — flip entries Skipped → Implemented.

**Size estimate.** `GroupQueryAttention` is the large piece:
approximately 600-800 LOC including KV-cache concatenation, RoPE
integration, grouped attention dispatch, and the causal mask. The
other four ops are smaller: `MultiHeadAttention` ~250 LOC (heavy reuse
of the SDPA helper), `RotaryEmbedding` ~150 LOC, the two norm variants
~50 LOC each, and the two shared helpers add ~200 LOC of their own.
Tests roughly double the implementation LOC.

**Out of scope:**
- Other `com.microsoft` ops not used by Llama, Gemma, or DeepSeek. The
  long tail (`QLinearAdd`, `QLinearSoftmax`, `QLinearSigmoid`,
  `QLinearLeakyRelu`, `QAttention`, `QEmbedLayerNormalization`,
  `GatherBlockQuantized`, `MatMulNBits`, `DequantizeLinear` with
  `axis`, etc.) stays classified as Skipped-vendor. If a future model
  target requires one, a new OpenSpec change will add it.
- GPU dispatch of any of these ops. CPU only for this tier.
- FP16 or BF16 fast paths. The ops accept fp16/bf16 inputs via the
  existing type-promotion shims but do not add specialized kernels.
- Llama / Gemma / DeepSeek training or fine-tuning. Inference only.

**Risks:**
- **`GroupQueryAttention` is not formally specified by ONNX.** It is a
  vendor contrib op. The canonical reference is the ORT contrib op
  implementation in the `onnxruntime` C++ source tree plus the
  Microsoft documentation page at
  <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention>.
  Our implementation reads from both. Mitigated by the end-to-end
  validation test (D9): the output must match a Python ORT reference
  within ±1 in the quantized integer domain.
- **KV-cache state management interacts with the sub-graph executor.**
  The Phase 2 sub-graph executor clears its inner `value_map` between
  `Loop` iterations. KV-cache tensors cannot live in that inner map —
  they must live in the outer scope and be passed in by outer-ref.
  This is already how ORT-exported Llama / Gemma write their graphs,
  but we must document the constraint so that future graph
  optimizations do not accidentally localize a KV-cache tensor.
  Mitigated by D8 in the design doc.
- **Interleaved vs non-interleaved RoPE.** HuggingFace / Llama use
  non-interleaved rotation (the second half rotates against the first
  half). DeepSeek uses interleaved rotation (alternating elements
  rotate pairwise). Both must be supported via the `interleaved` bool
  attribute. Mitigated by per-variant unit tests in tasks section 3.
- **Roadmap drift.** Flipping five Skipped entries to Implemented must
  happen in the same PR as the implementation, not a follow-up. An
  explicit task (11.2) enforces this.

## References

- `tools/coverage-probe/REPORT.md` — the empirical findings that
  motivate this change. The coverage probe walks every node of each
  HF ONNX export and reports unknown operators by domain. This is
  where the irreducible-union table in the Why section comes from.
- `openspec/changes/archive/<date>-onnx-full-coverage-roadmap-v1/` —
  the original roadmap that classified these five ops as Skipped-vendor.
  This change reverses that decision for these five specific ops only.
- `openspec/changes/archive/<date>-generative-llm-v1/` — Phase 2. The
  RMSNorm kernel this change reuses lives in `onnx-rt/src/ops/generative.rs`,
  added by that change. The sub-graph executor whose KV-cache
  lifecycle is discussed in D8 lives in `onnx-rt/src/sub_executor.rs`,
  also added there.
- `docs/sub-graph-executor-design.md` — the detailed spec for the
  sub-graph executor. Section on outer-ref value passing is the
  contract that KV-cache tensors rely on.
- <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md> —
  the ORT contrib op documentation, canonical reference for all five
  operators.
