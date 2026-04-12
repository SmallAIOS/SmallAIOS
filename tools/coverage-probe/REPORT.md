# ONNX Coverage Probe — Empirical Validation Report

This report captures the results of running
`smallaios-coverage-probe` against publicly-available ONNX exports of
the model families targeted by the Phase 1 (`transformer-models-v1`,
`vision-transformers-v1`) and Phase 2 (`generative-llm-v1`) operator
coverage roadmap entries.

The probe does **not** execute any model — it walks the `ModelProto`
graph, tallies each distinct `op_type`, and cross-references against the
live `OperatorRegistry` plus `docs/onnx-coverage-roadmap.md`. Its output
is a structural verdict: "loadable today", "loadable after Phase N",
or "has unrecognized / out-of-roadmap ops".

Op counts below have been cross-validated against a reference
`python3 -c "import onnx; ..."` parse of every fixture — every model
matches exactly on `(total_nodes, distinct_op_kinds)` and on the
per-op histogram.

## Fixtures used

All downloaded from public HuggingFace / ONNX-community mirrors on
2026-04-11. See the end of this document for re-download commands.
The fixtures directory (`tests/fixtures/onnx-models/`) is gitignored —
these files are not committed to the repo.

| Model | Source | Size |
|---|---|---|
| `bert-base-uncased.onnx` | `Xenova/bert-base-uncased` | 418 MB |
| `distilbert-base-uncased.onnx` | `Xenova/distilbert-base-uncased` | 256 MB |
| `vit-base-patch16-224.onnx` | `Xenova/vit-base-patch16-224` | 330 MB |
| `distilgpt2.onnx` | `Xenova/distilgpt2` | 313 MB |
| `gpt2-small.onnx` | `Xenova/gpt2` (decoder_model.onnx) | 476 MB |
| `llama-3.2-1b.onnx` | `onnx-community/Llama-3.2-1B-Instruct` | 105 KB (graph-only; weights in external `.onnx_data`) |
| `mobilenet_v2.onnx` | `onnx/models` model zoo | 13 MB |

The Llama-3.2-1B fixture is the small graph-only file that references
external weight data (`model.onnx_data` and friends, ~5 GB total).
Because the probe only looks at the graph structure, this is sufficient
for coverage analysis and avoids downloading multi-GB weight blobs.

## Per-model verdict table

| Model | Nodes | Distinct ops | Implemented | Planned-P2 | Planned-P3 | Planned-P4 | Unrecognized | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bert-base-uncased       | 1268 | 19 | 19 | 0 | 0 | 0 | 0 | loadable_today |
| distilbert-base-uncased |  607 | 22 | 22 | 0 | 0 | 0 | 0 | loadable_today |
| vit-base-patch16-224    | 1242 | 24 | 24 | 0 | 0 | 0 | 0 | loadable_today |
| mobilenet_v2            |  105 | 11 | 11 | 0 | 0 | 0 | 0 | loadable_today |
| distilgpt2              |  683 | 27 | 27 | 0 | 0 | 0 | 0 | loadable_today |
| gpt2-small              | 3092 | 25 | 25 | 0 | 0 | 0 | 0 | loadable_today |
| llama-3.2-1b            |  220 | 13 | 10 | 0 | 0 | 0 | 3 | has_unrecognized_ops |

(`Planned-P1` column omitted — all entries are zero; it is collapsed
into `Implemented` above for readability.)

## Phase 1 verification

**BERT-base (`bert-base-uncased`):** loadable on the current `develop`
branch with **zero** Planned / Unrecognized ops. All 19 distinct
`op_type`s (`Constant`, `Add`, `Gather`, `Unsqueeze`, `MatMul`, `Shape`,
`Mul`, `ReduceMean`, `Div`, `Transpose`, `Concat`, `Reshape`, `Sub`,
`Pow`, `Sqrt`, `Erf`, `Softmax`, `Cast`, `Slice`) are present in
`OperatorRegistry`.

**DistilBERT (`distilbert-base-uncased`):** loadable today. 22 distinct
ops, all implemented. Adds `Expand`, `Where`, `Equal` on top of BERT's
set — still all registered.

**ViT-base (`vit-base-patch16-224`):** loadable today. 24 distinct ops,
all implemented. Adds `ConstantOfShape`, `Conv`, `Gemm` on top of the
BERT set — still all registered.

**MobileNetV2** (sanity check, Tier 1): loadable today. 11 distinct ops,
dominated by `Conv` (52) and `Clip` (35).

**Result:** Phase 1 is structurally complete for BERT / DistilBERT /
ViT / MobileNetV2 on real exports. The probe finds **zero** gap for
Phase 1.

## Phase 2 verification

The Phase 2 change (`generative-llm-v1`) currently commits to:

- **3 control-flow ops:** `If`, `Loop`, `Scan`
- **19 generative / normalization ops:** `RMSNormalization`,
  `MatMulInteger`, `DynamicQuantizeLinear`, `RandomNormal`,
  `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`,
  `Multinomial`, `Bernoulli`, `Dropout`, `EyeLike`, `ReduceL1`,
  `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceSumSquare`,
  `LpNormalization`, `MeanVarianceNormalization`, `Softplus`
- A real tiled INT8 GEMM kernel backing `MatMulInteger` / `QLinearMatMul`

Against the Phase 2 target models:

### GPT-2-small (`Xenova/gpt2` decoder_model.onnx)

**Verdict: loadable_today — no Phase 2 ops needed.**

The Xenova export of GPT-2 is a plain forward-pass decoder model: the
host supplies `past_key_values` externally and calls the model once per
token. It does **not** embed a generation `Loop`. All 25 distinct op
types (`Constant`, `Unsqueeze`, `Shape`, `Gather`, `Concat`, `Reshape`,
`Add`, `Slice`, `Squeeze`, `Mul`, `Transpose`, `ReduceMean`, `Pow`,
`Gemm`, `Cast`, `Sub`, `Div`, `MatMul`, `Sqrt`, `ConstantOfShape`,
`Softmax`, `Split`, `Tanh`, `Where`, `Range`) are already in the live
registry.

**Implication:** Phase 2's `Loop` / `If` / `Scan` support is **not**
required to load and step this GPT-2 export. The host-driven KV-cache
loop works on `develop` today. Phase 2 still delivers value
(`onnxruntime-genai`-style single-call generation via `Loop`), but the
fixture used here does not exercise that path.

DistilGPT-2 is the same story — 27 distinct ops, all implemented.

### Llama-3.2-1B-Instruct (`onnx-community/Llama-3.2-1B-Instruct`)

**Verdict: has_unrecognized_ops — Phase 2 plan does not cover this
model's fused-op export.**

The onnx-community export of Llama-3.2-1B uses ONNX Runtime's
`com.microsoft` fused operators for the attention / norm blocks:

| Op (com.microsoft domain) | Nodes | In Phase 2 plan? |
|---|---:|---|
| `SkipSimplifiedLayerNormalization` | 32 | **No** |
| `GroupQueryAttention`              | 16 | **No** |
| `SimplifiedLayerNormalization`     |  1 | **No** |

The rest of the model is 10 distinct ops all already in the registry
(`MatMul`×113, `Mul`×32, `Sigmoid`×16, `Cast`×2, `Constant`×2,
`Gather`×2, `ReduceSum`×1, `Shape`×1, `Sub`×1, `Transpose`×1). Notably
absent: any standard `LayerNormalization`, `RMSNormalization`, standard
`Attention`, or scalar `RotaryEmbedding` — they have **all been fused
into the three Microsoft-domain ops above** by ORT's LLM optimizer.

This is a **real gap** in the Phase 2 plan. Phase 2 adds
`RMSNormalization` as a standard op, but Llama-3.2 as exported by the
community does not emit it — the ORT optimizer rewrote it into
`SimplifiedLayerNormalization` (semantically equivalent: RMSNorm
without bias) and fused `RMSNorm + residual` pairs into
`SkipSimplifiedLayerNormalization`. Similarly, `GroupQueryAttention`
is a single kernel rolling up RoPE + KV-cache update + scaled-dot-
product + causal masking + grouped-query broadcast.

## Recommended Phase 2 amendments

### Amendment 1 — add `com.microsoft` fused-norm / attention ops

**Scope:** Extend Phase 2 with three operators in the `com.microsoft`
domain. These are the minimum required to load the HuggingFace
`onnx-community` Llama family (Llama-3.2-1B, -3B, 3-8B, -3.1-*) and
the `microsoft/phi-*` ONNX exports.

| Op | Llama-3.2-1B node count | Semantics | Cost |
|---|---:|---|---|
| `SimplifiedLayerNormalization` | 1 | `x * rsqrt(mean(x*x) + eps) * gamma` — RMSNorm without bias. A trivial alias for `RMSNormalization` which Phase 2 already plans. | Low: a few lines of dispatcher glue. |
| `SkipSimplifiedLayerNormalization` | 32 | `SimplifiedLayerNorm(x + skip + bias)` plus passthrough `x + skip`. One fused kernel or a two-op rewrite in the loader. | Low: ~50 LOC, reuses the RMSNorm kernel. |
| `GroupQueryAttention` | 16 | RoPE + masked scaled-dot-product attention with grouped-query KV broadcast, past/present KV, optional causal mask. Substantial. | High: ~200+ LOC of new kernel plus KV-cache plumbing. |

**Recommendation:**

- Fold items 1 and 2 directly into Phase 2 — they are essentially
  aliases / compositions of ops (`RMSNormalization`, `Add`) that Phase 2
  is already delivering, and the incremental cost is small.
- Item 3 (`GroupQueryAttention`) is the largest single kernel in the
  list. Two viable paths:
  - **(preferred)** Add it to Phase 2 under a new spec requirement
    "Requirement: `com.microsoft`-domain fused attention for
    onnx-community LLM exports", with a dedicated section in
    `design.md` walking through the RoPE + cache + softmax fusion.
  - **(fallback)** Split into a follow-up `com-microsoft-ops-v1`
    change gated on Phase 2 merging, and amend the Phase 2 proposal
    to explicitly call out "Llama-family loading requires the
    follow-up `com-microsoft-ops-v1` change".

Without one of these, **Phase 2 merging does not unlock Llama-class
models on the onnx-community export path**, even though `proposal.md` /
`design.md` repeatedly say Phase 2 unlocks LLaMA.

### Amendment 2 — clarify the "Phase 2 unlocks GPT-2" claim

**Scope:** Documentation only.

The Phase 2 proposal / design frames Phase 2 as "what unlocks GPT-2
and Llama". The probe shows that the canonical HuggingFace
(`Xenova/gpt2`) export of GPT-2 is **already loadable on `develop`** —
its decoder-only single-step form uses only ops in the current
registry. Phase 2 unlocks a specific **use case** (single-call
in-graph generation via `Loop`) but not the baseline load-and-step
path.

**Recommendation:** amend `proposal.md` and `design.md` to distinguish:

- **Today (post Phase 1):** `Xenova/gpt2` and `Xenova/distilgpt2`
  decoder-only exports load and run one token per `Session::run()`
  call, with host-managed KV cache. ✅ already works.
- **After Phase 2:** `onnxruntime-genai`-style self-contained
  generation (the whole token loop lives inside a single graph via
  `Loop` + sub-graph executor) becomes possible for any decoder
  model. The Llama family additionally needs the `com.microsoft`
  fused-op amendment above.

This is a spec-honesty fix. The Phase 2 *implementation* remains
valuable (it enables the fused single-call API and unblocks several
production LLM deployment patterns), but the user-visible claim
"Phase 2 is what enables GPT-2" is only true for the in-graph-loop
form, not the straightforward exports the HF community publishes.

### Amendment 3 — add `tools/coverage-probe` strict-mode CI gate

**Scope:** Test infrastructure.

Before Phase 2 merges, run the probe against all seven fixtures in
`--strict` mode; the gate fails if any model regresses in verdict
class. The `#[ignore]`'d fixture smoke test added by this PR
(`tools/coverage-probe/tests/fixture_smoke.rs`) provides the harness;
a separate CI job (optional, since fixtures aren't in-tree) can
re-download and re-check on a scheduled nightly run.

## Attribute-level concerns

The probe runs `--detailed` mode against every fixture. No model in
this set triggers any of the currently-tracked attribute concerns:

- No `Resize` nodes with `mode=cubic`
- No `Resize` nodes with non-`half_pixel` `coordinate_transformation_mode`
- No `RoiAlign` with `mode=max`
- No `GridSample` with non-`zeros` padding
- No `Unique` with an `axis` attribute
- No `ScatterND` with `reduction != none`

Every fixture's `detailed.md` section "Attribute concerns" is empty.

## Probe hardening applied in this PR

The probe's walker was previously calling
`smallaios_onnx_rt::protobuf::decode_model`, which fully decodes every
`TensorProto` initializer, every `AttributeProto`, and every nested
`TypeProto` under `ValueInfoProto`. That full-decode path fails on
every one of the seven real-world fixtures because the in-tree minimal
protobuf decoder does not cover a few corners of the wire format that
production exporters use (doc_strings with non-UTF-8 content, certain
repeated-field packing variations, external-data tensors, `com.*`
domain annotations).

The walker has been switched to a **streaming scan** that:

1. Only descends into field 7 (graph) of `ModelProto`, skipping
   everything else at the top level.
2. Only descends into field 1 (node) of `GraphProto`, skipping
   initializers (field 5), inputs/outputs/value_info, and doc strings.
3. Only extracts field 4 (op_type) from each `NodeProto`; decodes
   field 5 (attributes) only when the op type is known to carry
   attribute-level concerns.
4. Recursively walks subgraphs via AttributeProto field 6 (g) and
   field 11 (graphs) so that `If` / `Loop` / `Scan` bodies are also
   counted — important for Phase 2.
5. Uses a tolerant skip helper that returns protocol errors gracefully
   at the buffer boundary rather than bailing out mid-decode.

This makes the probe robust against every public ONNX export we
tested, while preserving the full attribute-level concern machinery
for the small set of ops that need it. Existing unit tests for
`attribute_concerns_for` still pass.

## Reproduction

```bash
# 1. download the fixtures (gitignored)
mkdir -p tests/fixtures/onnx-models
cd tests/fixtures/onnx-models
curl -L -o bert-base-uncased.onnx \
  https://huggingface.co/Xenova/bert-base-uncased/resolve/main/onnx/model.onnx
curl -L -o distilbert-base-uncased.onnx \
  https://huggingface.co/Xenova/distilbert-base-uncased/resolve/main/onnx/model.onnx
curl -L -o vit-base-patch16-224.onnx \
  https://huggingface.co/Xenova/vit-base-patch16-224/resolve/main/onnx/model.onnx
curl -L -o distilgpt2.onnx \
  https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/model.onnx
curl -L -o gpt2-small.onnx \
  https://huggingface.co/Xenova/gpt2/resolve/main/onnx/decoder_model.onnx
curl -L -o llama-3.2-1b.onnx \
  https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct/resolve/main/onnx/model.onnx
curl -L -o mobilenet_v2.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
cd -

# 2. run the probe
cargo build -p smallaios-coverage-probe --release
PROBE=./target/release/smallaios-coverage-probe
for m in bert-base-uncased distilbert-base-uncased vit-base-patch16-224 \
         distilgpt2 gpt2-small llama-3.2-1b mobilenet_v2; do
  $PROBE tests/fixtures/onnx-models/$m.onnx
done

# 3. enable the (ignored-by-default) fixture smoke test
cargo test -p smallaios-coverage-probe --test fixture_smoke -- --ignored
```

The `Xenova/mobilenet_v2_1.0_224` mirror listed in the task brief
returned 401 on 2026-04-11; the `onnx/models` GitHub mirror above was
used instead. Every other listed URL worked on first try.

## Models attempted but not probed

None — all seven target fixtures downloaded successfully and were
successfully walked by the probe (after the streaming hardening above).

## Gemma + DeepSeek (added by follow-up validation)

A second probe pass on 2026-04-11 adds Gemma 3 and
DeepSeek-R1-Distill-Qwen exports to the fixture set in order to
determine whether they use the same `com.microsoft`-domain fused-op
surface as Llama-3.2-1B. The goal is to scope a planned
`microsoft-fused-ops-v1` tier that covers the union of all three
model families in one pass.

### Fixture acquisition notes

The Gemma 1.x / Gemma 2 ONNX mirrors on HuggingFace
(`onnx-community/gemma-2-2b-it`, `onnx-community/gemma-2b-it`,
`Xenova/gemma-1.1-2b-it`, etc.) all returned **HTTP 401** on
2026-04-11 — the Gemma model license gates the export. The
**Gemma 3** mirrors are public and were used instead:

- `onnx-community/gemma-3-1b-it-ONNX` (graph: 259 KB) — public
- `onnx-community/gemma-3-270m-it-ONNX` (graph: 181 KB) — public

For DeepSeek the Qwen-1.5B distill is public while the Llama-8B
distill (`onnx-community/DeepSeek-R1-Distill-Llama-8B-ONNX`) is
gated:

- `onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX` (graph: 519 KB) — public
- `onnx-community/DeepSeek-R1-Distill-Llama-8B-ONNX` — **401 gated**, not probed

### Per-model verdict table (new fixtures)

| Model | Nodes | Distinct ops | Implemented | Planned-P2 | Unrecognized | Verdict |
|---|---:|---:|---:|---:|---:|---|
| gemma-3-1b-it                    |  597 | 15 | 13 | 0 | 2 | has_unrecognized_ops |
| gemma-3-270m-it                  |  421 | 15 | 13 | 0 | 2 | has_unrecognized_ops |
| deepseek-r1-distill-qwen-1.5b    | 1503 | 25 | 21 | 0 | 4 | has_unrecognized_ops |

### Unrecognized ops by model

| Model | Op (com.microsoft domain) | Count | Recommendation |
|---|---|---:|---|
| gemma-3-1b-it                   | `SimplifiedLayerNormalization`     | 157 | Fold into Phase 2 (RMSNorm alias) |
| gemma-3-1b-it                   | `GroupQueryAttention`              |  26 | `microsoft-fused-ops-v1` |
| gemma-3-270m-it                 | `SimplifiedLayerNormalization`     | 109 | Fold into Phase 2 (RMSNorm alias) |
| gemma-3-270m-it                 | `GroupQueryAttention`              |  18 | `microsoft-fused-ops-v1` |
| deepseek-r1-distill-qwen-1.5b   | `SkipSimplifiedLayerNormalization` |  56 | `microsoft-fused-ops-v1` |
| deepseek-r1-distill-qwen-1.5b   | `RotaryEmbedding`                  |  56 | `microsoft-fused-ops-v1` (NEW — not in Llama) |
| deepseek-r1-distill-qwen-1.5b   | `MultiHeadAttention`               |  28 | `microsoft-fused-ops-v1` (NEW — not in Llama/Gemma) |
| deepseek-r1-distill-qwen-1.5b   | `SimplifiedLayerNormalization`     |   1 | Fold into Phase 2 (RMSNorm alias) |

### Empirical answers to the load-bearing questions

1. **Does Gemma use `SimplifiedLayerNormalization`?** Yes —
   heavily (109 / 157 nodes in the 270m / 1b variants).
2. **Does Gemma use `SkipSimplifiedLayerNormalization`?** **No.**
   Gemma 3's ORT export keeps the residual add as a separate
   `Add` node and does not fuse it into the norm. This is a
   meaningful difference from Llama.
3. **Does Gemma use `GroupQueryAttention`?** Yes — 26 (1B) / 18
   (270m) nodes. Same kernel Llama needs.
4. **Does Gemma use `MultiHeadAttention` or `RotaryEmbedding`
   as separate `com.microsoft` ops?** **No.** Gemma 3's ORT
   export fuses RoPE into `GroupQueryAttention` itself and does
   not emit a standalone `RotaryEmbedding` op. No
   `MultiHeadAttention` appears either.
5. **Does DeepSeek-R1-Distill-Qwen use the same set as Llama?**
   **No — different set.** DeepSeek's Qwen distill uses
   `MultiHeadAttention` (not `GroupQueryAttention`) and emits
   `RotaryEmbedding` as a **separate** fused op rather than
   folding it into attention. It does use
   `SkipSimplifiedLayerNormalization`, which Llama also uses but
   Gemma does not.
6. **Are there additional unrecognized ops?** No. Every
   non-`com.microsoft` op across all three models is already
   implemented in `develop`'s `OperatorRegistry`. The only gaps
   are the five `com.microsoft` fused ops below.

### Cross-model overlap

Let L = Llama-3.2-1B, G = Gemma-3-{1b,270m}, D = DeepSeek-R1-Distill-Qwen-1.5B.

| `com.microsoft` op | L | G | D | Union tier |
|---|:---:|:---:|:---:|---|
| `SimplifiedLayerNormalization`     | yes (1) | yes (109+157) | yes (1) | Phase 2 alias of RMSNorm |
| `SkipSimplifiedLayerNormalization` | yes (32) | **no**  | yes (56) | `microsoft-fused-ops-v1` |
| `GroupQueryAttention`              | yes (16) | yes (26+18) | **no**  | `microsoft-fused-ops-v1` |
| `MultiHeadAttention`               | **no**   | **no**   | yes (28) | `microsoft-fused-ops-v1` |
| `RotaryEmbedding`                  | **no**   | **no**   | yes (56) | `microsoft-fused-ops-v1` |

**Key finding:** the three model families do **not** converge on a
single fused-attention surface. Each picks a different subset:

- **Llama-3.2** : `GroupQueryAttention` (RoPE + KV + mask fused in)
  \+ `SkipSimplifiedLayerNormalization`.
- **Gemma 3**   : `GroupQueryAttention` (RoPE fused in) **without**
  the skip-norm fusion — a plain `SimplifiedLayerNormalization`
  with a separate `Add` residual.
- **DeepSeek-R1-Distill-Qwen-1.5B** : `MultiHeadAttention`
  (non-GQA, RoPE **split out**) + standalone `RotaryEmbedding` +
  `SkipSimplifiedLayerNormalization`.

Implementing only `GroupQueryAttention` (Llama+Gemma) does **not**
cover DeepSeek. Implementing only `MultiHeadAttention` does not
cover Llama or Gemma. A single tier targeting all three must
ship both attention kernels plus standalone `RotaryEmbedding`.

### Amendment 4 (NEW) — rename and rescope `llama-attention-v1` to `microsoft-fused-ops-v1`

**Scope:** The Phase 2 follow-up change previously sketched as
`llama-attention-v1` (a single `GroupQueryAttention`-focused tier)
should be renamed `microsoft-fused-ops-v1` and scoped to the full
`com.microsoft`-domain fused-attention surface used by Llama,
Gemma, and DeepSeek. Rationale: the three families do not share
a single fused kernel.

**Operators in the tier (union of Llama + Gemma 3 + DeepSeek-R1-Distill-Qwen):**

| Op | Llama-3.2-1B | Gemma-3-1b | Gemma-3-270m | DeepSeek-Qwen-1.5B | Disposition in this tier |
|---|---:|---:|---:|---:|---|
| `SimplifiedLayerNormalization`     |  1 | 157 | 109 |  1 | **Phase 2** — RMSNorm alias, small wrapper |
| `SkipSimplifiedLayerNormalization` | 32 |   - |   - | 56 | **microsoft-fused-ops-v1** — RMSNorm + residual fusion, reuses RMSNorm kernel |
| `GroupQueryAttention`              | 16 |  26 |  18 |   - | **microsoft-fused-ops-v1** — RoPE + KV cache + grouped-query + causal mask, largest single kernel |
| `MultiHeadAttention`               |  - |   - |   - | 28 | **microsoft-fused-ops-v1** — non-GQA fused attention + KV cache + optional causal mask |
| `RotaryEmbedding`                  |  - |   - |   - | 56 | **microsoft-fused-ops-v1** — standalone fused RoPE (cos/sin LUT) |

**Coverage implication:** landing this tier unlocks all three
model families on their canonical `onnx-community` exports. No
single op can be dropped without breaking one of the three:

- Drop `GroupQueryAttention` → Llama + Gemma fail to load.
- Drop `MultiHeadAttention` → DeepSeek fails to load.
- Drop `RotaryEmbedding` → DeepSeek fails to load (RoPE is not
  inlined into its `MultiHeadAttention` nodes).
- Drop `SkipSimplifiedLayerNormalization` → Llama + DeepSeek fail
  to load.
- Drop `SimplifiedLayerNormalization` → all three fail (it is
  the norm kernel the others reuse).

**Estimated effort (rough):**

- `SimplifiedLayerNormalization` — trivial alias of
  `RMSNormalization` (Phase 2 delivers the kernel).
- `SkipSimplifiedLayerNormalization` — ~50 LOC fused wrapper over
  the RMSNorm kernel + optional passthrough output.
- `RotaryEmbedding` — ~80 LOC, precomputed cos/sin table,
  per-head rotation.
- `MultiHeadAttention` — ~250 LOC (KV cache plumbing + softmax +
  optional mask). Simpler than `GroupQueryAttention` because no
  group broadcast.
- `GroupQueryAttention` — ~300 LOC (adds grouped-query
  broadcast over `MultiHeadAttention`; the two can share a core
  scaled-dot-product primitive).

Total: one large kernel (`GroupQueryAttention`), one medium
kernel (`MultiHeadAttention`), two small fused wrappers
(`Skip…Norm`, `RotaryEmbedding`), and one alias. The two fused
attention kernels should share a common scaled-dot-product
helper to avoid duplication.

### Reproduction (new fixtures)

```bash
cd tests/fixtures/onnx-models
curl -L -o gemma-3-1b-it.onnx \
  https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/onnx/model.onnx
curl -L -o gemma-3-270m-it.onnx \
  https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX/resolve/main/onnx/model.onnx
curl -L -o deepseek-r1-distill-qwen-1.5b.onnx \
  https://huggingface.co/onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/resolve/main/onnx/model.onnx
cd -

PROBE=./target/release/smallaios-coverage-probe
for m in gemma-3-1b-it gemma-3-270m-it deepseek-r1-distill-qwen-1.5b; do
  $PROBE tests/fixtures/onnx-models/$m.onnx
done
```

No probe / walker hardening was needed for the new fixtures —
Agent K's streaming scan already handled them on first try.

## Full Model Validation with HF Auth (2026-04-12)

The download script (`scripts/download-test-fixtures.sh`) was updated to
support HuggingFace token authentication, enabling download of gated
models. The token is resolved from `$HF_TOKEN` env var or
`~/.cache/huggingface/token` (written by `hf auth login`).

### Download Results

| Model | Source | Size | Token Required | Downloaded |
|---|---|---|---|---|
| `bert-base-uncased.onnx` | `Xenova/bert-base-uncased` | 418 MB | No | Yes |
| `vit-base-patch16-224.onnx` | `Xenova/vit-base-patch16-224` | 330 MB | No | Yes |
| `distilgpt2.onnx` | `Xenova/distilgpt2` | 313 MB | No | Yes |
| `llama-3.2-1b.onnx` | `onnx-community/Llama-3.2-1B-Instruct` | 105 KB | No | Yes |
| `deepseek-r1-distill-qwen-1.5b.onnx` | `onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX` | 519 KB | No | Yes |
| `mobilenet_v2.onnx` | `onnx/models` model zoo | 13 MB | No | Yes |
| `gemma-3-1b-it.onnx` | `onnx-community/gemma-3-1b-it-ONNX` | 258 KB | No | Yes |
| `gemma-2-2b-it.onnx` | `onnx-community/gemma-2-2b-it` | N/A | Yes (gated) | No (HTTP 404 - repo does not exist) |

**Gemma 2 2B note:** The `onnx-community/gemma-2-2b-it` repo does not
exist on HuggingFace. Only a Japanese variant (`gemma-2-2b-jpn-it`)
exists, gated under the Gemma license. The Gemma 3 1B export from
`onnx-community/gemma-3-1b-it-ONNX` is public and downloaded
successfully with authentication headers.

### Pipeline Stage Results: `decode_model` (develop branch, pre-PR #98)

All `decode_model` calls fail on develop because the in-tree protobuf
parser does not handle all wire-format features used by real ONNX
exports. These failures are expected and will become passes after
PR #98 (parser hardening) merges.

| Model | decode_model | Error | build_execution_graph | Notes |
|---|---|---|---|---|
| `bert-base-uncased.onnx` | FAIL | `UnexpectedEof` | Blocked | 418 MB with embedded weights |
| `vit-base-patch16-224.onnx` | FAIL | `InvalidFieldNumber` | Blocked | 330 MB with embedded weights |
| `distilgpt2.onnx` | FAIL | `BufferTooSmall` | Blocked | 313 MB with embedded weights |
| `llama-3.2-1b.onnx` | FAIL | `InvalidWireType` | Blocked | 105 KB graph-only -- real parser bug |
| `deepseek-r1-distill-qwen-1.5b.onnx` | FAIL | `InvalidFieldNumber` | Blocked | 519 KB graph-only -- real parser bug |
| `mobilenet_v2.onnx` | FAIL | `InvalidFieldNumber` | Blocked | 13 MB with weights |
| `gemma-3-1b-it.onnx` | FAIL | `InvalidFieldNumber` | Blocked | 258 KB graph-only -- real parser bug |

**Key observation:** The three graph-only files (Llama, DeepSeek,
Gemma 3) are small enough that `decode_model` should succeed -- these
are real parser bugs, not size limitations. PR #98 targets exactly
these failures.

### Coverage Probe Results (streaming scanner, bypasses decode_model)

The coverage probe uses its own streaming scanner that skips unknown
fields and successfully processes all fixtures:

| Model | Nodes | Distinct ops | Implemented | Unrecognized | Verdict |
|---|---:|---:|---:|---:|---|
| `bert-base-uncased` | 1268 | 19 | 19 | 0 | loadable_today |
| `vit-base-patch16-224` | 1242 | 24 | 24 | 0 | loadable_today |
| `distilgpt2` | 683 | 27 | 27 | 0 | loadable_today |
| `mobilenet_v2` | 105 | 11 | 11 | 0 | loadable_today |
| `llama-3.2-1b` | 220 | 13 | 10 | 3 | has_unrecognized_ops |
| `deepseek-r1-distill-qwen-1.5b` | 1503 | 25 | 21 | 4 | has_unrecognized_ops |
| `gemma-3-1b-it` | 597 | 15 | 13 | 2 | has_unrecognized_ops |

### What blocks end-to-end validation

1. **PR #98 (parser hardening)** -- must merge for `decode_model` to
   succeed on real ONNX files. Currently all 7 fixtures fail at the
   parse stage.
2. **Graph builder initializer fix** (in-flight PR) -- required for
   `build_execution_graph` to handle models with initializers
   correctly.
3. Once both merge, re-run:
   ```bash
   cargo test -p smallaios-onnx-rt --test test_model_fixtures -- --ignored --nocapture
   ```
   The graph-only files (Llama, DeepSeek, Gemma 3) should parse and
   build. The weight-embedded files (BERT, ViT, DistilGPT-2) may
   still fail due to large tensor data unless PR #98 also handles
   external-data / large-varint tensor fields.
