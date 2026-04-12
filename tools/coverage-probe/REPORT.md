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
