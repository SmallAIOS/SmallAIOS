# ONNX Operator Coverage Roadmap

This is the canonical plan for SmallAIOS's ONNX operator coverage. Every
future operator-coverage OpenSpec change must slot into a tier defined
here. Cite this document from each tier proposal.

**Last reviewed:** 2026-04-11
**Current coverage in `develop`:** 29 / ~190 standard ops (~15%)
**Coverage after Phase 1 PRs (#74, #76, #77) merge:** 109 / ~190 (~57%)
**Owning OpenSpec change:** `onnx-full-coverage-roadmap-v1`

> The "Implemented" section below lists ops that are *either* in
> `develop` today *or* in flight on the Phase 1 PRs (#74 Tier 2,
> #76 vision, #77 transformer). The roadmap forward-looks so it can
> serve as the merge target spec; numbers will become true as the
> PRs land.

## Goals

- Catalog **every** standard ONNX operator (opset 21) so nothing slips
  through ad-hoc per-model work.
- Drive coverage by **target model class**, not by spec compliance —
  every tier ships a real, demoable workload.
- Make "remaining work" a queryable, testable number rather than
  folklore.
- Be honest about what we will **not** implement (training ops,
  ecosystem ops, sequence/optional/control-flow subsystems).

## Plan Shape: Phases, Not a Long Queue

Earlier drafts of this roadmap listed 10 sequential tiers. After
analyzing the actual gap each tier closes, that ordering is wrong:

- The only **material** missing capability in standard ONNX is
  `Loop` / `If` / `Scan`. Without `Loop`, autoregressive generation
  (GPT/LLaMA-class models) cannot run as a single graph call. Every
  other "deferred" item is either dead code, vendor-specific, or a
  separate workload (classical ML).
- `int8-kernels` is **performance for LLMs**, not a coverage gap.
  It only matters once we have LLMs running, so it belongs *with* the
  generative tier, not as a separate later tier.
- `transformer-models` (BERT) and `vision-transformers` (ViT) are
  **independent** — different `ops/` submodules, different validation
  models — so they can run in **true parallel** worktrees.

The revised plan is **4 phases** instead of 10 tiers. Phases run
sequentially; tiers within a phase run in parallel.

| Phase | OpenSpec changes (parallel within phase) | Target models | Op delta | Cumulative | % of ONNX |
|---|---|---|---|---|---|
| 1 ✅ | `onnx-cpu-runtime-v1` (archived) | ResNet, MobileNet, YOLO | 29 | 29 | 15% |
| 2 ✅ | `additional-operators-v1` | Building blocks | +36 | 65 | 34% |
| **P1** | `transformer-models-v1` ‖ `vision-transformers-v1` | BERT, DistilBERT, ViT, Swin, DeiT, ConvNeXt | +44 | 109 | 57% |
| **P2** | `generative-llm-v1` (combined: control-flow + generative ops + i8 kernels) | GPT-2, T5, single-pass autoregressive generation, real i8 LLM inference | +22 + sub-graph executor | 131 | 69% |
| **P3** | `audio-models-v1` ‖ `detection-models-v1` | Whisper-tiny, Wav2Vec2, DETR, RetinaNet | +21 | 152 | 80% |
| **P4** | `long-tail-v1` (reactive) | User-driven additions | +~30 | ~182 | ~96% |

The remaining ~14 ops are explicitly **Deferred-subsystem** (sequence
types, optional types, string tensors, ONNX-ML classical ML) or
**Skipped** (training, deprecated, vendor). They are not on the path
unless a real user model demands one of them. See "What we're
explicitly not doing" below.

### Why this ordering

- **Phase 1 first** because BERT and ViT are the highest-leverage
  encoder workloads and the two changes can be implemented in
  parallel agent worktrees. Phase 1 also adds the
  `OperatorStatus` / `SUPPORTED_OPS_INVENTORY` machinery that every
  later phase depends on.
- **Phase 2 is the breakthrough phase** — GPT-2-class LLMs cannot
  run without `Loop`, and `int8-kernels` is only valuable once an
  LLM is running. Bundling control-flow + generative ops + i8
  kernels into one tier prevents the half-finished state where we
  have generative ops but still need an external Python loop. This
  is the most ambitious phase and runs sequentially after P1.
- **Phase 3 covers specialty model classes** (audio, detection)
  that are narrower in scope and runnable in parallel after the
  Phase 2 dispatcher work has stabilized.
- **Phase 4 is reactive maintenance**, not a scheduled change.
  Long-tail ops are pulled when a real user model fails to load.

### What we're explicitly not doing (and why)

| Bucket | Op count | Why we're not doing it |
|---|---|---|
| Training-only ops (Adam, Gradient, …) | ~10 | SmallAIOS is inference-only by charter. These ops appear only in training checkpoints, not inference models. |
| Deprecated ops (Affine, Upsample, Scatter, …) | ~7 | Converters rewrite them at export time. Implementing them would be implementing dead code. |
| Vendor (`com.microsoft`) QLinear* ops | ~8 | Microsoft-specific quantization format. Standard QDQ format works on every runtime and uses ops we already implement. |
| Sequence types (`SequenceConstruct`, …) | 9 | Requires a `Tensor`-of-`Tensor` value type. Niche; no current target model needs them. |
| Optional types (`Optional`, …) | 3 | Coupled with control-flow; would land in Phase 2 only if a target model demands it. |
| String / tokenizer ops | 5 | Requires a string tensor type. Most NLP pipelines tokenize before hitting the ONNX graph. |
| ONNX-ML classical ML (`TreeEnsembleClassifier`, `LinearRegressor`, …) | ~18 | Different ML domain entirely (trees, SVMs, feature pipelines). Out of scope for a neural inference runtime; would warrant its own decision rather than being lumped in. |

Total deliberately not on the path: **~60 ops**, none of which
materially limit any neural model class we care about.

## Operator Inventory

The inventory tracks every standard operator with one of:

- **Implemented** — works today.
- **Planned-P<n>** — will land in phase `n` (P1-P4).
- **Deferred-subsystem** — needs a non-trivial subsystem we haven't
  built yet (sequence types, optional types, control-flow executor,
  string tensors).
- **Skipped-training** — training-only op; SmallAIOS is inference-only.
- **Skipped-deprecated** — superseded by a newer op; converters
  rewrite at export time.
- **Skipped-vendor** — non-standard op (e.g., `com.microsoft` domain).

The Phase 1 change (`transformer-models-v1`) encodes this inventory
as a Rust constant `SUPPORTED_OPS_INVENTORY` with a CI test that fails
if the implemented set drifts from the inventory.

### Implemented (29 in develop today; 65 after #74; 109 after Phase 1)

**Tier 1 (29):** Add, Sub, Mul, Div, MatMul, Relu, Sigmoid, Tanh,
Softmax, Conv, MaxPool, AveragePool, BatchNormalization, Reshape,
Transpose, Flatten, Squeeze, Unsqueeze, Concat, Gather, Slice, Pad,
Gemm, GlobalAveragePool, LayerNormalization, Cast, Clip, ReduceMean,
ReduceSum

**Tier 2 (36):** Pow, Sqrt, Exp, Log, Erf, Neg, Abs, Floor, Ceil,
Round, Equal, NotEqual, Less, LessOrEqual, Greater, GreaterOrEqual,
Where, Min, Max, Not, Gelu, LeakyRelu, Elu, Swish, RNN, LSTM, GRU,
Split, Expand, Tile, OneHot, Einsum, QuantizeLinear, DequantizeLinear,
QLinearMatMul, QLinearConv

### Math / Elementwise

| Op | Status | Rationale |
|---|---|---|
| Mod | Implemented | Integer/float modulo; positional encoding |
| Sin | Implemented | Sinusoidal positional embeddings |
| Cos | Implemented | Sinusoidal positional embeddings |
| Reciprocal | Implemented | 1/x; LayerNorm and attention scaling |
| Sign | Implemented | Sign extraction; quant + transformers |
| Sum | Implemented | Variadic elementwise sum |
| Mean | Implemented | Variadic elementwise mean |
| And | Implemented | Boolean mask composition |
| Or | Implemented | Boolean mask composition |
| LogSoftmax | Implemented | Common classifier head |
| PRelu | Implemented | Parametric ReLU; older vision nets |
| HardSigmoid | Implemented | MobileNet/EfficientNet activation |
| HardSwish | Implemented | MobileNetV3 activation |
| Softplus | Planned-P2 | Smooth ReLU; generative models |
| Sinh | Planned-P3 | Hyperbolic trig; audio models |
| Cosh | Planned-P3 | Hyperbolic trig; audio models |
| Tan | Planned-P4 | Rare in inference |
| Asin | Planned-P4 | Rare inverse trig |
| Acos | Planned-P4 | Rare inverse trig |
| Atan | Planned-P4 | Rare inverse trig |
| Asinh | Planned-P4 | Long tail |
| Acosh | Planned-P4 | Long tail |
| Atanh | Planned-P4 | Long tail |
| Xor | Planned-P4 | Rare boolean op |
| BitwiseAnd | Planned-P4 | Integer bitwise (opset 18+) |
| BitwiseOr | Planned-P4 | Integer bitwise (opset 18+) |
| BitwiseXor | Planned-P4 | Integer bitwise (opset 18+) |
| BitwiseNot | Planned-P4 | Integer bitwise (opset 18+) |
| BitShift | Planned-P4 | Integer bit shift |
| ThresholdedRelu | Planned-P4 | Rare activation |
| Selu | Planned-P4 | Rare self-normalizing activation |
| Celu | Planned-P4 | Rare continuous ELU |
| Softsign | Planned-P4 | Rare activation |
| Shrink | Planned-P4 | Soft thresholding; rare |
| Hardmax | Planned-P4 | One-hot argmax variant |
| IsNaN | Planned-P4 | Debug/validation op |
| IsInf | Planned-P4 | Debug/validation op |
| MeanVarianceNormalization | Planned-P2 | MVN normalization |
| LpNormalization | Planned-P2 | L1/L2 normalization layer |
| Affine | Skipped-deprecated | Replaced by Mul+Add |
| ImageScaler | Skipped-deprecated | Replaced by Mul+Add |
| Scale | Skipped-deprecated | Replaced by Mul |
| ParametricSoftplus | Skipped-deprecated | Legacy activation |
| ScaledTanh | Skipped-deprecated | Legacy activation |

### Reduction

| Op | Status | Rationale |
|---|---|---|
| ReduceMax | Implemented | Attention / classifier max pooling |
| ReduceMin | Implemented | Counterpart to ReduceMax |
| ReduceProd | Implemented | Shape/volume computations |
| ArgMax | Implemented | Classifier decode |
| ArgMin | Implemented | Counterpart to ArgMax |
| ReduceL1 | Planned-P2 | L1 norm reductions |
| ReduceL2 | Planned-P2 | L2 norm reductions |
| ReduceLogSum | Planned-P2 | Log-domain reductions |
| ReduceLogSumExp | Planned-P2 | Stable softmax building block |
| ReduceSumSquare | Planned-P2 | Variance / norm computation |

### Pooling

| Op | Status | Rationale |
|---|---|---|
| GlobalMaxPool | Implemented | Common classifier head |
| RoiAlign | Implemented | ViT detection / Mask R-CNN |
| MaxRoiPool | Planned-P3 | Object detection ROI pooling |
| GlobalLpPool | Planned-P4 | Rare Lp pooling |
| LpPool | Planned-P4 | Rare Lp pooling |
| MaxUnpool | Planned-P4 | Segmentation decoder; rare |

### Normalization

| Op | Status | Rationale |
|---|---|---|
| InstanceNormalization | Implemented | Style transfer / vision transformers |
| GroupNormalization | Implemented | ConvNeXt / Swin / diffusion |
| RMSNormalization | Planned-P2 | LLaMA / T5 normalization |
| LRN | Planned-P3 | Legacy AlexNet; whisper preprocess |

### Shape / Data Movement

| Op | Status | Rationale |
|---|---|---|
| Shape | Implemented | Dynamic shape queries; BERT |
| Size | Implemented | Element count queries |
| Identity | Implemented | Pass-through; constant folding |
| ConstantOfShape | Implemented | Masking / attention patterns |
| Constant | Implemented | Inline constant tensors |
| Range | Implemented | Positional indices |
| Trilu | Implemented | Causal mask for attention |
| CumSum | Implemented | Positional offsets; attention |
| GatherND | Implemented | N-dim gather; transformers |
| ScatterND | Implemented | N-dim scatter; transformers |
| GatherElements | Implemented | Element-wise gather |
| ScatterElements | Implemented | Element-wise scatter |
| TopK | Implemented | Beam search; classifier top-k |
| NonZero | Implemented | Mask indexing |
| Compress | Implemented | Boolean mask selection |
| Unique | Implemented | Deduplication |
| DepthToSpace | Implemented | Super-resolution; pixel shuffle |
| SpaceToDepth | Implemented | YOLO / ViT patchify |
| Resize | Implemented | Image scaling; ViT/U-Net |
| GridSample | Implemented | Spatial transformers / STN |
| CenterCropPad | Implemented | Vision preprocessing (opset 18+) |
| EyeLike | Planned-P2 | Identity matrix generation |
| Reverse | Planned-P4 | Rare axis reversal |
| ReverseSequence | Planned-P4 | Bi-directional RNN support |
| ImageDecoder | Planned-P4 | Raw image decode (opset 20+) |
| AffineGrid | Planned-P4 | Spatial transformer grid (opset 20+) |
| Col2Im | Planned-P4 | Inverse im2col (opset 18+) |
| Scatter | Skipped-deprecated | Replaced by ScatterElements |
| Upsample | Skipped-deprecated | Replaced by Resize |
| SplitToSequence | Deferred-subsystem | Sequence type required |
| ConcatFromSequence | Deferred-subsystem | Sequence type required |

### Convolution Variants

| Op | Status | Rationale |
|---|---|---|
| ConvTranspose | Planned-P3 | Upsampling decoder; audio |
| ConvInteger | Planned-P4 | Integer convolution; rare quant path |
| DeformConv | Planned-P4 | Deformable convolution (opset 19+) |

### Quantized

| Op | Status | Rationale |
|---|---|---|
| MatMulInteger | Planned-P2 | Int8 LLM inference |
| DynamicQuantizeLinear | Planned-P2 | Runtime quantization for LLMs |
| QLinearAdd | Skipped-vendor | `com.microsoft` contrib op |
| QLinearMul | Skipped-vendor | `com.microsoft` contrib op |
| QLinearConcat | Skipped-vendor | `com.microsoft` contrib op |
| QLinearSigmoid | Skipped-vendor | `com.microsoft` contrib op |
| QLinearLeakyRelu | Skipped-vendor | `com.microsoft` contrib op |
| QLinearAveragePool | Skipped-vendor | `com.microsoft` contrib op |
| QLinearGlobalAveragePool | Skipped-vendor | `com.microsoft` contrib op |
| QLinearSoftmax | Skipped-vendor | `com.microsoft` contrib op |

### Random / Sampling

| Op | Status | Rationale |
|---|---|---|
| RandomNormal | Planned-P2 | Generative sampling |
| RandomNormalLike | Planned-P2 | Generative sampling |
| RandomUniform | Planned-P2 | Generative sampling / dropout |
| RandomUniformLike | Planned-P2 | Generative sampling / dropout |
| Multinomial | Planned-P2 | Token sampling for LLMs |
| Bernoulli | Planned-P2 | Stochastic masking |
| Dropout | Planned-P2 | No-op at inference; must parse |

### Audio / Signal

| Op | Status | Rationale |
|---|---|---|
| STFT | Planned-P3 | Whisper spectrogram frontend |
| DFT | Planned-P3 | Spectral transforms |
| MelWeightMatrix | Planned-P3 | Mel filterbank for Whisper |
| HannWindow | Planned-P3 | Audio windowing |
| HammingWindow | Planned-P3 | Audio windowing |
| BlackmanWindow | Planned-P3 | Audio windowing |

### Object Detection

| Op | Status | Rationale |
|---|---|---|
| NonMaxSuppression | Planned-P3 | All detection models need this |

### Sequence (Deferred)

`SequenceEmpty`, `SequenceConstruct`, `SequenceAt`, `SequenceInsert`,
`SequenceErase`, `SequenceLength`, `SequenceMap` — all
**Deferred-subsystem**. Requires a Tensor-of-Tensor type and a graph
executor that can hold variable-length lists. Not on the critical path
for any current target model.

### Optional (Deferred)

`Optional`, `OptionalHasElement`, `OptionalGetElement` —
**Deferred-subsystem**. Requires a sum-type wrapper around tensors.
Used by some control-flow models; will be addressed alongside
`If`/`Loop` in P2 alongside the sub-graph executor.

### Control Flow (Phase 2)

| Op | Status | Rationale |
|---|---|---|
| If | Planned-P2 | Conditional subgraph execution; needs sub-graph executor |
| Loop | Planned-P2 | Iterative subgraph execution; needs sub-graph executor |
| Scan | Planned-P2 | Recurrent subgraph execution; needs sub-graph executor |

Phase 2 is the only phase whose design budget will be larger than its
implementation budget. The ops themselves are simple; the runtime
machinery (sub-executor, scope management, carried dependencies) is
substantial. See `openspec/changes/generative-llm-v1/` and
`docs/sub-graph-executor-design.md` for the architecture.

### String / Tokenizer (Deferred)

`StringNormalizer`, `StringConcat`, `StringSplit`, `RegexFullMatch`,
`Tokenizer` — all **Deferred-subsystem**. Requires a string-tensor
data type. Niche even in NLP since most pipelines tokenize before
hitting the ONNX graph.

### ONNX-ML Classical ML (Deferred)

`LabelEncoder`, `CategoryMapper`, `TfIdfVectorizer`, `Binarizer`,
`ArrayFeatureExtractor`, `DictVectorizer`, `FeatureVectorizer`,
`Imputer`, `LinearClassifier`, `LinearRegressor`, `Normalizer`,
`OneHotEncoder`, `SVMClassifier`, `SVMRegressor`,
`TreeEnsembleClassifier`, `TreeEnsembleRegressor`, `ZipMap`, `Scaler`
— all **Deferred-subsystem**. These live in the `ai.onnx.ml` domain
and target classical ML (trees, SVMs, feature engineering) rather
than neural networks. Out of scope for an inference runtime focused
on neural workloads, but cataloged for completeness.

### Training (Skipped)

`Adam`, `Adagrad`, `Momentum`, `Gradient`, `GraphCall`,
`SoftmaxCrossEntropyLoss`, `NegativeLogLikelihoodLoss`, `TrainingInfo`
— all **Skipped-training**. SmallAIOS is inference-only by charter.
`BatchNormalization` and `Dropout` have inference-mode paths
implemented; their training-mode behaviors are skipped.

## Phase 1 Detail — `transformer-models-v1` ‖ `vision-transformers-v1`

These two changes run in parallel agent worktrees as soon as
`additional-operators-v1` merges.

### `transformer-models-v1`

**Target models:** BERT-base (`bert-base-uncased`), DistilBERT
(`distilbert-base-uncased`), TinyBERT.

**New operators (25):**
- **Math (10):** Mod, Sin, Cos, Reciprocal, Sign, Sum, Mean, And, Or,
  LogSoftmax
- **Reduction (5):** ReduceMax, ReduceMin, ReduceProd, ArgMax, ArgMin
- **Shape/Data (10):** Shape, Size, Identity, Constant,
  ConstantOfShape, Range, Trilu, CumSum, GatherND, ScatterND

**Inventory machinery (also added in this change):**
```rust
pub enum OperatorStatus {
    Implemented,
    Planned(Phase),
    DeferredSubsystem(&'static str),
    SkippedTraining,
    SkippedDeprecated,
    SkippedVendor,
}

pub const SUPPORTED_OPS_INVENTORY: &[(&str, OperatorStatus)] = &[
    ("Add", OperatorStatus::Implemented),
    ("Sin", OperatorStatus::Planned(Phase::P1)),
    // … one entry per standard ONNX op
];
```
A unit test asserts every `Implemented` entry has a matching `OpKind`
variant and vice versa, preventing the inventory from drifting.

**Validation:** loads BERT-base end-to-end via the integration harness.

### `vision-transformers-v1`

**Target models:** ViT, Swin, DeiT, ConvNeXt, MobileViT.

**New operators (19):** Resize, DepthToSpace, SpaceToDepth, RoiAlign,
GridSample, GroupNormalization, InstanceNormalization, TopK, Compress,
NonZero, Unique, GatherElements, ScatterElements, GlobalMaxPool,
HardSigmoid, HardSwish, PRelu, Mish, CenterCropPad.

**Validation:** loads ViT-base end-to-end via the integration harness.

### Phase 1 coordination

The two changes touch separate `ops/` submodules and only collide on
`OpKind` / `parse_str` / `dispatch_node`. The append-only file rules
in the "Agent-Team Execution Playbook" section below prevent merge
conflicts; whichever PR lands first, the other rebases.

## Phase 2 Detail — `generative-llm-v1`

This is a single combined tier covering control-flow ops, generative
ops, and the real i8 GEMM kernel. It is the **breakthrough phase**:
shipping it makes SmallAIOS usable for autoregressive LLM inference.

It runs sequentially after Phase 1 because it touches the dispatcher
extensively and adds new runtime machinery (sub-graph executor) that
later phases will build on.

### Target models
- GPT-2-small (`gpt2`)
- T5-small (`t5-small`)
- LLaMA-style int8 (a small open-weight LLM in int8 quantization)

### Sub-graph executor (the design-heavy part)

**~1500 lines of new runtime work** before any op implementation:

- Sub-graph compilation and caching — each `If` branch and `Loop`
  body is its own sub-graph that needs the full topological sort,
  memory planning, and dispatcher applied to it
- Scope management — outer-graph values referenced from inside a
  loop body must be visible without copying
- Carried-state semantics for `Loop` — loop-carried dependencies and
  scan outputs follow specific iteration rules
- Iteration limits + WCET integration — `Loop` must respect the
  per-operator hard time budget across all iterations, not per
  iteration
- Termination condition evaluation per iteration

This needs its own design document inside the OpenSpec change.

### Control-flow operators (3)
`If`, `Loop`, `Scan`.

### Generative operators (~17)
RMSNormalization, MatMulInteger, DynamicQuantizeLinear, RandomNormal,
RandomNormalLike, RandomUniform, RandomUniformLike, Multinomial,
Bernoulli, Dropout, EyeLike, ReduceL1, ReduceL2, ReduceLogSum,
ReduceLogSumExp, ReduceSumSquare, LpNormalization,
MeanVarianceNormalization, Softplus.

### Real int8 kernels (no new ops)
Replaces the dequantize → f32 → requantize shim in
`op_qlinear_matmul` and `op_qlinear_conv` with a real i8 GEMM kernel
using saturating accumulation, proper output scale handling, and
zero-point folding. `MatMulInteger` gets the same kernel.

### Validation
- GPT-2-small generates a 64-token completion in a **single** ONNX
  graph call (no external Python loop)
- T5-small encoder + decoder runs end-to-end
- An int8-quantized LLM produces output within 1% relative error of
  its f32 baseline

## Phase 3 Detail — `audio-models-v1` ‖ `detection-models-v1`

Two parallel changes covering specialty model classes. Lower priority
than Phases 1-2 because the workloads are narrower than generic
NLP/vision.

### `audio-models-v1`

**Target models:** Whisper-tiny, Wav2Vec2-base.

**New operators (~13):** STFT, DFT, MelWeightMatrix, HannWindow,
HammingWindow, BlackmanWindow, ConvTranspose, Sinh, Cosh, LRN, plus
opset-14 audio-related shape adjustments.

### `detection-models-v1`

**Target models:** DETR, RetinaNet, Mask R-CNN.

**New operators (~8):** NonMaxSuppression, MaxRoiPool, MaxUnpool,
ReverseSequence, Det, plus any detection-specific shape ops not
already in P1.

## Phase 4 — `long-tail-v1` (reactive)

No fixed scope. Operators tagged `Planned-P4` in the inventory are
pulled into small, focused PRs only when a real user model fails to
load because of them. Phase 4 is **reactive maintenance**, not a
scheduled change.

When a Phase 4 PR lands:
1. Add the missing op to its appropriate `ops/` submodule
2. Update the inventory entry from `Planned-P4` to `Implemented`
3. Add the user's model to the integration test suite if practical

## Agent-Team Execution Playbook

Within a phase, changes run in **parallel worktrees** following the
established pattern:

```
../SmallAIOS-Design-worktrees/
├── transformer-models-v1/    (Phase 1 — agent A)
├── vision-transformers-v1/   (Phase 1 — agent B)
└── audio-models-v1/          (Phase 3 — agent C)
```

### File-ownership rules

| Shared file | Rule |
|---|---|
| `onnx-rt/src/ops/<submodule>.rs` | Single tier owns the file. New tier creates a new file when possible. |
| `onnx-rt/src/operators.rs` (`OpKind`, `parse_str`, `name`, `ALL_OPS`) | **Append-only.** Merges sequenced. Later tiers rebase. |
| `onnx-rt/src/executor.rs` (dispatch helpers) | **Append-only.** Each tier adds new dispatch helpers; existing ones are not edited. |
| `onnx-rt/src/profile.rs` (`classify_op`) | **Append-only.** Add to the existing match arm with new op names. |
| `docs/onnx-coverage-roadmap.md` | Updated by every change — flip status from `Planned-Pn` to `Implemented`. |
| `SUPPORTED_OPS_INVENTORY` (T3+) | Append/update inline. CI test enforces consistency. |

### Per-tier validation gates

Every tier PR must pass before merging to `develop`:

1. `just fmt` clean
2. `just clippy --all-targets` clean (with `-D warnings`)
3. `just test` all green, including any new unit tests
4. `openspec validate <tier-name>` clean
5. `cargo bench` no regressions on Tier 1 ops (sanity)
6. The tier's target model loads end-to-end via the integration
   harness (`container/tests/e2e_*.rs` style)
7. Roadmap document updated to reflect new statuses
8. PR title follows conventional commits with semver scope

### Agent kickoff checklist

When starting an agent on a tier:

1. Create a worktree from `develop`:
   `git worktree add ../SmallAIOS-Design-worktrees/<tier> change/<tier>`
2. Run `openspec new change "<tier>"`
3. Read the relevant tier section of this roadmap and copy the op list
   into the change's `tasks.md`
4. Read the previously merged tier's PR for the dispatch-wiring pattern
5. Implement each op in its own commit if practical, or one commit
   per category if the ops are tiny
6. Run the validation gates above before opening the PR
7. Update this roadmap in the same PR — flip Planned-Pn → Implemented
   for every shipped op

### Coordination

Tiers that share files (3 + 4, or 5 + 8) sequence their merges through
the OpenSpec change queue. Independent tiers (e.g., T3 and T6) can run
truly in parallel because their ops live in different submodules and
their dispatch additions are append-only.

If a merge conflict arises, the **earlier-numbered tier wins** the
merge slot and the later tier rebases. This rule keeps merge order
deterministic.

## GPU operator coverage (Apple Metal)

The CPU operator inventory above is the canonical "what runs on
SmallAIOS" list. GPU coverage is a separate axis: it tells you which
of those CPU ops also have a Metal kernel that the runtime will
dispatch to when a `Session` is built with `.with_gpu(GpuConfig::metal())`.
Unsupported ops fall back to the CPU path transparently within the same
graph.

Owning OpenSpec change: `metal-operator-kernels-v1` (archived). All
shaders live in `arch/apple/src/shaders.rs`; dispatch plans in
`onnx-rt/src/metal_dispatch.rs`.

### Tier 1 — elementwise + matmul + softmax + conv (11 ops)

Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Gemm, Softmax, Conv.

Comparison tests vs. CPU live in `onnx-rt/src/metal_dispatch.rs`
(macOS-gated). MatMul has a tiled shared-memory variant that runs on
M1/M2 (no `simdgroup_matrix`); the M3+ path uses `simdgroup_matrix` via
`#if __METAL_VERSION__ >= 310`.

### Tier 2 — transformer ops (8 ops)

LayerNormalization, RMSNormalization, RotaryEmbedding,
ScaledDotProductAttention, GroupQueryAttention, BatchNormalization,
GEMM(int8) with `simdgroup_matrix` on M3+ and shared-mem fallback on
M1/M2, GEMM(f16).

### What's not on the GPU yet

Reductions (ReduceMean, ReduceSum, …), Cast, reshape-family
(Reshape, Transpose, Flatten, Squeeze, Unsqueeze), Concat, Gather,
Slice, Pad, GlobalAveragePool, MaxPool, AveragePool, Clip,
quantize/dequantize, and all P1+ ops fall back to CPU. They will
land alongside their respective phases as needed by the target
workloads.

### M1/M2 vs. M3+

Detected at runtime via `MTLDevice` family. M3+ uses
`simdgroup_matrix` for matmul/gemm; M1/M2 uses shared-memory tiling.
The dispatch layer is identical — the difference is only inside the
shader.

## Review checkpoints

This document must be reviewed:

- **Before each release** — flip statuses, sanity-check counts.
- **After each ONNX opset bump** — add any new ops to the inventory
  with a default status of `Planned-P4`.
- **After each tier merges** — verify the cumulative count and
  percentage in the tier table match reality.

A future Phase 1 follow-up will add a `just check-roadmap` recipe that
diffs the roadmap inventory against `OperatorRegistry` and fails CI
on drift.
