# ONNX Operator Coverage Roadmap

This is the canonical plan for SmallAIOS's ONNX operator coverage. Every
future operator-coverage OpenSpec change must slot into a tier defined
here. Cite this document from each tier proposal.

**Last reviewed:** 2026-04-11
**Current coverage:** 65 / ~190 standard ops (~34%)
**Owning OpenSpec change:** `onnx-full-coverage-roadmap-v1`

## Goals

- Catalog **every** standard ONNX operator (opset 21) so nothing slips
  through ad-hoc per-model work.
- Drive coverage by **target model class**, not by spec compliance —
  every tier ships a real, demoable workload.
- Make "remaining work" a queryable, testable number rather than
  folklore.
- Be honest about what we will **not** implement (training ops,
  ecosystem ops, sequence/optional/control-flow subsystems).

## Tier Sequence

| Tier | OpenSpec change | Target models | Op delta | Cumulative | % of ONNX |
|---|---|---|---|---|---|
| 1 ✅ | `onnx-cpu-runtime-v1` (archived) | ResNet, MobileNet, YOLO | 29 | 29 | 15% |
| 2 ✅ | `additional-operators-v1` | Transformer/recurrent/quant building blocks | +36 | 65 | 34% |
| 3 | `transformer-models-v1` | BERT-base, DistilBERT, TinyBERT | +18 | 83 | 44% |
| 4 | `vision-transformers-v1` | ViT, Swin, DeiT, ConvNeXt | +14 | 97 | 51% |
| 5 | `generative-models-v1` | GPT-2-small, T5-small, LLaMA-style | +18 | 115 | 60% |
| 6 | `audio-models-v1` | Whisper-tiny, Wav2Vec2 | +13 | 128 | 67% |
| 7 | `detection-models-v1` | DETR, RetinaNet, Mask R-CNN | +8 | 136 | 72% |
| 8 | `int8-kernels-v1` | (real i8 GEMM/Conv perf, no new ops) | 0 | 136 | 72% |
| 9 | `control-flow-v1` | Models w/ If/Loop/Scan + sequence types | +3 + subsystems | 139 | 73% |
| 10 | `long-tail-completion-v1` | Reactive, user-driven | +~25 | ~164 | ~86% |

The remaining ~26 ops are explicitly **Deferred-subsystem** (sequence,
optional, string, tokenizer, ONNX-ML classical ML) or **Skipped**
(training, deprecated, vendor). They are not in the path to 100%
unless a real user model demands one of them.

## Operator Inventory

The inventory tracks every standard operator with one of:

- **Implemented** — works today.
- **Planned-T<n>** — will land in tier `n`.
- **Deferred-subsystem** — needs a non-trivial subsystem we haven't
  built yet (sequence types, optional types, control-flow executor,
  string tensors).
- **Skipped-training** — training-only op; SmallAIOS is inference-only.
- **Skipped-deprecated** — superseded by a newer op; converters
  rewrite at export time.
- **Skipped-vendor** — non-standard op (e.g., `com.microsoft` domain).

The Tier 3 change (`transformer-models-v1`) will encode this inventory
as a Rust constant `SUPPORTED_OPS_INVENTORY` with a CI test that fails
if the implemented set drifts from the inventory.

### Implemented (65)

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
| Mod | Planned-T3 | Integer/float modulo; positional encoding |
| Sin | Planned-T3 | Sinusoidal positional embeddings |
| Cos | Planned-T3 | Sinusoidal positional embeddings |
| Reciprocal | Planned-T3 | 1/x; LayerNorm and attention scaling |
| Sign | Planned-T3 | Sign extraction; quant + transformers |
| Sum | Planned-T3 | Variadic elementwise sum |
| Mean | Planned-T3 | Variadic elementwise mean |
| And | Planned-T3 | Boolean mask composition |
| Or | Planned-T3 | Boolean mask composition |
| LogSoftmax | Planned-T3 | Common classifier head |
| PRelu | Planned-T4 | Parametric ReLU; older vision nets |
| HardSigmoid | Planned-T4 | MobileNet/EfficientNet activation |
| HardSwish | Planned-T4 | MobileNetV3 activation |
| Softplus | Planned-T5 | Smooth ReLU; generative models |
| Sinh | Planned-T6 | Hyperbolic trig; audio models |
| Cosh | Planned-T6 | Hyperbolic trig; audio models |
| Tan | Planned-T10 | Rare in inference |
| Asin | Planned-T10 | Rare inverse trig |
| Acos | Planned-T10 | Rare inverse trig |
| Atan | Planned-T10 | Rare inverse trig |
| Asinh | Planned-T10 | Long tail |
| Acosh | Planned-T10 | Long tail |
| Atanh | Planned-T10 | Long tail |
| Xor | Planned-T10 | Rare boolean op |
| BitwiseAnd | Planned-T10 | Integer bitwise (opset 18+) |
| BitwiseOr | Planned-T10 | Integer bitwise (opset 18+) |
| BitwiseXor | Planned-T10 | Integer bitwise (opset 18+) |
| BitwiseNot | Planned-T10 | Integer bitwise (opset 18+) |
| BitShift | Planned-T10 | Integer bit shift |
| ThresholdedRelu | Planned-T10 | Rare activation |
| Selu | Planned-T10 | Rare self-normalizing activation |
| Celu | Planned-T10 | Rare continuous ELU |
| Softsign | Planned-T10 | Rare activation |
| Shrink | Planned-T10 | Soft thresholding; rare |
| Hardmax | Planned-T10 | One-hot argmax variant |
| IsNaN | Planned-T10 | Debug/validation op |
| IsInf | Planned-T10 | Debug/validation op |
| MeanVarianceNormalization | Planned-T5 | MVN normalization |
| LpNormalization | Planned-T5 | L1/L2 normalization layer |
| Affine | Skipped-deprecated | Replaced by Mul+Add |
| ImageScaler | Skipped-deprecated | Replaced by Mul+Add |
| Scale | Skipped-deprecated | Replaced by Mul |
| ParametricSoftplus | Skipped-deprecated | Legacy activation |
| ScaledTanh | Skipped-deprecated | Legacy activation |

### Reduction

| Op | Status | Rationale |
|---|---|---|
| ReduceMax | Planned-T3 | Attention / classifier max pooling |
| ReduceMin | Planned-T3 | Counterpart to ReduceMax |
| ReduceProd | Planned-T3 | Shape/volume computations |
| ArgMax | Planned-T3 | Classifier decode |
| ArgMin | Planned-T3 | Counterpart to ArgMax |
| ReduceL1 | Planned-T5 | L1 norm reductions |
| ReduceL2 | Planned-T5 | L2 norm reductions |
| ReduceLogSum | Planned-T5 | Log-domain reductions |
| ReduceLogSumExp | Planned-T5 | Stable softmax building block |
| ReduceSumSquare | Planned-T5 | Variance / norm computation |

### Pooling

| Op | Status | Rationale |
|---|---|---|
| GlobalMaxPool | Planned-T4 | Common classifier head |
| RoiAlign | Planned-T4 | ViT detection / Mask R-CNN |
| MaxRoiPool | Planned-T7 | Object detection ROI pooling |
| GlobalLpPool | Planned-T10 | Rare Lp pooling |
| LpPool | Planned-T10 | Rare Lp pooling |
| MaxUnpool | Planned-T10 | Segmentation decoder; rare |

### Normalization

| Op | Status | Rationale |
|---|---|---|
| InstanceNormalization | Planned-T4 | Style transfer / vision transformers |
| GroupNormalization | Planned-T4 | ConvNeXt / Swin / diffusion |
| RMSNormalization | Planned-T5 | LLaMA / T5 normalization |
| LRN | Planned-T6 | Legacy AlexNet; whisper preprocess |

### Shape / Data Movement

| Op | Status | Rationale |
|---|---|---|
| Shape | Planned-T3 | Dynamic shape queries; BERT |
| Size | Planned-T3 | Element count queries |
| Identity | Planned-T3 | Pass-through; constant folding |
| ConstantOfShape | Planned-T3 | Masking / attention patterns |
| Constant | Planned-T3 | Inline constant tensors |
| Range | Planned-T3 | Positional indices |
| Trilu | Planned-T3 | Causal mask for attention |
| CumSum | Planned-T3 | Positional offsets; attention |
| GatherND | Planned-T3 | N-dim gather; transformers |
| ScatterND | Planned-T3 | N-dim scatter; transformers |
| GatherElements | Planned-T4 | Element-wise gather |
| ScatterElements | Planned-T4 | Element-wise scatter |
| TopK | Planned-T4 | Beam search; classifier top-k |
| NonZero | Planned-T4 | Mask indexing |
| Compress | Planned-T4 | Boolean mask selection |
| Unique | Planned-T4 | Deduplication |
| DepthToSpace | Planned-T4 | Super-resolution; pixel shuffle |
| SpaceToDepth | Planned-T4 | YOLO / ViT patchify |
| Resize | Planned-T4 | Image scaling; ViT/U-Net |
| GridSample | Planned-T4 | Spatial transformers / STN |
| CenterCropPad | Planned-T4 | Vision preprocessing (opset 18+) |
| EyeLike | Planned-T5 | Identity matrix generation |
| Reverse | Planned-T10 | Rare axis reversal |
| ReverseSequence | Planned-T10 | Bi-directional RNN support |
| ImageDecoder | Planned-T10 | Raw image decode (opset 20+) |
| AffineGrid | Planned-T10 | Spatial transformer grid (opset 20+) |
| Col2Im | Planned-T10 | Inverse im2col (opset 18+) |
| Scatter | Skipped-deprecated | Replaced by ScatterElements |
| Upsample | Skipped-deprecated | Replaced by Resize |
| SplitToSequence | Deferred-subsystem | Sequence type required |
| ConcatFromSequence | Deferred-subsystem | Sequence type required |

### Convolution Variants

| Op | Status | Rationale |
|---|---|---|
| ConvTranspose | Planned-T6 | Upsampling decoder; audio |
| ConvInteger | Planned-T10 | Integer convolution; rare quant path |
| DeformConv | Planned-T10 | Deformable convolution (opset 19+) |

### Quantized

| Op | Status | Rationale |
|---|---|---|
| MatMulInteger | Planned-T5 | Int8 LLM inference |
| DynamicQuantizeLinear | Planned-T5 | Runtime quantization for LLMs |
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
| RandomNormal | Planned-T5 | Generative sampling |
| RandomNormalLike | Planned-T5 | Generative sampling |
| RandomUniform | Planned-T5 | Generative sampling / dropout |
| RandomUniformLike | Planned-T5 | Generative sampling / dropout |
| Multinomial | Planned-T5 | Token sampling for LLMs |
| Bernoulli | Planned-T5 | Stochastic masking |
| Dropout | Planned-T5 | No-op at inference; must parse |

### Audio / Signal

| Op | Status | Rationale |
|---|---|---|
| STFT | Planned-T6 | Whisper spectrogram frontend |
| DFT | Planned-T6 | Spectral transforms |
| MelWeightMatrix | Planned-T6 | Mel filterbank for Whisper |
| HannWindow | Planned-T6 | Audio windowing |
| HammingWindow | Planned-T6 | Audio windowing |
| BlackmanWindow | Planned-T6 | Audio windowing |

### Object Detection

| Op | Status | Rationale |
|---|---|---|
| NonMaxSuppression | Planned-T7 | All detection models need this |

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
`If`/`Loop` in Tier 9 if at all.

### Control Flow (Deferred / Tier 9)

| Op | Status | Rationale |
|---|---|---|
| If | Tier 9 (subsystem) | Conditional subgraph execution |
| Loop | Tier 9 (subsystem) | Iterative subgraph execution |
| Scan | Tier 9 (subsystem) | Recurrent subgraph execution |

Tier 9 is the only tier whose design budget will be larger than its
implementation budget. The ops themselves are simple; the runtime
machinery (sub-executor, scope management, carried dependencies) is
substantial.

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

## Tier 3 Detailed Plan — `transformer-models-v1`

This is the next tier and is fully scoped here so an agent team can
start immediately after this roadmap merges.

### Target models
- BERT-base (`bert-base-uncased`)
- DistilBERT (`distilbert-base-uncased`)
- TinyBERT (`huawei-noah/TinyBERT_General_4L_312D`)

### New operators (18)
- **Math (10):** Mod, Sin, Cos, Reciprocal, Sign, Sum, Mean, And, Or,
  LogSoftmax
- **Reduction (5):** ReduceMax, ReduceMin, ReduceProd, ArgMax, ArgMin
- **Shape/Data (10):** Shape, Size, Identity, Constant,
  ConstantOfShape, Range, Trilu, CumSum, GatherND, ScatterND

(Note: Tier 3 actually adds 25 ops; the cumulative table above used a
conservative 18 to leave headroom. Refine when the change opens.)

### Inventory machinery
Tier 3 adds:
```rust
pub enum OperatorStatus {
    Implemented,
    Planned(Tier),
    DeferredSubsystem(&'static str),
    SkippedTraining,
    SkippedDeprecated,
    SkippedVendor,
}

pub const SUPPORTED_OPS_INVENTORY: &[(&str, OperatorStatus)] = &[
    ("Add", OperatorStatus::Implemented),
    ("Sin", OperatorStatus::Planned(Tier::T3)),
    // … one entry per standard ONNX op
];
```
A unit test asserts every `Implemented` entry has a matching `OpKind`
variant and vice versa.

## Tier 4 Sketch — `vision-transformers-v1`

**Target models:** ViT, Swin, DeiT, ConvNeXt, MobileViT

**New ops (~14):** Resize, DepthToSpace, SpaceToDepth, RoiAlign,
GridSample, GroupNormalization, InstanceNormalization, TopK, Compress,
NonZero, Unique, GatherElements, ScatterElements, GlobalMaxPool,
HardSigmoid, HardSwish, PRelu, CenterCropPad

## Tier 5 Sketch — `generative-models-v1`

**Target models:** GPT-2-small, T5-small, LLaMA-style int8 (with Tier 8)

**New ops (~18):** RMSNormalization, MatMulInteger, DynamicQuantizeLinear,
RandomNormal/NormalLike/Uniform/UniformLike, Multinomial, Bernoulli,
Dropout, EyeLike, ReduceL1/L2/LogSum/LogSumExp/SumSquare,
LpNormalization, MeanVarianceNormalization, Softplus

## Tier 6 Sketch — `audio-models-v1`

**Target models:** Whisper-tiny, Wav2Vec2-base

**New ops (~13):** STFT, DFT, MelWeightMatrix, Hann/Hamming/Blackman
windows, ConvTranspose, Sinh, Cosh, LRN, plus opset-14 BatchNorm
adjustments for audio preprocessing

## Tier 7 Sketch — `detection-models-v1`

**Target models:** DETR, RetinaNet, Mask R-CNN

**New ops (~8):** NonMaxSuppression, MaxRoiPool, MaxUnpool,
ReverseSequence, Det, EyeLike (if not in T5), MeanVarianceNormalization
(if not in T5), and any detection-specific shape ops

## Tier 8 — `int8-kernels-v1`

**No new ops.** This tier replaces the dequantize→f32→requantize
shim in `op_qlinear_matmul` and `op_qlinear_conv` with a real i8 GEMM
kernel using saturating arithmetic and proper output scale handling.
Adds `MatMulInteger`/`DynamicQuantizeLinear` perf paths if those landed
in Tier 5.

## Tier 9 — `control-flow-v1`

**New ops (3):** If, Loop, Scan
**New subsystems:** sub-graph executor, scope/carried-state management,
optional/sequence type wrappers if needed

This is the only tier with significant runtime-architecture work.
The proposal will need its own design document covering:
- How sub-graphs are compiled and cached
- Scope rules for outer-graph values referenced inside loop bodies
- Iteration carry semantics for Loop and Scan
- Termination conditions and bound limits (WCET budget interaction)

## Tier 10 — `long-tail-completion-v1`

Reactive tier. Operators are pulled from the "Planned-T10" inventory
on demand when a real user model fails to load. No fixed scope; merge
when no user-facing op gaps remain in the active model targets.

## Agent-Team Execution Playbook

Tiers run in **parallel worktrees** following the established pattern:

```
../SmallAIOS-Design-worktrees/
├── transformer-models-v1/    (Tier 3 — agent A)
├── vision-transformers-v1/   (Tier 4 — agent B)
└── audio-models-v1/          (Tier 6 — agent C)
```

### File-ownership rules

| Shared file | Rule |
|---|---|
| `onnx-rt/src/ops/<submodule>.rs` | Single tier owns the file. New tier creates a new file when possible. |
| `onnx-rt/src/operators.rs` (`OpKind`, `parse_str`, `name`, `ALL_OPS`) | **Append-only.** Merges sequenced. Later tiers rebase. |
| `onnx-rt/src/executor.rs` (dispatch helpers) | **Append-only.** Each tier adds new dispatch helpers; existing ones are not edited. |
| `onnx-rt/src/profile.rs` (`classify_op`) | **Append-only.** Add to the existing match arm with new op names. |
| `docs/onnx-coverage-roadmap.md` | Updated by every tier — change status from `Planned-Tn` to `Implemented`. |
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
7. Update this roadmap in the same PR — flip Planned-Tn → Implemented
   for every shipped op

### Coordination

Tiers that share files (3 + 4, or 5 + 8) sequence their merges through
the OpenSpec change queue. Independent tiers (e.g., T3 and T6) can run
truly in parallel because their ops live in different submodules and
their dispatch additions are append-only.

If a merge conflict arises, the **earlier-numbered tier wins** the
merge slot and the later tier rebases. This rule keeps merge order
deterministic.

## Review checkpoints

This document must be reviewed:

- **Before each release** — flip statuses, sanity-check counts.
- **After each ONNX opset bump** — add any new ops to the inventory
  with a default status of `Planned-T10`.
- **After each tier merges** — verify the cumulative count and
  percentage in the tier table match reality.

A future Tier 3 task will add a `just check-roadmap` recipe that
diffs the roadmap inventory against `OperatorRegistry` and fails CI
on drift.
