# Design — ONNX Full Operator Coverage Roadmap

## Context

The ONNX standard operator set as of opset 21 contains roughly 190
operators. SmallAIOS currently implements 65 (Tier 1 = 29 classic CNN
ops, Tier 2 = 36 transformer/recurrent/quantized building blocks).
Naïvely targeting "all 190" is the wrong frame because:

- ~25 ops are **training-only** (Adam, Momentum, Gradient, GraphCall,
  …) and irrelevant to an inference runtime.
- ~10 ops are **deprecated** or have superseded variants.
- ~15 ops require **subsystems we don't yet have** (control-flow,
  sequence types, optional types) and need their own design effort.
- The remaining ~115 ops can be cleanly grouped by **what model they
  unlock**, which gives us a natural delivery rhythm.

This roadmap turns "implement everything" into "implement the next
batch of models", which is testable, demoable, and reviewable.

## Decisions

### D1: Tiered model-driven phasing

We commit to a **tier sequence** where each tier targets a concrete
model class and ships when that class loads end-to-end. The full
sequence:

| Tier | OpenSpec change | Target models | Op delta | Cumulative | % of ONNX |
|---|---|---|---|---|---|
| 1 ✅ | `onnx-cpu-runtime-v1` (archived) | ResNet, MobileNet, YOLO | 29 | 29 | 15% |
| 2 ✅ | `additional-operators-v1` (this PR) | Building blocks | +36 | 65 | 34% |
| 3 | `transformer-models-v1` | BERT-base, DistilBERT | +18 | 83 | 44% |
| 4 | `vision-transformers-v1` | ViT, Swin, DeiT | +12 | 95 | 50% |
| 5 | `generative-models-v1` | GPT-2-small, T5-small | +14 | 109 | 57% |
| 6 | `audio-models-v1` | Whisper-tiny, Wav2Vec2 | +14 | 123 | 65% |
| 7 | `detection-models-v1` | DETR, RetinaNet | +10 | 133 | 70% |
| 8 | `int8-kernels-v1` | (real i8 GEMM, no new ops) | 0 | 133 | 70% |
| 9 | `control-flow-v1` | Models w/ If/Loop/Scan | +3 + subsystem | 136 | 72% |
| 10 | `long-tail-completion-v1` | Reactive, user-driven | +~50 | 186 | 98% |

Roughly 98% of the standard spec is achievable. The final ~2% (training
ops, deprecated variants, exotic string/sequence ops) is **explicitly
skipped** unless a real user model demands it.

**Rationale:** every tier delivers a working demo. We can stop at any
tier and have shipped value. The alternative — implementing ops in
spec order — produces months of work with no observable improvement.

### D2: Operator inventory as source of truth

We add a small Rust function `OperatorRegistry::status(name) ->
OperatorStatus` returning one of:

```rust
pub enum OperatorStatus {
    Implemented,        // works today
    Planned(Tier),      // listed for a future tier
    Deferred,           // intentionally not on the roadmap (e.g. training)
    Skipped,            // will not implement (deprecated, vendor)
}
```

Plus a `SUPPORTED_OPS_INVENTORY: &[(&str, OperatorStatus)]` constant
listing every standard ONNX op. This becomes the canonical inventory
that tools, docs, and CI can introspect. A unit test verifies the
inventory matches the actual implemented set so the two cannot drift.

**Rationale:** without a single source of truth, "remaining work"
becomes a folklore number. Encoding it in code makes it queryable,
testable, and auditable.

**Note:** the inventory itself is added in Tier 3 (`transformer-models-
v1`), not in this roadmap PR. The roadmap only declares the *intent*
for Tier 3 to add it.

### D3: Agent-team execution model

Tiers run in **parallel worktrees** following the established pattern:

```
../SmallAIOS-Design-worktrees/
├── transformer-models-v1/    (Tier 3 — agent A)
├── vision-transformers-v1/   (Tier 4 — agent B)
└── audio-models-v1/          (Tier 6 — agent C)
```

Each worktree branches from `develop`. Tiers that share files (e.g.,
`OpKind` enum, `dispatch_node`) sequence their merges to `develop`
to avoid conflicts. Independent additions (separate `ops/` submodules)
can run truly in parallel.

**File-ownership rules** (added to the agent-team playbook):

| Shared file | Ownership during a tier |
|---|---|
| `onnx-rt/src/ops/<submodule>.rs` | Single agent owns the file. |
| `onnx-rt/src/operators.rs` (OpKind enum, parse_str, name) | One PR at a time touches; later tiers rebase. |
| `onnx-rt/src/executor.rs` (dispatch helpers) | Append-only per tier; new helpers, no edits to existing ones. |
| `onnx-rt/src/profile.rs` (classify_op) | Append-only per tier. |

**Rationale:** the friction in parallel ops work isn't the
implementation — it's the few shared files. The append-only rule keeps
merges trivial.

### D4: Tier 3 op catalog (transformer-models-v1)

To anchor the roadmap, the next tier's op list is fully enumerated here
so an agent team can start immediately after this roadmap merges:

**Math/elementwise (5):** Mod, Sin, Cos, Reciprocal, Sign

**Reductions (3):** ReduceMax, ReduceMin, ReduceProd

**Shape/data (10):** Shape, Size, Identity, ConstantOfShape, Range,
Trilu, CumSum, ScatterND, GatherND, ArgMax

These 18 ops + the operator-status-inventory machinery from D2 are
sufficient to load BERT-base end-to-end.

### D5: Tier catalogs for Tiers 4-7 (sketch)

Detailed enumeration is left to each tier's own OpenSpec change, but
the roadmap names the targets so agents can pre-stage work:

- **Tier 4 — vision-transformers-v1:** Resize, Upsample, DepthToSpace,
  SpaceToDepth, RoiAlign, GridSample, GroupNormalization, InstanceNorm,
  TopK, Compress, NonZero, Unique
- **Tier 5 — generative-models-v1:** ArgMin, LogSoftmax, HardSigmoid,
  HardSwish, HardMax, PRelu, Mish, Softplus, Softsign, Selu,
  ThresholdedRelu, Celu, Shrink, Bernoulli (sampling)
- **Tier 6 — audio-models-v1:** STFT, DFT, MelWeightMatrix,
  BlackmanWindow, HannWindow, HammingWindow, ConvTranspose, Sinh, Cosh,
  Tanh variants, Asinh, Acosh, Atanh, LRN
- **Tier 7 — detection-models-v1:** NonMaxSuppression, MaxRoiPool,
  MaxUnpool, ReverseSequence, ScatterElements, GatherElements, Det,
  EyeLike, MeanVarianceNormalization, LpNormalization

### D6: Explicitly deferred / skipped operators

These ops are listed in the inventory but will **not** receive an
implementation slot unless a future model target demands one:

**Training-only (skip permanently):** Gradient, Adam, Momentum,
GraphCall, BatchNormalization (training mode), Dropout (training mode)

**Sequence types (deferred — needs subsystem design):**
SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert,
SequenceLength, SequenceMap, SequenceAt, SplitToSequence,
ConcatFromSequence

**Optional types (deferred — needs subsystem design):** Optional,
OptionalGetElement, OptionalHasElement

**String ops (deferred — niche):** RegexFullMatch, StringNormalizer,
StringConcat, StringSplit, ImageDecoder

**Random ops (Tier 5 covers Bernoulli; rest deferred):** Multinomial,
RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike

This list is expected to shrink as user needs emerge. Each item is
labeled with the rationale in the inventory constant.

## Alternatives Considered

### A1: Spec-order implementation

Implement operators in alphabetical or opset order. Rejected: produces
no shippable milestones, no model-driven validation, and risks
implementing rare ops before common ones.

### A2: Single mega-change for all remaining ops

One OpenSpec change containing 100+ operator implementations.
Rejected: PR would be unreviewable (~10k lines), couldn't be tested
incrementally, and would block all other work for weeks.

### A3: Reactive-only (no roadmap)

Implement ops only when a user reports a model failure. Rejected:
produces inconsistent coverage, blocks demo work, and has no way to
predict effort.

### A4: Adopt an external runtime (ONNX Runtime, tract)

Vendor in an existing Rust ONNX runtime. Rejected per project
charter: SmallAIOS is `#![no_std]` clean-room, and the existing
runtimes either depend on `std`/C bindings or violate the size
budget. Confirmed in `CLAUDE.md`.

## Open Questions

1. **Do we want a CI gate that fails when a new opset is released?**
   Probably yes — it prevents the inventory from silently going stale.
   To be specified in Tier 3.
2. **How are control-flow ops (`If`, `Loop`, `Scan`) implemented?**
   They need a sub-graph executor. Tier 9 will spend most of its
   design budget on this, not on the ops themselves.
3. **GPU coverage parity.** This roadmap is CPU-only. The GPU
   operator-coverage roadmap is a separate workstream and should be
   stubbed out under the existing compute-abstraction OpenSpec
   eventually.
