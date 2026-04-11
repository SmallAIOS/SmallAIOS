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

### D1: Phased plan, not a long tier queue

We commit to a **4-phase** plan where each phase delivers a concrete
model capability and the changes within a phase run in parallel agent
worktrees:

| Phase | Changes (parallel within phase) | Target models | Op delta | Cumulative | % of ONNX |
|---|---|---|---|---|---|
| Done ✅ | `onnx-cpu-runtime-v1`, `additional-operators-v1` | CNNs + transformer/recurrent/quant building blocks | 65 | 65 | 34% |
| **P1** | `transformer-models-v1` ‖ `vision-transformers-v1` | BERT, DistilBERT, ViT, Swin, DeiT | +39 | 104 | 55% |
| **P2** | `generative-llm-v1` (combined: control-flow + generative ops + i8 kernels) | GPT-2, T5, single-pass autoregressive generation, real i8 LLM inference | +21 + sub-graph executor | 125 | 66% |
| **P3** | `audio-models-v1` ‖ `detection-models-v1` | Whisper-tiny, Wav2Vec2, DETR, RetinaNet | +21 | 146 | 77% |
| **P4** | `long-tail-v1` (reactive) | User-driven additions when models fail to load | +~30 | ~176 | ~93% |

The remaining ~14 ops are deliberately not on the path: training-only,
deprecated, vendor-specific, or requiring subsystems (sequence types,
string tensors, ONNX-ML classical ML) that no current target model
needs.

**Rationale for the phase shape (vs. the original 10-tier sequence):**

1. The only **material** missing capability in standard ONNX is
   `Loop`/`If`/`Scan`. Without `Loop`, autoregressive generation
   cannot run as a single graph call. Everything else we marked
   "deferred" is dead code, vendor-specific, or a separate workload.
2. `int8-kernels` is **performance for LLMs**, not coverage. It only
   matters once LLMs are running, so it belongs **with** the
   generative tier, not as a separate later tier.
3. `transformer-models` (BERT) and `vision-transformers` (ViT) touch
   independent submodules and can run in **true parallel** worktrees.
4. Bundling control-flow + generative ops + i8 kernels into one
   Phase 2 prevents the half-finished state where we have generative
   ops but still need an external Python loop.
5. Long-tail ops are reactive maintenance, not a scheduled change.

Every phase delivers a working demo. We can stop at any phase
boundary and have shipped value.

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

Within a phase, changes run in **parallel worktrees** following the
established pattern:

```
../SmallAIOS-Design-worktrees/
├── transformer-models-v1/    (Phase 1 — agent A)
├── vision-transformers-v1/   (Phase 1 — agent B)
└── audio-models-v1/          (Phase 3 — agent C)
```

Each worktree branches from `develop` (or from the previous phase's
merged state). Changes within a phase share `OpKind` and
`dispatch_node` but follow append-only rules so merges are trivial.
Phases run sequentially: P2 cannot start until P1 has stabilized the
dispatcher, and P3 benefits from any cleanup P2 introduces.

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

### D4: Phase 1 op catalogs

#### `transformer-models-v1` (~25 ops)
- **Math (10):** Mod, Sin, Cos, Reciprocal, Sign, Sum, Mean, And, Or,
  LogSoftmax
- **Reduction (5):** ReduceMax, ReduceMin, ReduceProd, ArgMax, ArgMin
- **Shape/Data (10):** Shape, Size, Identity, Constant,
  ConstantOfShape, Range, Trilu, CumSum, GatherND, ScatterND
- Plus the `OperatorStatus` enum and `SUPPORTED_OPS_INVENTORY`
  constant from D2
- **Validates by loading BERT-base end-to-end**

#### `vision-transformers-v1` (~14 ops)
Resize, DepthToSpace, SpaceToDepth, RoiAlign, GridSample,
GroupNormalization, InstanceNormalization, TopK, Compress, NonZero,
Unique, GatherElements, ScatterElements, GlobalMaxPool, HardSigmoid,
HardSwish, PRelu, CenterCropPad.
- **Validates by loading ViT-base end-to-end**

### D5: Phase 2 catalog — `generative-llm-v1`

Combined tier with three sections:

1. **Sub-graph executor** (~1500 lines of runtime work) — sub-graph
   compilation/caching, scope management, carried-state for `Loop`,
   iteration limits with WCET integration. This needs its own design
   document inside the change.
2. **Control-flow ops (3):** If, Loop, Scan.
3. **Generative ops (~17):** RMSNormalization, MatMulInteger,
   DynamicQuantizeLinear, RandomNormal/Like, RandomUniform/Like,
   Multinomial, Bernoulli, Dropout, EyeLike, ReduceL1/L2/LogSum/
   LogSumExp/SumSquare, LpNormalization, MeanVarianceNormalization,
   Softplus.
4. **Real i8 GEMM kernel** replacing the dequantize→f32→requantize
   shim from `additional-operators-v1`.

**Validates by:** GPT-2-small generates a 64-token completion in a
single ONNX graph call (no external Python loop), and an int8 LLM
produces output within 1% relative error of its f32 baseline.

### D6: Phase 3 sketches

- **`audio-models-v1` (~13 ops):** STFT, DFT, MelWeightMatrix,
  Hann/Hamming/Blackman windows, ConvTranspose, Sinh, Cosh, LRN.
  Validates with Whisper-tiny.
- **`detection-models-v1` (~8 ops):** NonMaxSuppression, MaxRoiPool,
  MaxUnpool, ReverseSequence, Det. Validates with DETR.

### D7: Explicitly deferred / skipped operators

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
