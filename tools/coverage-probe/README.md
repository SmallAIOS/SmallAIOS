# smallaios-coverage-probe

Structural coverage analysis for ONNX models against the SmallAIOS
runtime roadmap.

## What it does

Given an ONNX model file, this tool:

1. Parses the `ModelProto` using the in-tree `smallaios-onnx-rt`
   protobuf parser (no external ONNX dependency).
2. Tallies every op kind in the graph.
3. Cross-references each op against two sources of truth:
   - the live `OperatorRegistry` from `smallaios-onnx-rt` — what the
     runtime *actually* supports today,
   - `docs/onnx-coverage-roadmap.md` — what we plan to support, and
     when.
4. Emits a Markdown (default), text, or JSON report showing which ops
   are implemented, which are planned in each phase, which are deferred
   or skipped, and which are unrecognized entirely.

The tool is used to **empirically validate** that Phase 1 actually
covers BERT / ViT and that Phase 2 actually covers GPT-2 / Llama
**before** spending implementation effort on those phases.

## Build

```bash
cargo build -p smallaios-coverage-probe
cargo test  -p smallaios-coverage-probe
```

## Usage

```bash
smallaios-coverage-probe model.onnx
smallaios-coverage-probe --json model.onnx
smallaios-coverage-probe --detailed model.onnx
smallaios-coverage-probe --strict model.onnx       # CI gate
smallaios-coverage-probe --inventory path/to/roadmap.md model.onnx
```

### Flags

| Flag | Purpose |
|---|---|
| `--json` | Emit JSON instead of Markdown/text |
| `--detailed` | Include per-node attribute-level concerns (Resize cubic, RoiAlign max, …) |
| `--quiet` | Summary only, no per-op breakdown |
| `--strict` | Exit with code 2 if any op is unrecognized or otherwise unsupported |
| `--inventory <path>` | Override the default roadmap-doc location |
| `-h`, `--help` | Print help |

### Example output

```
Coverage report: bert-like.onnx

Total nodes:        6
Distinct op kinds:  5

Implemented in develop:      4
Planned Phase 1:             1
Planned Phase 2:             0
Planned Phase 3:             0
Planned Phase 4:             0
Deferred subsystem:          0
Skipped (training/deprecated/vendor): 0
Unrecognized:                0

Verdict: Model becomes loadable after Phase 1 PRs merge.

Per-op breakdown:
  MatMul                        2 nodes   Implemented
  Add                           1 nodes   Implemented
  Gelu                          1 nodes   Planned-P1
  LayerNormalization            1 nodes   Implemented
  Softmax                       1 nodes   Implemented
```

## Adding new attribute concerns

Attribute-level limitations live in `src/walker.rs` under
`attribute_concerns_for`. To add a new one, match on the op type and
push a human-readable concern string for any unsupported attribute
combination. See the existing `Resize`, `RoiAlign`, `GridSample`,
`Unique`, and `ScatterND` entries for examples.

## Doc drift detection

If the roadmap doc claims an op is `Planned-P1` (or any non-Implemented
status) but the live `OperatorRegistry` already supports it, the probe
records a `doc_drifts` entry. The effective status is always the
registry's — the drift is surfaced as a warning so roadmap authors can
keep the doc honest.

## CI integration

Future: a CI job will run the probe against
`tests/fixtures/onnx-models/*.onnx` under `--strict` and fail if any
model regresses from `loadable_today` (or `loadable_after_phase<N>`) to
a less-loadable state. That ensures operator-coverage PRs cannot
accidentally break the baseline models.
