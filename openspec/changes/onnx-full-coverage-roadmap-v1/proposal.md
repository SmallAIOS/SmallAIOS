# ONNX Full Operator Coverage Roadmap

## Why

SmallAIOS currently implements 29 of the ~190 operators in the standard
ONNX specification (~15% coverage), with 80 more in flight on Phase 1
PRs (#74 Tier 2, #76 vision, #77 transformer) bringing the total to
109 (~57%). This is enough for classic CNN
inference (Tier 1) plus the building blocks for transformers, recurrent
models, and quantized inference (Tier 2). It is **not** enough to load
arbitrary real-world models off the shelf — every additional model
class (transformers, vision transformers, audio, object detection, …)
requires a new burst of operator implementation work.

We need a single, durable roadmap that:

1. **Catalogs every remaining ONNX operator** so nothing slips through
   the cracks during ad-hoc per-model work.
2. **Groups operators into tiers driven by real model targets** rather
   than spec compliance, so each tier delivers user-visible value.
3. **Defines an explicit execution model** for parallel agent-team
   implementation so multiple tiers can be in flight at once.
4. **Identifies operators we will deliberately defer or skip** (training
   ops, deprecated ops, ecosystem ops we may never support) so the
   "remaining work" number is honest.

The goal is *not* to ship all 190 ops in one change — that would be a
multi-month effort spread across many PRs. The goal is to have a single
authoritative plan that **every future operator-coverage PR slots into**.

## What Changes

This change adds a planning artifact set, not code:

- **Roadmap document** (`docs/onnx-coverage-roadmap.md`) — long-form
  prose describing the tier sequence, target models, op catalog, and
  agent-team execution model. This becomes the canonical reference
  cited by every subsequent operator-coverage OpenSpec change.
- **Operator inventory spec** (`onnx-cpu-execution`) — adds a
  requirement that the runtime SHALL track every standard ONNX
  operator's status as one of `Implemented`, `Planned`, `Deferred`, or
  `Skipped`, and that the inventory MUST be discoverable from the
  operator registry source of truth.
- **Tier definitions** — each subsequent tier (3-9) gets its own future
  OpenSpec change name reserved here, so PR titles and discussion can
  reference them before they exist.
- **Agent-team execution playbook** — captures the worktree-per-tier
  pattern, the file ownership rules for parallel agents, and the
  validation gates each tier must pass before merge.

This change does **not** implement any operators. It only sets up the
plan. Operator implementation happens in the per-tier follow-up changes.

## Impact

**Affected specs:**
- `onnx-cpu-execution` — adds a single non-implementation requirement
  about the operator inventory.

**Affected code:**
- New file: `docs/onnx-coverage-roadmap.md`
- No source code changes.

**Risks:**
- **Roadmap drift.** The ONNX spec evolves (new opset versions add
  ops). The roadmap must be revisited when new opsets land. Mitigated
  by an explicit "review checkpoint" task that runs before each new
  release.
- **Premature commitment.** Listing all tiers up front risks locking
  in choices that turn out to be wrong (e.g. control-flow ops may need
  a more invasive runtime change than we expect). Mitigated by treating
  Tiers 3-9 as *named slots*, not binding contracts — the per-tier
  OpenSpec proposals are still required and can revise the list.

**Out of scope:**
- Implementing any operators. Each tier gets its own follow-up change.
- GPU coverage. The CPU executor is the only target here; GPU
  operator coverage is tracked separately under the compute abstraction
  workstream.
- Custom / vendor / non-standard ops (e.g., ONNX Runtime contrib ops).
  These are explicitly **Skipped** unless a future model target requires
  them.
