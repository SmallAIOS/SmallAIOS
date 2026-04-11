## 1. Roadmap Document

- [ ] 1.1 Create `docs/onnx-coverage-roadmap.md` with the tier table from
  `design.md` D1
- [ ] 1.2 Add the full operator inventory: every standard ONNX op with
  status (`Implemented` / `Planned <tier>` / `Deferred` / `Skipped`)
- [ ] 1.3 Add the "Agent-Team Execution" section with worktree pattern,
  file-ownership rules, and per-tier validation gates
- [ ] 1.4 Add the deferred / skipped sections (D6) with rationale per item
- [ ] 1.5 Add Tier 3 (`transformer-models-v1`) op catalog from D4
- [ ] 1.6 Add Tier 4-7 sketches from D5 (these are not binding but reserve
  the slot names)
- [ ] 1.7 Cross-reference: link from `README.md` "Features" section to the
  roadmap

## 2. Tier Slot Reservation

- [ ] 2.1 Document the tier name → OpenSpec change mapping in the roadmap
  so future PRs can cite it
- [ ] 2.2 Verify each reserved tier name is unused in `openspec/changes/`
  and `openspec/changes/archive/`

## 3. Operator Inventory Hooks (deferred to Tier 3)

- [ ] 3.1 Note in the roadmap that the `OperatorStatus` enum and
  `SUPPORTED_OPS_INVENTORY` constant are added by `transformer-models-v1`,
  not by this change. (This is a planning marker only.)

## 4. Validation

- [ ] 4.1 `openspec validate onnx-full-coverage-roadmap-v1 --strict`
  passes
- [ ] 4.2 The roadmap document renders cleanly in the docs build
- [ ] 4.3 Operator counts in the roadmap match
  `OperatorRegistry::supported_count()` (currently 65 after merging
  `additional-operators-v1`)
- [ ] 4.4 Every "Implemented" entry in the roadmap matches an `OpKind`
  variant in `onnx-rt/src/operators.rs`
- [ ] 4.5 No code changes outside `docs/`
