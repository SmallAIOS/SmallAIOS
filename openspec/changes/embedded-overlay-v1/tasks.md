## 1. Phase 1 — Merge logic (read-only)

- [ ] 1.1 `fs/src/overlay/mod.rs` — module skeleton, `#![no_std]`, gated by `fs-overlay-mounts` cargo feature
- [ ] 1.2 `fs/src/overlay/lookup.rs` — upper-wins precedence resolver
- [ ] 1.3 Whiteout detection (`<name>.whiteout` regular file)
- [ ] 1.4 Opaque-directory detection (`<dir>/.opaque` regular file)
- [ ] 1.5 Reserved-suffix rejection (`.whiteout`, `.opaque`, `.sha3`, `.sig`)
- [ ] 1.6 `readdir` merging (union of upper + lower entries minus whiteouts/opaque hides)
- [ ] 1.7 Synthetic fixture builder: CI script that runs `mksquashfs` + `mkfs.f2fs` to produce small lower + upper images
- [ ] 1.8 Property-based tests sweep all upper/lower content combinations
- [ ] 1.9 Kani harness for the merge-lookup precedence invariant
- [ ] 1.10 TLA+ model `overlay_state.tla` — add/remove/whiteout/A-B-swap interleavings

## 2. Phase 2 — Upper-layer write path

- [ ] 2.1 `fs/src/overlay/write.rs` — write routing (upper-only, lower-only-rejected)
- [ ] 2.2 Stage-and-rename atomic write helper
- [ ] 2.3 Per-name advisory lock (`HashMap<&str, OverlayLock>` keyed by name)
- [ ] 2.4 Concurrent-add `-EBUSY` path
- [ ] 2.5 Aborted-add cleanup (orphan `<name>.tmp` removal on writer-fd close)
- [ ] 2.6 SPIN model proving no two concurrent adds for the same name both succeed
- [x] 2.7 Boot-cleanup pass: on mount, remove any orphan `<name>.tmp` from `/data/models-upper/` — `fs/src/overlay/boot_cleanup.rs::run_boot_cleanup`; wired into the mount path via `fs/src/mount.rs::sweep_overlay_upper` and the composite `run_overlay_boot_handlers`; integration tests in `fs/tests/integration_overlay_boot.rs`
- [ ] 2.8 Tests: stage-rename atomicity, per-name lock, concurrent on different names allowed

## 3. Phase 3 — model_add / model_remove syscalls

- [x] 3.1 Add `model_add` syscall (`ONNX_MODEL_ADD = 0x36`) per `kernel-syscalls`
- [x] 3.2 Add `model_remove` syscall (`ONNX_MODEL_REMOVE = 0x37`) per `kernel-syscalls`
- [x] 3.3 Hook into existing `min_role` capability dispatch
- [x] 3.4 SHA-3-256 fingerprint sidecar writer integrated with `model_add`
- [x] 3.5 Optional ML-DSA-65 signature sidecar (always written when payload provides one; validity checked at load time per integrity spec)
- [x] 3.6 `model_remove` mode 0/1/2 (delete-upper / hide-lower / unhide)
- [x] 3.7 `passwd`-style user-space CLI: `container/src/bin/model.rs` wrapping `model_add` + `model_remove`
- [x] 3.8 Tests for ABI conformance and per-mode behavior

## 4. Phase 4 — RBAC + Zenoh admin verbs

- [x] 4.1 Operator + Root permitted on `model_add`
- [x] 4.2 Root only permitted on `model_remove` (modes 0, 1)
- [x] 4.3 `model_remove` mode 2 (unhide) gated on `fs.overlay.allow_operator_unhide`
- [ ] 4.4 Zenoh admin verbs: `smallaios/admin/model/add`, `smallaios/admin/model/remove`
- [ ] 4.5 Streaming upload over Zenoh queryable for `model_add` payloads
- [x] 4.6 Tests across the RBAC matrix (every role × every verb)
- [x] 4.7 Audit `DENY:model_add` / `DENY:model_remove` records on RBAC failure

## 5. Phase 5 — Integrity layer

- [ ] 5.1 SHA-3-256 verify on every read of an upper file (per `fs-overlay-integrity`)
- [ ] 5.2 Missing-sidecar fail-closed
- [ ] 5.3 ML-DSA-65 signature verify when `require_signed = true`
- [ ] 5.4 Default-off mode (signature ignored if not required, even when present)
- [ ] 5.5 Audit records on hash mismatch and signature failure
- [ ] 5.6 Property-based fuzz on corrupted sidecars (truncated, malformed hex, wrong-length signature)

## 6. Phase 6 — Audit + cap enforcement + boot conflict

- [x] 6.1 Capacity-cap pre-flight on `model_add` (using `expected_size` when provided)
- [x] 6.2 Capacity-cap mid-flight (running cumulative bytes vs cap)
- [x] 6.3 `-ENOSPC` on cap violation; staged tmp file unlinked
- [x] 6.4 Audit `model_add_capacity_exceeded` with declared/actual bytes
- [x] 6.5 Boot-time conflict scan: walk upper, check each non-whiteout name against lower — `fs/src/overlay/conflict_scan.rs::run_boot_conflict_scan`; wired into the mount path via `fs/src/mount.rs::scan_overlay_conflicts` and composed inside `run_overlay_boot_handlers`
- [x] 6.6 One audit record per new conflict per boot (no re-audit if already-audited) — `ConflictMemo` suppression honored by the integration; covered by `fs/tests/integration_overlay_boot.rs::second_boot_with_memo_filled_suppresses_existing_conflicts` and `boot_handlers_emit_conflict_for_real_f2fs_upper_shadowing_lower`
- [ ] 6.7 `mgmt/policy.toml` fields wired with `#[reload("live")]`
- [x] 6.8 First-boot `/data/models-upper/` directory creation — `fs/src/mount.rs::ensure_models_upper_dir` creates `/data/models-upper/` with mode 0700 and emits `data_models_upper_initialized` (audit tag `AUDIT_TAG_MODELS_UPPER_INITIALIZED`); idempotent on subsequent boots; integration tests in `fs/tests/integration_overlay_boot.rs::ensure_models_upper_*`

## CHECKPOINT — Single PR into `develop` once all 6 phases land

This change ships as **two PRs**:

- **PR 1 (this PR — scaffolding only):** proposal.md, design.md, the 7 spec deltas (4 new + 3 modified), tasks.md. Merges to `develop` so agent teams can spawn worktrees off develop and pick up the design.
- **PR 2 (later — implementation):** all 6 phases in one cohesive merge, behind the default-off `fs-overlay-mounts` cargo feature.

The scaffolding (PR 1) gating list:

- [ ] CA.1 `openspec validate embedded-overlay-v1 --strict` returns clean
- [ ] CA.2 PR title follows conventional-commit semver convention
- [ ] CA.3 No production code changes (scaffolding only)
- [ ] CA.4 Documentation references `embedded-filesystem-v1` as the foundation

The implementation (PR 2) gating list:

- [ ] CB.1 Behind cargo feature `fs-overlay-mounts`, default off
- [ ] CB.2 `cargo fmt --check`, `cargo clippy -- -D warnings`
- [ ] CB.3 `cargo test --workspace` total ≥ 5250 (post-`embedded-filesystem-v1` baseline + ~150)
- [ ] CB.4 Kani harness for merge-lookup precedence passes
- [ ] CB.5 SPIN concurrent-add interleaving model verifies clean
- [ ] CB.6 TLA+ overlay-state model verifies clean
- [ ] CB.7 Cyclic-dep check passes (no new edges; reuses `fs/` and `security/`)
- [ ] CB.8 `cargo audit` / `cargo deny` clean

## 7. Cross-phase verification

- [ ] 7.1 Documentation updates: `docs/architecture.md` ONNX-category syscall table includes `ONNX_MODEL_ADD = 0x36` and `ONNX_MODEL_REMOVE = 0x37`; new `docs/embedded-overlay.md` operator runbook (model_add / model_remove flows)
- [ ] 7.2 Image-size regression check: overlay code stays ≤ 30 KB compiled growth budget
- [ ] 7.3 Reserved-suffix CI lint to catch accidental introduction of conflicting test fixture names
- [ ] 7.4 Update SmallAIOS user-space CLI tools to know about `/models/` upper layer (for `model list --include-whiteouts`)
- [x] 7.5 First-boot `/data/models-upper/` directory creation tested on a fresh GPT image — `fs/tests/integration_overlay_boot.rs::ensure_models_upper_creates_dir_on_first_boot`, `ensure_models_upper_uses_mode_0700`, `ensure_models_upper_creates_data_parent_if_missing`, `three_phase_workflow_first_boot_orphan_clean_boot`
- [ ] 7.6 Round-trip with external `mount -t f2fs`: external reader sees `/data/models-upper/` contents directly (operator-added files visible from a recovery laptop)
