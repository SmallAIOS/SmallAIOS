## Context

`embedded-filesystem-v1` ships A/B whole-image squashfs swap as
the only way to change `/models/`. That works for OS releases
but it doesn't handle "operator drops a 200 MB ONNX into the
running appliance without re-imaging the box." Whole-image swap
for that case is wasteful and friction-y.

This change introduces an OverlayFS-style writable layer mounted
on top of the read-only squashfs `/models/`. It is a clean-room,
SmallAIOS-specific design — not a port of Linux overlayfs. We
pick the subset of overlayfs semantics that matter for the
inference appliance and skip the rest (xattrs, char-device
whiteouts, copy-up of large files, multi-layer overlays).

## Goals / Non-Goals

**Goals:**
- Two-layer merged view at `/models/`: lower = active squashfs
  slot, upper = `/data/models-upper/` on F2FS.
- Clean-room Rust `#![no_std]` merge logic in the existing `fs/`
  crate (no new crate).
- File-based whiteout (`<name>.whiteout`) and opaque-dir
  (`<name>/.opaque`) markers — no xattrs, no char devices.
- Persistence across A/B swaps: operator-added models survive
  OS updates.
- Conflict resolution rule: upper wins; lower-side changes
  audit-logged, never silently merged.
- New `model_add` (49) and `model_remove` (50) syscalls with
  RBAC: Operator+ for add, Root for remove.
- Per-file SHA-3-256 fingerprint sidecar (`<name>.sha3`) and
  optional ML-DSA-65 signature sidecar (`<name>.sig`) controlled
  by `fs.overlay.require_signed`.
- Capacity cap (`fs.overlay.upper_max_bytes`, default 2 GiB).
- ≥5250 total tests after change (~+150 new).
- Formal coverage: Kani (merge precedence), SPIN (concurrent
  add), TLA+ (overlay state machine).

**Non-Goals (v1):**
- Hard links across upper/lower.
- POSIX `trusted.overlay.*` xattr compatibility.
- Multi-layer overlays (lower-1 + lower-2 + upper).
- copy-up of large lower files when modified.
- Custom caching / pinning hints (block cache from
  `embedded-filesystem-v1` covers reads).

## Decisions

The 15 questions resolved during the walkthrough.

### Q1. Whiteout encoding

**Decision:** Sidecar regular file `<name>.whiteout` next to
where the hidden lower entry would be. Empty contents; mtime is
the only metadata. Rejected reserved suffix at `model_add` time.
Opaque-directory marker: `<dir>/.opaque` regular file, hides the
entire lower-side subtree at that path.

### Q2. ML-DSA-65 signing policy

**Decision:** Configurable `fs.overlay.require_signed: bool`
(default `false`). When `true`, every `model_load` from the
upper SHALL find and verify a `<name>.sig` ML-DSA-65 signature
over the file's SHA-3-256 fingerprint; missing or invalid sig
returns `-EAUTH`. Verification happens at `model_load` time, not
`model_add`, so flipping the policy on later rejects existing
unsigned models without forcing re-upload.

### Q3. Upper capacity cap default

**Decision:** `fs.overlay.upper_max_bytes = 2147483648` (2 GiB).
Configurable up to (`/data/` partition size − 1 GiB headroom for
audit + config). Below-floor (`< 64 MiB`) rejected.

### Q4. Audit cadence

**Decision:** Every `model_add` and `model_remove` is audited
with full context (`who`, `name`, `sha3`, `size`). Every shadow
conflict detected at boot (upper has a name the new lower also
has) appends one record per conflict per boot. Reads are not
audited.

### Q5. Whiteout removal RBAC

**Decision:** Root only by default. Configurable knob
`fs.overlay.allow_operator_unhide: bool` (default `false`)
allows Operator to remove a whiteout. Audit record names the
role and the affected entry.

### Q6. Whiteout listing visibility

**Decision:** Hidden — `readdir(/models)` skips entries with
active whiteouts. Operator-facing diagnostic via the `model
list --include-whiteouts` user-space CLI; the kernel-level
listing remains POSIX-overlay-style hidden.

### Q7. Boot-ordering / feature flag

**Decision:** Behind cargo feature `fs-overlay-mounts`, default
off. Mirrors `embedded-filesystem-v1`'s `fs-on-disk-mounts`
pattern. While off, `/models/` continues to be a direct squashfs
mount.

### Q8. Test fixtures

**Decision:** Synthetic — CI runs `mksquashfs` and `mkfs.f2fs`
to produce small fixture images at every CI run. Keeps git
history clean, reproducible from source, matches the existing
`embedded-filesystem-v1` interop CI tooling.

### Q9. Reserved name suffixes

**Decision:** `model_add` rejects names ending in `.whiteout`,
`.opaque`, `.sha3`, `.sig` with `-EINVAL` and a clear message.

### Q10. `model_add` upload-fd source

**Decision:** Any readable fd. Zenoh stream, pre-staged
`/data/upload/...` file, or `/dev/null` (for empty-file tests)
all work the same way: kernel reads bytes from the fd until EOF,
hashes on the fly, writes to upper, writes sidecar.

### Q11. Concurrent `model_add` of same name

**Decision:** Per-name advisory lock in the kernel. Second
concurrent `model_add` on a name already locked returns
`-EBUSY` with a `Retry-After` hint. Lock released on success or
failure of the first writer.

### Q12. Phase ordering

**Decision:** 6-phase bottom-up:
1. Merge logic (read-only over synthetic upper+lower fixtures).
2. Upper-layer write path on F2FS.
3. `model_add` / `model_remove` syscalls.
4. RBAC bindings + Zenoh admin verbs.
5. SHA-3-256 fingerprint sidecars + optional ML-DSA-65 verify.
6. Audit + capacity-cap enforcement + boot-time conflict
   detection.

### Q13. Test target

**Decision:** ≥5250 total tests after change (~+150 new). Cover
merge correctness, conflict-on-A/B-swap, capacity cap,
signed-policy on/off, fingerprint mismatch fail-closed, RBAC
matrix, concurrent-add lock, reserved-name reject, whiteout
removal, audit emission.

### Q14. Formal verification

**Decision:** Kani (merge-lookup precedence invariant), SPIN
(concurrent-add interleaving), TLA+ (overlay state machine
covering add/remove/whiteout/A-B-swap interleavings). Maximum
assurance, matching the rest of `fs/`.

### Q15. PR strategy

**Decision:** This change ships as **two** PRs:

- **PR 1 (this PR):** scaffolding only — proposal, design,
  specs, tasks. Merges to `develop` so agent teams can spawn
  worktrees off develop and pick up the design.
- **PR 2 (later):** implementation, behind the
  `fs-overlay-mounts` cargo feature, single PR covering all 6
  phases since the change is small enough to review cohesively.
  Posts back to `develop` from a fresh `change/embedded-overlay-v1-impl`
  branch.

Note: Q15's "single PR once all 6 phases land" answer means the
implementation lands in one cohesive merge — it does NOT block
the scaffolding-PR-merging-now from going first. The wording in
the question covered the implementation step; the scaffolding
step is what we're delivering today.

## Risks / Trade-offs

- **[Risk] Merge logic correctness under directory iteration
  with mixed upper/lower entries** — Mitigation: property-based
  tests sweep all combinations; Kani harness proves the
  precedence invariant.
- **[Risk] Whiteout naming collision (operator literally adds
  `foo.whiteout`)** — Mitigation: reserved-suffix rejection at
  `model_add` (Q9) with a clear error.
- **[Risk] Upper-layer growth blowing out `/data/`** —
  Mitigation: capacity cap default 2 GiB (Q3), enforced at
  write time, sized below the F2FS partition.
- **[Risk] Concurrent-add race producing two half-written upper
  files** — Mitigation: per-name advisory lock (Q11) prevents
  it; SPIN model proves no two adds can both succeed.
- **[Risk] Lower changes during A/B swap silently shadowing
  operator's models** — Mitigation: boot-time conflict detection
  (Q4) audits every conflict so the operator can review and
  decide whether to un-hide the new lower version.
- **[Risk] `fs.overlay.require_signed` default off leaves
  unsigned models acceptable in default deployments** —
  Mitigation: documented as the v1 trade-off; regulated
  environments flip the policy on. Future v2 may flip the
  default; flagged in `Open Questions`.

## Migration Plan

This change is purely additive:
- Builds on `embedded-filesystem-v1`'s F2FS `/data/` mount and
  squashfs A/B layout. No existing data layout changes.
- First boot of an image carrying this change with the
  `fs-overlay-mounts` feature enabled creates the empty
  `/data/models-upper/` directory if absent (per the existing
  `mgmt-config-layout` directory-tree-creation requirement).
- An image without the feature enabled (or running on a system
  that doesn't support it yet) sees `/models/` as a direct
  squashfs mount, exactly as before.
- No format on `/data/` is changed by this addition; an
  operator can flip the feature flag in either direction
  without re-formatting.

## Open Questions

All fifteen design walkthrough questions are resolved (Q1–Q15
above).

Items deferred to a future change with explicit decision:

- **`fs.overlay.require_signed` default flip to `true`** — v1
  defaults off for friction-free dev/test workflows. v2 may
  default on once a model-signing tooling story is documented
  in the runbook.
- **Multi-layer overlay (lower-1 + lower-2 + upper)** — out of
  scope. If a use case appears (e.g., per-customer models on
  top of base + tenant-specific models on top of that), capture
  as `embedded-overlay-v2`.
- **Hard links across upper/lower** — explicit non-goal in v1.
  Linux overlayfs supports them via xattr-tracked origin; not
  worth the xattr machinery for our use case.
- **copy-up on modify** — out of scope. Operators add new files
  rather than editing squashfs files. Writes that require
  copy-up of an existing lower file return `-EROFS` with a
  message instructing the operator to `model_add` under a new
  name.
