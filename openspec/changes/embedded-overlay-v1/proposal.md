## Why

`embedded-filesystem-v1` ships A/B whole-image squashfs swap as
the only way to change `/models/`. That works for the
periodic-update cadence of OS releases, but it doesn't handle the
"operator drops a 200 MB ONNX model into the running appliance
without re-imaging the box" workflow. Whole-image swap for that
case is wasteful (gigabytes of unchanged base image written for
a small per-deployment delta) and friction-y (operators want
`cp model.onnx /models/custom/` semantics, not "produce a fresh
squashfs delta and stage it through the update pipeline").

This change introduces an OverlayFS-style writable layer mounted
on top of the read-only squashfs `/models/`. Adding, replacing,
or hiding a model becomes a normal POSIX write to the upper
layer; the base squashfs remains unchanged and continues to be
A/B-swappable per `embedded-filesystem-v1`.

The implementation is **clean-room Rust, `#![no_std]`** —
matching the rest of the project's fs stack. It is **not** a port
of the Linux kernel's overlayfs; it is an in-tree
SmallAIOS-specific design that picks the subset of overlayfs
semantics that matter for the inference appliance use case and
leaves the rest out.

## What Changes

### Two-layer mount at `/models/`

`/models/` becomes a merged view of two underlying mounts:

```text
/models/                  (visible)
  ├── lower/              (RO squashfs from active A/B slot)
  │   ├── llama-3.onnx
  │   └── whisper.onnx
  └── upper/              (RW F2FS subdir at /data/models-upper/)
      ├── custom-tuned.onnx       ← operator-added
      └── llama-3.onnx.whiteout   ← marker that hides lower's llama-3
```

Reads consult the upper first, then the lower. Writes go
exclusively to the upper. Whiteout markers in the upper hide
lower entries.

### Whiteout & opaque-directory semantics

A whiteout SHALL be encoded as a regular file `<name>.whiteout`
in the upper layer (not as a special inode type — F2FS has no
character-device support in our `#![no_std]` write path, and we
don't want to introduce one for this).

An opaque directory (operator wants to completely shadow a
lower-layer directory) SHALL be marked by a `.opaque` regular
file in the corresponding upper directory.

This is a deliberate divergence from Linux overlayfs, which uses
character-device whiteouts and `trusted.overlay.opaque` xattrs.
The simplification trades some POSIX-overlayfs interop for a much
smaller implementation and zero xattr dependency.

### Upper layer storage on `/data/`

The upper layer's backing storage SHALL be a subdirectory of the
existing F2FS `/data/` partition: `/data/models-upper/`. No new
partition is added. The kernel SHALL ensure the directory exists
on first boot (per `embedded-filesystem-v1`'s
directory-tree-creation requirement) with mode 0700 owned by
kernel.

### Persistence across A/B swaps

The upper layer SHALL persist across A/B swaps of the lower
squashfs. Operator-added models survive OS updates. The
`embedded-filesystem-v1` boot sequence already mounts `/data/`
before `/models/`, so the upper is available when the merged
mount happens.

### Conflict resolution on lower-side changes

When an A/B swap brings in a new lower squashfs that contains a
file the upper already shadows (e.g., the operator added a custom
`llama-3.onnx` to the upper and the new base image also has
`llama-3.onnx`), the upper's version SHALL continue to win. No
automatic merge or "lower changed, what now" prompt; the
operator's explicit action takes precedence over implicit OS
updates. An audit record SHALL flag the conflict at boot so the
operator can review.

### RBAC integration

Adding a model SHALL require `Role::Operator` or `Role::Root`
(matches `model_load` permissions from `management-login-v1`).
Whiting out or removing an upper-layer entry SHALL require
`Role::Root` (this is destructive across reboots and SHOULD be a
deliberate admin action).

### Capacity and quota

The upper layer SHALL have a configurable size cap
(`fs.overlay.upper_max_bytes`, default 2 GiB) enforced at write
time. Exceeding the cap SHALL fail writes with `-ENOSPC` so a
runaway uploader does not exhaust `/data/`. The cap SHALL count
toward the F2FS `/data/` partition's overall budget.

### Integrity for upper-layer files

Files added to the upper SHALL gain a SHA-3-256 fingerprint
recorded in a sidecar `<name>.sha3` file written atomically with
the model file. Loading any model from `/models/` (whether from
lower or upper) SHALL hash-verify before the bytes flow into the
ONNX runtime. The lower's per-block hashes (from
`embedded-filesystem-v1`) cover the squashfs side; the upper's
per-file fingerprints cover the operator-added side.

ML-DSA-65 signing of upper-layer files SHALL be optional in v1
(behind `fs.overlay.require_signed = bool`, default `false`).
Operators in regulated environments can flip the policy on; the
verifier expects a `<name>.sig` sidecar with an ML-DSA-65
signature over the model file's SHA-3-256 fingerprint.

### `model_add` / `model_remove` syscalls

Two new syscalls in the kernel's auth-gated table (numbers 49 and
50, after `boot_success = 48` from `embedded-filesystem-v1`):

```text
model_add(name_ptr, name_len, contents_fd) -> 0 | -errno
   -- Operator+
model_remove(name_ptr, name_len) -> 0 | -errno
   -- Root only
```

`model_add` reads from the supplied fd (typically a network
upload), writes to `/data/models-upper/<name>`, computes the
SHA-3-256, writes the sidecar fingerprint, and (if signed-policy
is on) verifies the signature. `model_remove` writes the
appropriate whiteout / removes the upper entry.

### Capabilities

#### New Capabilities
- `fs-overlay-mount`: two-layer merge view at `/models/`, lookup
  precedence (upper → lower), whiteout / opaque-dir markers,
  POSIX read semantics across the merged tree.
- `fs-overlay-write`: write path to the upper layer, conflict
  resolution rules, audit-record-on-shadow, capacity cap
  enforcement.
- `fs-overlay-integrity`: SHA-3-256 fingerprint sidecars,
  optional ML-DSA-65 signature verification, fail-closed read
  on hash mismatch.
- `fs-overlay-syscalls`: `model_add`, `model_remove`, RBAC
  bindings.

#### Modified Capabilities
- `posix-vfs` (from `embedded-filesystem-v1`): the `/models/`
  mount becomes a merged view rather than a direct squashfs
  mount.
- `kernel-syscalls`: adds `model_add` (49), `model_remove` (50);
  bumps documented count.
- `mgmt-config-layout` (from `management-login-v1` /
  `embedded-filesystem-v1`): adds `fs.overlay.*` configuration
  fields with `#[reload("live")]` annotations.

## Impact

- **Code:**
  - `fs/src/overlay.rs` — merge logic, whiteout/opaque scanning,
    upper-priority lookup.
  - `fs/src/overlay/write.rs` — upper-layer write path,
    capacity-cap enforcement, audit record on lower-shadow.
  - `fs/src/overlay/integrity.rs` — SHA-3-256 fingerprint sidecar
    writer + verifier, optional ML-DSA-65 signature check.
  - `kernel/src/syscalls/model.rs` — `model_add`, `model_remove`
    syscalls.
  - `container/src/bin/model.rs` — user-space `model add` /
    `model remove` CLI.
  - `mgmt/src/config.rs` — new `fs.overlay.*` fields.
- **Tests:** ~150 new tests targeted: merge correctness (upper
  shadows lower, lower visible when upper absent, whiteout
  hides lower, opaque-dir hides whole subtree), conflict
  resolution on A/B swap (operator-added file persists across
  swap), capacity-cap behavior, signed-policy on/off,
  fingerprint mismatch fail-closed, RBAC enforcement
  (Operator can add, Operator cannot remove, Viewer cannot
  add). Aim to keep the post-`embedded-filesystem-v1` baseline
  of `≥5100` growing to `≥5250`.
- **Boot footprint:** Negligible — overlay merge logic is small
  (~1 kLOC), no new decompressors, no new on-disk format. Reuses
  existing F2FS write path for the upper.
- **External interop:** The upper layer is a normal F2FS
  subdirectory under `/data/models-upper/`. A recovery laptop
  mounting the F2FS partition sees the operator-added files
  directly. The merged view at `/models/` is a SmallAIOS-runtime
  construct only; external readers see lower (squashfs) and
  upper (F2FS) separately, which is the right behavior for
  recovery.
- **Downstream:** Unblocks the "operator drops in a custom model"
  workflow without forcing it through the OS update pipeline.
  Independent from `remote-update-v1` (the A/B swap mechanism).
- **Dependencies:** No new external Rust crates. Reuses
  `security` (SHA-3, ML-DSA-65) and the F2FS write path.
- **Risks:**
  1. Merge logic correctness under directory iteration with
     mixed upper/lower entries. Property-based tests sweep this.
  2. Whiteout naming collision — a real model literally named
     `foo.whiteout` would be ambiguous. Mitigation: reject names
     ending in `.whiteout`, `.opaque`, `.sha3`, `.sig` at
     `model_add` time with `-EINVAL` and a clear message.
  3. Upper-layer growth blowing out `/data/` if cap not enforced.
     Capacity cap (Q from walkthrough) is the answer.

## Out of scope for v1 (flagged)

- Hard links across upper/lower (Linux overlayfs supports
  these via xattr-tracked origin; we deliberately don't).
- POSIX `trusted.overlay.*` xattr compatibility — our markers
  are file-based.
- Multi-layer overlays (lower-1 + lower-2 + upper). v1 is
  strictly two layers.
- copy-up of large lower files when modified (Linux overlayfs
  copies the whole file on first write). Our model is "operator
  adds new files" not "operator edits squashfs files." Reject
  writes that would require copy-up of an existing lower file
  with `-EROFS` and a clear message; if the operator wants a
  modified version, they `model_add` it under a new name.
- Caching / pinning hints. The block cache from
  `embedded-filesystem-v1` covers reads.

## Open Questions

1. Whiteout encoding — sidecar file (`<name>.whiteout`),
   directory entry naming convention, or both?
2. Should `model_add` accept signed-only at the syscall level,
   or always allow upload and verify-on-load?
3. Capacity cap default — 2 GiB, percentage of `/data/`, or
   operator-mandatory at first boot?
4. Audit cadence — every `model_add` and every shadow conflict
   on A/B swap, or only the conflicts?
5. Whiteout removal — does removing a whiteout (un-hiding a
   lower file) require Root, or Operator like other adds?
6. Should the merged `/models/` view show whiteout markers as
   "missing" or as a special `.whiteouts/` virtual subdirectory
   for diagnostics?
7. Boot ordering — should the overlay mount be deferred behind
   a feature flag like `embedded-filesystem-v1`'s
   `fs-on-disk-mounts`?
8. Test fixture strategy — synthetic squashfs + F2FS images
   built in CI, or a small set of golden images committed?
