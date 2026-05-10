## ADDED Requirements

### Requirement: model_add syscall
The kernel SHALL expose a new `model_add` syscall (`ONNX_MODEL_ADD = 0x36`, in the existing ONNX category) callable by `Role::Operator` and `Role::Root`. Signature:

```text
model_add(name_ptr, name_len, contents_fd, expected_size) -> 0 | -errno
```

Behavior: validates `name` against the reserved-suffix list, acquires the per-name advisory lock, opens `<name>.tmp` in `/data/models-upper/`, copies bytes from `contents_fd` until EOF, computes SHA-3-256 on the fly, enforces the capacity cap (using `expected_size` for pre-flight when non-zero, else current cumulative bytes), writes the `<name>.sha3` sidecar, optionally writes `<name>.sig` if signed-policy is on (signature provided as a follow-on field of the protocol), `fsync`s, atomically `rename`s over `<name>`, releases the lock, appends an audit record `model_added`.

#### Scenario: Operator successfully adds a model
- **WHEN** an Operator calls `model_add("custom.onnx", fd, 50_000_000)`
- **THEN** the syscall SHALL return `Ok(0)` after copying 50 MB from `fd`
- **AND** `/models/custom.onnx` SHALL serve the new bytes
- **AND** the audit ring SHALL contain `model_added{ who, name, sha3, size }`

#### Scenario: Viewer denied
- **WHEN** a Viewer calls `model_add(...)`
- **THEN** the syscall SHALL return `-EPERM`
- **AND** an audit `DENY:model_add` SHALL be appended

#### Scenario: Reserved suffix rejected
- **WHEN** an Operator calls `model_add("foo.whiteout", ...)`
- **THEN** the syscall SHALL return `-EINVAL` with the suffix-reserved message
- **AND** SHALL NOT acquire the per-name lock

### Requirement: model_remove syscall
The kernel SHALL expose a new `model_remove` syscall (`ONNX_MODEL_REMOVE = 0x37`, in the existing ONNX category) callable by `Role::Root` only. Signature:

```text
model_remove(name_ptr, name_len, mode: u8) -> 0 | -errno
```

`mode` selects the removal kind:
- `0` — Remove an upper-layer entry only (an operator-added file). If the lower has the same name, it becomes visible after removal.
- `1` — Hide a lower-layer entry by writing the whiteout marker. The lower entry remains on disk but is invisible from `/models/`.
- `2` — Remove a whiteout (un-hide), restoring lower visibility. Subject to `fs.overlay.allow_operator_unhide` policy.

#### Scenario: Root removes upper file
- **WHEN** Root calls `model_remove("custom.onnx", 0)` and the upper has `custom.onnx`
- **THEN** the upper file and its `.sha3`/`.sig` sidecars SHALL be deleted
- **AND** an audit `model_removed{ who, name, mode=0 }` SHALL be appended

#### Scenario: Root hides lower file
- **WHEN** Root calls `model_remove("llama-3.onnx", 1)` and the lower has `llama-3.onnx`
- **THEN** `<name>.whiteout` SHALL be written to the upper
- **AND** the audit record SHALL be `model_hidden{ who, name }`

#### Scenario: Operator denied remove
- **WHEN** an Operator calls `model_remove(...)` with any mode
- **THEN** the syscall SHALL return `-EPERM`
- **AND** an audit `DENY:model_remove` SHALL be appended

#### Scenario: Operator may unhide if policy permits
- **WHEN** `fs.overlay.allow_operator_unhide = true` is set
- **AND** an Operator calls `model_remove("foo.onnx", 2)`
- **THEN** the syscall SHALL succeed and the whiteout SHALL be deleted
- **AND** an audit record SHALL name the role and entry

### Requirement: Boot-time conflict detection
On every boot after a successful A/B swap into a new lower squashfs, the kernel SHALL scan the upper layer and SHALL append one audit record `overlay_conflict` for each upper entry that shadows a name now present in the new lower. The audit record SHALL include the name, the lower's SHA-3-256 (from the squashfs manifest), and the upper's SHA-3-256 (from the sidecar). No automatic resolution SHALL be attempted — the operator's upper continues to win.

#### Scenario: Newly-shadowing entry audited at boot
- **WHEN** before A/B swap the upper has `custom.onnx` and the lower has no `custom.onnx`
- **AND** an A/B swap brings in a new lower that has `custom.onnx`
- **THEN** on first boot of the new image an audit record `overlay_conflict{ name=custom.onnx, lower_sha3, upper_sha3 }` SHALL be appended

#### Scenario: Pre-existing conflict not re-audited
- **WHEN** an upper-shadow conflict was audited on a prior boot and neither upper nor lower has changed
- **THEN** the conflict SHALL NOT be re-appended on subsequent boots
