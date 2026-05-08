## ADDED Requirements

### Requirement: Upper-layer write path
Write operations under `/models/` SHALL be directed exclusively to the upper layer at `/data/models-upper/`. Lower-layer (squashfs) entries SHALL never be modified. Writes that would require modification of an existing lower-layer file (e.g., `truncate("/models/llama-3.onnx")` where `llama-3.onnx` is in the lower) SHALL return `-EROFS` with a user-readable message instructing the operator to `model_add` under a new name.

#### Scenario: Write to upper-only file succeeds
- **WHEN** an Operator writes to `/models/custom-tuned.onnx` (no lower entry)
- **THEN** the write SHALL go to `/data/models-upper/custom-tuned.onnx`
- **AND** the lower squashfs SHALL be untouched

#### Scenario: Write to lower-only file rejected
- **WHEN** an Operator attempts to truncate `/models/llama-3.onnx` (lower-only)
- **THEN** the syscall SHALL return `-EROFS`
- **AND** the message SHALL instruct using `model_add` under a new name

### Requirement: Per-name advisory lock during model_add
The kernel SHALL hold a per-name advisory lock during `model_add`. A second concurrent `model_add` for the same name SHALL return `-EBUSY` and SHALL include a `Retry-After` hint of 1 second. The lock SHALL be released on success, on failure, or on the writing fd being closed without completing.

#### Scenario: Concurrent add on same name rejected
- **WHEN** Operator A's `model_add("foo.onnx", ...)` is in progress
- **AND** Operator B issues `model_add("foo.onnx", ...)` simultaneously
- **THEN** B's syscall SHALL return `-EBUSY` with `Retry-After: 1`
- **AND** A's add SHALL proceed unaffected

#### Scenario: Concurrent add on different names allowed
- **WHEN** Operator A's `model_add("foo.onnx", ...)` is in progress
- **AND** Operator B issues `model_add("bar.onnx", ...)`
- **THEN** both adds SHALL proceed in parallel

#### Scenario: Lock released on writer-side abort
- **WHEN** Operator A's `model_add` aborts mid-write (Zenoh source disconnects)
- **THEN** the per-name lock SHALL be released
- **AND** any pending bytes in the upper SHALL be unlinked
- **AND** Operator B's subsequent `model_add` for the same name SHALL succeed

### Requirement: Atomic write via stage-and-rename
Each `model_add` SHALL stage to `<name>.tmp` in the upper layer, write the SHA-3-256 fingerprint sidecar (`<name>.sha3`) and optional ML-DSA-65 signature sidecar (`<name>.sig`), `fsync` the model file, then atomically `rename` over the final name. A crash before the rename SHALL leave no partially-visible model under `/models/`.

#### Scenario: Crash before rename leaves no visible file
- **WHEN** `model_add` writes bytes to `<name>.tmp` and the system crashes
- **THEN** on next boot `/models/<name>` SHALL not be visible (only the lower entry, if any)
- **AND** the orphan `<name>.tmp` SHALL be removed during boot cleanup

#### Scenario: Successful add visible after rename
- **WHEN** `model_add` completes the rename
- **THEN** `/models/<name>` SHALL serve the new bytes immediately
- **AND** `fsync(fd)` on the rename SHALL have made the change durable

### Requirement: Capacity cap enforcement
The total bytes occupied by `/data/models-upper/` SHALL be tracked and SHALL NOT exceed `fs.overlay.upper_max_bytes` (default 2 GiB; floor 64 MiB; ceiling = `/data/` partition size − 1 GiB headroom). Writes that would exceed the cap SHALL return `-ENOSPC`. The check SHALL include the in-progress staged file's expected size when known (operator-declared via the `model_add` ABI).

#### Scenario: Write within cap succeeds
- **WHEN** the upper holds 1 GiB and a 500 MiB add arrives (cap 2 GiB)
- **THEN** the add SHALL succeed
- **AND** subsequent reads SHALL return the new bytes

#### Scenario: Write exceeding cap rejected
- **WHEN** the upper holds 1.8 GiB and a 500 MiB add arrives (cap 2 GiB)
- **THEN** the add SHALL return `-ENOSPC` after the first ~200 MiB
- **AND** the staged tmp file SHALL be unlinked
- **AND** an audit record `model_add_capacity_exceeded` SHALL be appended

#### Scenario: Cap below floor rejected at config time
- **WHEN** an operator writes `fs.overlay.upper_max_bytes = 32 MiB`
- **THEN** the validator SHALL reject with `-EINVAL`
- **AND** the previous value SHALL remain in effect

### Requirement: Whiteout removal RBAC
Removing a whiteout marker (un-hiding a lower-layer entry) SHALL require `Role::Root` by default. When `fs.overlay.allow_operator_unhide = true`, `Role::Operator` MAY also remove whiteouts. Adding or modifying whiteouts SHALL always require Root via `model_remove`.

#### Scenario: Operator cannot remove whiteout by default
- **WHEN** an Operator calls `model_unhide("foo.onnx")` and the policy is default
- **THEN** the syscall SHALL return `-EPERM`
- **AND** an audit `DENY:model_unhide` SHALL be appended

#### Scenario: Operator unhides when policy permits
- **WHEN** `fs.overlay.allow_operator_unhide = true` is set
- **AND** an Operator calls `model_unhide("foo.onnx")`
- **THEN** the syscall SHALL succeed (subject to other preconditions)
- **AND** the audit record SHALL name the role and entry
