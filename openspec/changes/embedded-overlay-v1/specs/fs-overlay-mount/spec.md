## ADDED Requirements

### Requirement: Two-layer merged mount at /models/
When the `fs-overlay-mounts` cargo feature is enabled, `/models/` SHALL present a merged view of two underlying mounts: a read-only **lower** layer backed by the active squashfs slot (per `embedded-filesystem-v1`'s `fs-squashfs-readonly`) and a read-write **upper** layer backed by `/data/models-upper/` on F2FS (per `embedded-filesystem-v1`'s `fs-f2fs-readwrite`). When the feature is disabled, `/models/` SHALL continue to be a direct squashfs mount with no upper layer.

#### Scenario: Feature off behaves identically to embedded-filesystem-v1
- **WHEN** the kernel is built without `fs-overlay-mounts`
- **THEN** `/models/` SHALL be a direct squashfs mount
- **AND** all writes SHALL return `-EROFS`
- **AND** behavior SHALL be byte-identical to `embedded-filesystem-v1`

#### Scenario: Feature on creates merged view
- **WHEN** the kernel is built with `fs-overlay-mounts`
- **AND** `/data/models-upper/` exists
- **THEN** `/models/` SHALL present the merged view
- **AND** `readdir(/models)` SHALL return the union of upper and lower entries (with upper-wins precedence and whiteouts honored)

### Requirement: Upper-wins lookup precedence
Path lookup under `/models/` SHALL consult the upper layer first. If the upper has a regular file at the requested path, that file SHALL be returned. If the upper has a whiteout marker `<name>.whiteout` at the parent path, the lookup SHALL return `-ENOENT`. Otherwise the lookup SHALL fall through to the lower layer.

#### Scenario: Upper file shadows lower file
- **WHEN** `/data/models-upper/llama-3.onnx` exists with content X
- **AND** the lower squashfs has `llama-3.onnx` with content Y
- **THEN** `read("/models/llama-3.onnx")` SHALL return X

#### Scenario: Lower visible when upper absent
- **WHEN** `/data/models-upper/` has no entry for `whisper.onnx`
- **AND** the lower squashfs has `whisper.onnx`
- **THEN** `read("/models/whisper.onnx")` SHALL return the lower's content

#### Scenario: Whiteout hides lower
- **WHEN** `/data/models-upper/llama-3.onnx.whiteout` exists
- **AND** the lower squashfs has `llama-3.onnx`
- **THEN** `open("/models/llama-3.onnx", O_RDONLY)` SHALL return `-ENOENT`
- **AND** `readdir("/models/")` SHALL NOT list `llama-3.onnx`

#### Scenario: Opaque dir hides whole subtree
- **WHEN** `/data/models-upper/custom/.opaque` exists
- **AND** the lower has files under `custom/`
- **THEN** the lower entries under `custom/` SHALL NOT be visible
- **AND** only upper entries under `/data/models-upper/custom/` SHALL be listed

### Requirement: Whiteouts and opaque markers are file-based
Whiteout markers SHALL be regular empty files named `<name>.whiteout` placed at the parent path. Opaque-directory markers SHALL be regular empty files named `.opaque` placed inside the subdirectory whose lower-side contents are to be hidden. The implementation SHALL NOT use POSIX xattrs, character-device whiteouts, or any other mechanism beyond regular files.

#### Scenario: Empty whiteout file enough to hide
- **WHEN** an empty file `/data/models-upper/foo.onnx.whiteout` exists
- **THEN** lower-layer `foo.onnx` SHALL be hidden
- **AND** the whiteout file's contents (zero bytes) SHALL not be inspected

#### Scenario: Non-empty whiteout still functional
- **WHEN** the whiteout file contains arbitrary bytes (e.g., comment text)
- **THEN** the file's mere presence SHALL still hide the lower entry
- **AND** the bytes SHALL be ignored

### Requirement: Reserved-suffix names rejected
The merge layer and `model_add` syscall SHALL reject names ending in `.whiteout`, `.opaque`, `.sha3`, or `.sig` with `-EINVAL` and a user-readable message naming the conflicting suffix. This ensures operator-supplied names cannot collide with overlay-internal markers.

#### Scenario: Reserved suffix on add
- **WHEN** `model_add("foo.whiteout", ...)` is called
- **THEN** the syscall SHALL return `-EINVAL`
- **AND** the audit record SHALL note the rejected suffix

#### Scenario: Reserved suffix on direct write
- **WHEN** an Operator attempts `open("/data/models-upper/foo.opaque", O_CREAT)`
- **THEN** the operation SHALL be rejected at the VFS write boundary
