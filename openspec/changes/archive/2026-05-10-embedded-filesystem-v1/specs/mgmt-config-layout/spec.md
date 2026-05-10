## ADDED Requirements

### Requirement: Per-file permissions enforced against real F2FS inodes
The per-file permission table introduced in `management-login-v1` SHALL be enforced against real F2FS inode mode bits, not the in-memory VFS placeholders. The "loader refuses mode laxer than declared" rule SHALL operate on the F2FS-recorded mode at open time.

#### Scenario: F2FS mode 0644 on shadow rejected
- **WHEN** `/data/auth/shadow` exists in F2FS with mode 0644 (declared 0600)
- **THEN** the loader SHALL refuse to read it
- **AND** SHALL treat the file as corrupt per `auth-shadow`'s boot-state requirement

#### Scenario: F2FS mode 0600 on shadow accepted
- **WHEN** `/data/auth/shadow` exists in F2FS with mode 0600
- **THEN** the loader SHALL parse it normally


### Requirement: First-boot creation of /data/ directory tree
On first boot of a freshly-formatted `/data/` partition, the kernel SHALL create the canonical directory tree with the declared permissions:

```text
/data/auth/         mode 0700
/data/audit/        mode 0700
/data/mgmt/         mode 0700
/data/network/      mode 0755
/data/update/       mode 0700
/data/automotive/   mode 0700
```

Directory creation SHALL be atomic across the set: either the entire tree exists with the declared modes, or none does and the kernel halts.

#### Scenario: Directories created on fresh /data/
- **WHEN** the kernel formats `/data/` per `fs-f2fs-readwrite`'s first-boot path
- **THEN** all six top-level directories SHALL exist with their declared modes
- **AND** an audit record `data_tree_initialized` SHALL be appended

#### Scenario: Partial tree on previous-boot crash auto-completes
- **WHEN** boot finds `/data/auth/` but no `/data/audit/`
- **THEN** the kernel SHALL create the missing directories with their declared modes
- **AND** SHALL append a `data_tree_repaired` audit record

### Requirement: Cache-budget configuration fields
`mgmt/policy.toml` SHALL expose two new live-reload-able cache-budget fields:

- `fs.cache.models_bytes` — LRU budget for the `/models/` mount, default 16 MiB.
- `fs.cache.data_bytes` — LRU budget for the `/data/` mount, default 4 MiB.

Both SHALL have a hard floor (4 MiB for models, 1 MiB for data); values below the floor SHALL be rejected with `-EINVAL` by the validator.

#### Scenario: Default cache budgets present
- **WHEN** a fresh `mgmt/policy.toml` is generated on first boot
- **THEN** `fs.cache.models_bytes = 16777216` and `fs.cache.data_bytes = 4194304` SHALL be present

#### Scenario: Below-floor budget rejected
- **WHEN** an operator writes `fs.cache.models_bytes = 1048576` (below 4 MiB floor)
- **THEN** the validator SHALL reject the write with `-EINVAL`

#### Scenario: Live reload of cache budget
- **WHEN** an operator writes a new `fs.cache.models_bytes` value
- **THEN** the new budget SHALL apply to subsequent allocations without remount
