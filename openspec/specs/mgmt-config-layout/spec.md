# mgmt-config-layout Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: /data/ directory layout
All operator-tunable configuration SHALL live under `/data/` in the documented hybrid layout: a top-level `system.toml` for cross-cutting knobs and per-subsystem files for substantive configuration. v1 layout:

```text
/data/
├── system.toml              # hostname, time zone, log level, mDNS default
├── auth/
│   └── shadow               # 0600 root — passwords, role table
├── network/                 # populated by network-management-v1
│   ├── eth0.toml
│   ├── eth1.toml
│   └── bond0.toml
├── mgmt/
│   ├── zenoh.toml           # listen endpoints, PSK paths
│   └── policy.toml          # role defs, rate limits, lockout, idle, audit, password policy, metrics cadence
├── update/                  # populated by remote-update-v1
│   └── policy.toml
└── automotive/              # populated by automotive-bus-management-v1
    └── uds.toml
```

The hybrid layout SHALL be preferred over a monolithic file because it: (a) gives permission granularity (`auth/shadow` 0600, `network/*.toml` viewer-readable); (b) prevents a partial write-failure on one subsystem from corrupting another's config.

#### Scenario: Fresh /data/ contains expected v1 paths
- **WHEN** the system runs first-boot completion
- **THEN** `/data/system.toml`, `/data/auth/shadow`, `/data/mgmt/zenoh.toml`, and `/data/mgmt/policy.toml` SHALL exist with conservative defaults

### Requirement: Per-file permission declaration
Each declared file SHALL have a per-file permission declared in the schema. The loader SHALL refuse to read a file whose mode is laxer than declared.

| File | Mode | Owner |
|------|:----:|:-----:|
| `/data/system.toml` | 0644 | kernel |
| `/data/auth/shadow` | 0600 | kernel |
| `/data/mgmt/zenoh.toml` | 0640 | kernel |
| `/data/mgmt/policy.toml` | 0640 | kernel |
| `/data/network/*.toml` | 0644 | kernel |
| `/data/update/policy.toml` | 0640 | kernel |
| `/data/automotive/uds.toml` | 0640 | kernel |

#### Scenario: Stricter-than-declared mode accepted
- **WHEN** `/data/mgmt/zenoh.toml` exists with mode 0600 (declared 0640)
- **THEN** the loader SHALL accept the file

#### Scenario: Laxer-than-declared mode rejected
- **WHEN** `/data/auth/shadow` exists with mode 0644 (declared 0600)
- **THEN** the loader SHALL refuse and SHALL treat the file as corrupt

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

### Requirement: /data/models-upper/ directory
On first boot of an image carrying the `fs-overlay-mounts` feature enabled, the kernel SHALL ensure `/data/models-upper/` exists with mode 0700 owned by kernel. The directory SHALL be created atomically alongside the rest of the `/data/` tree per the existing first-boot directory-tree-creation requirement.

#### Scenario: Directory created on first boot with overlay feature
- **WHEN** the kernel formats `/data/` per `embedded-filesystem-v1`
- **AND** the `fs-overlay-mounts` cargo feature is enabled
- **THEN** `/data/models-upper/` SHALL exist with mode 0700
- **AND** an audit record `overlay_upper_initialized` SHALL be appended

#### Scenario: Directory not created without feature
- **WHEN** the kernel formats `/data/` and the overlay feature is disabled
- **THEN** `/data/models-upper/` SHALL NOT be created
- **AND** the absence SHALL not block boot

### Requirement: fs.overlay.* configuration fields
`mgmt/policy.toml` SHALL expose the following live-reload-able fields when the overlay feature is enabled:

- `fs.overlay.upper_max_bytes` — capacity cap, default 2 GiB. Floor 64 MiB; ceiling = `/data/` partition size − 1 GiB headroom.
- `fs.overlay.require_signed` — boolean, default `false`. When `true`, every `model_load` from upper requires a valid `<name>.sig` ML-DSA-65 signature.
- `fs.overlay.allow_operator_unhide` — boolean, default `false`. When `true`, `Role::Operator` MAY remove whiteouts via `model_remove(_, 2)`.

All three fields SHALL carry `#[reload("live")]` so changes apply without remount.

#### Scenario: Default cap present in fresh policy
- **WHEN** a fresh `mgmt/policy.toml` is generated
- **THEN** `fs.overlay.upper_max_bytes = 2147483648` SHALL be present
- **AND** `fs.overlay.require_signed = false`, `fs.overlay.allow_operator_unhide = false` SHALL be present

#### Scenario: Cap below floor rejected
- **WHEN** an operator writes `fs.overlay.upper_max_bytes = 33554432` (32 MiB, below 64 MiB floor)
- **THEN** the validator SHALL reject with `-EINVAL`

#### Scenario: Live reload of require_signed
- **WHEN** an operator writes `fs.overlay.require_signed = true`
- **THEN** subsequent `model_load` calls SHALL begin enforcing signed requirement immediately
- **AND** no remount of `/models/` SHALL be required

### Requirement: Per-file permission table extended
The per-file permission declaration table SHALL include an entry for the overlay sidecar files in `/data/models-upper/`. Sidecars (`<name>.sha3`, `<name>.sig`) SHALL be mode 0600 owned by kernel; main model files SHALL be mode 0640 (group-readable for the inference runtime). Whiteout markers (`<name>.whiteout`, `<name>/.opaque`) SHALL be mode 0600.

#### Scenario: Sidecar mode 0600
- **WHEN** `model_add` writes `<name>.sha3` and `<name>.sig`
- **THEN** the resulting files SHALL have mode 0600

#### Scenario: Model file mode 0640
- **WHEN** `model_add` completes
- **THEN** the resulting `<name>` SHALL have mode 0640
- **AND** the inference runtime (running as the appropriate group) SHALL be able to read it

