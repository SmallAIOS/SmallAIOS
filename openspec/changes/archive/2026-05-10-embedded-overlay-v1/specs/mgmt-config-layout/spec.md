## ADDED Requirements

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
