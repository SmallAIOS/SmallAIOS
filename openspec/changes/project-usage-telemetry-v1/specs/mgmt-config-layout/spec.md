## ADDED Requirements

### Requirement: telemetry/usage.toml configuration file

The `/data/` layout SHALL include `telemetry/usage.toml` holding the usage-telemetry configuration:

```toml
[usage_telemetry]
enabled              = false        # disabled-by-default invariant
consent_recorded_at  = 0            # 0 = never; >0 = unix timestamp
consent_revoked_at   = 0
endpoint             = "https://usage.smallaios.invalid/v1/ingest"
                                    # public; baked-in default; no auth
schema_version       = "0"          # bumped only when the schema changes
install_id           = ""           # generated at opt-in; UUID

[usage_telemetry.opt_outs]
crashes  = false
counters = false
features = false
```

A missing `enabled` field SHALL parse as `false` per the disabled-by-default invariant.

#### Scenario: Layout includes the usage-telemetry file

- **WHEN** the config layout schema is loaded
- **THEN** `/data/telemetry/usage.toml` SHALL be a declared file carrying the `[usage_telemetry]` table with the `enabled`, `consent_recorded_at`, `consent_revoked_at`, `endpoint`, `schema_version`, and `install_id` fields and the `[usage_telemetry.opt_outs]` table

#### Scenario: Missing enabled field defaults to false

- **WHEN** `/data/telemetry/usage.toml` exists without the `enabled` field
- **THEN** the loader SHALL treat `enabled` as `false`

### Requirement: Per-file permission declared for telemetry/usage.toml

The per-file permission declaration table SHALL include an entry for `/data/telemetry/usage.toml` with mode 0600 owned by kernel. The file carries the per-install identifier (`install_id`) and the consent record, and no group consumer needs to read it, so the strictest config-file mode applies. Per the existing per-file permission declaration rule, the loader SHALL refuse to read the file when its on-disk mode is laxer than declared.

| File | Mode | Owner |
|------|:----:|:-----:|
| `/data/telemetry/usage.toml` | 0600 | kernel |

#### Scenario: usage.toml created with declared mode

- **WHEN** first boot generates `/data/telemetry/usage.toml` with its documented defaults
- **THEN** the file SHALL have mode 0600 owned by kernel

#### Scenario: Laxer mode on usage.toml rejected

- **WHEN** `/data/telemetry/usage.toml` exists with mode 0644 (declared 0600)
- **THEN** the loader SHALL refuse to read it
- **AND** SHALL treat the file as corrupt per the per-file permission declaration rule

### Requirement: usage.toml endpoint field is read-only to operators

The `endpoint` field of `telemetry/usage.toml` SHALL be read-only to operators: the build-time constant `USAGE_TELEMETRY_ENDPOINT` is authoritative, and operator overrides SHALL be rejected by the loader. Operators wanting self-telemetry to their own backend use `telemetry-otel-export-v1`, not this channel — pointing usage telemetry at an arbitrary endpoint would defeat the schema-enforcement-at-the-edge guarantee.

#### Scenario: Operator endpoint override rejected

- **WHEN** an operator edits `telemetry/usage.toml` to set `endpoint` to a value different from the build-time `USAGE_TELEMETRY_ENDPOINT` constant
- **THEN** the loader SHALL reject the override
- **AND** the exporter SHALL only ever target the build-time constant

#### Scenario: Build-time constant is authoritative

- **WHEN** the usage-telemetry exporter resolves its target endpoint
- **THEN** the value SHALL come from the `USAGE_TELEMETRY_ENDPOINT` build-time constant
- **AND** no configuration file value SHALL substitute for it

### Requirement: install_id generated at opt-in and never re-rolled by updates

The `install_id` field of `telemetry/usage.toml` SHALL be generated (as a random UUID) only at opt-in time and SHALL never be re-rolled by a software update. It persists in `/data/` across reboots and updates; only reflashing `/data/` regenerates it (at the next opt-in), and that asymmetry SHALL be documented.

#### Scenario: Update preserves install_id

- **WHEN** a software update is applied to a system with a populated `install_id`
- **THEN** the `install_id` value SHALL be byte-for-byte unchanged after the update

#### Scenario: install_id empty until opt-in

- **WHEN** a system has never completed the opt-in flow
- **THEN** the `install_id` field SHALL remain empty
- **AND** no UUID SHALL be generated for it outside the opt-in flow

## MODIFIED Requirements

### Requirement: First-boot creation of /data/ directory tree
On first boot of a freshly-formatted `/data/` partition, the kernel SHALL create the canonical directory tree with the declared permissions:

```text
/data/auth/         mode 0700
/data/audit/        mode 0700
/data/mgmt/         mode 0700
/data/network/      mode 0755
/data/update/       mode 0700
/data/automotive/   mode 0700
/data/telemetry/    mode 0700
```

Directory creation SHALL be atomic across the set: either the entire tree exists with the declared modes, or none does and the kernel halts. `/data/telemetry/` SHALL be part of the atomic set so that the first-boot opt-in prompt (per `console-login`) can write `telemetry/usage.toml` on a freshly formatted `/data/`.

#### Scenario: Directories created on fresh /data/
- **WHEN** the kernel formats `/data/` per `fs-f2fs-readwrite`'s first-boot path
- **THEN** all seven top-level directories — including `/data/telemetry/` with mode 0700 — SHALL exist with their declared modes
- **AND** an audit record `data_tree_initialized` SHALL be appended

#### Scenario: Partial tree on previous-boot crash auto-completes
- **WHEN** boot finds `/data/auth/` but no `/data/audit/` or `/data/telemetry/`
- **THEN** the kernel SHALL create the missing directories with their declared modes
- **AND** SHALL append a `data_tree_repaired` audit record
