## ADDED Requirements

### Requirement: Closed-Set Allowed-Field Schema

The usage-telemetry schema SHALL be a closed set of enums and bounded counters. The allowed fields SHALL be exactly: the SmallAIOS version string (e.g. `0.2.3`); the architecture class (`aarch64` / `x86_64` / `riscv64`); a coarse hardware class (`jetson-orin`, `nvidia-discrete`, `cpu-only`, `unknown` — not model number, not SKU); the cumulative boot count (resets on reflash); the compiled-in Cargo feature flags (e.g. `verified-boot=on`, `gpu-profile=off`, `bus-can=on`); capability counters (inferences run since install rounded to the nearest power of 2, models currently loaded — not their names, unique sessions opened rounded, audit-record category counts for auth-fail / power-control / update / config-change with no actors and no targets); crash categories from a documented enum list (no stack frames, no code paths, no addresses); and the stable per-install random UUID generated at opt-in.

#### Scenario: Allowed fields accepted by the validator

- **WHEN** a payload containing only the allowed fields — version string, architecture class, coarse hardware class, boot count, Cargo feature flags, rounded capability counters, crash-category counts, and install ID — is presented to the schema validator
- **THEN** the validator SHALL accept the payload

#### Scenario: Hardware class stays coarse

- **WHEN** the anonymizer classifies the running hardware
- **THEN** the emitted hardware-class value SHALL be one of `jetson-orin`, `nvidia-discrete`, `cpu-only`, or `unknown`
- **AND** no device model number or SKU SHALL appear in any field

#### Scenario: Feature flags are Cargo features only

- **WHEN** the feature-flag field is populated
- **THEN** it SHALL contain only compiled-in Cargo feature names and their on/off state
- **AND** SHALL NOT contain any value derived from user code or user configuration

#### Scenario: No free-text fields exist in the schema

- **WHEN** a reviewer reads the machine-readable schema in `docs/usage-telemetry.schema.json`
- **THEN** every field SHALL be a closed-set enum or a bounded counter
- **AND** no field SHALL accept free-form text

### Requirement: Forbidden Fields Enforced at the Schema Layer

The schema layer SHALL reject the following, with each rejection covered by a unit test: model file names, hashes, sizes, or contents; inference inputs, outputs, or shapes; any IP address (SmallAIOS SHALL never include its own; the relay strips connecting IPs before ingest); user names, passwords, or role definitions; account counts beyond the booleans "≥1 viewer exists" / "≥1 operator exists"; hostnames; any free-text field; anything from `automotive/uds.toml`; anything from `network/*.toml` beyond the bonded-mode enum; and configuration values from `auth/`, `mgmt/`, or `update/`.

#### Scenario: Model information rejected

- **WHEN** a payload carrying a model file name, hash, size, or content bytes reaches the schema validator
- **THEN** the validator SHALL reject the payload

#### Scenario: Inference content rejected

- **WHEN** a payload carrying inference inputs, outputs, or tensor shapes reaches the schema validator
- **THEN** the validator SHALL reject the payload

#### Scenario: Identity-bearing fields rejected

- **WHEN** a payload carrying an IP address, a hostname, a user name, a password, a role definition, or an account count beyond the "≥1 viewer exists" / "≥1 operator exists" booleans reaches the schema validator
- **THEN** the validator SHALL reject the payload

#### Scenario: Sensitive configuration sources rejected

- **WHEN** a payload carrying any value sourced from `automotive/uds.toml`, from `network/*.toml` other than the bonded-mode enum, or from `auth/`, `mgmt/`, or `update/` configuration reaches the schema validator
- **THEN** the validator SHALL reject the payload

### Requirement: Schema Version Field

The schema SHALL carry a `schema_version` field, bumped only when the schema changes. The `schema_version` value SHALL be present in `telemetry/usage.toml` (initial value `"0"`) and in every emitted payload, so the relay can accept known versions forward-compatibly and drop unknown versions.

#### Scenario: Initial schema version is zero

- **WHEN** a fresh `telemetry/usage.toml` is generated
- **THEN** it SHALL contain `schema_version = "0"`

#### Scenario: Schema change bumps the version

- **WHEN** any field is added to, renamed in, or removed from the usage-telemetry schema
- **THEN** the `schema_version` SHALL be bumped in the same change
- **AND** every payload emitted afterwards SHALL carry the new version

### Requirement: Category Opt-Outs

Even with usage telemetry enabled, an operator SHALL be able to suppress whole categories without disabling the channel, via the `[usage_telemetry.opt_outs]` table with boolean fields `crashes`, `counters`, and `features`, each defaulting to `false`.

#### Scenario: Crash category suppressed while channel stays up

- **WHEN** telemetry is enabled with consent recorded and the operator sets `opt_outs.crashes = true`
- **THEN** crash-category data SHALL NOT be emitted
- **AND** the remaining categories (counters, features) SHALL continue to be emitted
- **AND** the channel SHALL remain enabled

#### Scenario: Opt-out defaults are false

- **WHEN** a fresh `telemetry/usage.toml` is generated
- **THEN** `opt_outs.crashes`, `opt_outs.counters`, and `opt_outs.features` SHALL all be `false`

### Requirement: Metrics and Counter Events Only

The usage-telemetry channel SHALL carry only metrics and counter events. Logs and traces SHALL NOT be carried on this channel — permanently, not as a deferral. Counters SHALL be bucketed and rounded per the documented rules; no per-feature granularity beyond the documented enum SHALL be emitted.

#### Scenario: No log or trace path exists

- **WHEN** a reviewer reads the public API of the `telemetry/src/usage/` module
- **THEN** there SHALL be no code path that accepts log records or trace spans for usage-telemetry export

#### Scenario: No fine-grained usage timeline

- **WHEN** counters are emitted
- **THEN** they SHALL be bucketed and rounded values
- **AND** no minute-by-minute or per-event feature-usage timeline SHALL be derivable from a single payload

### Requirement: Machine-Readable Schema Dump and Review Gate

A `cargo xtask telemetry-schema-dump` recipe SHALL produce a machine-readable schema file checked into the repository at `docs/usage-telemetry.schema.json`, paired with the human-readable `docs/usage-telemetry.md`. Any change to the schema SHALL require updating both files, gated by code review. CI SHALL fail if the checked-in dump file is out of date with the code.

#### Scenario: xtask produces the schema dump

- **WHEN** `cargo xtask telemetry-schema-dump` is run
- **THEN** it SHALL emit the machine-readable schema for `docs/usage-telemetry.schema.json`
- **AND** the output SHALL enumerate every allowed field with its enum values or counter bounds

#### Scenario: Stale dump file fails CI

- **WHEN** the schema in code changes but `docs/usage-telemetry.schema.json` is not regenerated in the same change
- **THEN** the CI schema-dump check SHALL fail

#### Scenario: Schema change requires the privacy doc

- **WHEN** a change updates `docs/usage-telemetry.schema.json`
- **THEN** the same change SHALL update `docs/usage-telemetry.md` to match
- **AND** both SHALL pass through code review before merge
