# mgmt-config-surface-trait Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: ConfigSurface trait shape
The `mgmt` crate SHALL define a `ConfigSurface` trait with the operations:

```rust
trait ConfigSurface {
    fn read(&self, path: &ConfigPath) -> Result<Value, Error>;
    fn write(&mut self, path: &ConfigPath, value: Value) -> Result<(), Error>;
    fn subscribe(&self, path: &ConfigPath) -> Result<Stream<Value>, Error>;
}
```

Every active management surface SHALL implement this trait. Adding a new surface (UDS over CAN, future REST proxy, etc.) SHALL be one trait impl; the option taxonomy, validators, audit records, and persistence SHALL be reused unchanged via the apply lifecycle in `mgmt-config-model`.

#### Scenario: TOML surface implements ConfigSurface
- **WHEN** the TOML loader is the surface
- **THEN** `read` SHALL parse `/data/<file>.toml` and return the value
- **AND** `write` SHALL go through the apply lifecycle
- **AND** `subscribe` SHALL emit on every successful write

#### Scenario: Zenoh surface implements ConfigSurface
- **WHEN** the Zenoh admin handler is the surface
- **THEN** `read` SHALL serve `smallaios/admin/config/get` queries
- **AND** `write` SHALL serve `smallaios/admin/config/set` queries
- **AND** `subscribe` SHALL forward notify events to `smallaios/admin/config/changed`

### Requirement: Universal-exposure invariant
Every option in `Config` SHALL be reachable from **all** active surfaces. There SHALL be no surface-specific knob, no "you can only set this from the console" exception unless explicitly declared. A build-time CI test SHALL walk the `Config` schema and SHALL fail compilation if any field is missing a handler in any surface. Additionally, a runtime sanity check at boot SHALL log a warning if the invariant is violated under the active feature combination.

#### Scenario: Build fails on missing handler
- **WHEN** a developer adds a `Config` field but only wires the TOML loader
- **THEN** the build-time CI walker SHALL fail with an error naming the field and the surfaces that lack handlers

#### Scenario: Runtime warns on feature-flag escape
- **WHEN** boot completes under a feature combination that omits a surface handler
- **THEN** a warning SHALL be emitted naming the unreachable field and the absent surface
- **AND** the boot SHALL continue (the build-time gate is the primary safety; runtime is defense in depth)

### Requirement: Surface-only escape hatch
Fields that genuinely cannot be exposed everywhere (e.g., a one-shot recovery action only valid on physically-present TTY) SHALL declare a `#[surface(only = "tty")]` attribute. The build-time walker SHALL honor the attribute and SHALL only require handlers on the named surfaces.

#### Scenario: TTY-only field accepted
- **WHEN** a field is annotated `#[surface(only = "tty")]`
- **THEN** the walker SHALL only require a TTY handler and SHALL NOT flag missing handlers on other surfaces

#### Scenario: Mistyped surface name fails build
- **WHEN** a field is annotated `#[surface(only = "stty")]` (typo)
- **THEN** the walker SHALL fail compilation with a message listing the recognized surface names

