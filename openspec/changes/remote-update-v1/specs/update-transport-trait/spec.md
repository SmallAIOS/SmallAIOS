## ADDED Requirements

### Requirement: `update::Transport` Plugin Contract

The `update` crate SHALL define a `Transport` trait that every update wire format satisfies:

```rust
fn begin(manifest: &Manifest) -> Result<SessionId>;
fn recv_chunk(session: SessionId, index: u32) -> Result<&[u8]>;
fn commit(session: SessionId) -> Result<()>;
fn abort(session: SessionId);
```

This change SHALL ship two implementations, `TtyYmodemTransport` and `ZenohChunkedTransport`. A third, `UdsIsoTpTransport`, is deferred to `automotive-bus-management-v1` and SHALL be addable without modifying the trait — same manifest, same signature check, same boot pointer; only the wire framing differs.

#### Scenario: Both v1 implementations satisfy the trait

- **WHEN** a reviewer reads the public API of the `update` crate
- **THEN** `TtyYmodemTransport` and `ZenohChunkedTransport` SHALL both implement `update::Transport`
- **AND** the manifest parser, signature verifier, and slot writer SHALL be invoked through the trait, not through transport-specific paths

#### Scenario: Transport-trait conformance suite runs per implementation

- **WHEN** the test suite runs
- **THEN** a shared transport-conformance test SHALL execute against each `Transport` implementation
- **AND** SHALL cover the begin → recv_chunk → commit happy path and the abort path for each

#### Scenario: Future UDS transport plugs in without trait changes

- **WHEN** `automotive-bus-management-v1` later adds `UdsIsoTpTransport`
- **THEN** the `Transport` trait signatures SHALL NOT need to change
- **AND** the shared manifest/signature/boot-pointer pipeline SHALL be reused as-is

### Requirement: New Layer-1 `update` Crate

The workspace SHALL gain a new `update/` crate at Layer 1 (Core Services) containing the manifest parser, the slot writer, the `Transport` trait, and the watchdog wiring. The crate SHALL respect the strict 4-layer acyclic dependency model, depending only on Layer 0/1 crates.

#### Scenario: Crate sits at Layer 1 without cycles

- **WHEN** `just arch-check` and the crate-level cycle check run after the crate is added
- **THEN** the `update` crate SHALL appear in Layer 1
- **AND** it SHALL introduce zero production-dependency cycles
- **AND** it SHALL NOT depend on any Layer 2/3 crate
