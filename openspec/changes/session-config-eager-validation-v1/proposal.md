## Why

`Session::new(config: SessionConfig) -> Self` is infallible today, but
`SessionConfig` can carry invalid values. `StreamConfig::Overlap {
transfer_streams }` is only checked (`<= 2`) lazily in
`Session::ensure_stream_pool`, on the first multi-stream inference — and
that check is additionally `#[cfg(feature = "cuda")]`-gated, so a
non-cuda build accepts `transfer_streams: 5` silently and forever. A
user passing an invalid config learns about it (if at all) at first
inference, not at construction.

GitHub Copilot Code Review flagged this on PR #125; it was deferred from
`codeql-quality-cleanup-v1` because it is a real API-design decision,
not a code-quality cleanup (tracked as issue #127).

## What Changes

- **New `SessionConfig::validate(&self) -> Result<(), SessionError>`** —
  the single, **feature-independent** validation authority for a
  configuration. v1 validates the known constraint (`transfer_streams
  <= 2`); the method is the extension point for future config
  invariants. It does not depend on the `cuda` feature.
- **`Session::new` becomes fallible:** `Session::new(config) ->
  Result<Self, SessionError>`, calling `validate()` before constructing.
  All in-repo callers (`container/src/main.rs`, the `onnx-rt` test
  suite — ~26 sites) are updated to `?`/`unwrap`. This is a breaking
  change to the constructor signature; pre-1.0 with every caller in-tree
  it is a `feat!` minor bump (see design.md for the alternatives
  considered).
- **`ensure_stream_pool` keeps its check** as a defence-in-depth
  backstop (its `SessionError::InvalidConfig` path is unreachable once
  construction validates, but harmless and cheap to retain).

## Capabilities

### Modified Capabilities

- `onnx-runtime`: `Session` construction gains an eager, fail-closed
  configuration-validation requirement, and the constructor contract
  changes from infallible to fallible.

## Impact

- **Code:** `onnx-rt/src/session.rs` (add `SessionConfig::validate`,
  change `Session::new` signature); ~26 call sites in
  `container/src/main.rs` and `onnx-rt/tests/*`.
- **API:** breaking signature change on `Session::new`
  (`cargo-semver-checks` will flag it; PR title carries `!`).
- **Tests:** a unit test asserting `SessionConfig::validate` rejects
  `transfer_streams > 2` in both cuda and non-cuda builds; a test that
  `Session::new` surfaces the error eagerly.
- **Dependencies:** none.
