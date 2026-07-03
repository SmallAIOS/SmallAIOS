# Design — session-config-eager-validation-v1

## Context

`Session::new` (`onnx-rt/src/session.rs`) is a pure, infallible field
initializer. The only `SessionConfig` invariant enforced anywhere is
`transfer_streams <= 2`, and it lives in `ensure_stream_pool`
(`#[cfg(feature = "cuda")]`, lazy). Two problems: (1) invalid configs
survive construction and surface late or never; (2) the check is
cuda-gated, so non-cuda builds never validate at all. Issue #127 asks
for eager validation and enumerates four API options; this design picks
one.

## Goals / Non-Goals

**Goals:**

- Reject invalid `SessionConfig` at construction, feature-independently.
- One validation authority (`SessionConfig::validate`) that future
  config invariants extend, rather than scattered ad-hoc checks.

**Non-Goals:**

- New config constraints beyond the existing `transfer_streams <= 2`.
- Removing the `ensure_stream_pool` check (kept as a backstop).
- Changing `SessionConfig`'s fields or `StreamConfig`'s shape.

## Decisions

### D1. Make `Session::new` fallible (Option 1)

`Session::new(config) -> Result<Self, SessionError>`, validating before
construction. Chosen over the three alternatives:

| Option | Verdict | Why |
|---|---|---|
| **1. `new` → `Result`** | **Chosen** | One eager, honest constructor. Only cost is caller churn — all ~26 sites are in-repo, and pre-1.0 the repo treats this as a `feat!` minor bump. |
| 2. Add `try_new`, keep `new` | Rejected | Two constructors doing the same job; `new` stays a silent-accept footgun, which is the defect we are fixing. |
| 3. Panic in `new` | Rejected | A library panicking on data-derived input is the wrong failure mode; it also can't be handled by callers. |
| 4. `validate()` callers must remember to call | Rejected | Implicit contract, trivially forgotten — no better than today in practice. |

The winning shape still *produces* Option 4's `SessionConfig::validate`
— but as the internal authority `new` calls, not a method callers must
remember. Best of both.

### D2. `validate` is feature-independent and total over known invariants

`SessionConfig::validate(&self) -> Result<(), SessionError>` lives
outside any `cfg(feature)` gate and matches on `stream_config`,
returning `SessionError::InvalidConfig` for `Overlap { transfer_streams
}` with `transfer_streams > 2`. Rationale: the invalid value is invalid
regardless of whether a CUDA stream pool will ever be built from it;
gating validation on `cuda` is exactly the current bug.

### D3. Keep the `ensure_stream_pool` check as a backstop

Its `InvalidConfig` branch becomes unreachable once construction
validates, but removing it would make the cuda path depend on the
invariant being enforced elsewhere. Retained (cheap, defence-in-depth);
a comment notes construction is the primary gate.

## Risks / Trade-offs

- [Breaking `new` signature churns every caller] → all ~26 are in-repo
  (`container/src/main.rs`, `onnx-rt/tests/*`); the change is mechanical
  (`Session::new(cfg)` → `Session::new(cfg)?` or `.unwrap()` in tests).
  `cargo-semver-checks` flags it; the PR title carries `!` so the semver
  gate passes.
- [Downstream/external callers (none known) would break] → pre-1.0,
  no stability guarantee; acceptable and documented in the changelog.
- [Backstop check now dead code under coverage] → it stays exercised by
  a direct `ensure_stream_pool` unit test if one exists; otherwise it is
  a cheap unreachable guard, not counted against the coverage gate
  because the eager test covers the same branch in `validate`.

## Migration Plan

Single PR against `develop`. Steps: add `validate`, change `new`, update
callers, add tests. Rollback = revert. No data migration.

## Open Questions

- None. The API-shape decision (the issue's central question) is
  resolved as D1.
