# Design — crypto-validation-strategy-v1

## Context

The wolfSSL/wolfCrypt question (raised during `security-ecdsa-p256-v1`
planning, 2026-07-02) exposed that SmallAIOS's crypto strategy —
clean-room Rust, official-vector validation, no C dependencies — is
enforced only by convention and reviewer memory. The proposal records
the trade study; this design covers how the policy becomes durable
and mechanical. All heavy analysis lives in the proposal; this change
ships documentation and one config edit.

## Goals / Non-Goals

**Goals:**

- One authoritative in-tree document answering "why not wolfSSL/a
  validated library?" with explicit revisit triggers.
- Mechanical enforcement of the no-C-crypto rule through the existing
  `cargo-deny` CI gate — no new CI jobs.
- A stated corpus-replay bar that future crypto change proposals are
  reviewed against.

**Non-Goals:**

- Changing any `security/` code or test.
- Pursuing FIPS/CMVP validation now (the doc enumerates options for
  when a trigger fires).
- Banning C libraries generally — the ban list targets crypto
  libraries specifically; other C dependencies are governed by the
  existing clean-room conventions, not this change.

## Decisions

### D1. Enforce via `deny.toml` bans, not a bespoke check

The `[bans]` section of the existing `cargo-deny` config is the
enforcement point: it already runs as a blocking PR gate (Supply
Chain Security), understands transitive dependencies, and produces
actionable errors. Rejected: a grep-based CI script (blind to
transitive deps, another job to maintain) and convention-only (the
status quo this change exists to fix). Ban entries carry a `reason`
string pointing at `docs/crypto-validation.md`.

### D2. Decision record in `docs/`, not an ADR directory

The repo's documentation convention is flat topical files under
`docs/` (`docs/architecture.md`, `docs/scheduling-model.md`) linked
from CLAUDE.md — no ADR directory exists, and one policy record does
not justify inventing that structure. `docs/crypto-validation.md`
follows the existing pattern.

### D3. Corpus-replay requirement is review-enforced, not tooling-enforced

Requirement 1 (official corpus per primitive) cannot be checked
mechanically without heavy scaffolding (how would CI know a corpus is
"official"?). It is stated as a spec requirement so `/opsx`-driven
proposals get reviewed against it, and the existing primitives are
audited once (task 1.2) to confirm the claim the spec makes about
them. Accepted trade-off: enforcement quality depends on review
discipline; the spec scenario gives reviewers concrete wording to
point at.

## Risks / Trade-offs

- [Ban list is enumerative, not exhaustive — a new C-crypto binding
  crate under an unlisted name slips through] → the policy doc states
  the rule generally; the ban list is a tripwire for the common
  cases, and review covers the rest. Adding a name later is a
  one-line PR.
- [A future legitimate need for a banned crate (e.g. a dev-only tool
  pulling `openssl-sys` transitively)] → `cargo-deny` supports
  scoped `wrappers`/exceptions; the exception PR is itself the
  policy-review moment.
- [Existing tree might already pull a banned crate transitively,
  making the bans fail on day one] → task 2.2 runs the check before
  the PR lands; if a hit appears, the exception is documented in the
  same PR rather than silently widening the ban.

## Migration Plan

Docs + config only; single PR against `develop`. Rollback = revert.
No sequencing constraints with other changes.

## Open Questions

- None. Revisit triggers are enumerated in `docs/crypto-validation.md`
  rather than left open here.
