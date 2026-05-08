## Why

`telemetry-otel-export-v1` solves the problem of *operators*
shipping their *own* SmallAIOS metrics to *their own* backend.
That is first-party self-telemetry: the operator pays the bill,
holds the credentials, and trusts themselves with the data.

This proposal is for a different problem: when SmallAIOS has a
user base, **the project** will want a small amount of
anonymized signal about how the software is actually being
used in the wild. Which architectures see real traffic? Which
GPU backends are exercised on real hardware (vs. tested only
on the developer's bench)? Which features are present-but-
never-used? Which feature combinations correlate with crashes?

The conventional names for this are "product analytics" or
"usage telemetry." Sentry, PostHog, Mixpanel, Mozilla's Glean,
Brave's P3A, VS Code's telemetry, and Honeycomb all solve
varying flavors of it. The common ingredients:

1. A **public, write-only ingest endpoint** — no embedded
   shared secret that, if extracted, opens the door to anyone.
2. A **strict, documented, anonymized schema** — no PII, no
   inference content, no model names, no IP addresses. Only
   feature flags, version strings, hardware class, error
   categories, capability counters.
3. **Explicit opt-in** — never on by default, never enabled
   silently, never re-enabled by an update.
4. **Off by default in v1.** The whole machinery is wired and
   testable but the runtime feature is `enabled = false` until
   the project formally launches user telemetry collection.

The hard parts of this change are **not the code**. The code
is straightforward — most of it is the same exporter pattern
as `telemetry-otel-export-v1` minus the credential handling.
The hard parts are the **policy decisions**: which pattern
(relay / Faro / SaaS DSN), what schema, what privacy policy,
what opt-in UX, what the first-boot prompt actually says,
what happens when the user says no, how an operator finds
and reads the schema document. This proposal exists to
capture those decisions, lock them in, ship the
implementation in a disabled state, and make "flip the
switch when ready" a one-line config change rather than a
six-month design exercise.

**Disabled-by-default invariant.** The proposal scopes a
runtime feature whose default value is `false`, with three
layers of defense to prevent accidental flip-on:

1. The `Config` field defaults `enabled = false` in code.
2. The TOML schema defaults `enabled = false` if the field
   is missing.
3. A boot-time assertion logs a prominent warning and
   refuses to start the exporter if `enabled = true` *and*
   no `consent_recorded_at` timestamp exists in the same
   file (preventing a hand-edit that flips the switch
   without going through the opt-in flow).

## What Changes

### Pattern decision (open question, recommendation in v1)

Three credible patterns:

| Pattern | How | Tradeoffs |
|---|---|---|
| **A. SaaS-with-DSN** (Sentry, PostHog) | Embed a project-scoped, write-only token in the binary. Backend rate-limits per-tenant. | Easiest. Locked into a vendor. Their schema, their UI, their pricing. |
| **B. Project-run relay** | Project hosts a public unauth'd ingest endpoint (Cloudflare Worker / Fly / Lambda). Relay re-stamps with the project's real backend credential and forwards. | Most control. Full schema validation at the edge. ~free at small scale. Project becomes responsible for an ops endpoint. |
| **C. Grafana Faro** | Grafana's RUM product gives a public-write endpoint per project. Same data model the operator-side `telemetry-otel-export-v1` already speaks. | Reuses our OTLP work. Locked into Grafana Cloud (paid tier likely needed at modest scale). Faro is RUM-flavored; we are not RUM. |

**Recommendation: Pattern B (project-run relay).** Reasons:

- **Optionality.** A relay is just a thin auth + filter
  layer; the backend behind it can be Grafana Cloud,
  self-hosted Mimir, ClickHouse, or any combination.
  Switching backends is a relay-config change, no client
  update required.
- **Schema enforcement at the edge.** The relay can reject
  requests that exceed the documented schema, strip
  unexpected attributes, drop oversized payloads. This is
  the strongest defense against "a developer accidentally
  added a sensitive field."
- **Cost control.** Free tier on Cloudflare Workers is
  100k requests/day — comfortably more than a hobby-scale
  user base produces — and scales linearly with paid plans.
- **Audit-friendly.** The relay's source is the same repo
  as SmallAIOS; reviewers can see exactly what the relay
  accepts, transforms, and drops.

Pattern A and C are both viable fallbacks if running an
ops endpoint becomes a problem; the wire format we ship is
designed to be retargetable.

### Schema (anonymized, capability-coarse)

The exact field list lives in `design.md` once the pattern
is locked. The proposal-level rules:

**Allowed:**

- SmallAIOS version (`0.2.3`).
- Architecture class (`aarch64` / `x86_64` / `riscv64`).
- Hardware class — coarse: `jetson-orin`, `nvidia-discrete`,
  `cpu-only`, `unknown`. **Not** model number, **not** SKU.
- Boot count (cumulative; resets on reflash).
- Feature flags compiled in (e.g. `verified-boot=on`,
  `gpu-profile=off`, `bus-can=on`) — Cargo features only,
  not user code.
- Capability counters: number of inferences run since
  install (rounded to nearest power of 2), number of
  models *currently* loaded (not their names), number of
  unique sessions opened (rounded), audit-record
  *categories* hit (auth-fail / power-control / update /
  config-change — counts only, no actors, no targets).
- Crash categories — category enum from a documented list,
  no stack frames, no code paths, no addresses.
- A stable per-install random ID (UUID, generated at
  opt-in time, persistable across reboots, **not** the
  same UUID `telemetry-otel-export-v1` uses for `host.id`
  — those leak if combined).

**Forbidden, enforced at the schema layer:**

- Model file names, hashes, sizes, contents.
- Inference inputs / outputs / shapes.
- Any IP address (the relay strips connecting IPs before
  ingest; SmallAIOS never includes its own).
- User names, passwords, role definitions, account counts
  beyond "≥1 viewer exists / ≥1 operator exists" booleans.
- Hostnames.
- Any free-text field. The schema is closed-set enums
  and bounded counters only.
- Anything from `automotive/uds.toml` (CAN-bus deployments
  are inherently identifiable).
- Anything from `network/*.toml` beyond the bonded-mode
  enum.
- Configuration values for `auth/`, `mgmt/`, `update/`.

**Opt-out tier:** even with telemetry on, an operator can
set `[opt_outs] crashes = true / counters = true / ...` to
suppress whole categories without disabling the channel.
Useful for high-security customers who want to participate
in "is the bug fixed in 0.3?" but not "how often is the
crash hit."

### Consent / opt-in flow

- **First boot:** if `enabled` is unset (the file does not
  yet contain the field), the TTY first-boot setup —
  *after* the root password is set — prompts:

      Help improve SmallAIOS by sharing anonymized usage
      counters? See /docs/usage-telemetry.md for the full
      schema. [y/N]:

  Default is **N**. Answering `y` writes
  `enabled = true` *and* `consent_recorded_at = <unix
  timestamp>` to the file.

- **Zenoh admin verb:** `smallaios/admin/telemetry/usage/opt_in`
  with body `{accept: true, version: "<schema-version>"}`.
  Records the same consent timestamp.

- **Zenoh admin verb:** `smallaios/admin/telemetry/usage/opt_out`
  flips `enabled = false` and writes
  `consent_revoked_at = <unix timestamp>`. Future flip-back
  requires a new explicit opt-in.

- **Privacy policy** (gating). `docs/usage-telemetry.md`
  must exist and be reviewed before the project flips the
  default to `enabled = true`. The proposal explicitly
  *does not* propose flipping the default; that is a
  separate, gated event and not a code change.

### Configuration: `telemetry/usage.toml`

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

The endpoint is a constant baked into the build (the
relay URL). Operators *cannot* point the usage telemetry
at an arbitrary endpoint — that defeats the schema-
enforcement-at-the-edge guarantee. Operators who want
*self*-telemetry use `telemetry-otel-export-v1`.

### Implementation in v1 (disabled state)

The implementation lands in v1, but the `enabled = false`
invariant means the exporter never runs by default. The
goal of v1 is:

1. The on-box anonymizer + schema validator + serializer
   exist and are unit-tested against the documented schema.
2. The opt-in UX (TTY + Zenoh verb) exists and is tested.
3. The endpoint URL is a build-time constant pointing at a
   project-controlled DNS name (the actual relay does not
   need to exist *yet* — the URL just needs to be wired so
   that flipping the switch is a relay deployment, not a
   client release).
4. CI tests assert the disabled-by-default invariant
   cannot regress: a build that defaults `enabled = true`
   without a corresponding policy change fails the build.
5. A `cargo xtask telemetry-schema-dump` produces a
   machine-readable schema file checked into the repo;
   any change to the schema requires updating the file
   *and* the privacy doc, both gated by code review.

### Out of scope for v1 (flagged)

- **Operating the relay.** Cloudflare Worker (or
  whichever runtime is chosen) lives in a separate repo
  with separate review. This change ships only the
  client-side wiring.
- **Server-side aggregation tooling, dashboards, alerting.**
  Out of scope for SmallAIOS proper.
- **Logs / traces.** v1 is metrics + counter events only.
  Logs are too easy to mishandle PII. Traces are
  implementation-internal and high-cardinality. Both
  permanently out-of-scope for project usage telemetry —
  not a deferral.
- **Per-feature granularity beyond the documented
  enum.** No "minute-by-minute feature usage." Counters
  are bucketed and rounded to discourage fine-grained
  inference.
- **A/B experiment assignment / feature-flag delivery
  back-channel.** This change is one-way: client →
  relay. No remote configuration of SmallAIOS via this
  channel.
- **The actual decision to flip `enabled = true` by
  default.** That is a project-leadership decision
  involving privacy review, community communication, and
  release-notes language; it is **not** a code change in
  this proposal.

## Capabilities

### New Capabilities

- `project-usage-telemetry-disabled-by-default`: the three-
  layers-of-defense invariant (code default, schema default,
  boot-time assertion-with-warning), the CI test that
  enforces it, and the rule that flipping the default is a
  policy event, not a refactor.
- `project-usage-telemetry-schema`: the closed-set enum +
  bounded-counter rules, the explicit allow-list, the
  explicit forbid-list, the schema-version field, the
  `cargo xtask telemetry-schema-dump` artifact and its
  review-gate.
- `project-usage-telemetry-anonymizer`: the on-box
  anonymizer and validator, the rule that anonymization
  happens *before* serialization (defense-in-depth against
  a serializer bug leaking unanonymized fields), the
  rounding / bucketing rules, the install-ID-vs-host-ID
  separation.
- `project-usage-telemetry-opt-in-ux`: TTY first-boot
  prompt wording + default, Zenoh opt-in / opt-out verbs,
  consent timestamp persistence, opt-out preserving the
  install ID for "did the bug-fix work" longitudinal
  tracking *only if the operator opts back in later*.
- `project-usage-telemetry-pattern-decision`: documents the
  recommended Pattern B (project-run relay), the rationale,
  and the deliberate non-decisions (which backend behind
  the relay, which hosting platform, the actual relay
  source code).

### Modified Capabilities

- `mgmt-config-layout`: adds `telemetry/usage.toml`, the
  rule that its `endpoint` field is **read-only** to
  operators (the build-time constant is authoritative;
  operator overrides are rejected by the loader), and the
  rule that the `install_id` field is generated at opt-in
  and never re-rolled by an update.
- `console-login` (from `management-login-v1`): adds the
  first-boot opt-in prompt to the post-root-password setup
  sequence.

## Impact

- **Code:**
  - Reuses `telemetry/` crate from
    `telemetry-otel-export-v1` for the protobuf encoder
    and HTTP transport — *not* for the resource model
    (different identity rules) or the buffer (we accept
    drops more aggressively here; usage telemetry is best-
    effort).
  - New `telemetry/src/usage/` module: anonymizer, schema
    validator, opt-in state machine, ~400 LOC.
  - Schema definition file checked into
    `docs/usage-telemetry.schema.json` (machine-readable)
    + `docs/usage-telemetry.md` (human-readable).
  - `cargo xtask` recipe: `telemetry-schema-dump`.
  - Build-time constant `USAGE_TELEMETRY_ENDPOINT` defaulting
    to a project-controlled hostname (the relay URL).
- **Tests:** ~50 new tests targeted: disabled-by-default
  invariant CI test, schema validator (every forbidden
  field is rejected; every allowed field is accepted),
  rounding / bucketing math, opt-in state machine, opt-out
  state machine, consent timestamp persistence,
  anonymizer-before-serializer ordering test, the
  schema-dump check (CI fails if the dump file is out of
  date).
- **Boot footprint:** ~25 KB code, ~64 KiB live. Zero
  runtime cost while disabled (which is always, until the
  project flips it).
- **Network:** zero in default state.
- **Downstream:** unblocks project-leadership decision to
  collect anonymized usage data when ready, without that
  decision becoming a six-month engineering project. Sets
  the precedent that any future telemetry path goes
  through the same anonymizer + schema-validator + opt-in
  pipeline.
- **Dependencies:** `management-login-v1` (auth, surface
  convention, audit log, TTY first-boot flow);
  `telemetry-otel-export-v1` (protobuf encoder, HTTP
  transport — reused, not re-implemented). No dependency
  on `system-power-control-v1`, `remote-update-v1`,
  `network-management-v1`, `automotive-bus-management-v1`,
  or `console-monitor-v1`.
- **Risks:**
  (1) **The most consequential risk in the entire OpenSpec
  set.** A bug that exfiltrates a forbidden field is a
  privacy incident, not a software bug. Mitigation: the
  anonymizer-before-serializer rule, the schema-dump
  CI check, the relay-side schema validation as a
  second-line defense. Reviewer attention warranted.
  (2) Default-flip-on by accident. Mitigation: the three-
  layer disabled-by-default invariant + CI test.
  (3) Install-ID collision with `host.id` from
  `telemetry-otel-export-v1`. The two must be different
  random UUIDs so that joining the two telemetry streams
  cannot re-identify a host. Anonymizer test asserts
  this.
  (4) Schema drift between client and relay. Versioning
  via `schema_version`; relay accepts versions ≤ N + 1
  and drops with a logged metric otherwise.

## Open Questions

1. **Pattern A / B / C — confirm Pattern B?** Recommendation
   is the relay. Decision needed before design.md.
2. **Where does the relay live?** Cloudflare Worker is the
   default recommendation; alternatives are Fly.io (closer
   to a real Rust app), Lambda, Deno Deploy. Out of this
   change's scope but informs the endpoint URL we bake in.
3. **What does the first-boot prompt actually say?** Open
   for word-by-word review when ready. The proposal
   commits to "default N, link to a doc, two-character
   answer" but the wording will go through review.
4. **Counter buckets**: nearest power of 2 is conservative;
   nearest order of magnitude is more conservative.
   Pick one and document.
5. **Should the install ID survive a `git pull && reflash`?**
   It lives in `/data/`, so yes by default — but if the
   user reflashes `/data/`, it regenerates. Acceptable
   asymmetry, document it.
6. **What is the schema-version cadence?** Bumped on every
   added field (relay accepts N+1 forward-compatibly), or
   only on breaking changes (renamed / removed fields)?
   Leaning every-change for simplicity; the relay's
   forward-compat rule absorbs the noise.
7. **Crash category enum** — initial set: `panic`,
   `oom`, `gpu-fault`, `verified-boot-fail`,
   `update-rollback`, `auth-lockout`, `other`. Is `other`
   acceptable for v1? It's a slight abstraction leak.
   Leaning yes — categorization can be refined as we
   collect data.
