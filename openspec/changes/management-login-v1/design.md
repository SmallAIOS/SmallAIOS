## Context

SmallAIOS boots cleanly across x86-64, AArch64, and the Jetson Orin
KVM path, but has no notion of "who is on the box." There is no
local login, no password store, no audit identity on syscalls, no
remote management or monitoring channel, and no way to give a
serving-team SRE a read-only view of the running model server
without handing them full kernel privileges. Pulling in OpenSSH
would balloon the attack surface and fight the unikernel design.
This change wires the existing PQC-backed Zenoh + QUIC + TLS 1.3
stack and the serial UART driver into a small management layer:
local console login with three roles (`Root`, `Operator`,
`Viewer`), a shadow-style password file with Argon2id, a Zenoh
admin/telemetry channel, an in-kernel audit log, and the
**management surface convention** every follow-on management
feature inherits.

This document captures every decision made during the open-questions
walkthrough so design.md is the single source of truth that the
spec deltas, tasks.md, and implementation crates all derive from.

## Goals / Non-Goals

**Goals:**
- Three-role identity/access model (`Root`, `Operator`, `Viewer`) on
  every reachable management surface.
- Shadow-style password store with Argon2id (per-tier parameters
  embedded in the PHC string).
- Local TTY login with first-boot setup, lockout, idle auto-logout,
  echo-off prompt, recovery boot arg gated on physical presence.
- Remote Zenoh admin/telemetry surface bound to PQC peer identity.
- Single typed `Config` source of truth in a new `mgmt/` Layer 1 crate
  with the `ConfigSurface` trait and universal-exposure invariant.
- In-kernel audit ring with SHA-3-256 hash chain, periodic flush to
  `/data/audit/log.jsonl`, hybrid rotation, optional ML-DSA-65
  signed checkpoints.
- TOTP (RFC 6238) opt-in second factor, off by default.
- New `auth/` Layer 1 crate (workspace 21 → 22) and new `mgmt/`
  Layer 1 crate (workspace 22 → 23).
- ≥4500 total tests after change (≈ +360 new tests).
- Formal models: TLA+ session-state, Kani audit-chain & shadow
  rewrite, SPIN lockout interleaving.

**Non-Goals (v1):**
- SSH / sshd. Re-evaluate in v2 only with a hard requirement.
- Multi-tenant federation, group-based RBAC, sudoers, capability
  inheritance across exec.
- PAM / Kerberos / LDAP.
- Off-box audit log shipping. Local persistence + chain only.
- Hardware-backed credential storage (TPM key unwrap).
- gRPC. Zenoh request/response is the only RPC surface.

## Decisions

The 40 questions resolved during the design walkthrough. Each
decision is the answer to one question and is the source of the
corresponding requirement(s) in the spec deltas.

### Q1. Crate boundary for auth code

**Decision:** New `auth/` Layer 1 crate, depending only on
`security` (Layer 0) + `kernel` (Layer 0).

**Rationale:** Clear separation of identity/access from crypto
primitives. Owns shadow parsing, Argon2id wrapper, role table,
session-table API. Workspace 21 → 22.

### Q2. Bearer-token format for Zenoh admin

**Decision:** Default is an opaque random 16-byte ID with
server-side session-table lookup. ML-DSA-65 signed-token mode is
available behind the `mgmt-token-mldsa` cargo feature for
higher-assurance deployments. An Ed25519 signed-token mode is
gated behind the `mgmt-token-ed25519-legacy` feature, which is
**off by default and excluded at compile time** when off (no
Ed25519 code linked into the production binary).

**Rationale:** Opaque IDs are simpler, allow instant revocation,
and keep the kernel session table the single source of session
truth. ML-DSA-65 is offered for clients that need to reconnect
without a kernel round-trip and matches the project's PQC stance.
Ed25519 is preserved only as a transition path and must be
removable at compile time.

### Q3. Universal-exposure invariant enforcement

**Decision:** Build-time CI test (hard fail) + runtime boot-time
sanity check (warn-and-continue, defense in depth).

**Rationale:** Build time prevents regressions reaching production.
Runtime check catches the rare case where a feature flag composition
subverts the build-time walker. Pair with `cargo xtask
scaffold-config-field` to ease developer friction.

### Q4. `mgmt/` crate placement

**Decision:** New `mgmt/` Layer 1 crate, separate from `auth/`.

**Rationale:** Identity/access (`auth`) and config-of-everything
(`mgmt`) are conceptually distinct and `mgmt` will grow as more
features add fields. Workspace 22 → 23.

### Q5. Per-role idle-timeout defaults

**Decision:** `Root` 15 min, `Operator` 60 min, `Viewer` 60 min.
Any keypress resets the timer (so a viewer running
`console-monitor-v1` `top`-style remains logged in while actively
watching). All three thresholds live in `mgmt/policy.toml`.

### Q6. First-boot recovery path

**Decision:** `auth.skip-firstboot` boot arg honored only when a
target-specific physical-presence indicator is asserted. See Q7
for the indicator definition.

### Q7. Physical-presence indicator

**Decision:** `arch::physical_presence()` is a trait
(`PhysicalPresenceProvider`) that returns `false` by default;
each arch contributes one or more provider impls. v1 ships:
- x86: configurable GPIO line + ACPI `\_SI._SST` hint provider
- AArch64 / Jetson: `peripheral::gpio` line provider
- QEMU: `fw_cfg` flag (`opt/smallaios/phys-presence`) provider
- RISC-V: GPIO line provider
- `verified-boot`-trusted boot arg provider (honored only if the
  kernel is signature-verified by the `verified-boot` feature)

A TPM-backed provider is the planned next addition. The trait
allows future architectures to register without changing the gate.

### Q8. Auth syscall ABI

**Decision:** Match the existing kernel syscall convention used by
the current ~46 syscalls. Pointers + lengths in successive arg
registers, return value in the standard return register. Zenoh
admin is a thin authenticated wrapper that ultimately calls the
same syscalls via the `mgmt` adapter — single source of truth for
auth checks.

### Q9. Error encoding

**Decision:** Numeric POSIX-aligned `errno` codes (`-EPERM`,
`-EACCES`, `-EAGAIN`, `-EINVAL`) reused from the `posix` crate.
Auth-specific cases that have no POSIX equivalent (e.g.,
`-EAUTHEXPIRED` for must-change-password gating) are added with the
same convention. A shared `error_string()` table provides uniform
messages across TTY, Zenoh, and future UDS surfaces.

Constant-time-equivalent rejection of "user does not exist" vs
"password incorrect" prevents user enumeration on every surface.

### Q10. Audit ring format

**Decision:** Fixed 4 KiB × 4096 = 16 MiB in-memory ring of
structured records `{ ts, who, surface, action, before, after,
code }`. Background flusher writes append-only JSONL to
`/data/audit/log.jsonl` every 1 s or on auth events.

### Q11. Audit log rotation

**Decision:** Hybrid (size OR time) configurable rotation plus a
hard failsafe.

`mgmt/policy.toml` exposes:
- `audit.rotate_size_bytes` (default 64 MiB)
- `audit.rotate_age_hours` (default 24)
- `audit.keep_archives` (default 8)
- `audit.max_total_disk_bytes` (default 512 MiB) — **enforced
  failsafe**: when exceeded, the oldest archive is evicted
  regardless of retention to guarantee the writer never blocks on
  a full disk

### Q12. Skip-firstboot credentials

**Decision:** Default is a cryptographically random 16-char root
password, hashed with Argon2id, written to `/data/auth/shadow`,
printed once on the serial console, and recorded in the audit log
with `must_change_password_on_login` set. An optional
`auth.skip-firstboot=<argon2id-hash>` boot-arg variant accepts a
pre-computed hash so operators can pre-bake credentials without a
race against attackers reaching TTY first.

### Q13. UART raw / echo-off mode

**Decision:** v1 ships per-read `ReadOptions { echo: bool, raw:
bool }` (no global state). Defaults are `echo=true, raw=false`.
RAII guard pattern (option C from the question) and a separate
`LineDiscipline` layer (option D) are documented evolution paths.

### Q14. Kernel session table

**Decision:** Fixed-size `[Option<Session>; SESSION_TABLE_CAP]`,
`SESSION_TABLE_CAP` = 32 by default. A build-time const allows
small-RAM targets (8) and large appliances (256). `mgmt/policy.toml`
allows runtime resize within the build-time cap. Heap-allocated
`Vec<Session>` is a v2 evolution path.

### Q15. Zenoh admin handler placement

**Decision:** Implemented inside the `mgmt/` crate as a
`ConfigSurface` impl. `Mgmt::serve(&Session)` is the entry point
`container/` calls during boot.

### Q16. Spec delta granularity

**Decision:** All 9 new + 4 modified capabilities receive their
own spec deltas as listed in the proposal:
- New: `auth-shadow`, `auth-roles`, `console-login`,
  `mgmt-zenoh-admin`, `mgmt-zenoh-telemetry`, `mgmt-config-model`,
  `mgmt-config-surface-trait`, `mgmt-config-layout`,
  `mgmt-audit-log`
- Modified: `kernel-syscalls`, `security`, `peripheral-uart`,
  `ipc-zenoh`

### Q17. Shadow boot states

**Decision:** Missing → enter first-boot setup. Corrupt (parse
fail / mode laxer than declared / truncated) → kernel halts with
an explicit recovery hint pointing at `auth.skip-firstboot`.
Refuses to silently regenerate a corrupt shadow because the
corruption may be tampering.

### Q18. must-change-password gating

**Decision:** Whitelist: `auth_change_password`, `auth_whoami`,
`auth_logout` only. All other syscalls return `-EAUTHEXPIRED`
until the password is rotated.

### Q19. Audit-on-denial

**Decision:** Every `min_role` denial is appended to the audit
ring with full call context (`who, surface, action=DENY:<syscall>,
code=-EPERM`). Rate-limited to 10/s per user to prevent log
flooding from a buggy or hostile client.

### Q20. Lockout policy

**Decision:** 5 consecutive failed login attempts per source → 60 s
lock for that source. Counter resets on success. TTY counts as one
source; Zenoh counts per remote PQC peer identity. A locked
remote attacker cannot lock out the local TTY operator.

### Q21. Second factor

**Decision:** TOTP (RFC 6238) optional in v1, off by default.
Implemented in `security/src/totp.rs` (or `auth/src/totp.rs`),
exposes a `totp_setup` syscall, adds a `totp_secret` column to
shadow and a `totp_required` per-user policy bit.

`auth_login` ABI accepts `factor2_ptr/factor2_len`; values are
empty/zero when TOTP is not enrolled. (Standard TOTP uses SHA-1; a
SHA-3 variant is documented as a possible extension that does not
break interoperability.)

### Q22. Zenoh per-identity caps

**Decision:** 4 concurrent admin sessions per remote identity, 16
total Zenoh slots out of the 32-slot session table. Excess login
requests get `-EAGAIN`.

### Q23. Telemetry cadence

**Decision:** Default 1 Hz per metric key, configurable per-key in
`mgmt/policy.toml`. Bounds: min 100 ms, max 60 s.

Optional adaptive mode: `metrics.<key>.adaptive = { threshold,
fast_hz, slow_hz }` upgrades to fast cadence when the threshold is
crossed and falls back when it clears.

### Q24. Audit tamper resistance

**Decision:** Always-on SHA-3-256 hash chain — every record
contains `prev_hash`. Latest fingerprint is exposed via
`audit_read` and as a streamed metric so an external watcher can
detect rollback or excision.

Optional ML-DSA-65 signed-checkpoint mode in `mgmt/policy.toml`:
`audit.signed_checkpoints = { enabled, interval = 1024 }`. When
enabled, every Nth fingerprint is signed by the kernel-held
ML-DSA-65 key; off-box verifiers with the public key can detect
tampering even if the signing key is compromised after the fact.

### Q25. Live-reload field annotation

**Decision:** Per-field `#[reload("live"|"boot")]` attribute on
the `Config` struct, default `live`. The build-time schema walker
also enforces every field has a declared kind, plus the
universal-exposure invariant from Q3.

### Q26. Argon2id parameters

**Decision:** Per-tier defaults stored in the PHC-format hash so
verification is self-describing:
- `tiny` — `m=8 MiB, t=3, p=1` for ≤256 MiB RAM
- `default` — `m=64 MiB, t=3, p=1`
- `strong` — `m=128 MiB, t=4, p=2` for Jetson-class

Kernel measures available RAM at first boot and selects.

### Q27. Shadow record format

**Decision:** Colon-separated, PHC hash, append-only fields:

```text
username:$argon2id$...:role=<root|operator|viewer>:flags=<u32>:last_changed=<unix-day>:totp_secret=<base32-or-empty>:lockout_until=<unix>
```

Parsers ignore unknown trailing fields so future versions can
extend without breaking existing shadows.

### Q28. Zenoh admin wire contract

**Decision:** JSON request/response over Zenoh queryables at
`smallaios/admin/<verb>`:

```json
// request
{ "token": "<opaque-id>", "args": { ... } }
// response
{ "ok": true, "payload": { ... } }
{ "ok": false, "code": -1, "reason": "user-readable" }
```

Schema documented in the `mgmt-zenoh-admin` spec.

### Q29. Bearer-token TTL

**Decision:** Sliding TTL = configurable per-role idle window from
Q5. Each authenticated request resets the clock. Once a request
enters the kernel its token is treated as live for the duration of
that call (no mid-call expiry).

Optional industry-standard two-tier (access + refresh) mode is
available for orchestration-class clients via configuration.

A `smallaios/admin/heartbeat` keepalive verb is documented for UI
long-progress cases.

### Q30. Argon2id implementation

**Decision:** Production: clean-room `#![no_std]` Argon2id in
`security/src/argon2id.rs` with per-arch NEON / AVX2 SIMD shims.
Tested against RFC 9106 KAT vectors. The external `argon2` crate
may appear in `dev-dependencies` only as a validation oracle in
tests; it MUST NOT enter the production dep graph.

### Q31. TTY line editing

**Decision:** Backspace, ^U (kill line), ^C (abort). No echo of
any character (not even `*`) to avoid leaking length. Newline
submits.

### Q32. Password strength policy

**Decision:** Defaults: min 16 chars; 3 of 4 character classes
(upper, lower, digit, symbol); dictionary check; reject username,
role names. All thresholds configurable in `mgmt/policy.toml`
(`password.min_length`, `password.max_length`,
`password.require_classes`, `password.dictionary_check`).

High-entropy token mode: passwords ≥ 40 chars whose Shannon
entropy meets a configurable threshold bypass the character-class
rule so machine-generated tokens are not rejected.

### Q33. `auth_create_user` role argument

**Decision:** `u8` enum: 0 = `Root`, 1 = `Operator`, 2 = `Viewer`.
Reject all other values with `-EINVAL`. Future roles append (3,
4, ...) without breaking existing callers.

### Q34. Peer-identity binding

**Decision:** `auth_login` over Zenoh records the peer's TLS-1.3
cert fingerprint or PSK identity. Subsequent admin requests check
`request.peer == session.peer` and reject mismatches. Token
replayed from a different peer fails closed.

### Q35. `passwd` user-space tool

**Decision:** `container/src/bin/passwd.rs`. Reads old + new with
echo-off prompt, calls `auth_change_password` syscall.

### Q36. Implementation phasing

**Decision:** Bottom-up across 10 phases, each ending green:

1. Argon2id + KAT tests (`security/`).
2. `auth/` crate scaffold + shadow parser + role enum.
3. Kernel syscalls + session table.
4. Console-login (TTY first-boot, login, lockout, idle sweep).
5. `mgmt/` crate scaffold + `Config` + `ConfigSurface` trait.
6. TOML loader + universal-exposure CI walker.
7. Zenoh admin keyspace + bearer-token wrapper.
8. Zenoh telemetry keyspace.
9. TOTP (RFC 6238) + `totp_setup` syscall.
10. Audit chain + signed checkpoints + denial audit.

### Q37. Test target

**Decision:** Aim for ≥4500 total tests after change (~+360 new):
- Argon2id KATs + property-based parameter sweep
- Shadow parse round-trip + corruption fuzz
- Lockout timer + concurrent-login race
- Per-syscall role enforcement matrix (every syscall × every role)
- Zenoh admin auth (peer mismatch, token replay, expired)
- Telemetry schema round-trip + adaptive cadence
- ConfigSurface conformance per impl + universal-exposure CI test
- Atomic-rewrite under simulated power-fail (Kani harness)
- Audit-chain verify + tamper-detect tests
- TOTP RFC-6238 vectors + clock-skew tolerance
- JSON request fuzz harness

### Q38. Formal verification

**Decision:** All three formalisms exercised:
- TLA+ `session_state.tla` — login → active → idle-expired →
  logged-out, including concurrent same-user logins and
  force-logout on password change.
- Kani — bounded model checks on `audit_chain_verify` (no silent
  corruption admitted) and `shadow_atomic_rewrite` (crash mid-rename
  never yields half-written shadow).
- SPIN — Promela model of the 5-fail lockout proving it is not
  bypassable via interleaved attempts.

### Q39. Branch / PR strategy

**Decision:** New branch `change/management-login-v1` from
`develop`. One PR per phase into the change branch (10 PRs). After
all 10 land, the change branch is rebased on `develop` and merged
in one final PR. The current `claude/add-login-management-dc8DG`
worktree branch is the working surface for the proposal/design
artifacts and will be replaced by the convention-named branch
before implementation begins.

### Q40. Session next steps

**Decision:** Draft `design.md`, the 13 spec deltas, and
`tasks.md`; commit on the worktree branch; run `openspec validate
--strict`; push to origin so the PR/branch updates. No `develop`
touched.

## Risks / Trade-offs

- **[Risk] Argon2id memory tuning on small boards** — Mitigation:
  per-tier parameters (Q26), with the `tiny` tier validated on
  256 MiB-RAM x86. The PHC string carries the parameters so
  verification works regardless of which tier produced the hash.
- **[Risk] First-boot UX over a flaky serial console** — Mitigation:
  explicit retry on parse fail; `auth.skip-firstboot` recovery boot
  arg gated on physical presence (Q6, Q7); pre-baked-hash variant
  for race-free lab provisioning (Q12).
- **[Risk] Universal-exposure CI gate breaks the build when a
  developer adds a `Config` field without wiring all surfaces** —
  Mitigation: clear error message identifying the missing surfaces;
  `cargo xtask scaffold-config-field <name>` helper that emits the
  trait impls for every active surface.
- **[Risk] Cross-feature cargo-feature explosion** —
  `mgmt-token-opaque` × `mgmt-token-mldsa` × `mgmt-token-ed25519-legacy`
  × `signed-checkpoints` × `totp` could create a combinatorial
  build matrix. Mitigation: `cargo deny` rule banning combinations
  that disable opaque tokens entirely; a single `mgmt-default`
  feature covers the common path.
- **[Risk] Audit log fingerprint published as a metric is itself
  forgeable by a compromised kernel** — Mitigation: signed-checkpoint
  mode uses a kernel-held ML-DSA-65 key. v2 sealing of that key in
  TPM/secure-enclave is captured as future work, not v1 scope.
- **[Risk] Build-time schema walker misses a `#[cfg(feature)]`-gated
  field** — Mitigation: walker runs once per active feature
  combination in CI; runtime sanity check (Q3) covers the gap.
- **[Risk] Workspace count growth (21 → 23)** — Mitigation:
  Q1 / Q4 explicitly chose two crates over one because identity and
  config-of-everything have different ownership and grow at
  different rates. Documented as a deliberate trade-off.

## Migration Plan

This is a v0.x prototype change; no on-disk format from prior
versions to migrate. First boot of any image carrying this change:
- If `/data/auth/shadow` does not exist, enter first-boot setup
  (Q12, Q17).
- If `/data/auth/shadow` exists but lacks the new TOTP / lockout
  trailing fields, parsers ignore the gap and treat the user as
  TOTP-not-enrolled and not-locked (Q27).
- If existing builds have policy files in unrelated locations, this
  change does not touch them; `mgmt/policy.toml` is a new file with
  conservative defaults.

## Open Questions

All six questions raised in the proposal are now resolved (see
Q1–Q6 above). No remaining open questions for v1.

Two items deferred to v2 with explicit decision:
- Hardware-backed credential sealing (TPM key unwrap). Listed as
  a non-goal; design accommodates it (PhysicalPresenceProvider
  trait already supports a TPM provider).
- Off-box audit log shipping. v1 persists the chain locally; v2
  may add a Zenoh streaming sink without changing the chain
  format.
