## Why

SmallAIOS now boots cleanly on x86-64, AArch64, and the Jetson Orin
KVM path, but it has no notion of "who is on the box." There is no
login, no password store, no audit identity on syscalls, and no
remote management/monitoring channel. Any operator with serial
access has implicit root, and there is no way to give a
serving-team SRE a read-only view of the running model server
without handing them full kernel privileges.

Pulling in OpenSSH would balloon the attack surface (PTY, shell,
~MB of code, ~80 syscalls) and fight the unikernel design. We
already ship Zenoh + QUIC + TLS 1.3 + ML-KEM-768/ML-DSA-65 in
the IPC/network stack and a serial UART driver in `peripheral`.
This change wires those into a small management layer:

1. A **local console login** on the existing serial/TTY (boot
   recovery, first-boot setup, fully offline path) with explicit
   `logout` / `exit` to end a session cleanly.
2. A **shadow-style password file** with **Argon2id** hashing
   (memory-hard; SHA-3 alone is too fast).
3. A **three-role model** — `Root` (full access), `Operator`
   (load/unload models + read telemetry), and `Viewer`
   (read-only telemetry; the role the `console-monitor-v1`
   top-style monitor is designed for) — with a precise
   role-vs-syscall partition.
4. A **Zenoh admin/telemetry channel** (`smallaios/admin/**` for
   request/response, `smallaios/metrics/**` for streaming
   telemetry) reusing the existing PQC-backed transport.

This unblocks running SmallAIOS as an attended appliance instead of
"the binary that whoever has the keyboard owns."

Because every follow-on management feature
(`system-power-control-v1`, `remote-update-v1`,
`network-management-v1`, `automotive-bus-management-v1`, and
anything else that lands later) needs to be reachable from the
*same* set of surfaces — TTY console, Zenoh admin, future
automotive UDS, and a boot-time configuration file — this
foundational change also establishes the **management surface
convention** that all of them inherit. The architecture is
captured here so no follow-on change needs to redefine it.

## Management Surface Convention

This change establishes the architectural pattern every
SmallAIOS management feature follows. It is intentionally
captured in the foundational change so follow-on features
(known and unknown) inherit it without an extra dependency
edge.

### Single config model, multiple edge serializers

There is **one** Rust source of truth: a typed `Config` model
held in the `mgmt` crate. Every reachable management surface
is a thin (de)serializer over the same model:

```text
                  ┌──────────────────────────────────┐
                  │   Config (Rust types, validated) │ ← source of truth
                  └─────┬────────┬────────┬───────┬──┘
                        │        │        │       │
                  TOML loader  TTY      Zenoh    UDS
                              console   admin   (later)
```

A `ConfigSurface` trait defines `read(path) -> Value`,
`write(path, value) -> Result`, and `subscribe(path) -> Stream`.
Adding a new surface (UDS over CAN, a future REST proxy, etc.)
is one trait impl; the option taxonomy, validators, audit
records, and persistence are reused unchanged.

**Invariant**: every option in `Config` must be reachable from
**all** active surfaces — no surface-specific knobs, no "you
can only set this from the console." A build-time test walks
the `Config` schema and fails if any field is missing a
handler in any surface; CI enforces this. New options that
genuinely cannot be exposed everywhere (e.g. a one-shot
recovery action only valid on physically-present TTY) must
declare a `#[surface(only = "tty")]` attribute and the test
honors it explicitly — the default is universal exposure.

### Apply lifecycle

Every write — regardless of surface — goes through the same
path:

1. **Parse** into the typed `Config` field.
2. **Validate** (per-field rules + cross-field constraints).
3. **Stage** to a `.tmp` file in the same directory as the
   target.
4. `fsync` then atomic `rename`.
5. **Notify** subscribers via a broadcast channel; affected
   subsystems re-read and apply at their own pace
   (live-reconfigurable) or queue the change for next boot
   (boot-time-only).
6. **Audit**: append `(who, when, surface, path, before,
   after)` to the same audit ring used for `auth_login` and
   (later) `system_power`.

### On-disk layout

All configuration lives under `/data/`:

```text
/data/
├── system.toml              # hostname, time zone, log level, mDNS default
├── auth/
│   └── shadow               # 0600 root — passwords, role table
├── network/
│   ├── eth0.toml            # per-interface (network-management-v1)
│   ├── eth1.toml
│   └── bond0.toml           # bonding (network-management-v1)
├── mgmt/
│   ├── zenoh.toml           # listen endpoints, PSK paths
│   └── policy.toml          # role defs, rate limits, lockout
├── update/
│   └── policy.toml          # watchdog window, slot retention (remote-update-v1)
└── automotive/
    └── uds.toml             # CAN iface, DID table, SecOC key (automotive-bus-management-v1)
```

The layout is hybrid (a top-level `system.toml` for cross-
cutting knobs, plus per-subsystem files for substantive
config) — same convention systemd, OpenWRT, and most embedded
Linuxes converged on. The benefit over a monolithic file is
permission granularity: `auth/shadow` stays 0600 root-only,
`network/*.toml` can be viewer-readable, and a partial
write-failure on one subsystem cannot corrupt another's
config.

Permissions per file are declared in the schema; the loader
refuses to read a file with mode laxer than declared (defense
in depth against an accidental world-readable shadow).

### Where each follow-on change plugs in

- `system-power-control-v1`: adds verbs only, no `Config`
  fields.
- `remote-update-v1`: adds `update/policy.toml`.
- `network-management-v1`: adds `network/<iface>.toml` and
  `network/<bond>.toml`.
- `automotive-bus-management-v1`: adds `automotive/uds.toml`,
  and adds a `UdsConfigSurface` impl of the `ConfigSurface`
  trait (its on-bus equivalent of "set this option").

Future features the user has not yet enumerated follow the
same pattern: declare the typed field on `Config`, declare
the file path + permissions, write the TOML schema +
validators. TTY shell access, Zenoh admin keyspace, UDS
exposure (when applicable), and audit-log integration are all
automatic.

## What Changes

### Console login (Layer 2)

- Add a `console-login` component in `peripheral` (or a new
  `auth` Layer 1 crate — see design.md) that runs after kernel
  init and before any service that handles untrusted data.
- On boot:
  - If `/data/auth/shadow` does not exist or has no `root` entry,
    enter **first-boot setup**: prompt `Set initial root password:`,
    read with echo suppressed, confirm, write `shadow` atomically.
  - Otherwise prompt `username:` then `password:` with echo off,
    rate-limit to 1 attempt/sec and lock after 5 consecutive
    failures for 60 s.
- Successful login establishes a **session token** with a `Role`
  capability (`Role::Root`, `Role::Operator`, or `Role::Viewer`)
  attached to the current process / control plane.
- Explicit `logout` / `exit` console command (and Ctrl-D EOF
  handling) invalidates the session token, clears the audit
  identity, redraws the login prompt, and writes a `logout`
  audit record. Idle sessions are auto-logged-out after a
  configurable timeout (default 15 min for `Root`, 60 min for
  `Operator` and `Viewer`; lives in `mgmt/policy.toml`).

### Shadow file format and password hashing (Layer 0/1)

- `security` crate gains an `argon2id` module (clean-room Rust,
  `#![no_std]`). Defaults: `m_cost = 64 MiB`, `t_cost = 3`,
  `parallelism = 1`, 16-byte salt, 32-byte tag.
- New `auth` Layer 1 crate (depends only on `security` + `kernel`)
  owns the shadow file format, parsing, and verification:

  ```text
  username:$argon2id$v=19$m=65536,t=3,p=1$<base64-salt>$<base64-tag>:role=root:flags=0:last_changed=<unix-day>
  ```
- File lives at `/data/auth/shadow`, mode 0600, owned by kernel.
  Atomic replace via `shadow.tmp` + `rename`.
- `passwd` user-space command (Layer 3 in `container`) re-prompts
  for old password, verifies, prompts new (with confirmation), and
  rewrites the file via the new auth syscalls.

### Roles and capabilities (Layer 0)

- New `auth-roles` capability with three roles:
  - **`Role::Root`** — full kernel/user-space access. Auth
    administration, system power control, OS updates, network
    config, model load/unload, telemetry, audit-log read.
  - **`Role::Operator`** — model lifecycle and read access:
    `model_load` / `model_unload`, `metrics_read`,
    `audit_read` (own-actions only), `auth_change_password`
    (own only). May **not** create users, change other users'
    passwords, mutate `auth/`, `network/`, `mgmt/`, or
    `update/` config, reboot/shutdown the system, push OS
    updates, or write to `/data/automotive/`.
  - **`Role::Viewer`** — read-only telemetry. The role the
    `console-monitor-v1` top-style monitor is built around.
    `metrics_read`, `audit_read` (own-actions only),
    `auth_change_password` (own only), `auth_whoami`. May
    **not** load models, mutate any config, reboot, update,
    or call any `*_write` / `*_load` / `*_admin` syscall.

- **Role-vs-syscall partition** (precise matrix lives in
  `design.md`; v1 surface is ~30 entries). Sketch:

  | Operation | Root | Operator | Viewer |
  |-----------|:----:|:--------:|:------:|
  | `auth_login` / `auth_whoami` | ✓ | ✓ | ✓ |
  | `auth_change_password` (own) | ✓ | ✓ | ✓ |
  | `auth_create_user` / `auth_change_password` (other) | ✓ | — | — |
  | `model_load` / `model_unload` | ✓ | ✓ | — |
  | `metrics_read` / `audit_read` (own) | ✓ | ✓ | ✓ |
  | `audit_read` (all) | ✓ | — | — |
  | `system_power(STATUS)` | ✓ | ✓ | ✓ |
  | `system_power(REBOOT|SHUTDOWN)` | ✓ | — | — |
  | `system_update_*` | ✓ | — | — |
  | Config write — `auth/*`, `mgmt/*`, `update/*` | ✓ | — | — |
  | Config write — `network/*`, `automotive/*` | ✓ | — | — |
  | Config write — `system.toml` (hostname, log level) | ✓ | — | — |

- Role is attached to the active session token. Existing
  capability checks gain a `min_role: Role` field; rejecting a
  non-qualifying caller is a non-fatal `EPERM`-style error
  returned over whatever transport made the request.

- A pre-provisioned `Role::Operator` and `Role::Viewer` user
  do **not** exist by default — first boot creates only `root`,
  and `root` runs `auth_create_user` (TTY shell command,
  Zenoh admin verb) to add an operator and a viewer when
  needed. No latent service accounts.

### Syscall surface (Layer 0)

Five new syscalls (43–47, incrementing the current ~46):

- `auth_login(user_ptr, user_len, pass_ptr, pass_len) -> session_id`
- `auth_logout() -> 0|err`
- `auth_change_password(old_ptr, old_len, new_ptr, new_len, target_user_ptr, target_user_len) -> 0|err`
  — `target_user` may be `null` (own password); non-null requires `Role::Root`.
- `auth_create_user(user_ptr, user_len, role: u8, initial_password_ptr, initial_password_len) -> 0|err`
  — root only; sets the `must_change_password_on_login` flag for the new user.
- `auth_whoami() -> { role: u8, user_id: u32, login_unix_time: u64, idle_seconds: u32 }`

The shadow file is read **only** through these syscalls; user
space cannot map or read the file directly even as root (defense
against a compromised admin tool exfiltrating hashes).

### Remote management over Zenoh (Layer 1/2)

- New `mgmt` component (in `container` or a dedicated Layer 2
  crate — see design.md) subscribes to:
  - `smallaios/admin/login` — request/response, authenticates and
    returns a short-lived bearer token bound to the Zenoh session.
  - `smallaios/admin/passwd` — change password (root only).
  - `smallaios/admin/users/add_viewer` — root creates the
    viewer account (idempotent).
- Publishes on:
  - `smallaios/metrics/cpu` — per-core utilization, 1 Hz.
  - `smallaios/metrics/mem` — heap/page allocator stats, 1 Hz.
  - `smallaios/metrics/inference` — per-model QPS, latency p50/p99,
    error counts, 1 Hz.
  - `smallaios/metrics/log` — structured log records, streamed.
- Channel re-uses the existing PQC-backed Zenoh transport; no new
  TLS/PSK config. The bearer token is mandatory for every admin
  request after `login`.

### Out of scope for v1 (flagged)

- SSH / sshd. (Re-evaluate in v2 only if there is a hard
  requirement.)
- Multi-tenant accounts beyond `root` + a single `viewer`.
- Group-based RBAC, sudoers, capability inheritance across exec.
- PAM/Kerberos/LDAP federation.
- Session recording / audit log shipping (basic in-kernel audit
  ring is included; off-box shipping is v2).
- gRPC. Zenoh request/response is the only RPC surface;
  syslog-style line shipping is replaced by structured Zenoh pub.
- Hardware-backed credential storage (TPM/secure-enclave key
  unwrap). Argon2id over disk is the v1 baseline.

## Capabilities

### New Capabilities

- `auth-shadow`: shadow file format, Argon2id parameters, atomic
  rewrite semantics, on-disk layout, version field, file
  permissions.
- `auth-roles`: `Role::Root` / `Role::Operator` / `Role::Viewer`
  definitions, the role-vs-syscall partition matrix, the
  `min_role` capability guard, the rule that `Operator` and
  `Viewer` accounts are created explicitly (no latent service
  accounts), the per-role idle-timeout policy, and the rule
  that `auth_change_password` accepts a target only when the
  caller is `Root`.
- `console-login`: TTY login flow, first-boot setup, echo-off
  password entry, lockout / rate-limit policy, recovery boot
  argument, explicit `logout` / `exit` / Ctrl-D handling, and
  the per-role idle-timeout auto-logout.
- `mgmt-zenoh-admin`: `smallaios/admin/**` request/response
  contract, bearer-token lifecycle, error encoding.
- `mgmt-zenoh-telemetry`: `smallaios/metrics/**` schema and
  publication cadence.
- `mgmt-config-model`: the typed `Config` Rust source of truth,
  validation pipeline (per-field and cross-field), and the
  apply lifecycle (parse → validate → stage → fsync+rename →
  notify → audit).
- `mgmt-config-surface-trait`: the `ConfigSurface` trait
  (`read` / `write` / `subscribe`) every management surface
  implements, plus the universal-exposure invariant and the
  `#[surface(only = ...)]` attribute escape hatch.
- `mgmt-config-layout`: the `/data/` directory layout, per-file
  permission declarations, mode-stricter-than-declared loader
  rule, and the rationale for hybrid-rather-than-monolithic
  configuration.
- `mgmt-audit-log`: in-kernel audit ring format, fields
  `(who, when, surface, action, before, after)`, retention
  policy, and survival-across-reset guarantee. (Used by
  `auth_login`, `system_power`, every config write, every
  update commit.)

### Modified Capabilities

- `kernel-syscalls`: adds `auth_login`, `auth_change_password`,
  `auth_whoami`; bumps the syscall count in the architecture
  doc.
- `security`: adds Argon2id KDF alongside existing SHA-3 / AES /
  PQC primitives.
- `peripheral-uart`: gains an "echo-suppressed read" mode for
  password entry on the serial console.
- `ipc-zenoh`: adds the admin and metrics keyspaces and the
  bearer-token authentication wrapper.

## Impact

- **Code:**
  - New crate `auth/` (Layer 1) — shadow parser, Argon2id wrapper,
    role table.
  - New crate `mgmt/` (Layer 1) — `Config` type, `ConfigSurface`
    trait, validator pipeline, atomic-rewrite helper, audit-log
    ring.
  - `security/src/argon2id.rs` — clean-room Argon2id (no_std).
  - `kernel/` — five new syscalls (`auth_login`, `auth_logout`,
    `auth_change_password`, `auth_create_user`, `auth_whoami`)
    + role-tagged session table + per-role idle-timeout sweeper.
  - `peripheral/src/uart.rs` — echo-off read + raw-mode toggle.
  - `container/src/mgmt_zenoh.rs` — Zenoh admin/telemetry surface
    (one impl of `ConfigSurface`).
  - `container/src/mgmt_tty.rs` — console shell surface (another
    impl of `ConfigSurface`).
  - `mgmt/src/loader.rs` — TOML file surface (third impl).
  - `container/src/bin/passwd.rs` — user-space passwd tool.
- **Tests:** ~80 new tests targeted: Argon2id KAT vectors, shadow
  parse round-trip, lockout timer, role enforcement on every
  syscall, Zenoh admin request unauthorized → rejected, telemetry
  schema round-trip, `ConfigSurface` conformance per impl, the
  universal-exposure invariant CI test (walks the schema, fails
  if any field lacks a handler), audit-record persistence across
  reset, atomic-rewrite under simulated power-fail. Aim to keep
  the 4,143 → ≥4,220 passing.
- **Boot footprint:** Argon2id at `m=64 MiB, t=3, p=1` adds ~150
  ms to first login (acceptable; not on the hot inference path).
  Static memory: <1 MB shadow + role table + Config model +
  console buffers + audit ring.
- **Container image:** unchanged — Zenoh, QUIC, TLS 1.3, PQC are
  already linked. Adds ~50 KB code (auth + mgmt).
- **Downstream:** unblocks operating SmallAIOS as an attended
  appliance, gives serving teams a read-only telemetry channel
  without granting kernel privileges, **and establishes the
  surface convention every follow-on management feature
  (current and future) inherits without re-litigating
  TTY/Zenoh/file plumbing per change.**
- **Dependencies:** none new. (No `argon2` crate, no
  `serde_yaml` — clean-room rule. `serde` + `toml` are already
  in the workspace.)
- **Risks:** (1) Argon2id parameter tuning on small boards
  (Jetson Orin has plenty; bare-metal x86 with 256 MB RAM may
  need `m=16 MiB`). (2) First-boot UX over a flaky serial
  console — needs explicit retry + an `auth.skip-firstboot`
  recovery boot arg gated on physical presence (see design.md).
  (3) The universal-exposure invariant fails closed — adding a
  new `Config` field without wiring all surfaces breaks CI.
  Acceptable cost for the architectural guarantee, but a
  developer ergonomics issue worth a `cargo xtask scaffold-
  config-field` helper.

## Open Questions

1. Should `auth` be its own Layer 1 crate, or sit inside
   `security`? Leaning toward a separate crate for clarity, but it
   adds one to the 21-crate count.
2. Should the bearer token be a JWT (signed with ML-DSA-65) or an
   opaque random ID looked up server-side? Opaque is simpler;
   JWT lets the viewer cache the token across reconnects without a
   kernel round-trip.
3. *(resolved)* Three roles — `Root`, `Operator`, `Viewer` —
   committed. `Operator` can load/unload models and read
   telemetry; `Viewer` is read-only and is the role the
   `console-monitor-v1` top-style monitor is designed for.
4. Should the `Config` model live in its own `mgmt` Layer 1
   crate or fold into `auth`? They are conceptually distinct
   (config-of-everything vs identity-and-access) and mgmt will
   grow as more features are added, so a separate crate is
   probably right — but it raises the workspace count by two
   (auth + mgmt) instead of one. Final call deferred to design.md.
5. Should the `ConfigSurface` invariant be enforced at
   build time (CI test, hard fail) or runtime (boot-time check
   that logs a warning)? Build time is stricter and prevents
   regressions reaching production but increases friction during
   feature development. Leaning build time with a clear error
   message and the `cargo xtask` scaffolder above.
6. Per-role idle-timeout defaults — `Root` 15 min, `Operator`
   60 min, `Viewer` 60 min. Reasonable for an attended box;
   may be too aggressive for a viewer running `top`-style
   monitor for hours. Open: should `console-monitor-v1`
   reset the idle timer on any keypress (yes, almost certainly).
