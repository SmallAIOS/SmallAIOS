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
   recovery, first-boot setup, fully offline path).
2. A **shadow-style password file** with **Argon2id** hashing
   (memory-hard; SHA-3 alone is too fast).
3. A built-in **viewer** role with capability-scoped read-only
   access to monitoring data only.
4. A **Zenoh admin/telemetry channel** (`smallaios/admin/**` for
   request/response, `smallaios/metrics/**` for streaming
   telemetry) reusing the existing PQC-backed transport.

This unblocks running SmallAIOS as an attended appliance instead of
"the binary that whoever has the keyboard owns."

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
  capability (`Role::Root` or `Role::Viewer`) attached to the
  current process / control plane.

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

- New `auth-roles` capability:
  - `Role::Root` — full kernel/user-space access.
  - `Role::Viewer` — may **read** telemetry, model metadata,
    inference statistics, log tail; may **not** load models, mutate
    config, change passwords (other than its own), or call any
    `*_write` / `*_load` syscall.
- Role is attached to the active session token. Existing capability
  checks gain a `min_role: Role` field; rejecting a viewer is a
  non-fatal `EPERM`-style error returned over the channel.

### Syscall surface (Layer 0)

Three new syscalls (43, 44, 45 — incrementing the current ~46):

- `auth_login(user_ptr, user_len, pass_ptr, pass_len) -> session_id`
- `auth_change_password(old_ptr, old_len, new_ptr, new_len) -> 0|err`
- `auth_whoami() -> { role: u8, user_id: u32 }`

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
- `auth-roles`: `Role::Root` / `Role::Viewer` definitions, the
  read/write syscall partition, and the `min_role` capability
  guard.
- `console-login`: TTY login flow, first-boot setup, echo-off
  password entry, lockout / rate-limit policy, recovery boot
  argument.
- `mgmt-zenoh-admin`: `smallaios/admin/**` request/response
  contract, bearer-token lifecycle, error encoding.
- `mgmt-zenoh-telemetry`: `smallaios/metrics/**` schema and
  publication cadence.

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
  - `security/src/argon2id.rs` — clean-room Argon2id (no_std).
  - `kernel/` — three new syscalls + role-tagged session table.
  - `peripheral/src/uart.rs` — echo-off read.
  - `container/src/mgmt.rs` — Zenoh admin/telemetry handler.
  - `container/src/bin/passwd.rs` — user-space passwd tool.
- **Tests:** ~50 new tests targeted: Argon2id KAT vectors, shadow
  parse round-trip, lockout timer, role enforcement on every
  syscall, Zenoh admin request unauthorized → rejected, telemetry
  schema round-trip. Aim to keep the 4,143 → ≥4,190 passing.
- **Boot footprint:** Argon2id at `m=64 MiB, t=3, p=1` adds ~150
  ms to first login (acceptable; not on the hot inference path).
  Static memory: <1 MB shadow + role table + console buffers.
- **Container image:** unchanged — Zenoh, QUIC, TLS 1.3, PQC are
  already linked. Adds ~30 KB code.
- **Downstream:** unblocks operating SmallAIOS as an attended
  appliance and gives serving teams a read-only telemetry channel
  without granting kernel privileges.
- **Dependencies:** none new. (No `argon2` crate — clean-room
  rule.)
- **Risks:** (1) Argon2id parameter tuning on small boards
  (Jetson Orin has plenty; bare-metal x86 with 256 MB RAM may
  need `m=16 MiB`). (2) First-boot UX over a flaky serial
  console — needs explicit retry + an `auth.skip-firstboot`
  recovery boot arg gated on physical presence (rear of design.md).

## Open Questions

1. Should `auth` be its own Layer 1 crate, or sit inside
   `security`? Leaning toward a separate crate for clarity, but it
   adds one to the 21-crate count.
2. Should the bearer token be a JWT (signed with ML-DSA-65) or an
   opaque random ID looked up server-side? Opaque is simpler;
   JWT lets the viewer cache the token across reconnects without a
   kernel round-trip.
3. Do we want `Role::Operator` (can load models but not change
   passwords) in v1 as well, or strictly root + viewer? Spec is
   currently strictly two-role; adding a third is straightforward
   if requested.
