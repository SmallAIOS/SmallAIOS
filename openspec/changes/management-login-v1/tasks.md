## 1. Phase 1 — Argon2id (`security/`)

- [ ] 1.1 Add `security/src/argon2id.rs` with `#![no_std]` clean-room implementation
- [ ] 1.2 Implement `argon2id_hash`, `argon2id_verify`, `argon2id_format_phc`
- [ ] 1.3 Add RFC 9106 KAT test vectors (verify against archived reference output)
- [ ] 1.4 Add `argon2` crate to `dev-dependencies` only (validation oracle)
- [ ] 1.5 Add property-based tests across the parameter space (`m_cost`, `t_cost`, `p_cost`)
- [ ] 1.6 Add NEON SIMD shim behind `arch::aarch64::neon` cargo feature
- [ ] 1.7 Add AVX2 SIMD shim behind `arch::x86_64::avx2` cargo feature
- [ ] 1.8 Verify SIMD tags byte-identical to portable path (~30 fuzz iterations)
- [ ] 1.9 Document tier selection (`tiny`/`default`/`strong`) in module docs

## 2. Phase 2 — `auth/` Layer 1 crate scaffold

- [ ] 2.1 Create `auth/` crate, `Cargo.toml`, register in workspace (21 → 22)
- [ ] 2.2 `auth/src/role.rs` — `Role` enum (`Root=0`, `Operator=1`, `Viewer=2`), reject unknowns
- [ ] 2.3 `auth/src/shadow.rs` — colon-separated record parser/serializer
- [ ] 2.4 Shadow round-trip tests (round-trip, unknown-trailing-field preservation)
- [ ] 2.5 Shadow corruption fuzz harness (truncated, malformed PHC, mode laxer than 0600)
- [ ] 2.6 Atomic-rewrite helper (`stage → fsync → rename`)
- [ ] 2.7 Kani harness for `shadow_atomic_rewrite` (crash mid-rename never yields half-write)
- [ ] 2.8 Tier auto-selection from RAM
- [ ] 2.9 Cyclic-dep check passes; clippy-D-warnings clean

## 3. Phase 3 — Kernel syscalls + session table

- [ ] 3.1 `kernel/src/auth.rs` — session table `[Option<Session>; SESSION_TABLE_CAP]`
- [ ] 3.2 Build-time const `SESSION_TABLE_CAP` (default 32, overridable per target)
- [ ] 3.3 Implement `auth_login` (43)
- [ ] 3.4 Implement `auth_logout` (44)
- [ ] 3.5 Implement `auth_change_password` (45)
- [ ] 3.6 Implement `auth_create_user` (46), Root-only enforcement
- [ ] 3.7 Implement `auth_whoami` (47)
- [ ] 3.8 Wire `min_role` capability guard into existing syscall dispatch
- [ ] 3.9 Per-role idle-timeout sweeper (cooperative, runs on Core 0 between ONNX yields)
- [ ] 3.10 must-change-password whitelist gate (`auth_change_password`/`auth_whoami`/`auth_logout` only)
- [ ] 3.11 Reject direct shadow open/mmap from user space
- [ ] 3.12 Constant-time-equivalent rejection of unknown-user vs wrong-password
- [ ] 3.13 TLA+ `session_state.tla` model (login/active/idle-expired/logged-out, concurrent same-user)
- [ ] 3.14 Per-syscall role enforcement matrix tests (every syscall × every role)

## 4. Phase 4 — Console-login (TTY)

- [ ] 4.1 `peripheral::uart::ReadOptions { echo, raw, max_len }` + `read_line` overload
- [ ] 4.2 Backspace, ^U, ^C handling in echo-off mode
- [ ] 4.3 Stateless-across-reads test
- [ ] 4.4 First-boot prompt flow (`Set initial root password:` → confirm → atomic write)
- [ ] 4.5 First-boot retry on confirmation mismatch (3 attempts, then halt)
- [ ] 4.6 Steady-state login flow (`username:` → `password:` → banner)
- [ ] 4.7 Lockout: 5 fails → 60 s, per-source counters (TTY = 1 source, Zenoh = per peer)
- [ ] 4.8 SPIN model of lockout (5-fail interleaving non-bypass)
- [ ] 4.9 `auth.skip-firstboot` boot arg parser + physical-presence gate
- [ ] 4.10 `auth.skip-firstboot=<phc>` pre-baked-hash variant
- [ ] 4.11 `PhysicalPresenceProvider` trait + per-arch impls (GPIO x86, GPIO Jetson, fw_cfg QEMU, GPIO RISC-V, verified-boot trusted boot arg)
- [ ] 4.12 Explicit `logout` / `exit` / Ctrl-D handling, redraw prompt, audit record
- [ ] 4.13 Idle auto-logout with keypress reset (uses Phase 3 sweeper)
- [ ] 4.14 Corrupt-shadow halt with recovery hint
- [ ] 4.15 Missing-shadow drops to first-boot

## 5. Phase 5 — `mgmt/` Layer 1 crate scaffold

- [ ] 5.1 Create `mgmt/` crate, register in workspace (22 → 23)
- [ ] 5.2 `mgmt/src/config.rs` — typed `Config` struct (initial v1 fields: idle.{root|operator|viewer}_minutes, password.*, audit.*, metrics.*, mgmt.zenoh.*)
- [ ] 5.3 `#[reload("live"|"boot")]` proc-macro attribute + schema walker
- [ ] 5.4 Build-time test that fails compilation if any field lacks `#[reload]`
- [ ] 5.5 `mgmt/src/surface.rs` — `ConfigSurface` trait (`read`/`write`/`subscribe`)
- [ ] 5.6 Apply lifecycle helper (`parse → validate → stage → fsync → rename → notify → audit`)
- [ ] 5.7 Validation pipeline (per-field rules + cross-field constraints)
- [ ] 5.8 Broadcast notify channel for live-reload subscribers
- [ ] 5.9 `pending_reboot()` accumulator for boot-only fields

## 6. Phase 6 — TOML loader + universal-exposure CI

- [ ] 6.1 `mgmt/src/loader_toml.rs` — `ConfigSurface` impl over `/data/*.toml`
- [ ] 6.2 Per-file permission declaration table; refuse mode laxer than declared
- [ ] 6.3 Generate skeleton `system.toml` and `mgmt/policy.toml` on first boot with conservative defaults
- [ ] 6.4 Build-time CI walker that fails when any field lacks a handler in any active surface
- [ ] 6.5 Runtime boot-time sanity check (warn-and-continue) for the same invariant
- [ ] 6.6 `cargo xtask scaffold-config-field <name>` helper to emit trait impls for every active surface
- [ ] 6.7 `#[surface(only = "tty")]` escape-hatch attribute
- [ ] 6.8 Tests for each surface impl conformance (read/write/subscribe round-trip)

## 7. Phase 7 — Zenoh admin + bearer-token wrapper

- [x] 7.1 `mgmt/src/surface_zenoh/` — `ConfigSurface` impl on Zenoh (`ZenohSurface` + `ZenohConfigView` adapter)
- [x] 7.2 Register `smallaios/admin/**` queryables (login, logout, whoami, passwd, users/add, users/list, heartbeat, config/get, config/set, config/changed)
- [x] 7.3 JSON request/response codec — clean-room `mgmt::surface_zenoh::json` (`serde_json` is `std`-only and brings a substantial transitive tree, so we mirror Phase 5+6's clean-room TOML pattern; `net` did not have a JSON codec to reuse)
- [x] 7.4 Bearer-token wrapper: extract → lookup → peer-check → idle-check → reset → dispatch (`AdminDispatcher::dispatch`)
- [x] 7.5 Peer-identity binding (32-byte fingerprint recorded in kernel `Session::peer_identity` at login)
- [x] 7.6 Reject token replay from a different peer (`cross_peer_replay_eperm` test)
- [x] 7.7 Per-identity (4) and total (16) Zenoh session caps; `-EAGAIN` on exceed (`per_identity_cap_eagain_on_5th`, `total_cap_eagain_on_17th`)
- [x] 7.8 In-flight protection: long-running call survives idle-expiry mid-flight (`long_running_call_survives_idle_expiry_via_in_flight_flag`)
- [x] 7.9 `heartbeat` keepalive verb (`handle_heartbeat`, returns `expires_in`)
- [ ] 7.10 Optional two-tier (access + refresh) mode behind `mgmt.token.two_tier = true` policy [DEFERRED — orchestration-class clients land in Phase 8/10; opaque + ML-DSA modes ship in Phase 7 and the `mgmt.token_two_tier` Config field is already wired through Phase 5]
- [x] 7.11 `mgmt-token-mldsa` cargo feature (ML-DSA-65 signed tokens — `surface_zenoh::token::mldsa`)
- [x] 7.12 `mgmt-token-ed25519-legacy` cargo feature, off by default, compile-time excluded when off (`#[cfg]`-gated `ed25519_legacy` submodule + `binary_excludes_ed25519_when_feature_off` test)
- [ ] 7.13 Cargo-deny rule banning combinations that disable opaque tokens entirely [DEFERRED — opaque tokens are unconditionally compiled in Phase 7 (no feature can turn them off); the rule lands when a future `mgmt-disable-opaque` feature is contemplated]
- [x] 7.14 JSON request fuzz harness on the wire codec (`fuzz_smoke_random_bytes_dont_panic` + decode-rejection coverage; the libfuzzer corpus moves to `cargo-fuzz` in Phase 11 cross-cutting verification)

## 8. Phase 8 — Zenoh telemetry

- [ ] 8.1 `mgmt/src/telemetry.rs` — publishers for `cpu`, `mem`, `inference`, `log`, `audit_fingerprint`
- [ ] 8.2 Per-key configurable cadence (`metrics.<key>.interval_ms`), bounds 100 ms–60 s
- [ ] 8.3 Adaptive cadence (`metrics.<key>.adaptive = { threshold, fast_hz, slow_hz }`)
- [ ] 8.4 Schema round-trip tests (each metric publishes and parses cleanly)
- [ ] 8.5 Adaptive-mode test (cross threshold → fast_hz, fall below → slow_hz)
- [ ] 8.6 Log streaming integration with existing `tracing` / `defmt` infrastructure

## 9. Phase 9 — TOTP

- [x] 9.1 `security/src/totp.rs` — RFC 6238 TOTP-SHA1 implementation (clean-room SHA-1 + HMAC-SHA1 + dynamic-truncation, ~600 LOC across `sha1.rs`, `hmac_sha1.rs`, `totp.rs`)
- [x] 9.2 RFC 6238 test vectors (Appendix B all 6 SHA-1 timestamps; RFC 2202 all 7 HMAC-SHA1 vectors; FIPS 180-4 Appendix A SHA-1 KATs in `*_test_vectors.rs`)
- [x] 9.3 Clock-skew tolerance test (±1 step at default 30-s period; `verify_accepts_one_step_skew_within_window` exercises both +1 and −1)
- [x] 9.4 `totp_required` field added to shadow record; `totp_secret` already in Phase 2 format. Parser defaults missing field to `false`; serializer omits when `false` so pre-Phase-9 shadows round-trip byte-for-byte.
- [x] 9.5 `auth_totp_setup` syscall (0x95) — replaces Phase 3's `-ENOSYS` stub; CSPRNG-driven 20-byte secret, persisted via `ShadowProvider::write_totp_secret`, copied to caller-supplied buffer; cross-enrol requires Root.
- [x] 9.6 `auth_login` factor2 path — empty factor2 on `totp_required = true` user → `-EAUTHEXPIRED`; wrong code → `-EACCES`; valid code → session id. Constant-time-equivalent dummy verify on user-not-found / non-enrolled paths.
- [x] 9.7 `mgmt::Config::totp.enforced_for_roles: RoleSet` — wired through registry (Live reload), validate.rs (rejects unknown bits), TOML loader (`["root","operator","viewer"]` array form), and Zenoh JSON codec.
- [x] 9.8 Tests for enrolled-user-must-supply-code, unenrolled-user-may-omit, invalid-code-rejected, malformed-factor2 rejected, fail-closed when secret missing but `totp_required = true`, dummy-verify path on unknown user (kernel/syscall/auth.rs adds 12 new tests; auth/shadow adds 7; mgmt/config + validate + loader + zenoh add 16).

## 10. Phase 10 — Audit chain + signed checkpoints + denial audit

- [ ] 10.1 `mgmt/src/audit.rs` — fixed 16 MiB in-memory ring of records `{ ts, who, surface, action, before, after, code, prev_hash, hash }`
- [ ] 10.2 Background flusher to `/data/audit/log.jsonl` (1 s cadence + immediate on auth events)
- [ ] 10.3 Hybrid rotation (size OR age) configurable via `mgmt/policy.toml`
- [ ] 10.4 `audit.max_total_disk_bytes` failsafe with eviction-of-oldest-archive when exceeded
- [ ] 10.5 SHA-3-256 hash chain (every record contains `prev_hash` and `hash`)
- [ ] 10.6 Latest fingerprint exposed via `audit_read` and streamed on `smallaios/metrics/audit_fingerprint`
- [ ] 10.7 Optional ML-DSA-65 signed checkpoints every Nth record (configurable interval, default 1024)
- [ ] 10.8 Off-box verifier integration test (rewrite a record between checkpoints → tamper detected)
- [ ] 10.9 Denial audit on every `min_role` denial (`action = "DENY:<syscall>"`)
- [ ] 10.10 Denial rate-limit (10/s/user → coalesce excess into `DENY_BURST` record)
- [ ] 10.11 Kani harness for `audit_chain_verify` (no silent corruption admitted)
- [ ] 10.12 Atomic-rewrite under simulated power-fail tests for the JSONL flusher
- [ ] 10.13 `passwd` user-space tool (`container/src/bin/passwd.rs`)

## 11. Cross-phase verification

- [ ] 11.1 Workspace passes `cargo fmt --check`, `cargo clippy -- -D warnings`
- [ ] 11.2 `cargo test --workspace` total ≥ 4500 (proposal: 4143 + ~360 new)
- [ ] 11.3 Cyclic-dep check passes (workspace 21 → 23, all new edges respect Layer 1 → 0)
- [ ] 11.4 `just arch-check` clean (module-level acyclicity)
- [ ] 11.5 `cargo audit` advisory clean
- [ ] 11.6 `cargo deny` license/advisory/ban clean
- [ ] 11.7 `cargo geiger` shows no new unsafe outside well-justified arch SIMD
- [ ] 11.8 `cargo llvm-cov --fail-under-lines 80` passes (ratchet target 93%)
- [ ] 11.9 TLA+ session model verifies clean
- [ ] 11.10 Kani harnesses (audit chain, shadow rewrite) pass under default unwind bounds
- [ ] 11.11 SPIN lockout model verifies clean
- [ ] 11.12 Boot footprint regression check: container image growth ≤ 100 KB
- [ ] 11.13 First-login latency on Jetson Orin NX < 250 ms (Argon2id `strong` tier ~150 ms)
- [ ] 11.14 First-login latency on 256-MiB-RAM x86 < 250 ms (Argon2id `tiny` tier)
- [ ] 11.15 `openspec validate management-login-v1 --strict` returns clean
- [ ] 11.16 Zero CodeQL alerts on the new code (preserves the post-`codeql-remediation-v1` baseline)
