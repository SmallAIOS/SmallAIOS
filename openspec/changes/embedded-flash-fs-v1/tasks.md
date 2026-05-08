## 1. Phase 1 — `FlashDevice` trait + `FlashError` enum

- [x] 1.1 `fs/src/flash/mod.rs` — module skeleton, `#![no_std]`, gated by `fs-flash` cargo feature
- [x] 1.2 `fs/src/flash/device.rs` — `FlashDevice` trait
- [x] 1.3 `fs/src/flash/error.rs` — `FlashError` enum + POSIX errno conversion (folded into `device.rs`)
- [x] 1.4 Generic conformance test suite (program-on-dirty rejected, erase resets to 0xFF, alignment, etc.)
- [x] 1.5 Documentation: trait contract for NOR vs NAND parts

## 2. Phase 2 — Mock device

- [x] 2.1 `fs/src/flash/mock.rs` — in-memory `MockFlashDevice` behind `fs-flash-mock` cargo feature
- [x] 2.2 Bit-flip injection knob (configurable rate)
- [x] 2.3 Bad-block-on-erase injection knob
- [x] 2.4 Mock passes the generic conformance suite
- [x] 2.5 Mock available for downstream phases' integration tests

## 3. Phase 3 — littlefs v2.x read path

- [x] 3.1 `fs/src/flash/littlefs/superblock.rs` — superblock parser, format-version check, refuse non-v2.x (folded into `littlefs.rs`)
- [x] 3.2 `fs/src/flash/littlefs/metadata.rs` — metadata-pair reader (committed-half selection) (folded into `littlefs.rs`)
- [x] 3.3 `fs/src/flash/littlefs/dir.rs` — directory entry iteration (folded into `littlefs.rs`)
- [x] 3.4 `fs/src/flash/littlefs/file.rs` — file inline data + CTZ block list reader (folded into `littlefs.rs`)
- [ ] 3.5 `fs/src/flash/littlefs/global.rs` — global state reader (deferred to phase 5; not required for read-only path)
- [x] 3.6 Pin spec to a specific commit hash, document in `docs/embedded-flash-fs.md` (pinned to v2.10.0 SPEC.md in module doc)
- [ ] 3.7 Add `littlefs2` crate as `dev-dependencies` only (validation oracle) — deferred; pure-Rust fixture builder used instead
- [ ] 3.8 Round-trip test against `littlefs2`-generated images — deferred to follow-up PR
- [ ] 3.9 Weekly cron job: round-trip against C `mklittlefs`-generated images — deferred to phase 4 CI work

## 4. Phase 4 — littlefs v2.x write path

- [ ] 4.1 `fs/src/flash/littlefs/write.rs` — file create / extend / overwrite
- [ ] 4.2 Directory create / delete
- [ ] 4.3 Atomic rename via metadata-pair commit
- [ ] 4.4 Truncate
- [ ] 4.5 Round-trip test: SmallAIOS write → littlefs2 reads → cmp byte-exact
- [ ] 4.6 Round-trip test: SmallAIOS write → external `littlefs-fuse` reads → cmp byte-exact (weekly cron)
- [ ] 4.7 Property-based fuzz on write sequences

## 5. Phase 5 — fsync + metadata-pair commit semantics

- [ ] 5.1 `fs/src/flash/littlefs/fsync.rs` — `fsync(fd)` triggers metadata-pair commit
- [ ] 5.2 No background timer commit (different from F2FS — flash wear cost is too high)
- [ ] 5.3 Power-fail tests: kill-9 mid-write → mock simulates partial program → mount → verify last fsync intact
- [ ] 5.4 Kani harness for metadata-pair commit atomicity invariant
- [ ] 5.5 Tests for fadvise SEQUENTIAL write-batching behavior

## 6. Phase 6 — Wear-leveling + Bad Block Table

- [ ] 6.1 `fs/src/flash/littlefs/alloc.rs` — block allocator integrating BBT skip
- [ ] 6.2 BBT readers/writers at start AND end of flash (duplicated)
- [ ] 6.3 Both-BBT-corrupt halt path
- [ ] 6.4 Single-BBT-corrupt recovery path (rewrite from surviving copy)
- [ ] 6.5 Runtime-detected bad block path (program/erase failure → mark bad → update both BBTs → continue)
- [ ] 6.6 1M-cycle stress on mock validates per-block erase distribution within 5% of mean
- [ ] 6.7 TLA+ model `littlefs_wear.tla` proves wear-leveling progress invariant
- [ ] 6.8 Coq proof of BBT redundancy (loss of either copy survives)

## 7. Phase 7 — `/flash/` mount

- [ ] 7.1 Extend `posix-vfs` with `/flash/` mount point
- [ ] 7.2 Conditional compilation: feature off → no /flash/ in path tree
- [ ] 7.3 Boot-time mount sequence: flash device discovery → BBT load → littlefs mount
- [ ] 7.4 Coexistence with `/data/` F2FS mount (both available, distinct purposes)
- [ ] 7.5 Flash-only target: canonical `auth/`/`audit/`/`mgmt/` under /flash/
- [ ] 7.6 First-boot directory tree creation per `mgmt-config-layout`
- [ ] 7.7 Format-on-physical-presence path (mirrors F2FS first-boot logic)
- [ ] 7.8 Boot-cleanup: littlefs orphan recovery on mount

## 8. Phase 8 — Per-arch QSPI/ONFI driver stubs

- [ ] 8.1 `arch/aarch64/src/flash/qspi.rs` — QSPI NOR controller binding stub (TODO when first MCU/FPGA target arrives)
- [ ] 8.2 `arch/aarch64/src/flash/onfi.rs` — ONFI NAND controller binding stub
- [ ] 8.3 `arch/riscv64/src/flash/qspi.rs` — QSPI stub
- [ ] 8.4 `arch/riscv64/src/flash/onfi.rs` — ONFI stub
- [ ] 8.5 Documentation: `docs/embedded-flash-fs.md` operator runbook (format choice, BBT semantics, fadvise hints, recovery via littlefs-fuse)

## CHECKPOINT — Two PRs into `develop`

This change ships as **two PRs**:

- **PR 1 (this PR — scaffolding only):** proposal.md, design.md, the 5 spec deltas (3 new + 2 modified), tasks.md. Merges to `develop` so agent teams can spawn worktrees off develop and pick up the design.
- **PR 2 (later — implementation):** all 8 phases in one cohesive merge, behind the default-off `fs-flash` cargo feature. Per-arch QSPI/ONFI bringup ships as documented stubs only; full hardware bringup happens when the first MCU/FPGA target arrives.

Scaffolding (PR 1) gating list:

- [ ] CA.1 `openspec validate embedded-flash-fs-v1 --strict` returns clean
- [ ] CA.2 PR title follows conventional-commit semver convention
- [ ] CA.3 No production code changes (scaffolding only)
- [ ] CA.4 Documentation references `embedded-filesystem-v1` and `embedded-overlay-v1` as sibling foundations

Implementation (PR 2) gating list:

- [ ] CB.1 Behind cargo feature `fs-flash`, default off
- [ ] CB.2 `cargo fmt --check`, `cargo clippy -- -D warnings`
- [ ] CB.3 `cargo test --workspace` total ≥ 5600 (post-overlay baseline + ~350)
- [ ] CB.4 Kani harness for metadata-pair commit atomicity passes
- [ ] CB.5 TLA+ wear-leveling progress model verifies clean
- [ ] CB.6 Coq BBT redundancy proof checked
- [ ] CB.7 1M-cycle wear stress on mock device passes
- [ ] CB.8 Round-trip via `littlefs2` Rust port (per-PR) and via C `mklittlefs` (weekly cron) both clean
- [ ] CB.9 Cyclic-dep check passes (no new edges; reuses `fs/`)
- [ ] CB.10 `cargo audit` / `cargo deny` clean

## 9. Cross-phase verification

- [ ] 9.1 Image-size regression check: littlefs code stays ≤ 80 KB compiled growth budget
- [ ] 9.2 Documentation: `docs/embedded-flash-fs.md` includes spec commit hash, recovery laptop tooling instructions, BBT theory of operation
- [ ] 9.3 Operator runbook covers: how to read /flash/ on a recovery laptop via `littlefs-fuse`; how to format on first boot with physical presence asserted
- [ ] 9.4 Mode-stricter-than-declared loader integration: refuses lax mode on /flash/secrets/sign-key.priv etc.
- [ ] 9.5 Architectural docs reflect that /flash/ is OPT-IN per target via `fs-flash` feature, NOT enabled by default for the current x86/AArch64/RISC-V/Jetson lineup
