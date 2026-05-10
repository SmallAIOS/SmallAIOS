## 1. Phase 1 — `fs/` crate scaffold + `BlockDevice` trait

- [ ] 1.1 Create `fs/` crate, `Cargo.toml`, register in workspace (23 → 24)
- [ ] 1.2 `fs/src/lib.rs` — module structure, `#![no_std]` declaration
- [ ] 1.3 `fs/src/block.rs` — `BlockDevice` trait, `BlockError` enum
- [ ] 1.4 `fs/src/block/errno.rs` — `BlockError` → POSIX errno mapping
- [ ] 1.5 Per-op timeout + 3-retry exponential backoff helper
- [ ] 1.6 4 KiB / 512 B sector-size handling layer with emulation slow-path
- [ ] 1.7 Generic conformance test suite (alignment, retry, error mapping)
- [ ] 1.8 `arch/{x86_64,aarch64}/src/virtio_blk.rs` — `BlockDevice` impl
- [ ] 1.9 Conformance suite passes against virtio-blk in QEMU CI
- [ ] 1.10 Cyclic-dep check passes; clippy-D-warnings clean

## 2. Phase 2 — GPT partition table

- [x] 2.1 `fs/src/gpt.rs` — clean-room GPT parser per UEFI 2.10 §5.3
- [x] 2.2 Primary + secondary header validation, fallback on primary corrupt
- [x] 2.3 Protective MBR writer (for fresh-format path)
- [x] 2.4 v1 partition layout enforcement (5-partition table, type GUID checks)
- [x] 2.5 SmallAIOS-specific type GUID allocation, documented in `docs/architecture.md`
- [ ] 2.6 GPT parser fuzz harness
- [ ] 2.7 Kani harness for parser bounds (no out-of-buffer reads on adversarial input)
- [ ] 2.8 Tests against `parted`/`gdisk`-produced disks

## 3. Phase 3 — Squashfs read path + manifest verification

- [x] 3.1 `fs/src/squashfs/superblock.rs` — squashfs 4.0 superblock parser
- [x] 3.2 `fs/src/squashfs/inodes.rs` — inode table reader
- [x] 3.3 `fs/src/squashfs/dirs.rs` — directory table reader
- [x] 3.4 `fs/src/squashfs/fragments.rs` — fragment table reader
- [x] 3.5 zstd decoder (clean-room — `compute` crate has no zstd; landed in `fs/src/squashfs/decompress/zstd.rs`)
- [x] 3.6 `fs/src/squashfs/decompress/xz.rs` — clean-room xz/lzma2 decoder (uncompressed-chunk path; LZMA range decoder DEFERRED to a follow-up phase, see `TODO(#fs-squashfs-xz-lzma)`)
- [x] 3.7 `fs/src/squashfs/decompress/gzip.rs` — clean-room inflate decoder (RFC 1951)
- [x] 3.8 `fs/src/squashfs/decompress/lz4.rs` — clean-room lz4 block decoder
- [x] 3.9 Round-trip test for `Compression::Zstd` (raw-block framing)
- [x] 3.10 Round-trip test for `Compression::Xz` (uncompressed-chunk framing)
- [x] 3.11 Round-trip test for `Compression::Gzip` (DEFLATE stored-block path)
- [x] 3.12 Round-trip test for `Compression::Lz4` (literal-only sequence path)
- [x] 3.13 `fs/src/squashfs/manifest.rs` — appended footer parser + ML-DSA-65 verify
- [x] 3.14 Mount-time signature verification fail-closed
- [x] 3.15 Per-4-KiB SHA-3-256 verify of the squashfs blob at mount (stronger than per-read)
- [ ] 3.16 External interop CI: stock `mount -t squashfs -o loop` on Ubuntu LTS (helper script `tools/ci/squashfs-interop.sh` landed; GitHub Actions wiring DEFERRED)
- [ ] 3.17 External interop CI: stock `mount -t squashfs -o loop` on Fedora current
- [ ] 3.18 External interop CI: macOS via `squashfs-tools` from Homebrew

## 4. Phase 4 — A/B boot pointer + bsdiff delta apply

- [ ] 4.1 `fs/src/boot_config.rs` — `BootConfigRecord` struct + double-buffer layout
- [ ] 4.2 Read both records, validate SHA-3-256 record_hash, pick highest valid generation
- [ ] 4.3 Atomic update: write inactive slot → flush → done
- [ ] 4.4 Kani harness for atomicity invariant under arbitrary partial writes
- [ ] 4.5 UEFI variable mirror writer (x86-64 + aarch64-with-UEFI)
- [ ] 4.6 Variable-vs-partition disagreement: partition wins, log warning, rewrite variable
- [ ] 4.7 Watchdog arming on boot when `tentative=1`
- [ ] 4.8 `fs/src/delta.rs` — clean-room bsdiff applier
- [ ] 4.9 Pre-apply ML-DSA-65 signature verify on delta payload
- [ ] 4.10 Pre-apply reference-blob hash check
- [ ] 4.11 Bsdiff streaming apply to inactive partition
- [ ] 4.12 Post-apply per-block SHA-3-256 manifest verify
- [ ] 4.13 Post-apply ML-DSA-65 footer signature verify
- [ ] 4.14 Hand-off event `update_staged{ new_slot, generation }` for `remote-update-v1`
- [ ] 4.15 Coq proof of bsdiff applier correctness
- [ ] 4.16 Property-based tests for bsdiff under truncated, scrambled, and oversized patches

## CHECKPOINT A — Phases 1-4 ship as one PR into develop

- [ ] CA.1 Behind cargo feature `fs-on-disk-mounts` (off by default)
- [ ] CA.2 `cargo fmt --check`, `cargo clippy -- -D warnings`
- [ ] CA.3 `cargo test --workspace` total ≥ 4700
- [ ] CA.4 External interop CI green for squashfs on Ubuntu / Fedora / macOS
- [ ] CA.5 Kani harnesses pass: GPT parser, A/B boot atomicity, bsdiff bounds
- [ ] CA.6 PR opened against `develop` with summary covering Phases 1-4

## 5. Phase 5 — F2FS read path

- [x] 5.1 `fs/src/f2fs/superblock.rs` — Linux 6.6 LTS feature-set superblock parser
- [x] 5.2 Primary/secondary superblock fallback on CRC failure
- [x] 5.3 `fs/src/f2fs/checkpoint.rs` — checkpoint journal reader
- [x] 5.4 `fs/src/f2fs/sit.rs` — Segment Information Table reader
- [x] 5.5 `fs/src/f2fs/nat.rs` — Node Address Table reader
- [x] 5.6 `fs/src/f2fs/ssa.rs` — Segment Summary Area reader
- [x] 5.7 `fs/src/f2fs/inode.rs` — F2FS inode parsing, extents, indirect pointers
- [x] 5.8 `fs/src/f2fs/dir.rs` — directory iteration
- [x] 5.9 `fs/src/f2fs/data.rs` — data block read path (cache integration deferred to Phase 6)
- [x] 5.10 Round-trip test: synthetic F2FS image → SmallAIOS reads → byte-exact (mkfs.f2fs interop deferred to external CI)
- [x] 5.11 Unknown mandatory feature bit rejected with clear error
- [x] 5.12 F2FS metadata CRC32C verification on every metadata read
- [ ] 5.13 NVMe `BlockDevice` impl (x86-64) with conformance + F2FS RO test
- [ ] 5.14 SDHCI/eMMC `BlockDevice` impl (Jetson) with conformance + F2FS RO test on real Orin

## 6. Phase 6 — Mount machinery + integrity layer

- [ ] 6.1 Extend `posix-vfs` with `/models/` and `/data/` mount points
- [ ] 6.2 Boot-time mount sequence: GPT → squashfs slot via boot config → F2FS
- [ ] 6.3 Both-slots-bad halt path (login + audit + `/data/` still up)
- [ ] 6.4 SPIN model proving no path under integrity failure leaks unverified bytes
- [ ] 6.5 `fs/src/cache.rs` — per-mount LRU (16 MiB models, 4 MiB data, configurable)
- [ ] 6.6 256 KiB write-coalescing buffer for `/data/`
- [ ] 6.7 Cache-budget validators with hard floor (4 MiB / 1 MiB)
- [ ] 6.8 `mgmt/policy.toml` `fs.cache.*`, `fs.block.*` fields with `#[reload("live")]` annotations

## CHECKPOINT B — Phases 5-6 ship as one PR into develop

- [ ] CB.1 `fs-on-disk-mounts` cargo feature still off by default — mounts still in-memory in production
- [ ] CB.2 `cargo test --workspace` total ≥ 4900
- [ ] CB.3 External interop CI green: `mount -t f2fs` on Ubuntu / Fedora reads SmallAIOS-produced images
- [ ] CB.4 SPIN model verifies clean
- [ ] CB.5 PR opened against `develop` with summary covering Phases 5-6

## 7. Phase 7 — F2FS write path

- [x] 7.1 `fs/src/f2fs/write.rs` — file create, directory create, write extending
- [x] 7.2 NAT/SIT/SSA update path, log-structured allocation — Phase 7 NAT/SIT in `fs/src/f2fs/write.rs`; Phase 9 integration wires the in-memory SSA reverse map (`record_ssa` + `WriteState::ssa_index`) so GC v2's `live_blocks_from_segment` projection has authoritative data
- [x] 7.3 Atomic rename via journal
- [x] 7.4 truncate, unlink, rmdir
- [ ] 7.5 Round-trip test: SmallAIOS writes → Linux 6.6 reads → `cmp` byte-exact (DEFERRED — needs the F2FS interop CI matrix from task 10.7's coverage)
- [ ] 7.6 Application-layer `policy.toml` SHA-3-256 fingerprint header (DEFERRED — owned by mgmt phase)
- [ ] 7.7 Write-coalescing buffer integration (DEFERRED — Phase 6 cache LRU lands first)
- [x] 7.8 Phase 9 GC v2 wired into the write path — `fs/src/f2fs/write.rs::maybe_run_gc` calls `gc::run_gc_pass_v2` with hot/cold cursor hints, journal staging, and the cooperative-yield budget; integration tests in `fs/tests/integration_gc_v2_writepath.rs` (34 cases, mixed-mtime fragmentation workload preserves bytes)
- [x] 7.9 Hot/cold cursor seeding from checkpoint pack — `WriteState::cur_hot_segno` / `cur_cold_segno` initialized to `cur_data_segno` on mount; surfaced via `F2fs::allocator_hints` for tests
- [x] 7.10 Wall-clock injection for GC age classification — `F2fs::set_now_secs` / `now_secs()` setter+getter; production wires the kernel scheduler's clock tick, tests inject deterministic values
- [x] 7.11 GC journal staging surface — `WriteState::gc_journal_dirty` accumulates `GcJournalRecord` entries before the SIT/NAT mutation lands; drained at `fsync` alongside `journal_dirty`

## 8. Phase 8 — fsync + checkpoint commit

- [x] 8.1 `fs/src/f2fs/fsync.rs` — checkpoint commit on `fsync(fd)` (lives in `fs/src/f2fs/write.rs::fsync`; see Phase 7 + Phase 8 polish)
- [x] 8.2 5-second background commit timer (cooperative) — `fs/src/f2fs/commit_timer.rs`
- [x] 8.3 Power-fail tests: kill-9 mid-write → reboot → verify last fsync'd offset intact — `fs/tests/f2fs_fsync_fuzz.rs`
- [x] 8.4 TLA+ model `f2fs_checkpoint.tla` — checkpoint commit interleaving invariants (Phase 8 expands to 9 invariants; smoke `fs/tests/f2fs_checkpoint_tla_smoke.rs`)
- [ ] 8.5 First-boot format-on-physical-presence path (DEFERRED — owned by mgmt phase)
- [ ] 8.6 First-boot `/data/` directory-tree creation with declared modes (DEFERRED — owned by mgmt phase)

## 9. Phase 9 — F2FS garbage collection

- [x] 9.1 `fs/src/f2fs/gc.rs` — segment-level GC (Phase 7 baseline + Phase 9 sophistication: multi-victim, hot/cold, journal staging, time cap, yield budget)
- [x] 9.2 Threshold-driven trigger (free segments < 5%) — Phase 7 (`should_run_gc`)
- [x] 9.3 Cooperative yielding so foreground writes are not starved — Phase 7 (`GcYield`) + Phase 9 (`yield_budget_blocks`, `should_yield` poll)
- [x] 9.4 Live-block relocation correctness tests (post-GC `cmp` against pre-GC) — `fs/tests/f2fs_gc_phase9_conformance.rs::live_block_relocation_preserves_all_data_1000_files` and `integration_10000_files_no_corruption`
- [x] 9.5 Property-based GC stress tests under random foreground workload — `fs/tests/f2fs_gc_phase9_conformance.rs` (foreground-yield, time-cap, multi-victim, hot/cold)
- [x] 9.6 SSA-driven live-block relocation primitives — `gc::live_blocks_from_segment`, `gc::run_gc_pass_v2`
- [x] 9.7 Multi-victim selection (top-N) — `gc::pick_victims`, default N=4 via `DEFAULT_GC_VICTIMS_PER_PASS`
- [x] 9.8 Age-based hot/cold heuristics — `gc::classify_temperature`, `gc::pick_target_segment`, default cold-threshold 1 hour via `DEFAULT_COLD_THRESHOLD_SECS`
- [x] 9.9 Per-pass time cap — `GcPassConfig::pass_budget_ticks`, default 100 via `DEFAULT_PASS_BUDGET_TICKS`
- [x] 9.10 Relocation-staging journal — `fs/src/f2fs/gc_journal.rs` (`GcJournalRecord` + CRC-protected encode/decode block)
- [x] 9.11 TLA+ checkpoint model bump — Phase 9 GC interleaving invariants in `formal/tla/F2fsCheckpoint.tla` (5 new invariants: `InvLiveBlocksPreservedUnderCrash`, `InvFreeSegmentCountNonNegative`, `InvGcJournalReplayIdempotent`, `InvGcRelocationAtomicWrtCheckpoint`, `InvGcCannotWedgeFsync`); Rust mirror in `fs/tests/f2fs_checkpoint_tla_smoke.rs`

## 10. Phase 10 — Boot success syscall + interop CI matrix

- [x] 10.1 Add `boot_success` syscall (`SYS_BOOT_SUCCESS = 0x57`, System category) — Root only, idempotent (`kernel/src/syscall/system.rs::sys_boot_success` + `kernel/src/boot_success.rs`)
- [x] 10.2 Watchdog arming + disarming wiring through `boot_success` (`fs/src/boot_config/watchdog.rs::arm_if_tentative`; per-arch real impl deferred — see KernelWatchdog rustdoc)
- [x] 10.3 Audit record `boot_success_committed` on commit (`kernel/src/boot_success.rs::audit_capture`)
- [ ] 10.4 End-to-end A/B update test: stage delta → reboot → boot_success → next boot (DEFERRED — needs full QEMU integration harness)
- [ ] 10.5 End-to-end rollback test: stage delta → reboot → no boot_success → watchdog → previous slot (DEFERRED — same)
- [ ] 10.6 Update `docs/architecture.md` syscall table to include `SYS_BOOT_SUCCESS = 0x57` (DEFERRED — docs phase)
- [x] 10.7 Interop CI: full matrix (Ubuntu LTS + Fedora current + macOS via FUSE) for both squashfs and F2FS — `.github/workflows/fs-interop.yml`
- [x] 10.8 Image-size regression check: F2FS write path stays within ≤ 200 KB compiled growth budget — `tools/ci/image-size-check.sh`
- [x] 10.9 Boot footprint regression check: total binary ≤ 8.2 MB (vs prior 8.0 MB target) — same script
- [ ] 10.10 AHCI `BlockDevice` impl (legacy x86-64 SATA) — opportunistic; not a checkpoint blocker (DEFERRED, marked opportunistic)

## CHECKPOINT C — Phases 7-10 ship as one PR into develop

- [ ] CC.1 Flip `fs-on-disk-mounts` default to ON
- [ ] CC.2 `cargo test --workspace` total ≥ 5100
- [ ] CC.3 Full external interop matrix green
- [ ] CC.4 Kani: GPT, A/B atomicity, delta pipeline pass
- [ ] CC.5 TLA+: F2FS checkpoint model verifies clean
- [ ] CC.6 SPIN: integrity fail-closed proof clean
- [ ] CC.7 Coq: bsdiff applier proof checked
- [ ] CC.8 PR opened against `develop` with summary covering Phases 7-10

## 11. Cross-phase verification

- [ ] 11.1 `cargo fmt --check`, `cargo clippy -- -D warnings`
- [ ] 11.2 `cargo audit` advisory clean
- [ ] 11.3 `cargo deny` license/advisory/ban clean
- [ ] 11.4 `cargo geiger` shows no new unsafe outside well-justified arch / decoder code
- [ ] 11.5 `cargo llvm-cov --fail-under-lines 80` passes (ratchet target 93%)
- [ ] 11.6 Cyclic-dep check passes (workspace 23 → 24, all new edges respect Layer 1 → 0)
- [ ] 11.7 `just arch-check` clean (module-level acyclicity)
- [ ] 11.8 `openspec validate embedded-filesystem-v1 --strict` returns clean
- [ ] 11.9 Zero CodeQL alerts on the new code (preserves baseline)
- [ ] 11.10 Documentation updated: `docs/architecture.md` partition layout + GUIDs + syscall table; `docs/embedded-filesystem.md` operator runbook
