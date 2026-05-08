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

- [ ] 2.1 `fs/src/gpt.rs` — clean-room GPT parser per UEFI 2.10 §5.3
- [ ] 2.2 Primary + secondary header validation, fallback on primary corrupt
- [ ] 2.3 Protective MBR writer (for fresh-format path)
- [ ] 2.4 v1 partition layout enforcement (5-partition table, type GUID checks)
- [ ] 2.5 SmallAIOS-specific type GUID allocation, documented in `docs/architecture.md`
- [ ] 2.6 GPT parser fuzz harness
- [ ] 2.7 Kani harness for parser bounds (no out-of-buffer reads on adversarial input)
- [ ] 2.8 Tests against `parted`/`gdisk`-produced disks

## 3. Phase 3 — Squashfs read path + manifest verification

- [ ] 3.1 `fs/src/squashfs/superblock.rs` — squashfs 4.0 superblock parser
- [ ] 3.2 `fs/src/squashfs/inodes.rs` — inode table reader
- [ ] 3.3 `fs/src/squashfs/dirs.rs` — directory table reader
- [ ] 3.4 `fs/src/squashfs/fragments.rs` — fragment table reader
- [ ] 3.5 zstd decoder wired in from `compute`
- [ ] 3.6 `fs/src/squashfs/xz.rs` — clean-room xz/lzma2 decoder
- [ ] 3.7 `fs/src/squashfs/gzip.rs` — clean-room inflate decoder (RFC 1951)
- [ ] 3.8 `fs/src/squashfs/lz4.rs` — clean-room lz4 block decoder
- [ ] 3.9 Round-trip test against `mksquashfs -comp zstd` reference image
- [ ] 3.10 Round-trip test against `mksquashfs -comp xz` reference image
- [ ] 3.11 Round-trip test against `mksquashfs -comp gzip` reference image
- [ ] 3.12 Round-trip test against `mksquashfs -comp lz4` reference image
- [ ] 3.13 `fs/src/squashfs/manifest.rs` — appended footer parser + ML-DSA-65 verify
- [ ] 3.14 Mount-time signature verification fail-closed
- [ ] 3.15 Per-block SHA-3-256 verify on every read
- [ ] 3.16 External interop CI: stock `mount -t squashfs -o loop` on Ubuntu LTS
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

- [ ] 5.1 `fs/src/f2fs/superblock.rs` — Linux 6.6 LTS feature-set superblock parser
- [ ] 5.2 Primary/secondary superblock fallback on CRC failure
- [ ] 5.3 `fs/src/f2fs/checkpoint.rs` — checkpoint journal reader
- [ ] 5.4 `fs/src/f2fs/sit.rs` — Segment Information Table reader
- [ ] 5.5 `fs/src/f2fs/nat.rs` — Node Address Table reader
- [ ] 5.6 `fs/src/f2fs/ssa.rs` — Segment Summary Area reader
- [ ] 5.7 `fs/src/f2fs/inode.rs` — F2FS inode parsing, extents, indirect pointers
- [ ] 5.8 `fs/src/f2fs/dir.rs` — directory iteration
- [ ] 5.9 `fs/src/f2fs/data.rs` — data block read path with cache integration
- [ ] 5.10 Round-trip test: `mkfs.f2fs` populated by Linux → SmallAIOS reads → `cmp` byte-exact
- [ ] 5.11 Unknown mandatory feature bit rejected with clear error
- [ ] 5.12 F2FS metadata CRC32C verification on every metadata read
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

- [ ] 7.1 `fs/src/f2fs/write.rs` — file create, directory create, write extending
- [ ] 7.2 NAT/SIT/SSA update path, log-structured allocation
- [ ] 7.3 Atomic rename via journal
- [ ] 7.4 truncate, unlink, rmdir
- [ ] 7.5 Round-trip test: SmallAIOS writes → Linux 6.6 reads → `cmp` byte-exact
- [ ] 7.6 Application-layer `policy.toml` SHA-3-256 fingerprint header
- [ ] 7.7 Write-coalescing buffer integration

## 8. Phase 8 — fsync + checkpoint commit

- [ ] 8.1 `fs/src/f2fs/fsync.rs` — checkpoint commit on `fsync(fd)`
- [ ] 8.2 5-second background commit timer (cooperative)
- [ ] 8.3 Power-fail tests: kill-9 mid-write → reboot → verify last fsync'd offset intact
- [ ] 8.4 TLA+ model `f2fs_checkpoint.tla` — checkpoint commit interleaving invariants
- [ ] 8.5 First-boot format-on-physical-presence path
- [ ] 8.6 First-boot `/data/` directory-tree creation with declared modes

## 9. Phase 9 — F2FS garbage collection

- [ ] 9.1 `fs/src/f2fs/gc.rs` — segment-level GC
- [ ] 9.2 Threshold-driven trigger (free segments < 5%)
- [ ] 9.3 Cooperative yielding so foreground writes are not starved
- [ ] 9.4 Live-block relocation correctness tests (post-GC `cmp` against pre-GC)
- [ ] 9.5 Property-based GC stress tests under random foreground workload

## 10. Phase 10 — Boot success syscall + interop CI matrix

- [ ] 10.1 Add `boot_success` syscall (`SYS_BOOT_SUCCESS = 0x57`, System category) — Root only, idempotent
- [ ] 10.2 Watchdog arming + disarming wiring through `boot_success`
- [ ] 10.3 Audit record `boot_success_committed` on commit
- [ ] 10.4 End-to-end A/B update test: stage delta → reboot → boot_success → next boot
- [ ] 10.5 End-to-end rollback test: stage delta → reboot → no boot_success → watchdog → previous slot
- [ ] 10.6 Update `docs/architecture.md` syscall table to include `SYS_BOOT_SUCCESS = 0x57`
- [ ] 10.7 Interop CI: full matrix (Ubuntu LTS + Fedora current + macOS via FUSE) for both squashfs and F2FS
- [ ] 10.8 Image-size regression check: F2FS write path stays within ≤ 200 KB compiled growth budget
- [ ] 10.9 Boot footprint regression check: total binary ≤ 8.2 MB (vs prior 8.0 MB target)
- [ ] 10.10 AHCI `BlockDevice` impl (legacy x86-64 SATA) — opportunistic; not a checkpoint blocker

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
