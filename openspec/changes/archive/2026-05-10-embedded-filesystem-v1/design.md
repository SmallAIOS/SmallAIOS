## Context

SmallAIOS today has no on-disk read/write filesystem. The existing
[posix/src/vfs.rs](posix/src/vfs.rs) is a 527-line in-memory
read-only VFS that returns `EROFS` for every write. The boot
partition is FAT32 (UEFI ESP) and that is the entire on-disk
story. The recently-landed `management-login-v1` change writes
to `/data/auth/shadow`, `/data/audit/log.jsonl`, and
`/data/mgmt/policy.toml` with `stage → fsync → rename` atomicity
— a contract that has no implementation. The upcoming
`remote-update-v1` change assumes A/B partition swap. Both are
blocked by the absence of a real filesystem substrate.

This change introduces that substrate. The shape converged across
30 design-walkthrough questions:

- **Read-only `/models/` on squashfs**, A/B-partitioned, atomic
  whole-image swap with bsdiff block-level deltas applied to the
  inactive slot.
- **Read-write `/data/` on F2FS**, log-structured, journaled,
  power-fail-safe, mainline-readable.
- **GPT** as the partition table (universal modern standard).
- **Pure Rust, clean-room**, `#![no_std]`. No FFI to C drivers.
  Squashfs zstd+xz+gzip+lz4 decoders, F2FS read+write, GPT,
  bsdiff, A/B boot logic — all written or wrapped from scratch.
- **Universal external interop**. Per-PR CI mounts the produced
  images on stock Ubuntu LTS, Fedora current, and macOS via FUSE.
  Operators with a recovery laptop must not need custom tooling.

OverlayFS-style writable layering on `/models/` for
per-deployment model injection is **explicitly deferred** to a
follow-on `embedded-overlay-v1`. v1 ships A/B whole-image updates.

## Goals / Non-Goals

**Goals:**
- New `fs/` Layer 1 crate (workspace 23 → 24 with `auth` + `mgmt`
  from `management-login-v1`).
- Single `BlockDevice` trait; v1 impls for virtio-blk (QEMU CI),
  NVMe (x86), eMMC/SDHCI (Jetson), AHCI (legacy x86).
- GPT partition table; protective MBR for tool compatibility.
- Squashfs 4.0 read-only with zstd, xz, gzip, lz4 decoders;
  appended manifest + ML-DSA-65 signature footer.
- F2FS matching the Linux 6.6 LTS feature set; read → write →
  fsync/checkpoint → GC phasing.
- Dedicated 8 MiB GPT partition for the A/B boot pointer with
  double-buffered records; UEFI variable mirror.
- Watchdog timeout + explicit `boot_success` syscall as the
  rollback trigger.
- bsdiff block-level delta payloads with ML-DSA-65 signed
  pre/post integrity checks.
- Per-mount LRU block cache (16 MiB `/models/`, 4 MiB `/data/`)
  + 256 KiB write-coalescing buffer.
- Native 4 KiB block size; 512-byte slow-path emulation for
  legacy devices.
- ≥5100 total tests after the change (~+600 new); Kani + TLA+ +
  SPIN + Coq formal coverage.

**Non-Goals (v1):**
- OverlayFS / writable layer on `/models/`. `embedded-overlay-v1`.
- ext4 / xfs / btrfs as alternatives to F2FS.
- LittleFS, JFFS2, UBIFS for raw-flash MCU targets.
  `embedded-flash-fs-v1` if hardware drives the need.
- Multi-disk RAID, encryption-at-rest, fscrypt, NFS, virtio-fs.
- Quotas, ACLs, extended attributes beyond what F2FS provides.
- Trim / discard performance tuning.

## Decisions

The 30 questions resolved during the walkthrough. Each decision
is the source of one or more spec requirements.

### Q1. Block device trait shape

**Decision:** Single `BlockDevice` trait
(`read_block`/`write_block`/`block_size_bytes`/`block_count`/`flush`).
Per-arch impls. Layered queue trait (option B) and per-bus
traits (option C) are documented evolution paths if NVMe-class
parallelism or per-device quirks become binding.

### Q2. Partition table

**Decision:** GPT only. Protective MBR is written per the GPT
spec for tool compatibility. Hybrid-MBR layouts (some macOS
dual-boot disks) are honored as plain GPT.

### Q3. Squashfs compression algorithms

**Decision:** zstd + xz + gzip + lz4. Full compatibility with
`mksquashfs` defaults across all distros. zstd is reused from
`compute`; xz/gzip/lz4 are new clean-room decoders (~5 kLOC
combined).

### Q4. F2FS phasing

**Decision:** Read → write → fsync/checkpoint → GC, four sub-phases
each tested against `mkfs.f2fs`-produced reference images.

### Q5. A/B boot pointer storage

**Decision:** Dedicated 8 MiB GPT partition holds the source of
truth (works on every arch, atomically flippable, externally
inspectable with `dd`). UEFI variable holds a mirror so the UEFI
bootloader can pick a slot without parsing GPT first. On
disagreement, the partition wins.

### Q6. Boot rollback trigger

**Decision:** Watchdog timeout (60 s default) **and** explicit
`boot_success` syscall, both required. Bootloader sets
`tentative=true`; kernel calls `boot_success` after self-tests
and first successful auth pass; failure of either path on next
boot triggers slot rollback.

### Q7. Delta format

**Decision:** bsdiff. Better ratio than zstd `--patch-from` on
binary blobs (Mender's choice for the same reason). Adds ~500
LOC of clean-room bsdiff applier in `fs/src/delta.rs`.

### Q8. Manifest placement

**Decision:** Manifest + ML-DSA-65 signature appended as a sealed
footer to the squashfs blob. Single artifact; squashfs ignores
trailing bytes, so external `mount -t squashfs -o loop` still
works without offset gymnastics.

### Q9. VFS mount points

**Decision:** Fixed in kernel code: `/models/` → active squashfs
slot, `/data/` → F2FS partition, `/dev/` and `/proc/self/` → the
existing in-memory tree. Compile-time constants. Avoids the
chicken-and-egg of reading `/data/mgmt/...` to know how to mount
`/data/`.

### Q10. Block cache

**Decision:** Per-mount LRU (16 MiB `/models/`, 4 MiB `/data/`)
plus a 256 KiB write-coalescing buffer for `/data/`. Cache budgets
configurable via `mgmt/policy.toml` (`fs.cache.models_bytes`,
`fs.cache.data_bytes`).

### Q11. Crate split

**Decision:** Single `fs/` Layer 1 crate. Sub-crates
(`fs-core`/`fs-squashfs`/`fs-f2fs`/`fs-update`) are documented
evolution paths if cross-FS sharing becomes a problem.

### Q12. Block error encoding

**Decision:** Typed `BlockError` enum internally
(`MediaError`/`NotPresent`/`Timeout`/`BadCrc`/`Unaligned`/`OutOfRange`);
POSIX-aligned errno (`-EIO`/`-ENXIO`/`-ETIMEDOUT`) at the syscall
boundary. Matches `management-login-v1`'s pattern.

### Q13. Squashfs version

**Decision:** Squashfs 4.0 only. Reject older (1.x/2.x/3.x) with
a clear error. Forward-compatible reject-unknown-major-version
check guards future 5.x.

### Q14. F2FS feature-set target

**Decision:** Linux 6.6 LTS feature set. Forward-compatible:
unknown mandatory feature bits cause a clean refusal, not silent
mis-mounts.

### Q15. F2FS commit cadence

**Decision:** Inline on `fsync` plus a 5-s background commit
timer. Idle-detection commit (option D) is documented as an
evolution path once a power-state observer exists.

### Q16. Integrity check policy

**Decision:** Manifest signature verified once at mount;
per-block SHA-3-256 verified on every read before bytes leave
the kernel. Fail-closed.

### Q17. Both-slots-bad recovery

**Decision:** Refuse to mount `/models/`. Print recovery hint.
Login + audit + `/data/` remain available so the operator can
investigate and re-image. Inference is hard-disabled.

### Q18. Block I/O retry

**Decision:** Per-op timeout (250 ms read, 1 s write) + 3 retries
with exponential backoff (500 ms / 1 s / 2 s) → fail with `-EIO`.
Retry counts and timeouts configurable via `mgmt/policy.toml`.
The fail-after-retries behavior is **non-configurable**: an
unattended appliance must never wedge on a bad sector.

### Q19. F2FS crash safety

**Decision:** Standard POSIX/F2FS guarantee: all `fsync`-acknowledged
writes are durable; non-fsync data is lost up to the last
checkpoint.

### Q20. Boot-pointer atomicity

**Decision:** Invariant: after any sequence of writes followed
by an arbitrary power loss, the bootloader **always** finds at
least one slot whose `valid=true` and whose generation counter is
one of the two most recent. Generation counter is monotonically
increasing. Modeled in Kani.

### Q21. Delta-apply checks

**Decision:** Pre: ML-DSA-65 signature on delta payload + reference
blob hash check. Post: full SHA-3-256 manifest verify on the
inactive partition + ML-DSA-65 signature verify. Fail-closed leaves
inactive partition `unbootable`.

### Q22. External interop CI

**Decision:** Ubuntu LTS (current), Fedora (current), macOS via
FUSE. Every PR mounts the produced squashfs and F2FS images on
all three and round-trips data with `cmp`.

### Q23. Sector size

**Decision:** Native 4 KiB throughout. 512-byte logical / 4 KiB
physical Advanced Format devices accessed in 4 KiB chunks.
512-byte-only legacy devices handled by a slow-path emulation
layer.

### Q24. First-boot `/data/` formatting

**Decision:** Format only when a `PhysicalPresenceProvider`
indicator is asserted (mirrors `management-login-v1`'s
`auth.skip-firstboot` policy). Otherwise halt with recovery hint.
Format event audit-recorded.

### Q25. Phase ordering

**Decision:** 10-phase bottom-up sequence (block → GPT → squashfs
→ A/B + delta → F2FS-RO → F2FS-RW → fsync → GC → integrity →
interop CI). Each phase ends green.

### Q26. Test target

**Decision:** ≥5100 total tests after the change (~+600 new).
DO-178C-target depth: exhaustive error paths, F2FS spec
coverage, sector-size/cache/retry combinations.

### Q27. Formal verification

**Decision:**
- Kani — A/B boot atomicity (Q20 invariant), delta-apply
  pipeline (Q21), GPT bounds.
- TLA+ — F2FS checkpoint commit interleaving.
- SPIN — Promela model proving no path under integrity failure
  leaks unverified bytes to user space.
- Coq — bsdiff applier correctness proof.

### Q28. Block device order

**Decision:** virtio-blk → NVMe (x86) → eMMC/SDHCI (Jetson) →
AHCI (legacy x86). CI signal first via QEMU; real-hardware
bringup follows.

### Q29. PR strategy

**Decision:** Three checkpoint PRs into `develop`:
- A: `block + GPT + squashfs read + A/B + delta`
- B: `F2FS read-only + integrity`
- C: `F2FS read-write + fsync + GC + interop CI`

Each checkpoint is independently shippable. New on-disk mounts
are gated behind a `fs-on-disk-mounts` cargo feature (off by
default) until checkpoint C lands; `/data/` falls back to
in-memory until the RW path is ready.

### Q30. Session next step

**Decision:** Draft `design.md`, the 10 spec deltas, and
`tasks.md`; commit on `change/embedded-filesystem-v1`; run
`openspec validate --strict`; push to origin. No `develop`
touched.

## Risks / Trade-offs

- **[Risk] F2FS write path is the largest clean-room
  implementation in the project to date** — Mitigation: Q4 phasing
  delivers RO before RW; checkpoint B is shippable with reads
  only behind the cargo feature; Coq-style assurance carries a
  high bar but is restricted to bsdiff (a small applier), not the
  whole F2FS write path. The F2FS write path leans on TLA+
  checkpoint modeling + property-based fuzzing rather than full
  proof.
- **[Risk] Block-device drivers on bare-metal x86 (NVMe / AHCI)
  may uncover gaps in PCIe enumeration** — Mitigation: virtio-blk
  ships first (Q28) so CI is unblocked while hardware bringup
  proceeds; NVMe second because USB-NVMe carriers are the cheapest
  test bench; AHCI last because our deployment story doesn't lean
  on legacy SATA.
- **[Risk] Boot-pointer flip atomicity is the central correctness
  invariant for `remote-update-v1`** — Mitigation: Q20 invariant
  is Kani-modeled with a power-loss harness covering arbitrary
  partial writes across the 8 MiB boot-config partition. The
  proof carries across ABI changes.
- **[Risk] Five compression decoders (zstd/xz/gzip/lz4 + bsdiff)
  multiply the parser attack surface** — Mitigation: each decoder
  has its own fuzz harness; CI runs them on every PR; Linux's CVE
  history for these formats is the floor we measure against.
- **[Risk] Cache budgets configurable from `mgmt/policy.toml`
  introduces a feedback loop with `management-login-v1`'s
  universal-exposure invariant** — Mitigation: cache fields use
  `#[reload("live")]`; budget changes apply on next allocation
  without remount. Validators reject budgets below a hard floor
  (4 MiB `/models/`, 1 MiB `/data/`) so misconfiguration cannot
  starve the FS layer.
- **[Risk] Workspace count growth** — Mitigation: Q11 keeps it to
  one new crate (`fs/`), bringing the post-`management-login-v1`
  count from 23 → 24.
- **[Risk] FAT32 ESP remains as the boot partition** — Out of
  scope but worth flagging: this change does not replace the UEFI
  boot partition, which stays FAT32 because UEFI requires it.
  `verified-boot` covers the kernel image stored on FAT32; the
  rest of the system runs on squashfs + F2FS.

## Migration Plan

This is a v0.x prototype change. There is no on-disk data from
prior versions to migrate; the existing in-memory VFS holds no
state across reboots.

First boot of any image carrying this change:
- If the disk has a recognized GPT with the v1 partition layout
  and a valid F2FS superblock at partition 4 → mount normally.
- If the disk is pristine (no GPT) and physical presence is
  asserted → write the GPT, format `/data/` (Q24), populate slot
  A from the boot image, mark slot B `unbootable`, write the
  initial boot config.
- If the disk has an unrecognized layout and physical presence is
  not asserted → halt with a recovery hint.

`management-login-v1` writes to `/data/` previously routed through
the in-memory VFS will transparently flow to F2FS after the
`fs-on-disk-mounts` cargo feature flips on. No application code
changes.

## Open Questions

All ten of the proposal's open questions are now resolved (Q1–Q10
above). Twenty additional design / specs / sequencing questions
were resolved during the walkthrough (Q11–Q30).

Items deferred to follow-on changes with explicit decisions:
- **`embedded-overlay-v1`** — OverlayFS on top of `/models/` for
  per-deployment model injection without a full image swap.
- **`embedded-flash-fs-v1`** — LittleFS / JFFS2 / UBIFS support
  for raw-flash MCU/FPGA targets if those platforms enter the
  roadmap.
- **F2FS idle-detection commit** — Q15 documented this as a
  future addition once a power-state observer crate exists.
- **NVMe queue-depth-32+ parallel I/O** — Q1 documented the
  layered queue trait as a follow-on if NVMe performance becomes
  binding.
