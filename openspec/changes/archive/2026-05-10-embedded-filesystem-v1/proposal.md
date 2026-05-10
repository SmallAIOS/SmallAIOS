## Why

SmallAIOS today has no on-disk read/write filesystem driver. The
existing `posix/src/vfs.rs` is a 527-line in-memory read-only VFS
that returns `EROFS` for every write. The boot partition is FAT32
(UEFI ESP) and that is the entire on-disk story. The
`management-login-v1` change adds a write-heavy `/data/` layer
(`auth/shadow`, `audit/log.jsonl`, `mgmt/policy.toml`,
per-subsystem TOML config) that assumes POSIX `stage → fsync →
rename` atomicity — a contract that has no implementation. The
upcoming `remote-update-v1` change assumes A/B partition swap.
Both are blocked by the absence of a real on-disk filesystem
substrate.

This change introduces the embedded-filesystem layer SmallAIOS
needs to graduate from "the binary that runs in RAM" to "an
attended appliance that boots, persists state, and survives
power loss":

1. **Read-only `/models/` on squashfs**, A/B-partitioned for
   atomic update swap. Squashfs is universally readable on Linux
   (`mount -t squashfs` is preinstalled on every mainstream
   distro), provides per-block transparent decompression
   (zstd / xz), and matches the read-mostly inference workload.
2. **Read-write `/data/` on F2FS**. F2FS is a log-structured
   journaling filesystem designed for flash storage (eMMC, NVMe,
   SSD), mainline-readable since Linux kernel 3.8, gives
   power-fail-safe atomic rename semantics, and handles the
   wear-leveling story FAT32 does not.
3. **Block-level delta updates**. Updates ship as binary diffs
   (zstd `--patch-from` or bsdiff) applied against the active
   squashfs blob to produce the inactive partition; a single
   atomic boot-pointer flip activates the new image.
4. **Pure Rust, clean-room implementations**, `#![no_std]`, no
   FFI to C drivers. Squashfs and F2FS read paths first; F2FS
   write path; integrity verification on every read.
5. **Universal external interop**. A recovery laptop running
   stock Linux must be able to mount both filesystems without
   installing custom tooling. This is a hard requirement and
   constrains every format and parameter choice.

OverlayFS-style writable layering on `/models/` (so operators can
add models without a full image swap) is **explicitly deferred**
to a follow-on `embedded-overlay-v1`. v1 ships A/B whole-image
updates only.

## What Changes

### New `fs/` Layer 1 crate

A new Layer 1 workspace crate owns block-device I/O, partition
table parsing, and the on-disk filesystem implementations. Layer
1 placement keeps it accessible to `kernel` (Layer 0) for boot
mounts and to `container` (Layer 3) for runtime use; it depends
only on `kernel` and `security` (for hash-based integrity
verification). Workspace count grows by one (current `auth` +
`mgmt` from `management-login-v1` would take it to 23 → 24).

### Block-device abstraction

A `BlockDevice` trait (`read_block`, `write_block`,
`block_size_bytes`, `block_count`, `flush`) sits at the bottom
of the new crate. v1 implementations:

- **AHCI/NVMe** on x86-64 — reuses or extends existing PCIe
  bringup in `arch/x86_64`.
- **eMMC** on Jetson Orin — wraps the Tegra234 SDHCI controller
  exposed by `arch/aarch64`.
- **virtio-blk** on QEMU — for CI smoke tests and developer
  workflows.

### Squashfs (read-only) driver

`fs/src/squashfs.rs` — clean-room `#![no_std]` Rust parser for
the squashfs 4.0 on-disk format. Read-only. Supports zstd
decompression (already in the workspace via `compute`); xz
support is optional. Mounts at `/models/`. Integrity-verifies
every block against an SHA-3-256 manifest published with the
image.

### F2FS (read-write) driver

`fs/src/f2fs.rs` — clean-room `#![no_std]` Rust implementation
of the F2FS on-disk format. Read and write. Implements the
checkpoint / segment / SIT / NAT structures, atomic rename via
the journal, and `fsync`. Mounts at `/data/`. This is the
biggest piece of work in the change and is phased: read path
first (Phase 4), write path second (Phase 5), `fsync` and
checkpoint behavior third (Phase 6).

### Partition table

GPT (GUID Partition Table) parser with the v1 layout:

```text
Partition  Type     Size       Purpose
─────────────────────────────────────────────────────────────
1          ESP      256 MiB    UEFI FAT32 — bootloader + kernel
2          squashfs ~4 GiB     /models/ slot A
3          squashfs ~4 GiB     /models/ slot B
4          F2FS     remainder  /data/
```

GPT is universal. The two squashfs slots are equal-size and
swappable. The single `/data/` partition is shared across boots.

### A/B boot pointer

A small (4 KiB) "boot config" region in a fixed location (either
LBA 64-71 reserved, or a tiny dedicated partition) records:

- Active squashfs slot (A or B)
- Generation counter and boot-success counter
- SHA-3-256 of the boot config itself for integrity

The boot config is updated atomically by writing the alternate
slot's bytes and flipping a `valid` pointer (similar to the EFI
variable double-buffer pattern). On boot failure (watchdog
timeout), the pointer reverts to the previous slot.

### Block-level delta update

`fs/src/delta.rs` — applies a zstd `--patch-from` or bsdiff
delta against the active squashfs blob, writes the result to
the inactive partition, verifies the SHA-3-256 manifest, then
returns the new slot to `remote-update-v1` for the boot-pointer
flip. v1 picks one delta format (zstd patch is leaner for
embedded since zstd is already in the dep graph; bsdiff is
documented as an alternative).

### Integrity verification

Every squashfs block read SHALL be verified against the
manifest's SHA-3-256 hash before the bytes leave the kernel.
F2FS metadata SHALL be verified via its native CRC32C; data
blocks SHALL be checksummed at the application layer (the
audit-chain for `/data/audit/`, the typed `Config` for
`/data/mgmt/`).

### Capabilities

#### New Capabilities
- `fs-block-device`: `BlockDevice` trait, per-arch
  implementations, error encoding.
- `fs-partition-table`: GPT parser, partition discovery, v1
  layout enforcement.
- `fs-squashfs-readonly`: squashfs 4.0 reader, zstd
  decompression, manifest verification, `/models/` mount.
- `fs-f2fs-readwrite`: F2FS read + write, journal, checkpoint,
  atomic rename, `fsync` semantics, `/data/` mount.
- `fs-ab-boot`: A/B boot config region, atomic slot pointer
  flip, watchdog rollback on failed boot.
- `fs-delta-update`: block-level delta application against
  active squashfs to produce inactive squashfs, manifest
  verification, hand-off to `remote-update-v1`.
- `fs-integrity`: per-read hash verification, application-layer
  checksums on `/data/`.

#### Modified Capabilities
- `posix-vfs`: extends the existing in-memory VFS with two
  on-disk mounts (`/models/` squashfs, `/data/` F2FS); adds
  write paths returning real I/O errors instead of `EROFS`.
- `kernel-syscalls`: existing file syscalls (open, read, write,
  fsync, rename, stat) gain real backing on the two mounts.
- `mgmt-config-layout` (from `management-login-v1`): per-file
  permission declarations get a real filesystem to enforce
  against; mode-stricter-than-declared rule operates on F2FS
  inode permissions.

## Impact

- **Code:**
  - New crate `fs/` (Layer 1) — block device, GPT, squashfs,
    F2FS, A/B boot, delta apply.
  - `fs/src/squashfs.rs` — clean-room squashfs 4.0 reader.
    Estimate ~2k LOC.
  - `fs/src/f2fs.rs` — clean-room F2FS implementation.
    Estimate ~6k LOC for read; ~10k more for write +
    checkpoint. Largest piece in the change.
  - `arch/x86_64/src/ahci.rs` (or equivalent NVMe) — block
    device for x86 bare metal.
  - `arch/aarch64/src/sdhci.rs` — eMMC block device for Jetson.
  - `arch/{x86_64,aarch64}/src/virtio_blk.rs` — virtio-blk for
    QEMU.
  - `posix/src/vfs.rs` — extend with on-disk mount points.
  - `kernel/src/boot.rs` — mount `/models/` and `/data/` after
    `auth_login` is available.
- **Tests:** ~250 new tests targeted: squashfs round-trip
  against `mksquashfs`-produced images, F2FS round-trip against
  `mkfs.f2fs` images, GPT parser fuzz, A/B boot-pointer atomic
  flip under simulated crash, delta apply against known-good
  patches, manifest verification (correct + corrupted),
  per-arch block-device read/write KAT, mount-time integrity
  fail-closed. Aim to keep `≥4500` (already targeted in
  `management-login-v1`) growing to `≥4750`.
- **Boot footprint:** F2FS is the dominant new code — likely
  +100-150 KB compiled (no_std, `opt-level = "z"`, LTO). The
  base SmallAIOS binary will exceed the prior <8 MB target by
  ~2%; the trade is acceptable for the resilience and
  interop guarantees.
- **External interop:** A laptop running stock Ubuntu/Fedora
  with no extra packages MUST be able to: (a) mount the
  squashfs image read-only via `mount -t squashfs -o loop`,
  (b) mount the F2FS partition read-write via `mount -t f2fs`.
  This SHALL be CI-tested on every PR.
- **Downstream:** Unblocks the `management-login-v1` write
  paths (`/data/auth/shadow`, audit log, mgmt config) and the
  `remote-update-v1` A/B swap mechanism. Establishes the
  filesystem substrate every persistent feature inherits.
- **Dependencies:** No new external Rust crates in the
  production dep graph (clean-room rule). zstd is already
  present. `mksquashfs` and `mkfs.f2fs` are dev-only host
  tooling for producing test images.
- **Risks:**
  1. F2FS write path is the largest clean-room implementation
     in the project to date. Phased delivery with read path
     first lets us land squashfs end-to-end before the F2FS
     write story is final.
  2. Block-device drivers on bare-metal x86 (AHCI/NVMe) may
     uncover gaps in PCIe enumeration. Mitigate by landing
     virtio-blk first (works in QEMU CI) before bare-metal
     hardware bringup.
  3. Boot-pointer flip atomicity under power loss is the
     central correctness invariant for `remote-update-v1`.
     Cover with a Kani harness that models partial writes
     across the boot-config region.

## Out of scope for v1 (flagged)

- OverlayFS-style writable layer on `/models/` for
  per-deployment model injection. Captured as
  `embedded-overlay-v1` follow-on.
- ext4, xfs, btrfs as alternatives to F2FS. Re-evaluate only
  if a hard target requirement appears.
- LittleFS, JFFS2, UBIFS for raw-flash microcontroller
  targets. Captured as `embedded-flash-fs-v1` follow-on for
  future MCU/FPGA targets.
- Multi-disk RAID, encrypted-at-rest filesystems, fscrypt,
  network filesystems (NFS, 9P, virtio-fs).
- Quotas, ACLs, extended attributes beyond what F2FS metadata
  natively provides.
- Trim / discard support beyond what's needed for correctness
  (performance-tuning is a v2 concern).

## Open Questions

The following are filled in by the design walkthrough:

1. Block device layer — single `BlockDevice` trait, or per-bus
   trait hierarchy (NVMe vs AHCI vs SDHCI vs virtio-blk)?
2. Partition table — GPT only, or both MBR (legacy x86) and GPT?
3. Squashfs compression — zstd only (already in tree), or
   zstd + xz (broader external compatibility)?
4. F2FS implementation phasing — read path first, then write,
   then `fsync`, or vertical-slice (one config file
   end-to-end, then expand)?
5. Boot-pointer storage — reserved LBA, dedicated tiny
   partition, or UEFI variable?
6. Boot rollback trigger — watchdog timeout, explicit
   "boot success" syscall, or both?
7. Delta format — zstd `--patch-from`, bsdiff, both?
8. Manifest format — separate `.sig` file or inline at end of
   squashfs blob?
9. Where do mount points live in `posix-vfs`'s tree — fixed,
   or driven by a config file?
10. Block cache — single shared LRU, per-mount, or none in v1?
