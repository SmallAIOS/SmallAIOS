## Why

`embedded-filesystem-v1` standardizes on F2FS for the writable
`/data/` partition. F2FS is the right choice for SmallAIOS's
present targets — x86-64, AArch64 (Jetson Orin), RISC-V — which
all run on **block-device-backed flash** (eMMC, NVMe, SATA SSD,
virtio-blk). The block device abstracts away the raw flash
characteristics: the underlying NAND/NOR controller handles
wear-leveling, bad-block management, and ECC; F2FS just sees a
linear array of 4 KiB blocks.

But SmallAIOS's roadmap explicitly calls out **MCU-class and
FPGA-class targets** where there is no block device. The flash
sits **directly on a memory-mapped bus** (NOR via QSPI, NAND via
ONFI) with no firmware controller in between. The OS itself is
responsible for wear-leveling, bad-block remapping, and
power-fail safety. F2FS will not run on that hardware: it
assumes a block device that masks the raw-flash quirks F2FS does
not handle.

This change introduces a dedicated **raw-flash filesystem** for
those targets. It is **additive** to `embedded-filesystem-v1`,
not a replacement: most SmallAIOS deployments continue to use
F2FS on a block device. Targets that need raw-flash get the new
FS; targets with a block device keep F2FS. The kernel selects
which to use at boot based on what the architecture's
`BlockDevice` discovery returns.

The implementation is **clean-room Rust, `#![no_std]`** —
matching the rest of the project's fs stack. Format choice:
**littlefs (v2.x)** — pure-Rust port of the well-spec'd ARMmbed
format. Reasons: power-fail-safe by design (every operation is a
single block-erase boundary), minimal RAM footprint (~4 KiB
working set), small code size (~5 kLOC vs F2FS ~30 kLOC),
existing pure-Rust crate (`littlefs2`) as a learning aid for the
clean-room write, and Linux interop via the `littlefs-fuse` tool
(works, isn't preinstalled — interop trade-off documented).

JFFS2, UBIFS, and SPIFFS were considered:

- **JFFS2** — log-structured, mainline Linux kernel filesystem
  for raw NOR flash. Excellent maturity. But ~10 kLOC C
  implementation, journal scan at every mount (slow on large
  partitions), and no first-class power-fail proof we can point
  at. Dropped.
- **UBIFS** — JFFS2's successor for NAND. Sits on top of UBI
  (Unsorted Block Images) which itself is ~10 kLOC. Total scope
  is comparable to F2FS; over-engineered for our 1–256 MiB
  flash partitions. Dropped.
- **SPIFFS** — small, simple, designed for SPI NOR. Smaller
  code than littlefs. **Not power-fail-safe by spec** — explicit
  caveat in the docs. Dropped.

littlefs's biggest weakness is interop: an external recovery
laptop needs the `littlefs-fuse` userspace tool installed to
read the partition. F2FS mounts on every Linux distro out of
the box; littlefs does not. We accept this trade because the
target audience is MCU/FPGA developers who already have flash
tooling, not "operator with a generic recovery laptop."

## What Changes

### New `fs-flash` Layer 1 sub-module inside `fs/`

Rather than a new crate, the raw-flash code lives under
`fs/src/flash/` as a sibling of `fs/src/f2fs/`, `fs/src/squashfs/`,
and `fs/src/overlay/`. Same `BlockDevice`-style abstraction at
the bottom; different format on top. Workspace count stays at
24 (no new crate).

### Flash device abstraction (`FlashDevice` trait)

Raw-flash hardware does not match the `BlockDevice` trait — it
needs explicit erase-block accounting and bad-block reporting.
A new trait sits next to `BlockDevice`:

```rust
pub trait FlashDevice {
    fn read(&self, offset: u64, buf: &mut [u8])
        -> Result<(), FlashError>;
    fn program(&mut self, offset: u64, buf: &[u8])
        -> Result<(), FlashError>;
    fn erase(&mut self, block: u64) -> Result<(), FlashError>;
    fn block_size_bytes(&self) -> u32;       // erase-block size
    fn page_size_bytes(&self) -> u32;        // program-unit size
    fn block_count(&self) -> u64;
    fn is_bad(&self, block: u64) -> bool;
    fn mark_bad(&mut self, block: u64) -> Result<(), FlashError>;
}
```

`program` writes to a previously-erased page; bits can only
flip 1→0 within a page until the containing block is erased.
This matches NOR/NAND semantics exactly.

### Per-target flash drivers

v1 ships flash drivers for the targets currently on the
roadmap that need raw-flash:

- **QSPI NOR** (typical FPGA boot flash, MCU code flash)
  — `fs/src/flash/qspi.rs` + per-arch QSPI controller bindings.
- **ONFI NAND** (typical MCU data flash) — `fs/src/flash/onfi.rs`
  + per-arch NAND controller bindings.
- **In-memory mock** — for CI tests, simulates erase/program
  cycles, can be configured to inject bit flips and
  bad-block-on-erase events.

The proposal explicitly does NOT pre-bind these to a specific
physical platform. Architecture-side bringup happens when the
first real MCU/FPGA target lands; this change defines the API
the platform code targets.

### littlefs v2.x format

The on-disk format SHALL be littlefs v2.x with default block
size 4 KiB on QSPI NOR, 128 KiB on ONFI NAND (matching typical
erase-block sizes for each medium). The driver SHALL refuse to
mount images of any other major version. The format is
documented at <https://github.com/littlefs-project/littlefs/blob/master/SPEC.md>;
this change pins to a specific commit hash of that spec.

### Wear-leveling and bad-block management

The driver SHALL implement littlefs's wear-leveling
out-of-the-box (the format's built-in dynamic wear-leveling
within a block-allocator pool). On NAND, the driver SHALL also
maintain a Bad Block Table (BBT) in a dedicated reserved area;
blocks marked bad SHALL be skipped by the allocator and SHALL
not be reused.

### Mount points and target-specific layouts

For MCU/FPGA targets, the kernel SHALL mount the littlefs
partition at `/flash/` (distinct from `/data/` so application
code that targets MCU + larger boards can have a single
`/flash/` path that always works). On block-device targets where
F2FS provides `/data/`, the kernel MAY also mount a littlefs
partition at `/flash/` if one exists (e.g., a small QSPI NOR for
secrets/secure-config alongside an eMMC for bulk data). This
"both available" mode is allowed but optional.

### Format-on-first-boot

Same convention as `embedded-filesystem-v1`: format an
unrecognized partition only when a `PhysicalPresenceProvider`
indicator asserts. No silent overwrite of arbitrary flash
contents.

### `posix-vfs` extension

The VFS SHALL gain a `/flash/` mount point. POSIX semantics
(open / read / write / fsync / unlink / rename / mkdir / rmdir)
SHALL apply, with one caveat: writes to large files may incur
multi-block-erase latency. The driver SHALL expose `fadvise`
hints so applications can opt into write-batching (e.g., the
audit log can append a batch then fsync once per second).

### Capabilities

#### New Capabilities
- `fs-flash-device`: `FlashDevice` trait, error encoding,
  bad-block table semantics, retry policy on program/erase
  failures.
- `fs-flash-littlefs`: littlefs v2.x on-disk format reader and
  writer, wear-leveling, atomic-rename via the format's metadata
  pair commit, fsync semantics.
- `fs-flash-mount`: `/flash/` mount point in the VFS, format-on-
  physical-presence, both-available coexistence with `/data/`.

#### Modified Capabilities
- `posix-vfs`: adds the `/flash/` mount point.
- `mgmt-config-layout`: adds optional `flash/` directory tree
  declaration for targets where `/flash/` is the writable
  surface.

## Impact

- **Code:**
  - `fs/src/flash/mod.rs` — module root, gated by `fs-flash`
    cargo feature.
  - `fs/src/flash/device.rs` — `FlashDevice` trait + `FlashError`
    enum.
  - `fs/src/flash/qspi.rs` — QSPI NOR driver.
  - `fs/src/flash/onfi.rs` — ONFI NAND driver + BBT.
  - `fs/src/flash/mock.rs` — in-memory simulator for CI.
  - `fs/src/flash/littlefs.rs` — clean-room littlefs v2.x reader
    and writer (~5 kLOC).
  - `fs/src/flash/mount.rs` — `/flash/` mount integration.
  - `arch/{aarch64,riscv64}/src/flash/` — per-arch QSPI/ONFI
    controller bindings (stubs in v1; real targets land later).
- **Tests:** ~200 new tests targeted: littlefs format
  round-trip against the C reference implementation's output
  (read `mklittlefs`-produced images byte-for-byte), wear-
  leveling under simulated 100k erase cycles, bad-block
  remapping, power-fail safety (kill-9 mid-write, verify mount
  succeeds with no data loss past last sync), atomic-rename,
  `fadvise` write-batching. Aim to grow `≥5250` (post-overlay)
  to `≥5450`.
- **Boot footprint:** `fs-flash` cargo feature off by default.
  Targets that don't need raw-flash pay zero overhead. Targets
  that do pay ~50 KB compiled.
- **External interop:** Reading a SmallAIOS littlefs partition
  on a recovery laptop requires the `littlefs-fuse` tool from
  GitHub (one apt/brew/cargo-install away). This is a
  deliberate trade-off, documented in the runbook.
- **Downstream:** Unblocks the future MCU and FPGA targets in
  the roadmap. Independent from `embedded-overlay-v1` (which
  remains F2FS-specific via `/data/models-upper/`).
- **Dependencies:** No new external production dependencies.
  `littlefs2` crate may appear in `dev-dependencies` only as a
  validation oracle (mirroring how `argon2` is used as a KAT
  reference in `management-login-v1`).
- **Risks:**
  1. littlefs's interop story is weaker than F2FS — operators
     used to a generic recovery laptop will need new tooling.
     Captured in the runbook.
  2. v1 ships drivers and format with no real platform binding.
     QSPI / ONFI controller code is per-arch and lands when the
     first MCU/FPGA target arrives. The FS layer is testable
     standalone via the mock device.
  3. Wear-leveling correctness over millions of cycles is hard
     to test exhaustively. Mitigation: property-based tests
     simulate up to 1M cycles in CI on the mock device.

## Out of scope for v1 (flagged)

- JFFS2, UBIFS, SPIFFS, F3 — alternative flash filesystems.
  Captured in design as considered-and-rejected.
- TLC/QLC NAND with read-disturb mitigation. v1 targets SLC
  and MLC; advanced read-disturb is a v2 concern.
- Power-loss-recovery scrubbing on mount (we let littlefs's
  natural commit semantics handle it; an explicit fsck-on-mount
  is v2).
- Encrypted flash partitions. Captured as
  `embedded-flash-encrypt-v1` if a regulated target needs it.

## Open Questions

1. Format choice — littlefs v2.x is the proposed default. Open
   if any reader disagrees and prefers JFFS2 / UBIFS.
2. Mount point — `/flash/` or `/persist/` or per-target
   override?
3. NAND erase-block size default — 128 KiB matches typical
   small-page MLC; larger NAND (256 KiB) would need a config
   flag.
4. Bad Block Table location — at the start of the flash, end,
   or duplicated for redundancy?
5. Coexistence rule when both F2FS `/data/` and littlefs
   `/flash/` are present — single source of truth for `/data/auth/`,
   or duplicate?
6. fadvise hint set — minimal (`SEQUENTIAL`/`RANDOM`) or richer
   (`WILLNEED`, `DONTNEED`, `BATCH_WRITE`)?
7. Test fixtures — synthetic littlefs images via the C
   reference, or via a pure-Rust generator?
8. Should we ship the in-memory mock as part of the production
   binary (always-on for `fs-flash` feature) or behind a
   separate `fs-flash-mock` feature?
9. Wear-leveling algorithm tuning — accept littlefs defaults
   or expose `fs.flash.lookahead_size` etc. via
   `mgmt/policy.toml`?
10. Migration story from a future rev where the format pins to
    a different littlefs spec commit — refuse to mount and
    require reflash, or implement read-only legacy fallback?
