## Context

`embedded-filesystem-v1` standardizes on F2FS for the writable
`/data/` partition. F2FS assumes a block device that masks the
raw-flash quirks F2FS does not handle (wear-leveling, bad-block
management, ECC). Every current SmallAIOS target (x86-64,
AArch64-Jetson, RISC-V, virtio-blk QEMU) provides such a block
device — eMMC, NVMe, SATA, or virtio-blk all have a controller in
between the OS and the raw NAND/NOR.

But the roadmap explicitly calls out **MCU-class and FPGA-class
targets** where flash sits **directly on a memory-mapped bus**
(QSPI NOR, ONFI NAND) with no firmware controller. The OS owns
wear-leveling, bad-block remapping, and power-fail safety. F2FS
will not run there.

This change introduces the raw-flash filesystem layer SmallAIOS
needs for those targets. It is **additive** to
`embedded-filesystem-v1`, not a replacement. The shape converged
across 15 design-walkthrough questions.

## Goals / Non-Goals

**Goals:**
- New `fs/src/flash/` sub-module inside the existing `fs/` Layer 1
  crate (no new crate; workspace stays at 24).
- New `FlashDevice` trait with read / program / erase /
  bad-block semantics matching NOR and NAND hardware exactly.
- Clean-room Rust `#![no_std]` littlefs v2.x reader and writer;
  power-fail-safe by design, ~5 kLOC.
- In-memory `MockFlashDevice` for CI tests with bit-flip and
  bad-block-on-erase injection (behind a separate
  `fs-flash-mock` cargo feature, off in production).
- Per-arch QSPI NOR + ONFI NAND drivers as stubs in v1; real
  bringup lands when the first MCU/FPGA target arrives.
- `/flash/` mount point in the VFS, distinct from `/data/`,
  coexists with F2FS-`/data/` when both substrates are present.
- Behind `fs-flash` cargo feature (default off).
- ≥5600 total tests after change (~+350 new).
- Formal: Kani (metadata-pair commit atomicity), TLA+ (wear-
  leveling progress), Coq (BBT redundancy).

**Non-Goals (v1):**
- JFFS2, UBIFS, SPIFFS, F3 — alternative formats. Considered
  and rejected.
- TLC/QLC NAND with read-disturb mitigation (v2).
- Power-loss-recovery scrubbing on mount beyond what littlefs's
  natural commit semantics provides (v2).
- Encrypted flash partitions. Captured as
  `embedded-flash-encrypt-v1` for regulated targets.

## Decisions

### Q1. Format choice

**Decision:** littlefs v2.x. Power-fail-safe by design, ~5 kLOC
clean-room scope (vs ~10 kLOC JFFS2, ~20 kLOC UBIFS+UBI), well-
spec'd at <https://github.com/littlefs-project/littlefs/blob/master/SPEC.md>,
existing `littlefs2` Rust crate usable as a dev-dependency
oracle for tests. Interop weaker than F2FS — recovery laptop
needs the `littlefs-fuse` userspace tool. Trade accepted because
the audience is MCU/FPGA developers with flash tooling, not
generic-recovery-laptop operators.

JFFS2 / UBIFS / SPIFFS rejected: too large to clean-room (JFFS2
~10 kLOC + slow journal scan; UBIFS ~20 kLOC over UBI), or not
power-fail-safe by spec (SPIFFS).

### Q2. Mount point

**Decision:** `/flash/` — distinct from `/data/`. Application
code that targets MCU + larger boards uses `/flash/` for raw-
flash content (boot keys, attestation state, secure config) and
`/data/` for bulk writable state when available, with a clear
substrate signal in the path.

### Q3. Coexistence with F2FS `/data/`

**Decision:** Both available, distinct purposes. `/data/` holds
bulk writable state. `/flash/` is reserved for small high-
assurance content. `/data/auth/shadow` stays on F2FS;
`/flash/secrets/` is a separate location for keys. No mirroring;
single source of truth per file.

### Q4. NAND erase-block size default

**Decision:** 128 KiB. Matches typical small-page MLC NAND. Pages
of 4 KiB read/programmed; 32 pages erased at a time. Conservative
and works on most parts. Per-target ONFI parameter-page probing
(option B) is documented as an evolution path if a NAND device
reports something different.

### Q5. Bad Block Table location

**Decision:** Duplicated at start AND end of flash. Standard
JFFS2/UBIFS-style redundancy. Costs one erase-block per copy.
Either survives loss of the other; manufacturer factory BBT (if
present) is consulted as a third source on first format.

### Q6. fadvise hint set

**Decision:** Minimal — `SEQUENTIAL` and `RANDOM` only. POSIX-
standard, matches Linux defaults. SEQUENTIAL hint enables
write-batching for log-style workloads (audit log appends).
Richer hints (option B) deferred to a future addition if real
applications need them.

### Q7. Test fixtures

**Decision:** Pure-Rust generator using the `littlefs2` crate as
a dev-dependency. Faster CI than invoking the C `mklittlefs`
tool, and the Rust port's spec-conformance is itself tested
elsewhere. Loses the "we agree with the C reference" signal —
mitigation: a smaller subset of fixtures from the C reference
runs in a weekly cron job (not per-PR).

### Q8. Mock device gating

**Decision:** Behind separate `fs-flash-mock` cargo feature, off
in production. Mock is for CI tests and developer
experimentation only. Test profiles enable; release profiles do
not.

### Q9. Wear-leveling parameter exposure

**Decision:** Accept littlefs defaults: `lookahead_size=8192`,
`cache_size=block_size`, `block_cycles=500`. No operator-facing
knob in v1. Per-medium presets (option C) are a documented
evolution path.

### Q10. Format version migration

**Decision:** Refuse to mount on major-version mismatch with a
clear `reflash required` message. Operator runs the
manufacturer-provided reflash tool to install a fresh image.
Matches the `verified-boot` "fail closed on mismatch" philosophy.

### Q11. Boot-ordering / feature flag

**Decision:** Behind `fs-flash` cargo feature (default off).
Targets that need raw-flash enable the feature; everyone else
pays zero overhead. Mirrors `embedded-filesystem-v1`'s
`fs-on-disk-mounts` and `embedded-overlay-v1`'s
`fs-overlay-mounts` patterns.

### Q12. Phase ordering

**Decision:** 8 phases bottom-up:
1. `FlashDevice` trait + `FlashError` enum.
2. `MockFlashDevice` with bit-flip and bad-block injection.
3. littlefs v2.x read path.
4. littlefs v2.x write path.
5. `fsync` + metadata-pair commit semantics.
6. Wear-leveling integration + Bad Block Table.
7. `/flash/` mount point in the VFS.
8. Per-arch QSPI/ONFI driver stubs (real hardware bringup
   lands later when the first MCU/FPGA target arrives).

### Q13. Test target

**Decision:** ≥5600 total tests after change (~+350 new). Cover:
littlefs round-trip vs `littlefs2` Rust port, 1M-cycle wear
stress on mock, exhaustive bit-flip injection, BBT redundancy,
power-fail (kill-9 mid-write → mount → no data loss past last
sync), atomic-rename, fadvise behavior, multi-device coexistence
(F2FS `/data/` + littlefs `/flash/` both mounted).

### Q14. Formal verification

**Decision:**
- Kani — bounded model check on the littlefs metadata-pair
  commit invariant: under arbitrary partial writes during a
  pair commit, the next mount finds at least one valid pair
  with the correct generation.
- TLA+ — model the wear-leveling allocator's progress invariant:
  every erase-block is eventually erased before any specific
  block exceeds its cycle budget.
- Coq — proof that BBT redundancy (start + end copies) survives
  single-block-erase loss in either copy.

### Q15. PR strategy

**Decision:** Two PRs:
- **PR 1 (this PR):** scaffolding only — proposal, design,
  specs, tasks. Merges to `develop` so agent teams can spawn
  worktrees and pick up the design.
- **PR 2 (later):** all 8 implementation phases in one cohesive
  merge, behind the `fs-flash` cargo feature default-off.

## Risks / Trade-offs

- **[Risk] Interop story weaker than F2FS** — Mitigation:
  documented in operator runbook; `littlefs-fuse` is one
  package install away; target audience already has flash
  tooling for the MCU/FPGA hardware. Not a generic-laptop
  scenario.
- **[Risk] No real platform binding in v1** — Mitigation:
  per-arch QSPI/ONFI drivers ship as documented stubs; FS layer
  is fully testable on the mock device. First real target
  bringup lands the platform code without changing the format
  or trait contracts.
- **[Risk] Wear-leveling correctness over millions of cycles**
  — Mitigation: 1M-cycle property-based stress on the mock,
  TLA+ progress proof, littlefs's well-spec'd algorithm has
  been validated in production by ARMmbed and many vendors.
- **[Risk] Pure-Rust fixture generator may diverge from C
  reference over time** — Mitigation: weekly cron job runs the
  C-reference round-trip to catch drift early.
- **[Risk] BBT location duplication wastes 256 KiB on a 4 MiB
  flash** — Mitigation: 256 KiB is 6% of a 4 MiB flash; the
  redundancy is worth the cost. Below-1-MiB partitions can
  declare a single-BBT-only mode via a future config field if
  needed.

## Migration Plan

This change is purely additive. No existing data on any current
SmallAIOS deployment lives on raw flash; no migration is needed.
Future targets that adopt this filesystem will format from
scratch on first boot per the existing physical-presence policy.

The `mgmt-config-layout` adds an optional `/flash/` subtree
declaration that activates only when the `fs-flash` feature is
on AND a flash device is enumerated.

## Open Questions

All fifteen design walkthrough questions are resolved (Q1–Q15
above).

Items deferred with explicit decision:
- **Encrypted flash** — `embedded-flash-encrypt-v1` follow-on
  if a regulated target needs it.
- **TLC/QLC read-disturb mitigation** — v2.
- **Per-medium wear-leveling presets** — Q9 noted this as a
  documented evolution path.
- **Per-target ONFI parameter-page probe** — Q4 noted as
  evolution path.
- **fsck-on-mount scrubbing** — v2.
