# Design — ecc-scrubbing-v1

## Goal

A background ECC memory-scrubbing service that walks every kernel-managed DRAM region at a configurable interval, surfaces correctable / uncorrectable error counts per region, and integrates with the watchdog to detect scheduler hangs. Two backends: Tegra234 EMC hardware scrub (preferred when available) and software read-modify-write scrub (universal fallback). One async task per region, all running in the cooperative scheduler.

## Design decisions

### Decision 1: Per-region scrub state, not global

Different memory regions have different scrub priorities:

| Region | Scrub interval | Why |
|--------|---------------|-----|
| ONNX model weights | 24 h | Static once loaded, but a flipped weight bit persists indefinitely until corrected |
| Heap | 6 h | Transient — most allocations don't live long enough to accumulate error, but long-lived ones (graph builders, capability tables) do |
| DTB region | 7 d | Read-once at boot, kept around for debug |
| Kernel `.bss` / `.data` | 24 h | Mostly static after boot |
| GPU-shared (post `tegra-smmu-isolation-v1`) | 12 h | Live workload, but the GPU's own ECC may already cover |

Per-region state means we can apply per-region intervals. A single global scrub would either be too slow (24 h baseline on all regions = ONNX activations get under-protected) or wasteful (1 h on all regions = DTB scrubbed pointlessly).

### Decision 2: Cooperative async task, not interrupt-driven

The scrub task is a normal async task in the SmallAIOS cooperative scheduler. It walks pages one chunk (default 64 KiB) at a time and yields between chunks. This naturally interleaves with inference work — the scheduler doesn't preempt, so a long ONNX op completes before scrub resumes.

Alternative — driving scrub from a timer interrupt — was considered. Rejected because (a) interrupt handlers in `#![no_std]` should be short, and even a 64 KiB read-modify-write loop is too long; (b) cooperative async is the existing scheduling discipline and adding interrupt-driven exceptions to it adds complexity for marginal benefit; (c) the EMC hardware scrub *is* interrupt-driven internally — we just program its registers and poll completion in our task.

### Decision 3: TOML config at boot, programmatic API for runtime adjustment

Static config (`/etc/smallaios-scrub.toml` for container builds, embedded in the kernel image for unikernel builds) defines the default regions and intervals. A programmatic API (`scrub::add_region(...)` and `scrub::set_interval(...)`) lets the kernel re-tune at runtime — e.g., when a new ONNX model loads, register its weight region with a 24 h interval.

This mirrors how the existing `peripheral` config layer works.

### Decision 4: Backend abstraction — `ScrubBackend` trait

```rust
trait ScrubBackend {
    fn name(&self) -> &'static str;
    async fn scrub_region(&mut self, region: &Region) -> ScrubResult;
    fn supports_async_completion(&self) -> bool;
    fn error_counters(&self) -> (u64, u64); // (correctable, uncorrectable) since boot
}
```

Two implementations:

- `TegraEmcBackend` — programs the Tegra234 EMC patrol-scrub registers, polls completion, reads error counters from `EMC_ECC_STATUS`.
- `SoftwareBackend` — `usize`-stride read+write loop. Yields after every chunk. Doesn't actually report correctable-error counts (those come from the DRAM controller in any case; software backend just *triggers* corrections by reading); we wire a separate `ras_counters` API for that on platforms that expose them.

Backend is selected at runtime (`ScrubBackend::probe()` returns the best available) so the same binary runs on Orin NX (EMC path) and an x86 dev box (software path).

### Decision 5: Watchdog correlation — scrub-advance feeds watchdog

The watchdog "wants" to be fed periodically. Tying the feed to scrub-advance has a beneficial property: if the scheduler is starved, scrub stops advancing, and the watchdog fires. If the EMC stops responding, scrub stops advancing, and the watchdog fires. If the scrub task itself panics, scrub stops advancing, and the watchdog fires.

This is *too aggressive* without nuance — between scrub cycles the task is idle, and we don't want the watchdog firing during legitimate idle. Solution: the scrub task feeds the watchdog every loop iteration regardless of whether it advanced the cursor, but it tracks a separate "advanced recently" flag that toggles. The watchdog policy is configurable:

- **Aggressive (DAL A default)**: cursor must advance every `watchdog_threshold` seconds, else reset.
- **Permissive (development default)**: any scrub task heartbeat (even idle) feeds the watchdog.

### Decision 6: Boot-time hardware probe + demand-mode wipe

At boot, after the EMC is configured, run one full demand-mode scrub of the kernel image + heap region. This serves three purposes:

1. **Verify EMC responds** — if `EMC_ECC_STATUS` reads back garbage, fall back to software scrub with a warning.
2. **Establish a known-good baseline** — initial scrub corrects any error that accumulated in DRAM between power-on and kernel init.
3. **Surface UEFI-residual errors** — UEFI may have left correctable errors in its own pages; scrubbing surfaces them in the boot log.

Demand mode blocks until complete (no async yield) so we get a clean baseline. After boot, the patrol-mode service takes over.

## Alternatives considered

### Alt A: Rely on the DRAM controller's autonomous patrol scrub (no kernel involvement)

**Rejected.** Most DRAM controllers (Tegra234's EMC included) can run patrol scrub autonomously without kernel intervention, but:

- Error counters are still kernel-readable only via MMIO; without kernel polling we don't *see* the errors.
- Configuration (interval, region selection) is firmware-set on most platforms — we lose the per-region tuning.
- Watchdog correlation requires kernel-side visibility into scrub progress.

Autonomous patrol is the lowest-friction path but it gives up the certification evidence we need.

### Alt B: Memory mirroring (RAID-1 of DRAM regions) instead of scrubbing

**Rejected for now.** Some server platforms support memory mirroring at the BIOS level (write to two physical regions; reads consistency-check). It would catch double-bit errors as well. Cost: 2× memory, complex firmware setup, not available on Jetson Orin or any embedded part we target. Tracked for future server-platform-only consideration.

### Alt C: Stop the world during scrub

**Rejected.** Synchronous scrub of a 16 GB region takes seconds at memory-bus bandwidth. Pausing inference for seconds every interval is unacceptable. The cooperative async approach interleaves naturally.

### Alt D: Use Linux's existing scrub user-space tools and call them via FFI

**Rejected.** SmallAIOS is `#![no_std]` and the unikernel has no path to Linux user-space. The container build could in principle pipe out to a user-space scrub tool, but that's the wrong layering — scrub is a kernel concern.

## Risks

### Risk 1: ECC support varies by deployment

Orin NX (P3767-0000) ships with LPDDR5 ECC enabled in NVIDIA's standard config; verify on the specific board before relying. x86 server platforms vary widely. RISC-V dev boards rarely have ECC at all. Mitigation: `scrub::probe()` detects whether ECC is configured at boot; on non-ECC platforms it logs `[ecc-scrub] DRAM does not advertise ECC; service disabled` and the scrub task does not start. Software fallback is still available as a defense-in-depth correctness check, but its value is much lower without ECC.

### Risk 2: Scrub interval misconfiguration starves inference

Too-aggressive intervals on too-large regions could consume measurable memory bandwidth. Mitigation: (a) `scrub::stats` surfaces bandwidth usage; (b) the default intervals (6-24 h depending on region) are conservative; (c) the cooperative yield ensures inference progress regardless of scrub rate; (d) Phase 5 docs include a calibration table for different operational environments.

### Risk 3: Tegra234 EMC register documentation gaps

NVIDIA's published Tegra234 TRM covers `EMC_ECC_SCRUB_*` registers but some Linux-driver lore (the `tegra-mc` driver in L4T's NVIDIA fork) suggests there are undocumented fields that the firmware uses. Mitigation: (a) verify register semantics against L4T fork source code; (b) test against real Orin NX hardware with deliberate bit-error injection (some boards expose DBI registers); (c) if a register is too undocumented, fall back to software scrub on Tegra234 with a warning rather than guessing.

### Risk 4: Soak-test workload availability

Verifying the service requires running for days under realistic workload. Mitigation: post-merge, schedule a 7-day soak on a dedicated Orin NX with a fixed LLM workload; track the correctable-error counter as the baseline. Treat this as a "release readiness" gate for the avionics-cert track.

### Risk 5: Watchdog tuning is environment-specific

Aggressive-mode watchdog reset on scrub stall is right for DAL A; wrong for development where a debugger pause is legitimate. Mitigation: clearly-named Cargo features (`scrub-watchdog-aggressive`, `scrub-watchdog-permissive`) with explicit docs about which is appropriate when.

## Build/CI surface

- New Cargo feature `ecc-scrub` on `smallaios-kernel`, off by default. Default-on for `tegra234`-feature builds once Phase 2 lands hardware verification.
- New module `kernel/src/mem/scrub/{mod.rs, task.rs, config.rs, stats.rs, sw_backend.rs}`.
- New module `arch/aarch64/src/scrub/tegra_emc.rs` (when `tegra234` + `ecc-scrub` are both on).
- New `just ecc-scrub-test` recipe that boots a kernel with the service on, runs a 30-minute software-scrub cycle on a 1 MB test region, asserts no errors and correct cycle-count increment.
- New CI advisory job `ecc-scrub-smoke` (TCG-emulated, software backend only — hardware backend requires self-hosted Orin runner).
- New `docs/ecc-scrubbing.md`.

## What this change explicitly does NOT do

- Does not modify any non-mem code paths (syscalls, ONNX, networking, capability system).
- Does not change syscall ABI — scrub stats are surfaced via the telemetry path, not via a new syscall.
- Does not add a new dependency — all new code is `#![no_std]` core+alloc.
- Does not require any change to the existing DRAM init in `mem::init` — scrub runs *after* mem-init completes.
