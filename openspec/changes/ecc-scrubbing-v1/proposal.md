# ecc-scrubbing-v1

## Summary

Single-event upsets (SEUs) — DRAM bit flips caused by cosmic rays, alpha particles, or thermal noise — are a measurable failure mode in any system that runs continuously for weeks at altitude or in radiation-exposed environments (avionics > 30,000 ft, satellites, ground stations near reactor sites, high-altitude balloon platforms). Single-bit errors are detected *and* corrected by ECC DRAM at the controller level; double-bit errors are detected but uncorrectable. The mitigation against the second class is **periodic memory scrubbing**: walk every DRAM region in the background, force a read-modify-write at ECC-granularity, so single-bit errors are silently corrected before they accumulate into double-bit errors.

This change adds an ECC scrubbing service to the SmallAIOS kernel under a new `ecc-scrub` Cargo feature on `smallaios-kernel`, gated to platforms with ECC DRAM controllers (Jetson Orin AGX Industrial, modern x86 server platforms, certain RISC-V dev boards with LPDDR4-ECC). The service runs as a low-priority background task in the cooperative scheduler, scrubs a configurable region at a configurable interval, integrates with the existing watchdog, and surfaces per-region scrub-cycle counters + correctable-error counts via telemetry. New capability: `kernel-mem` extended with scrub-related Requirements.

## Why

- **DO-178C DAL A and DO-254 design-assurance both expect single-event-effect (SEE) mitigation in any system fielded above 30,000 ft.** Avionics environments accumulate ~1 SEU per gigabit per 1000 hours at cruise altitude (industry rule-of-thumb, varies by latitude and solar activity). A 16 GB Jetson Orin NX deployed in a UAV computing pod for a 1000-hour mission expects 16 single-bit corrections statistically. If we don't scrub, single-bit errors accumulate at the bit positions they occur — and when a second flip hits the same ECC word, the error becomes uncorrectable. RTCA DO-160G section 22 (lightning-induced upset) and section 25 (electrostatic discharge) explicitly call out memory-system error-rate budgets. The certification answer for "what is your scrub interval" cannot be "we don't scrub".
- **SmallAIOS targets long-running inference workloads where the math just works against us.** Continuous LLM inference on Jetson Orin in an avionics pod is exactly the workload where scrubbing matters most: every weight tensor sits in DRAM for weeks, every activation walks through DRAM hundreds of times per second, and any single-bit flip in a weight is permanent until corrected. ML workloads are also (helpfully) resilient to single-bit weight perturbations — the network keeps producing roughly-correct outputs even with a corrupted weight — which means errors go undetected by the workload itself. Scrubbing is the only mechanism that surfaces them.
- **The Tegra234 memory controller exposes scrub registers via the EMC (External Memory Controller).** NVIDIA's Tegra234 TRM documents the EMC's `EMC_CFG_DIG_DLL` and `EMC_ECC_SCRUB_*` register family for hardware-accelerated scrubbing. Linux's `tegra-mc` driver doesn't currently expose them (L4T treats them as firmware-managed). SmallAIOS as a unikernel can drive them directly — the EMC is part of the SoC's MMIO surface and the registers are documented. On x86 server platforms the equivalent is `ECC Scrub` controlled via the memory controller's PCIe-config registers; on AMD EPYC it's `SCRUBCTRL` MSRs. The interface is platform-specific but the abstraction is uniform: "scrub region R every interval I".
- **The cooperative scheduler is the right home for the scrub task.** SmallAIOS uses cooperative async with yields at ONNX op boundaries (`docs/scheduling-model.md`). A scrub task that yields after every N pages naturally interleaves with inference work without blocking it. The scrub is also a watchdog-correlated signal: if scrub progress halts (no advance in scrub cursor for >N seconds) the watchdog fires — because either the scheduler deadlocked or the EMC stopped responding.
- **This is niche, but it's the kind of niche that DAL A demands.** SmallAIOS's competitive position is not "fastest inference" — it's "inference you can ship into an avionics box and certify". ECC scrubbing is on the short list of mitigations every avionics OS has and every commodity OS lacks. Investing here strengthens the cert-track value proposition.

## Scope — phases

### Phase 1 — Scrub service core (~1 week)

Create `kernel/src/mem/scrub/` as a new module:

- `mod.rs` — public API: `init(config)`, `add_region(name, base, size, interval)`, `pause(name)`, `resume(name)`, `cursor(name) -> Position`, `stats(name) -> ScrubStats`.
- `task.rs` — async task driver: loops over registered regions, advances each region's cursor by `chunk_size` per scheduler tick, yields between chunks. Implements the per-region interval timer.
- `config.rs` — `ScrubConfig { regions: Vec<Region>, default_chunk_size, default_interval, watchdog_threshold }`. Loaded from a kernel-side TOML at boot or constructed programmatically.
- `stats.rs` — `ScrubStats { cycles_completed, last_cycle_duration, correctable_errors, uncorrectable_errors, cursor_position }`. Atomic counters readable from telemetry.

Boot wiring: `kernel_main` registers a scrub region covering the heap and the ONNX-resident weight region after `mem::init` completes. Default interval = 24 hours per region (industry-standard for DAL A workloads), configurable per region.

### Phase 2 — Tegra234 EMC backend (~1 week)

`arch/aarch64/src/scrub/tegra_emc.rs` — backend that drives the Tegra234 EMC's hardware-accelerated scrub. Probes the EMC registers (MMIO base in DTB), configures `EMC_ECC_SCRUB_*` for a region descriptor, kicks off the scrub, polls completion. Returns the count of correctable/uncorrectable errors observed in the cycle.

The Tegra234 EMC supports two scrub modes: **patrol** (background, autonomous, low priority — what we want) and **demand** (synchronous, blocking — used for boot-time wipe). We use patrol mode for the steady-state service; demand mode is used once at boot to verify the controller responds.

### Phase 3 — Software-fallback backend (~3-5 days)

For platforms without a hardware scrub controller (development boards, some RISC-V parts) the kernel can do software scrubbing: read every cache line, write the same value back. The DRAM controller's ECC engine then walks the data on the bus and corrects single-bit errors as a side effect of the read.

`kernel/src/mem/scrub/sw_backend.rs` — uses `core::ptr::read_volatile` + `core::ptr::write_volatile` in a `usize`-stride loop with a yield every N pages. Slower than the EMC patrol mode but always available.

### Phase 4 — Watchdog integration (~3-4 days)

If the scrub cursor doesn't advance for `watchdog_threshold` seconds (default 60s), the watchdog timer fires. Two scenarios:

- **Cooperative scheduler stuck** — some other task is monopolizing the scheduler. Watchdog reset.
- **EMC stopped responding** — register-level failure; treated as a hardware fault. Watchdog reset, with a serial log line documenting the suspected cause.

Wiring: the scrub task records the last-advance timestamp on every chunk. A separate `watchdog::feed` call happens *only when* the scrub cursor advances. A scrub that's idle (no work to do — between cycles) calls `feed` to keep the watchdog happy.

### Phase 5 — Telemetry + docs (~3-4 days)

- Surface `ScrubStats` per region via the same telemetry path as `telemetry-otel-export-v1` (or, if that hasn't landed, via the kernel console log).
- Document the service in `docs/ecc-scrubbing.md` covering: what SEUs are, why scrubbing matters for DAL A, the per-region config knobs, the recommended intervals for different operational environments (ground vs. cruise altitude vs. orbital), the supported platforms, and the failure modes.
- Update `docs/architecture.md` and `CLAUDE.md` to note the feature.

## Out of scope

- **ECC algorithm changes.** We use the DRAM controller's existing ECC (single-error-correction, double-error-detection — SECDED, the industry default). Stronger ECC (Chipkill, double-bit-correction) requires DRAM-vendor support we can't add in software.
- **Non-DRAM scrubbing.** SoC SRAM, cache, and register-file scrubbing are different mechanisms (often hardware-only) and out of scope here. We surface a TODO note for future work.
- **Bit-error injection for testing.** Validating the scrub service against real injected errors requires hardware support (some boards expose ECC error injection via debug registers). We document the test procedure for boards that support it but don't ship an injection harness.
- **Per-tenant scrub policy.** A future multi-tenant SmallAIOS might want per-tenant scrub guarantees. The unikernel single-tenant model makes this irrelevant for now.
- **RAS (Reliability/Availability/Serviceability) error injection compliance.** RAS frameworks like ACPI APEI on x86 standardize error reporting and recovery; integrating with them is a follow-up change.
- **Cache-line scrubbing.** The CPU's L1/L2/L3 caches typically have their own ECC (Cortex-A78AE has cache ECC enabled by default). We rely on that.

## Sequencing

Phase 1 (core service) lands first; it is fully testable on any platform via the software-fallback backend (Phase 3, which can land alongside Phase 1). Phase 2 (Tegra234 EMC) lands when on-Orin hardware verification is available — depends on `unikernel-orin-bringup-v1` Phase 2e. Phase 4 (watchdog) depends on Phase 1 completing. Phase 5 (telemetry / docs) closes the change.

The change is independent of `tegra-smmu-isolation-v1` (different MMIO surfaces), `aarch64-mte-pac-hardening-v1` (CPU vs DRAM), and `spec-exec-mitigations-v1` (data-path vs control-path). All four can land in parallel.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Scrub service core (task + config + stats) | ~1 week |
| 2 | Tegra234 EMC hardware backend | ~1 week |
| 3 | Software-fallback backend | ~3-5 days |
| 4 | Watchdog integration | ~3-4 days |
| 5 | Telemetry + docs | ~3-4 days |
| **Total** | | **~3-4 weeks** |

Bench-of-record: post-merge, run a 7-day soak test on an Orin NX with a 16 GB weight load, scrub interval 6 h, and capture the correctable-error count. Use that as the baseline for future regressions.
