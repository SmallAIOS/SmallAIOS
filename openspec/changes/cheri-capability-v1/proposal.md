# cheri-capability-v1

## Summary

CHERI (Capability Hardware Enhanced RISC Instructions) is a research-stage instruction-set extension that replaces raw pointers with 128-bit hardware *capabilities* — pointers that carry bounds, permissions, and an unforgeable tag bit. A CHERI capability cannot be forged, cannot exceed its installed bounds, and cannot be widened in scope. The result is hardware-enforced spatial + temporal memory safety at the *pointer* level, complementary to MTE (tag-per-allocation, see `aarch64-mte-pac-hardening-v1`) and SMMU isolation (per-peripheral DMA bounds, see `tegra-smmu-isolation-v1`).

This proposal is **exploratory and research-stage**: no production CHERI silicon exists for SmallAIOS to target today. The reference hardware is the **CHERI-RISC-V** family (Morello prototype boards from Arm Research and the FETT trial; the SCI's Sunburst board; the upcoming CHERIoT-Ibex variants for embedded). All run RISC-V — which aligns with SmallAIOS's existing `riscv64gc-unknown-none-elf` target — but none are production-ready and none are deployable in an avionics box today.

The change is framed as **research alignment**: SmallAIOS already uses a capability-based security model in software (`kernel/src/cap.rs`'s `ResourceType` + `Permissions` + handle-based authorization). Moving that model to *hardware-enforced* capabilities is a natural future step if CHERI matures. This proposal documents the alignment, defines a research-track Cargo feature (`cheri`, off by default, riscv64 only), and lays out what a SmallAIOS-on-CHERI port would look like — *as a planning document*, not an implementation timeline.

## Why

- **SmallAIOS's existing capability model is structurally CHERI-aligned.** `kernel/src/cap.rs` already treats every resource handle as a capability: `Capability { resource_type, permissions, ... }` is the kernel-side authorization primitive for every syscall. CHERI takes that same shape and moves it from "kernel-checked at every dereference" to "hardware-enforced at every load/store". The semantic match is unusually clean — there is no design impedance between our software capabilities and CHERI's hardware ones.
- **The DO-178C / avionics value of provable memory safety is enormous.** If CHERI silicon ever ships in a flight-qualified part, the certification value proposition shifts: instead of "we have done our best to write memory-safe Rust, plus MTE/PAC, plus SMMU, plus speculation barriers" we can say "every pointer in the kernel is hardware-enforced to its bounds, every capability is unforgeable, every cross-component pointer transfer is monotonic in permissions". That story changes the cert risk profile materially.
- **The cost of being CHERI-ready is low if we start now.** The SmallAIOS Rust code is small (~30 crates, mostly `#![no_std]`). Porting to CHERI's `cheri-rust` toolchain — which exists upstream as a research fork — is mechanical in many places (raw pointers become explicit capabilities) and exposes safety issues in others (any `unsafe` block that bypasses bounds becomes immediately visible). Even *not yet running on CHERI silicon*, the static-analysis value of "would this compile under CHERI?" is high. We can run the compile in CI without ever booting on CHERI hardware.
- **It is the right time to plan for it, not to ship it.** The CHERIoT-Ibex variant targets embedded workloads in particular. Industrial partners are beginning to evaluate flight-qualified CHERI silicon timelines. None of that is product-ready in 2026. But a proposal that lays out the alignment now lets us track the maturity from the inside — and lets us push back on alternative roadmaps that ignore CHERI.

## Scope — exploratory phases

This proposal does NOT commit to implementation. It commits to producing artifacts that prove SmallAIOS is CHERI-compatible-on-paper and ready to test on hardware as soon as such hardware is available.

### Phase 1 — Documentation + capability-model alignment doc (~1 week)

Produce `docs/cheri-alignment.md` covering:

- A side-by-side mapping of SmallAIOS's `Capability { resource_type, permissions, ... }` struct to CHERI's capability fields (`base`, `length`, `perms`, `otype`, `tag`).
- Which existing `unsafe` blocks in SmallAIOS would become CHERI-trapped if compiled under `cheri-rust` (e.g., raw pointer arithmetic in the allocator).
- Which CHERI permissions map to which SmallAIOS `Permissions` enum variants — `R` ↔ `LOAD`, `W` ↔ `STORE`, `X` ↔ `EXECUTE`, with notes on the CHERI-only permissions (`LOAD_CAP`, `STORE_CAP`, `SEAL`, etc.) that have no SmallAIOS analog yet.
- Suggested mapping of `ResourceType` discriminants to CHERI `otype` values for sealed capabilities (which would prevent cross-resource-type confusion at the hardware level — a hardware analog of the `pacda`-based confused-deputy defense in `aarch64-mte-pac-hardening-v1`).

### Phase 2 — Toolchain experiment (~1 week)

Attempt to compile a small subset of SmallAIOS (the `smallaios-security` crate's capability primitives) under the `cheri-rust` toolchain. Document what compiles, what doesn't, what changes would be needed. No CI integration — this is one-shot research evidence.

The deliverable is a short report (`notes/cheri-compile-experiment.md`) covering: build command used, errors encountered, hand-applied fixes, conclusion ("the capability core is N% CHERI-clean, with M issues clustered around X pattern").

### Phase 3 — Defer everything else

Anything beyond Phase 2 — actually running on Morello / CHERIoT silicon, integrating CHERI capabilities with the SMMU work, performance benchmarks — is **deferred until production-grade CHERI silicon is available for embedded workloads**. The proposal explicitly tracks this as "open question, revisit when hardware matures".

## Out of scope

- **All other implementation work.** Phase 3+ is explicitly deferred.
- **Toolchain CI integration.** `cheri-rust` is a research fork; adding it as a CI dependency adds maintenance burden for no current return. The compile experiment in Phase 2 is one-shot, not gated.
- **Hardware procurement.** Morello boards are available via Arm Research; CHERIoT FPGA targets via Microsoft Research. Procurement decisions are deferred.
- **CHERI for x86_64 or aarch64.** CHERI extensions to those ISAs exist as research (Morello is aarch64-based), but the path SmallAIOS prioritizes is CHERI-RISC-V because (a) our existing riscv64 target needs the most security work anyway, (b) CHERIoT-Ibex is the closest-to-production embedded variant, (c) Morello is an aarch64 prototype that is unlikely to ship in flight-qualified form.

## Sequencing

This change is fully independent of the four parallel changes (`tegra-smmu-isolation-v1`, `aarch64-mte-pac-hardening-v1`, `spec-exec-mitigations-v1`, `ecc-scrubbing-v1`). It can land anytime; it produces docs only. The right time to do Phase 1+2 is when there is a research-track sprint week available — for example, between major implementation pushes when the team has bandwidth for forward-looking work.

The proposal pre-positions SmallAIOS to move quickly *if and when* CHERI silicon becomes flight-qualified. The risk of not doing this now: another OS (a CHERI-native research kernel from Cambridge or SRI, for instance) becomes the default platform for CHERI inference workloads, and SmallAIOS plays catch-up.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Alignment documentation | ~1 week |
| 2 | Toolchain compile experiment | ~1 week |
| 3+ | All implementation work | Deferred indefinitely |
| **Total in-scope** | | **~2 weeks** |

Beyond Phase 2 — the actual hardware port — is a multi-quarter effort once silicon and a cert-viable timeline emerge. This proposal does not estimate that.
