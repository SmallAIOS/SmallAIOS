# Design — cheri-capability-v1

## Goal

Produce two artifacts that prove SmallAIOS is CHERI-aware on paper and CHERI-clean in critical code paths:

1. `docs/cheri-alignment.md` — the side-by-side mapping of SmallAIOS's capability model to CHERI's hardware capability model, with explicit gap analysis.
2. `notes/cheri-compile-experiment.md` — empirical evidence from a one-shot attempt to compile the capability primitives under the `cheri-rust` research toolchain.

This is a research-track change. It does not produce running code on CHERI silicon and does not block any other proposal.

## Design decisions

### Decision 1: Document-driven, not implementation-driven

A typical OpenSpec change ships running code. This one ships documentation. The justification: CHERI hardware is not deployable in any environment SmallAIOS targets in 2026, so producing running code would be either (a) only-useful-on-research-FPGA — limited audience, high maintenance cost, or (b) emulator-only — which doesn't prove anything about real silicon behavior. The documentation deliverable is the highest-leverage option for the current state of CHERI maturity.

If CHERI hardware matures (Phase 3+ becomes feasible), a future change `cheri-capability-v2` opens the implementation track. This proposal explicitly does not commit to that follow-up.

### Decision 2: CHERI-RISC-V target, not Morello (CHERI-ARM)

CHERI exists in two production research targets:

| Target | ISA | Status | Why pick / not pick |
|--------|-----|--------|--------------------|
| **CHERI-RISC-V** (CHERIoT-Ibex / Sail-RISC-V) | RISC-V | Multiple FPGA + ASIC variants | **Pick** — aligns with SmallAIOS's existing riscv64 target, embedded-friendly |
| Morello | aarch64 (Armv8.2-A extended) | Single Arm Research prototype, ~2021 | Reject — research-only, no embedded/avionics variant planned |

If a flight-qualified CHERI-ARM variant ever emerges, the trait shape designed here would let us add a second target without restructuring.

### Decision 3: Capability model alignment table — explicit field-by-field

The CHERI hardware capability is a 128-bit object with these fields (CHERI ISAv9):

| CHERI field | Size | Meaning | SmallAIOS analog |
|-------------|------|---------|------------------|
| `tag` | 1 bit | Validity bit, hardware-managed | Implicit — `Capability` exists or it doesn't |
| `base` | 64 bits | Lower bound of accessible region | No direct analog — handle indexes into pools |
| `length` | 64 bits | Region size | No direct analog — pool entry has its own size |
| `address` | 64 bits | Current pointer position | The handle's `resource_id` field |
| `perms` | 16 bits | R/W/X/LC/SC/SE/CINV/etc. | `Permissions` enum (much smaller domain) |
| `otype` | 16 bits | Sealed-capability type | `ResourceType` enum |
| `flags` | 8 bits | Implementation-defined | Unused in SmallAIOS analog |

The mapping is clean for the conceptual fields (perms ↔ Permissions, otype ↔ ResourceType) and somewhat awkward for the spatial fields (CHERI's base/length/address vs SmallAIOS's pool-index model). The alignment doc explicitly calls this out — porting to CHERI would require migrating from "handle indexes a pool" to "capability carries its own bounds", which is a kernel-side refactor of moderate scope (estimated 2-3 weeks if and when it happens).

### Decision 4: Sealed capabilities as the hardware analog of PAC-signed handles

CHERI's *sealing* mechanism lets a capability be wrapped with an `otype` so that it can only be unsealed (used as a normal pointer) by code that holds an unseal capability for that type. This is the hardware analog of the PAC-signed capability handles in `aarch64-mte-pac-hardening-v1` (sign with `pacda` using `ResourceType` as modifier). The alignment doc maps:

- `Capability { resource_type: TensorBuffer, ... }` (SmallAIOS today) → `CHERI capability sealed with otype = TENSOR_BUFFER_OTYPE` (CHERI future).
- `autda` fails because of a wrong `ResourceType` modifier (SmallAIOS today) → hardware unseal-fault because the otype doesn't match the unseal-capability's otype (CHERI future).

This is one of the cleanest semantic alignments — both designs catch confused-deputy attacks via the same conceptual mechanism (typed unforgeable handles).

## Alternatives considered

### Alt A: Don't write the proposal at all — wait for CHERI silicon to mature

**Rejected.** The cost of writing the alignment doc + compile experiment is low (~2 weeks) and the option-value of being CHERI-ready is high. Not having a proposal makes us reactive when silicon timelines firm up; having one makes us proactive.

### Alt B: Implement CHERI in software (capability emulation in the unikernel)

**Rejected.** Software-emulated CHERI capabilities (a `Capability<T>` newtype that does runtime bounds + permission checks) lose the entire performance + certification value proposition. The point of CHERI is hardware enforcement. A software emulation costs runtime cycles for no measurable security gain over what we already get from Rust's type system + MTE.

### Alt C: Skip RISC-V, focus on Morello

**Rejected.** Morello is a single prototype board, primarily a research vehicle, not a path to deployed hardware. Even though SmallAIOS's aarch64 path is more mature than riscv64 (Orin is our reference platform), the deployment path for CHERI is the embedded-RISC-V variants (CHERIoT-Ibex).

### Alt D: Make the CHERI feature a public roadmap commitment

**Considered, rejected.** Publicly committing to CHERI support before silicon timelines firm up creates implicit deadlines we can't meet. The proposal is internal-facing and the docs are internal — the roadmap commitment, if any, is a separate communications decision.

## Risks

### Risk 1: CHERI silicon never reaches a flight-qualified state

The largest risk. If CHERI remains research-only for the next decade, the alignment doc remains a planning artifact with no production value. Mitigation: the cost of the proposal is bounded (Phase 1+2 = ~2 weeks); the upside if silicon does mature is large. We treat it as a low-cost hedge.

### Risk 2: `cheri-rust` toolchain bit-rots

The CHERI Rust fork is maintained by SRI International / University of Cambridge research groups and lags upstream Rust by months. The Phase 2 compile experiment may need to use an older Rust target than `nightly-2026-02-01`. Mitigation: Phase 2 deliverable explicitly documents the toolchain version used; the experiment is reproducible from the documented invocation, not bit-perfect against a future toolchain.

### Risk 3: Capability model evolution

The SmallAIOS capability model itself is evolving. Major changes (e.g., adding revocation, hierarchical capabilities) would invalidate parts of the CHERI alignment doc. Mitigation: the doc is dated; future capability-model changes trigger a doc refresh.

### Risk 4: Misleading "CHERI-ready" claims

Saying "SmallAIOS is CHERI-ready" without running on hardware is over-promising. Mitigation: the alignment doc explicitly states "alignment-on-paper, not hardware-tested" in its opening section; the proposal restricts itself to "research-stage, exploratory".

## Build/CI surface

- No build changes. No new Cargo feature.
- No CI changes.
- New file: `docs/cheri-alignment.md`.
- New file (in this change's `notes/`): `notes/cheri-compile-experiment.md` (if Phase 2 produces evidence).

## What this change explicitly does NOT do

- Does not produce any Rust code.
- Does not add a CHERI-related Cargo feature to any crate.
- Does not commit to a CHERI implementation timeline.
- Does not modify any existing capability code.
- Does not require any hardware.
- Does not interact with any of the four parallel memory-safety changes — it is purely a forward-looking planning artifact.
