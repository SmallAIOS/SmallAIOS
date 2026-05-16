# Design — aarch64-mte-pac-hardening-v1

## Goal

Two CPU-side hardware memory-safety features active on every `--features tegra234` build of the SmallAIOS kernel:

1. **PAC** — return-address signing on every function entry/exit (catches ROP), signed indirect-call pointers in the ONNX op dispatch (catches JOP), and signed data pointers for capability handles (catches confused-deputy attacks on `ResourceType`).
2. **MTE** — synchronous tag-check on every load/store, with allocator-side tag assignment from a hardware RNG, and a kernel-side fault handler that converts mismatches into structured panics (or watchdog notifications on safety-critical builds).

Verification: on Orin NX hardware, a deliberately-injected use-after-free or stack-smash triggers a trapped fault before observable memory corruption, with a structured log line identifying the fault PC.

## Design decisions

### Decision 1: PAC algorithm choice — QARMA5 with all five keys

ARMv8.5-A defines five PAC keys: `APIA` (instruction address A), `APIB` (instruction address B), `APDA` (data address A), `APDB` (data address B), `APGA` (generic). We enable and key all five rather than the bare-minimum `APIA` (return-address only):

| Key | Use | Why enable |
|-----|-----|------------|
| **APIA** | `pacia` / `autia` — return addresses | Baseline ROP defense, free with compiler flag |
| **APIB** | `pacib` / `autib` — alternate inst key | Used by Rust for non-return indirect branches |
| **APDA** | `pacda` / `autda` — data pointers | We use for capability handle signing |
| **APDB** | (unused initially) | Wired but not used; available for future signed-FFI cases |
| **APGA** | `pacga` — generic 64-bit signing | We use for ONNX op dispatch indirect call signing |

Cost of enabling all five: 5 × 128 bits = 80 bytes of EL1 system register state. Worth it because compilers may emit any of them and we don't want a "key not installed" trap on instruction execution.

QARMA5 vs. QARMA3: silicon decides. Cortex-A78AE implements QARMA5 by default; we don't get to pick. We read `ID_AA64ISAR1_EL1.{APA, GPA}` to confirm at boot and refuse to enable PAC if the silicon reports unsupported.

### Decision 2: PAC keys are per-boot, derived from hardware RNG

Each boot derives all five keys from Orin's TRNG (`RNGSR_EL0` available on the security engine; falls back to `CNTPCT_EL0`-mixed PRNG on platforms without TRNG, with a boot-time warning). Keys never persist across reboots — an attacker who exfiltrates the key for one boot cannot use it after a reboot.

A build-time `mte-pac-deterministic` Cargo feature lets development builds use a fixed key (printed at boot) so a debugger can decode signed pointers manually. Off by default. Production builds always use the TRNG path.

### Decision 3: MTE in synchronous mode (sync TCF), not asynchronous

| TCF mode | Latency on hit | Precision | Use case |
|----------|---------------|-----------|----------|
| **Sync (0b01)** | Traps immediately | Exact fault PC | Pick this — safety-critical needs precise diagnosis |
| Async (0b10) | Trap on next exception entry | Approximate PC | Faster on workloads with many tag operations |
| Asymmetric (0b11) | Sync on read, async on write | Mixed | Compromise — not picked |

We pick sync for the primary build because (a) Cortex-A78AE's MTE implementation has very low sync overhead per Arm's published numbers, (b) async loses the offending PC on workloads with deep `unsafe` chains, which is exactly the workloads MTE is supposed to diagnose. We add an `mte-async` Cargo feature for non-safety-critical builds that prefer throughput.

### Decision 4: 4-bit tags chosen by per-allocation RNG, granule = 16 bytes

ARMv8.5-A fixes the tag width (4 bits = 16 values) and granule (16 bytes). We have no design freedom there. The choice is *how to assign tags*:

- Random per-allocation: 1/16 chance two adjacent allocations collide, which is fine for catching most UAF/OOB.
- Sequential (1, 2, 3, ..., 15, 1, 2, ...): better for spatial locality, slightly worse for entropy. Linux uses this on kernel allocations.

We pick **random** because the kernel allocator (`kernel/src/mem/heap.rs`) is called rarely (most ONNX-runtime allocation goes through a slab allocator that we tag separately) and the entropy is worth more than the locality.

Tag-zero is reserved for "untagged" memory regions (MMIO, stack, DTB). The allocator assigns from 1-15 only.

### Decision 5: MTE fault handler converts to structured panic, not silent retry

A sync MTE fault raises a Data Abort with `ESR_EL1.EC = 0x25` (data abort, current EL) and a specific FSC code (`0x11` for tag check fault). The handler in `interrupts.rs`:

1. Decodes `ESR_EL1`, `FAR_EL1`, and `ELR_EL1` to capture the fault PC, fault address, and the expected vs. actual tag (from address bits 56-59 and the granule tag).
2. Logs a structured fault line: `[mte-fault] pc=0x… addr=0x… tag_pointer=N tag_memory=M`.
3. Panics — the kernel does not attempt to continue execution because tag mismatches always indicate a real safety bug. (Linux's user-space MTE allows recovery by re-tagging; the kernel case is "this should not happen".)
4. On `mte-watchdog` Cargo feature builds, instead of panicking, signals the hardware watchdog and produces a coredump-shaped serial dump.

### Decision 6: Capability handle PAC signing — APDA, not APGA

The existing capability system in `kernel/src/cap.rs` stores `ResourceType` enums inside handles. A bug that lets a caller flip the `ResourceType` bits (via a confused `transmute` or a misused arg decoding) is a classic confused-deputy escalation. PAC signing fixes this: the kernel signs the handle with `pacda` using the resource-type-derived modifier; if anything in the handle flips, `autda` traps.

We pick `APDA` (data address A) for this rather than `APGA` (generic) because:

- `pacda` traps inside the kernel directly when `autda` fails — `pacga` only produces a 32-bit signature that the caller has to check explicitly with a compare-and-branch, which is more code and easier to forget.
- `APDA` is unused by Rust's compiler-emitted PAC instructions, so we own that key for application-defined purposes.

## Alternatives considered

### Alt A: Software-only ASLR + stack canaries instead of PAC

**Rejected.** Stack canaries are useful but they catch a narrower class of attacks (stack-buffer-overflow → return-address corruption only). PAC catches the same class plus indirect-call corruption plus heap-based ROP. ASLR is orthogonal — we don't ship ASLR today (the unikernel has a fixed kernel layout) and adding it costs more than just enabling PAC. PAC is the right primary defense.

### Alt B: Use AddressSanitizer (ASAN) instead of MTE

**Rejected.** ASAN-style red-zone instrumentation costs ~50% throughput and doubles memory use. MTE costs <2% throughput and ~3% memory (one tag byte per 16 bytes of allocation, packed in DRAM). ASAN is the right tool for development; MTE is the right tool for production safety-critical builds. We may opt-in ASAN for unit-test builds (`#[cfg(test)]`) as a follow-up.

### Alt C: Enable PAC and MTE conditionally based on runtime feature detection

**Rejected for the primary path.** SmallAIOS is `#![no_std]` and we know our target hardware at build time. The `mte-pac` Cargo feature is opt-in at build time (default-on for `tegra234`, off for other aarch64 targets). Runtime feature detection adds branch overhead and a "what if it's missing" code path we don't need. If we ever target a non-MTE aarch64 SoC, we build without the feature.

### Alt D: Defer until DO-178C tooling story is more mature

**Considered, rejected.** Waiting for full DO-178C tooling support before adding MTE/PAC means missing an entire generation of hardware mitigations. We document the features in the existing safety case structure and let the certification work catch up.

## Risks

### Risk 1: ONNX runtime perf regression from MTE allocator overhead

The ONNX runtime allocates working buffers per operator. Tagged allocations cost a `stg` instruction per 16-byte granule on alloc and a check on every access. Mitigation: (a) benchmark on representative ONNX workloads (existing `bench/` crate) before/after MTE, gate merge on <5% steady-state regression; (b) for the slab allocator path (high-frequency same-size allocs), assign one tag per *slab* rather than per allocation — slab-level MTE catches inter-slab OOB but not intra-slab; (c) document the `mte-async` opt-out.

### Risk 2: PAC key reuse across boots could weaken security

Mitigation handled in Decision 2 — per-boot TRNG-derived keys. The `mte-pac-deterministic` dev feature is opt-in only.

### Risk 3: MTE fault during early boot (before fault handler installed)

If the allocator or any tagged-pointer operation runs before `interrupts.rs` has installed the Data Abort vector, a tag mismatch becomes an unrecoverable hardware fault. Mitigation: ordering — `mte::init()` is called *after* `interrupts::init()` in the boot sequence (`arch/aarch64/src/main.rs`). Before `mte::init`, the kernel runs untagged (TCF=0), so no tag mismatch can fire.

### Risk 4: UEFI firmware leaves MTE/PAC state inconsistent

JetPack 6's UEFI may have programmed `SCTLR_EL1` with its own PAC/MTE settings before `ExitBootServices`. Mitigation: at `kernel_main` entry, we explicitly disable both (`SCTLR_EL1.{TCF, EnIA, EnIB, EnDA, EnDB} = 0`), then install our own keys, then re-enable. Single-source-of-truth: the kernel owns the keys; UEFI's choices are clobbered.

### Risk 5: Toolchain divergence — Rust nightly PAC support is moving

The Rust 2026-02-01 toolchain pinned in `rust-toolchain.toml` supports `-C codegen-options=branch-protection=pac-ret` stably. The MTE-aware allocator hook (`-Z sanitizer=memtag`) is nightly-only and has changed shape across recent toolchains. Mitigation: pin the toolchain explicitly in the new CI job; document the codegen flags in `docs/aarch64-security.md` with the exact `cargo build` invocation.

## Build/CI surface

- New Cargo feature `mte-pac` on `smallaios-arch-aarch64`. Default-on for `tegra234`, default-off elsewhere.
- New module `arch/aarch64/src/security/{mod.rs, pac.rs, mte.rs}` containing the boot-init code, key install, and fault decoder.
- Modify `arch/aarch64/src/interrupts.rs` Data Abort handler to dispatch tag-check faults into `security::mte::handle_fault`.
- Modify `arch/aarch64/src/main.rs` / `main_uefi.rs` boot sequence to call `security::init()` after `interrupts::init()` and before allocator init.
- New `mte-pac-deterministic` (dev) and `mte-async` (perf) and `mte-watchdog` (safety) sub-features.
- New CI advisory job `aarch64-mte-pac-build` — builds with the feature and runs the existing QEMU smoke (QEMU's `cortex-a78` model supports `+mte,+pauth` — we add `-cpu cortex-a78,mte=on,pauth=on` to the smoke recipe).
- New `just mte-fault-test` recipe that runs a tag-mismatch-injecting test binary under QEMU and asserts the fault handler fires.

## What this change explicitly does NOT do

- Does not change syscall ABI — PAC/MTE are CPU-internal, invisible to syscall callers.
- Does not modify any non-aarch64 architecture path. The `kernel-security` capability spec (separate from `arch-aarch64-security`) covers cross-arch mitigations under `spec-exec-mitigations-v1`.
- Does not enable user-space MTE (there is no user space).
- Does not change the existing PQC crypto stack — MTE/PAC are CPU memory protections, not cryptographic primitives.
