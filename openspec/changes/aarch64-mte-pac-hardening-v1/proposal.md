# aarch64-mte-pac-hardening-v1

## Summary

Cortex-A78AE (the CPU in Jetson Orin's Tegra234 SoC) implements ARMv8.5-A — including two memory-safety hardware features that SmallAIOS does not yet exercise: **Memory Tagging Extension (MTE)** and **Pointer Authentication (PAC)**. MTE hardware-tags every 16-byte allocation granule and traps mismatched memory accesses; PAC signs return addresses and indirect function pointers with a per-process secret, making ROP/JOP exploits orders of magnitude harder. Both features impose <2% steady-state runtime overhead on real workloads (Google's MTE measurements on Pixel 8, Apple's PAC measurements on M-series), neither requires source changes beyond a small `unsafe` allocator hook, and both are baseline hardware-mitigation expectations in any 2026-era safety-critical OS deployment.

This change enables MTE and PAC under a new `mte-pac` Cargo feature on `smallaios-arch-aarch64`, gated on Tegra234 / Orin builds (the only ARMv8.5-A platform SmallAIOS currently targets — `aarch64-unknown-uefi --features tegra234`). The feature splits into two cleanly-separable phases: PAC first (smaller, no allocator changes, ROP hardening only), then MTE (needs allocator integration to assign and check tags). A new `arch-aarch64-security` capability spec documents the contract.

## Why

- **SmallAIOS's safety-critical positioning demands hardware-level memory safety where it exists.** The kernel is `#![no_std]` Rust — the type system already prevents the most common UAF / OOB patterns at the source level. But the moment we touch `unsafe { ... }` (allocator, DMA, FFI to the NVIDIA HAL, hand-rolled UEFI bindings in `arch/aarch64/src/uefi.rs`, syscall argument decoding in `kernel/src/syscall/`), source-level guarantees evaporate. MTE catches whatever those `unsafe` blocks get wrong, at hardware speed, with no recompile of the offending code path. PAC catches return-address smashes regardless of source-language. Both are exactly the kind of *defense in depth* the DO-178C "design assurance" framework rewards.
- **The hardware is free on Orin and we are not using it.** Cortex-A78AE supports MTE (FEAT_MTE2) and PAC (FEAT_PAuth + FEAT_PAuth2) in silicon. JetPack 6's L4T kernel enables them — `cat /proc/cpuinfo | grep -E "(mte|paca|pacg)"` returns hits on a stock Orin NX. SmallAIOS's bare-metal aarch64 path (the `unikernel-orin-bringup-v1` Phase 2 BSP, currently in interim mode) does not yet program `SCTLR_EL1.{TCF, EnIB, EnIA}` or related control bits. Until we do, every CPU cycle that runs SmallAIOS on Orin runs without those features active.
- **PAC closes one of the few remaining low-friction exploit primitives in a Rust unikernel.** Even pure-Rust code is vulnerable to ROP if an attacker can corrupt a stack frame (e.g., a misused `transmute`, an `unsafe` slice indexing bug, a syscall arg with a missed bounds check). PAC's `pacia`/`autib` instruction pair makes the return address self-validating — a corrupted return address fails authentication on `RET` and traps. Cost: one instruction at function entry, one at exit. Net effect: return-oriented programming requires breaking PAC's per-process key, which on ARMv8.5-A requires either a side-channel or kernel-privileged register reads.
- **MTE is the right complement to the SMMU isolation work.** `tegra-smmu-isolation-v1` (parallel change) contains DMA-side corruption to specific stream IDs. MTE contains CPU-side corruption to specific allocations. Together they cover both the "GPU scribbles random DRAM" and "kernel scribbles random kernel struct" failure modes that any safety-critical certification will ask about. The two changes don't interact at the spec level — they just both reduce the blast radius of the same underlying class of memory-safety bugs.
- **The Rust target supports it already.** The standard Rust compiler emits BTI (branch target identification) and supports MTE-aware allocator hooks via `-Z sanitizer=memtag` (nightly). `aarch64-unknown-none` and `aarch64-unknown-uefi` both accept `-C target-feature=+mte,+pauth` on the toolchain pinned in `rust-toolchain.toml`. No upstream-rust changes are needed.

## Scope — phases

The two features ship in series because PAC is genuinely easier (no allocator changes, no fault paths to design) and de-risks the rest.

### Phase 1 — PAC (Pointer Authentication, ~1 week)

PAC has three independent sub-features in ARMv8.5-A. We enable all three:

- **PACIASP / AUTIASP** — sign + authenticate the link register on function entry/exit. Compiled in via `-C target-feature=+pauth -C codegen-options=branch-protection=pac-ret`. Catches stack-smashing ROP.
- **PACGA** — sign generic pointers. We use this for indirect call sites in ONNX op dispatch (`kernel/src/syscall/onnx.rs`) — every operator function pointer is signed when stored, authenticated when called. Catches indirect-call ROP/JOP.
- **PACDA / AUTDA** — sign data pointers. We use this for capability handles (`kernel/src/cap.rs`) — every capability is signed with its resource type, so confused-deputy attacks that smash a `ResourceType` field fail authentication.

Boot setup: write the per-boot PAC keys to `APIAKeyHi_EL1`, `APIBKeyHi_EL1`, `APDAKeyHi_EL1`, `APDBKeyHi_EL1`, `APGAKeyHi_EL1` (one 128-bit key per algorithm). Keys are derived from a hardware RNG (Orin has TRNG via `RNGSR_EL0`) plus a build-time per-image salt for development determinism.

`SCTLR_EL1.{EnIA, EnIB, EnDA, EnDB}` enable instructions in EL1 (kernel-only). `SCTLR_EL1.EnIB` enables `pacib`/`autib` for the secondary instruction key.

No allocator changes are needed. No syscall changes. The user-visible boot diff is a `[pac] keys installed, branch-protection=pac-ret active` line.

### Phase 2 — MTE (Memory Tagging Extension, ~1-1.5 weeks)

MTE tags every 16-byte allocation granule with a 4-bit tag and a 4-bit "log" tag stored in DRAM (Orin's DRAM controllers support this — LPDDR5 has the spare bits). Loads/stores check that the tag in the pointer matches the tag in memory; mismatches trap.

Three pieces:

1. **Tag assignment in the allocator** (`kernel/src/mem/heap.rs` or wherever the `GlobalAlloc` impl lives). Every `alloc` call generates a random 4-bit tag, writes it to all granules of the allocation (`stg` instruction), and embeds the tag in the returned pointer's bits 56-59 (the ARM "top byte ignore" address space, which MTE repurposes).
2. **Tag check enablement** — write `SCTLR_EL1.TCF = 0b01` (sync MTE) for the kernel. Sync mode traps immediately on mismatch; async mode batches and is faster but less precise. We pick sync for safety-critical correctness and re-evaluate if benchmarks show measurable overhead.
3. **Fault handler** — sync MTE faults raise a Data Abort with `ESR_EL1.EC = 0x24/0x25` and a specific FSC code. The handler in `arch/aarch64/src/interrupts.rs` decodes the fault, logs the offending PC + tag mismatch info, and panics (or, on safety-critical builds, calls into the watchdog).

A future polish: stack tagging (`-Z sanitizer=memtag-stack`). This requires Rust toolchain support that's nightly-only and gated; we defer it as Phase 2.5 follow-up.

## Out of scope

- **MTE on the GPU side.** MTE is a CPU feature. The GA10B GPU on Orin does not implement MTE — GPU-side memory safety is the SMMU's job (`tegra-smmu-isolation-v1` parallel change).
- **BTI (Branch Target Identification).** Already enabled in the default `aarch64-unknown-none` codegen via Rust's `-Z branch-protection=bti`. Not a delta this change introduces; we document it for completeness.
- **PAC on the userspace boundary.** SmallAIOS is a unikernel — there is no userspace transition, so the EL0/EL1 PAC key separation doesn't apply. We use EL1 keys exclusively.
- **Heap stack-tagging.** Defer to Phase 2.5 follow-up once the toolchain story for stack MTE on bare-metal is stable.
- **x86_64 / RISC-V analogs.** Intel's LAM (Linear Address Masking) is the rough equivalent of TBI, MPK has overlap with MTE, CET shadow stack overlaps with PAC. Each is its own change; we do aarch64 first because Orin is our reference platform.
- **Verified Boot integration.** Whether MTE-tagged kernel images need their own measurement is a question for the `verified-boot` feature owner; we don't touch it here.

## Sequencing

Phase 1 (PAC) is independent and lands first — small, low-risk, compiler-driven. Phase 2 (MTE) builds on Phase 1 only in that they share the boot-time RNG / key-install code path. The change can run fully in parallel with `tegra-smmu-isolation-v1` (different hardware features, different code paths). It depends only on `unikernel-orin-bringup-v1` Phase 2e landing (need a working AArch64 boot path on Orin to test on hardware).

The MTE benchmark gate (Phase 2 task 2.4) is the most likely point of friction: if our specific workloads hit higher overhead than the ~2% expected (e.g., due to ONNX runtime's allocation churn), we either accept it as the safety-critical price or add an `mte-async` Cargo feature for async-mode tag checking. We document the decision in `design.md`.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | PAC (compiler flag + key install + boot wiring) | ~1 week |
| 2 | MTE (allocator integration + fault handler + benchmark) | ~1-1.5 weeks |
| **Total** | | **~2-3 weeks** |
