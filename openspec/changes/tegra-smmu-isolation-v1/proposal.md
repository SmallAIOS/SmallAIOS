# tegra-smmu-isolation-v1

## Summary

The `sys_tensor_map_gpu` syscall (`kernel/src/syscall/memory.rs:361`) is the gateway between SmallAIOS-managed DRAM and the Ampere GA10B GPU's view of memory on Jetson Orin. Today it performs capability checks (`TensorBuffer:WRITE` + `GpuDevice:EXECUTE`) and then returns `NotSupported` because the NVIDIA HAL integration hasn't landed yet. When that HAL does land, the syscall path will hand a physical buffer pointer to the GPU command submission code — which on Tegra234 means **DMA from a peripheral that does not share the CPU's address space**. Without explicit System MMU (SMMU) programming, every DMA-capable engine on the SoC (GA10B GPU, host1x clients, PCIe root complex, USB, SDMMC, security engine) sees the full physical-address space, and a buggy or hostile peripheral can read or scribble anywhere in DRAM regardless of what page tables the CPU thinks are in effect. This is the SoC-level analog of running without an MMU.

Tegra234 ships a Cortex-A78AE-class **SMMU500** (Arm SMMUv2) for client peripherals and an **MMU400** (SMMUv1-shaped) for legacy host1x v6 stream IDs — both are documented in the Tegra234 TRM and exposed via the upstream Linux `arm,mmu-500` / `nvidia,tegra234-smmu` device-tree bindings. This change wires those IOMMUs into SmallAIOS so that every DMA path crosses a hardware translation step gated by an IOMMU stream-table entry the kernel controls. The Cargo feature `smmu` on `smallaios-arch-aarch64` (default off until Phase 2 lands hardware verification on Orin) selects the driver; a new `kernel-iommu` capability covers the per-stream-ID lifecycle.

## Why

- **DMA isolation is a DO-178C DAL A prerequisite, not a polish item.** Avionics safety reviewers ask "what stops a peripheral from corrupting the partition's address space" and the only acceptable answer is hardware-enforced IOMMU isolation. RTCA DO-254 (the hardware companion to DO-178C) and the IMA partitioning model in ARINC 653 both presume bus-master containment. Today SmallAIOS has no story for this on aarch64 — `arch/aarch64/src/paging.rs` programs the CPU MMU (translation regime EL1) but nothing touches the System MMU's stream tables (translation regime "non-secure stage-1/stage-2 for peripherals"). For x86_64 the situation is similar (Intel VT-d / AMD-Vi are stubbed). On Orin specifically, the GPU is the worst offender: a single bad GR class submission can DMA-write anywhere in DRAM unless `arm-smmu-500` is the gatekeeper.
- **The syscall surface is already shaped for it.** `sys_tensor_map_gpu` returns a `GpuPtr` (`u64`) that the GPU sees. Today that's expected to be a physical address. Once the SMMU is in the loop, the `GpuPtr` becomes an *IO virtual address* (IOVA) — a number that means something only inside the GPU's translation context. That re-typing is mechanical once the SMMU driver exists; the syscall ABI stays binary-compatible because both forms are `u64`. The capability checks already in place (`require_capability(GpuDevice, EXECUTE)`) become the authorization gate for "may this caller install an SMMU mapping at all".
- **Tegra234's SMMU500 is well-documented and Linux-tested.** Upstream Linux's `drivers/iommu/arm/arm-smmu/arm-smmu.c` plus `drivers/iommu/arm/arm-smmu-v3/` provide a working reference for SMMUv2 + SMMUv3 programming. NVIDIA's L4T fork exposes the SMMU on Tegra234 through the standard `arm,mmu-500` binding with extra `nvidia,memory-controller` properties for stream-ID-to-engine assignment. We are not reinventing the protocol — we are porting a known-good driver to `#![no_std]` with no allocator (or with the page-aligned allocator from `kernel/src/mem/`).
- **A unikernel with hardware-enforced DMA isolation is a stronger story than a Linux-based competitor.** Linux-based ML appliances (NVIDIA's own JetPack 6 host stack included) leave the SMMU "passthrough" by default for performance reasons; userspace can request DMA-API mappings but the kernel does not enforce that *every* DMA-capable engine is contained. A unikernel that ships SMMU-enforced isolation by default differentiates SmallAIOS on the same hardware NVIDIA ships, without giving up performance (SMMU500 walks page tables in hardware — overhead is a single-digit-percent TLB-miss tax once warm).

## Scope — phases

The work is sequenced so the highest-risk pieces (initial SMMU init, GPU stream-ID isolation) land first, with peripheral coverage following once the framework is proven.

### Phase 1 — SMMU500 driver scaffold (~1 week)

Create `arch/aarch64/src/smmu/` as a new module:

- `mod.rs` — public API: `init(dtb)`, `attach_stream(stream_id, page_table_root)`, `detach_stream(stream_id)`, `flush_tlb(stream_id)`. Trait-shaped so the SMMUv3 variant slots in later.
- `smmu_v2.rs` — SMMU500 implementation (SMMUv2-shaped per the arm-smmu Linux driver): probe register banks at the Tegra234 MMIO base, allocate context banks (CB0-CB15), program global address translation tables (GR0/GR1), wire stream-match registers (SMR/S2CR). Identity-mapped + stage-1 only initially.
- `stream_id.rs` — type-safe `StreamId(u16)` newtype with Tegra234-specific constants extracted from the Tegra234 TRM (GA10B GPU = `0x7`, host1x = `0x1`, PCIe controllers = `0x10-0x17`, USB = various). One source of truth so syscall-side code never hand-writes literals.

Boot-time sequence (called from `kernel_main` after `mem::init` but before any DMA-capable driver init): probe SMMU registers, install a permissive identity map as a transient default, then immediately switch to "no stream mapped = no DMA allowed" once each peripheral is explicitly attached.

Verification: read back the SMR/S2CR registers and assert the kernel-side stream-table matches programmed expectations. Boot output prints `[smmu] SMMU500 attached, N stream IDs reserved`.

### Phase 2 — GPU stream-ID enforcement (~1-1.5 weeks)

Tie `sys_tensor_map_gpu` to the SMMU. Today the syscall short-circuits at `NotSupported`; once the NVIDIA HAL lands, the syscall must:

1. Compute an IOVA from the tensor's physical pages (allocator-friendly: pick from an SMMU IOVA pool, *not* return raw physical).
2. Install an SMMU stage-1 mapping for the GPU's stream ID (`StreamId(0x7)` on Tegra234) covering those pages with the requested permissions (`R`, `RW`).
3. Invalidate the SMMU TLB for that stream.
4. Return the IOVA as the `GpuPtr`.

`sys_tensor_unmap_gpu` is the symmetric tear-down: invalidate + free the IOVA region.

A new `kernel-iommu` capability spec (this change) governs stream-table modifications. Only the kernel itself (via the syscall handler) may write to SMMU registers; userspace never has direct register access. The cooperative-async scheduler treats SMMU TLB invalidation as a yield point because the invalidate-and-wait sequence can take microseconds.

### Phase 3 — Peripheral coverage (~1 week)

Extend the same machinery to every other DMA-capable Tegra234 peripheral surfaced by SmallAIOS today or planned in the next two releases:

- **host1x v6** — the video/ISP/encoder controller. Stream IDs `0x1-0x6`. Shares the GPU's GR class submission shape, so the same `attach_stream` API works.
- **PCIe root complex** — stream IDs `0x10-0x17`. NVMe and Wi-Fi M.2 cards arrive here.
- **USB XHCI** — stream ID `0x18`. The USB stack is a current SmallAIOS deliverable (`smallaios-usb` crate) so this matters even for "no GPU" deployments.
- **SDMMC** — stream IDs `0x20-0x21`. eMMC and SD card access.

For each peripheral, the driver init code calls `smmu::attach_stream(<id>, this_driver's_page_table_root)` before enabling DMA. Peripherals without an attached stream are simply unable to DMA — register access still works (CPU MMIO), DMA is hardware-blocked.

### Phase 4 — Fault handling + telemetry (~0.5-1 week)

SMMU500 raises a fault interrupt when a peripheral attempts an untranslated or denied access (FSR register + GICv3 SPI). We wire that to a kernel-side handler that:

- Logs the offending stream ID, address, fault syndrome (translation fault vs. permission fault vs. external abort).
- Increments per-stream-ID fault counters surfaced via the existing telemetry / OTEL export path (`telemetry-otel-export-v1` proposal).
- Optionally panics under a `smmu-fatal-fault` Cargo feature for safety-critical builds where any DMA fault is a definite bug.

## Out of scope

- **x86_64 IOMMU (VT-d / AMD-Vi).** Different protocol, different driver. Tracked separately as `iommu-x86-vtd-v1` when the x86 path is prioritized.
- **RISC-V IOMMU.** The riscv64 target doesn't currently have a documented IOMMU on the bring-up hardware. When we cross that bridge, the trait shape introduced here (`Iommu::attach_stream` etc.) is the right re-use point.
- **SMMUv3 promotion.** Tegra234 ships SMMUv2 (SMMU500) — the AGX Orin Industrial / Thor parts use SMMUv3. We pick SMMUv2 first because that is what Orin NX ships; the abstraction allows a v3 driver to slot in later without touching syscall callers.
- **Per-process page tables.** SmallAIOS is a single-address-space unikernel by design (`docs/architecture.md`); the SMMU stage-1 page tables are kernel-managed and shared across "tasks". We are not introducing per-process page tables on the IOMMU side either; we are partitioning the *peripheral* view of DRAM, not the CPU view.
- **Userspace IOMMU API (VFIO-style).** No syscall surfaces SMMU programming directly; the kernel is the sole gatekeeper. A future change might expose a `sys_iommu_*` family for advanced use cases but the default story is "kernel-only".
- **Live SMMU reconfiguration during inference.** Streams are attached at driver-init time and tensor-map mappings are added/removed per `sys_tensor_map_gpu` call, but global SMMU configuration (which stream IDs exist, fault-handler installation) does not change after boot.

## Sequencing

Phase 1 lands first and is independent — the SMMU driver compiles, boots on the Orin via the existing `unikernel-orin-bringup-v1` path, and prints the attach banner without changing any existing syscall behavior. Phase 2 is the value-delivery phase but depends on Phase 1 and on the NVIDIA HAL integration sub-PR (out of scope for this change but a soft dependency). Phase 3 can run partially in parallel with Phase 2 (different peripherals, same `attach_stream` API). Phase 4 closes the change once enough drivers are attached to make per-stream-ID telemetry meaningful.

This change can run in parallel with `aarch64-mte-pac-hardening-v1` (CPU-side memory safety, no SMMU touch) and `spec-exec-mitigations-v1` (instruction-level mitigations, no MMU touch). It is logically antecedent to any future GPU bring-up change (`unikernel-orin-gpu-v1`): the GPU HAL must call into `smmu::attach_stream` from day one.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | SMMU500 driver scaffold, boot-time init | ~1 week |
| 2 | GPU stream-ID + `sys_tensor_map_gpu` wiring | ~1-1.5 weeks |
| 3 | Peripheral coverage (host1x, PCIe, USB, SDMMC) | ~1 week |
| 4 | Fault handler + telemetry | ~0.5-1 week |
| **Total** | | **~3-4 weeks** |
