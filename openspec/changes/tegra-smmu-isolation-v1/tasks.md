# Tasks — tegra-smmu-isolation-v1

## 0. Hardware verification + reference reading (prereq)

- [ ] 0.1 Capture and record the SMMU MMIO bases from a stock JetPack 6 / L4T R36.4 Orin NX boot: `cat /sys/firmware/devicetree/base/smmu*/reg | hexdump`, `dmesg | grep -i smmu`, `cat /proc/iomem | grep -i smmu`. Paste in the change PR description.
- [ ] 0.2 Read Tegra234 TRM section "System MMU (ARM SMMU500)" — confirm register bank layout (GR0, GR1, SMR, S2CR, CB0-CB15), stream-ID assignments for every DMA-capable engine, and fault syndrome encoding.
- [ ] 0.3 Read upstream Linux `drivers/iommu/arm/arm-smmu/arm-smmu.c` + `arm-smmu-impl.c` + Tegra-specific glue in `drivers/iommu/tegra-smmu.c`. Document which initialization steps are silicon-mandated vs. Linux-policy.
- [ ] 0.4 Confirm SMMU500 fault routing GICv3 SPI number by inspecting the Orin NX device tree and matching against the GICv3 driver landing in `unikernel-orin-bringup-v1` Phase 2e.

## 1. Phase 1 — SMMU500 driver scaffold

### 1a. Module scaffolding

- [ ] 1.1 Create `arch/aarch64/src/smmu/mod.rs` with the public API surface: `Iommu` trait + `pub fn init(dtb: *const u8)` + `pub fn attach_stream(stream_id, root_pa, perms)` + `pub fn detach_stream(stream_id)` + `pub fn flush_tlb(stream_id)`.
- [ ] 1.2 Create `arch/aarch64/src/smmu/stream_id.rs` defining the `StreamId(u16)` newtype and Tegra234 constants (GPU, host1x clients, PCIe, USB, SDMMC) from the TRM, with doc-comments citing the TRM section for each constant.
- [ ] 1.3 Add `smmu` Cargo feature to `arch/aarch64/Cargo.toml`. Off by default. Doc-comment distinguishes it from the parent `tegra234` feature (this one toggles the IOMMU driver; `tegra234` toggles the whole BSP).
- [ ] 1.4 Gate the new module in `arch/aarch64/src/lib.rs` behind `#[cfg(feature = "smmu")]`.

### 1b. SMMUv2 register driver

- [ ] 1.5 Create `arch/aarch64/src/smmu/regs.rs` — typed MMIO accessor structs for `GR0`, `GR1`, `SMR[0..=N]`, `S2CR[0..=N]`, `CB[0..=15]` per the SMMUv2 spec. Use `volatile-register` style accessors (already used in `arch/aarch64/src/uart.rs`).
- [ ] 1.6 Create `arch/aarch64/src/smmu/smmu_v2.rs` implementing the `Iommu` trait. `init`: probe `GR0::IDR0/IDR1/IDR2` for capability detection (number of context banks, S2 supported, etc.); program a transient identity-map state; install fault-handler interrupt vector via `interrupts.rs`.
- [ ] 1.7 Implement `attach_stream` — pick a free context bank (CB), program SMR with the stream ID, point S2CR at the chosen CB, install the page-table root (a stage-1 walk with the kernel's existing `paging.rs` 4 KiB-page format), set CB context attributes (`TCR`, `SCTLR`, `TTBR0`).
- [ ] 1.8 Implement `detach_stream` and `flush_tlb` — TLBIALL on the context bank, poll status register, return when complete. Cooperative-yield while polling per design decision 5.
- [ ] 1.9 Wire `smmu::init(dtb)` into the boot sequence (`arch/aarch64/src/main.rs` or `main_uefi.rs`) after `mem::init` and before any DMA-driver init.

### 1c. Unit tests + QEMU virt-smmu smoke

- [ ] 1.10 Add unit tests under `arch/aarch64/src/smmu/tests.rs` exercising the register-level state machine (mocked MMIO via `volatile-register` test shims).
- [ ] 1.11 Add a `just smmu-qemu-smoke` recipe that boots SmallAIOS under `qemu-system-aarch64 -M virt,iommu=smmuv2 -cpu cortex-a72 ...` (QEMU's virt-smmu model is SMMUv2-compatible) and asserts the boot banner includes `[smmu] SMMU attached`.
- [ ] 1.12 Add an `smmu-qemu-smoke` CI job (advisory initially) that runs the recipe.

## 2. Phase 2 — GPU stream-ID enforcement

### 2a. IOVA allocator

- [ ] 2.1 Create `arch/aarch64/src/smmu/iova.rs` — per-stream bitmap IOVA allocator with a 2-GiB window per stream, 4 KiB page granularity. API: `alloc_iova(stream, n_pages, align) -> Iova`, `free_iova(stream, iova, n_pages)`.
- [ ] 2.2 Unit-test the IOVA allocator: random alloc/free patterns, alignment edge cases, exhaustion behavior.

### 2b. sys_tensor_map_gpu wiring

- [ ] 2.3 Modify `kernel/src/syscall/memory.rs::sys_tensor_map_gpu` to, instead of returning `NotSupported`, call into a HAL-level `gpu_map_tensor` that: (a) resolves the tensor handle to its physical pages, (b) allocates an IOVA range from the GPU stream's allocator, (c) calls `smmu::map_pages(StreamId::GPU, iova, &phys_pages, perms)`, (d) flushes the GPU stream TLB, (e) returns the IOVA as `GpuPtr`. Gated by a new `gpu` feature on `smallaios-kernel` so non-GPU builds compile unchanged.
- [ ] 2.4 Add `smmu::map_pages` + `smmu::unmap_pages` (page-table edit helpers — the stage-1 walk happens in hardware, but the kernel owns the page-table memory).
- [ ] 2.5 Modify `sys_tensor_unmap_gpu` symmetrically: unmap pages, free IOVA, flush TLB.
- [ ] 2.6 Add unit + integration tests covering capability rejection (no `EXECUTE` on `GpuDevice` -> error) and IOVA non-collision (two tensors get non-overlapping IOVA ranges).

### 2c. kernel-iommu capability spec

- [ ] 2.7 Add `ResourceType::IommuStream` to `kernel/src/cap.rs` (or wherever `ResourceType` lives). Permissions: `BIND` (attach a stream), `WRITE` (modify mappings under an attached stream).
- [ ] 2.8 Document in the new `kernel-iommu` capability spec that only the syscall handler and HAL driver code may construct an `IommuStream` capability — never user-graph nodes.

## 3. Phase 3 — Peripheral coverage

- [ ] 3.1 host1x v6 — attach `StreamId::HOST1X_*` in the host1x driver's `init` path (if/when the driver lands). Update `arch/nvidia/src/tegra/` notes.
- [ ] 3.2 PCIe root complex — attach `StreamId::PCIE_*` in the PCIe driver's `init`. (Driver may not exist yet; add a TODO comment and a unit-test stub.)
- [ ] 3.3 USB XHCI — attach `StreamId::USB` from the `smallaios-usb` crate's controller init. This is the highest-impact non-GPU stream: USB is the easiest physical attack surface.
- [ ] 3.4 SDMMC — attach `StreamId::SDMMC_*` from the eMMC/SD driver init.
- [ ] 3.5 Document in `docs/architecture.md` (Layer 2 section) the SMMU-enforced DMA isolation model with a table of stream-ID → driver mappings.

## 4. Phase 4 — Fault handler + telemetry

- [ ] 4.1 Implement `arch/aarch64/src/smmu/fault.rs` — handler for SMMU global fault SPI. Reads FSR, FAR, syndrome registers; produces a structured `SmmuFault` value.
- [ ] 4.2 Wire the handler into the GICv3 dispatch (depends on `unikernel-orin-bringup-v1` Phase 2e landing first).
- [ ] 4.3 Add per-stream fault counters surfaced via the same telemetry path as the `telemetry-otel-export-v1` change (or, if that hasn't landed, via the existing console-log path).
- [ ] 4.4 Add a `smmu-fatal-fault` Cargo feature that turns any SMMU fault into a kernel panic — opt-in for safety-critical builds.
- [ ] 4.5 Test: write a deliberately-broken in-kernel test driver that attempts a DMA outside its mapped region; assert the fault handler fires, the counter increments, and (with `smmu-fatal-fault` on) the kernel panics with a structured message.

## 5. Docs

- [ ] 5.1 Add `docs/smmu-isolation.md` covering: the IOMMU model, stream-ID table, how a new DMA-capable driver attaches itself, fault triage, the per-stream telemetry counters, and the `smmu-disable` / `smmu-fatal-fault` Cargo features.
- [ ] 5.2 Update `docs/architecture.md` Layer 2 section to call out SMMU-enforced isolation.
- [ ] 5.3 Update `CLAUDE.md` to note SMMU isolation is enabled on `tegra234` builds.

## 6. Verify + archive

- [ ] 6.1 Run `openspec validate tegra-smmu-isolation-v1 --strict` after all phases land.
- [ ] 6.2 Capture on-Orin-NX evidence: boot output showing `[smmu] SMMU500 attached, 5 stream IDs reserved`, fault-test output, telemetry counter snapshot. Paste in the final PR description.
- [ ] 6.3 Open archive PR moving the change to `openspec/changes/archive/YYYY-MM-DD-tegra-smmu-isolation-v1` and sync the spec deltas to main specs.
