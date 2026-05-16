# Design — tegra-smmu-isolation-v1

## Goal

Wire Tegra234's System MMU (SMMU500 / arm-mmu-500, SMMUv2-shaped) into the SmallAIOS bare-metal aarch64 path so that **every DMA-capable peripheral on the SoC is hardware-contained**: the GPU, host1x clients, PCIe root complex, USB, and SDMMC each see only the pages the kernel has explicitly mapped into their stream ID. By the end of this change, an exploit or bug in any DMA-capable driver cannot scribble outside the pages that driver was authorized to touch.

Verification: SMMU fault register reads back as "no faults" during a green boot, and a deliberately-broken driver (test-only) triggers a fault that is logged with the correct stream ID, fault address, and syndrome.

## Design decisions

### Decision 1: SMMUv2 (SMMU500), not SMMUv3

| Variant | Tegra234 silicon | Driver complexity | Linux upstream maturity | Why we pick / reject |
|---------|------------------|-------------------|-------------------------|---------------------|
| **SMMUv2 (SMMU500)** | Yes — this is what Orin NX ships | Lower (register-bank model) | `drivers/iommu/arm/arm-smmu/` is mature since ~2014 | **Pick** — matches the silicon and is the simpler driver |
| SMMUv3 | Future Orin variants (AGX Industrial, Thor) | Higher (queue-based command/event interface) | `drivers/iommu/arm/arm-smmu-v3/` is mature since ~2017 | Reject for Phase 1 — not in target silicon |

We design the `Iommu` trait surface so that a future `smmu_v3.rs` can implement it without changing callers. This is the same shape Linux uses (`struct iommu_ops`).

### Decision 2: Identity map first, then enforce

Phase 1 brings up the SMMU in a transient "identity map everything" mode before any peripheral has attached. This is unsafe but lets us validate that the SMMU driver itself works (registers respond, fault handler installs) before we start denying DMA. As soon as `init` returns, the policy flips: **no stream ID is mapped, so no peripheral can DMA**. Each driver then opts in via `attach_stream`.

The alternative — bring up SMMU in deny-everything mode from cold — was rejected because it makes early-boot DRAM training and UEFI hand-off harder to debug. UEFI itself programs the SMMU passthrough during its boot services phase; we don't want to fight that until our own drivers are ready. Flipping to enforced mode after `init` is a single register write (S2CR `Type` field) so the transition is atomic.

### Decision 3: Stream IDs are kernel-constants, not allocated dynamically

The Tegra234 TRM (and matching `tegra234.dtsi` upstream) assigns stream IDs statically to hardware engines:

```
GPU (GA10B)         = 0x07
host1x clients      = 0x01 - 0x06
PCIe controllers    = 0x10 - 0x17
USB XHCI            = 0x18
SDMMC               = 0x20 - 0x21
```

These are baked into silicon — we don't allocate them. The new `arch/aarch64/src/smmu/stream_id.rs` defines them as `const StreamId`s and the type system prevents driver code from passing the wrong one. The Linux `tegra-mc` driver does the same thing in a less type-safe way.

### Decision 4: IOVA pool per stream, not per-tensor

`sys_tensor_map_gpu` returns a `GpuPtr` that is an IOVA. The naive design — "the IOVA equals the physical address" — works but creates collisions when two tensors come from different physical regions but the GPU's IOVA space wraps. The safer design is a per-stream IOVA allocator that hands out non-overlapping IOVAs and installs the SMMU page-table entries to point at the (possibly disjoint) physical pages.

We use a bitmap allocator over a 2-GiB IOVA window per stream, page-aligned. 2 GiB matches Orin NX's typical model-side working set (16 GB total DRAM, ~8 GB available, ~2 GB ML-resident at peak). Larger windows are a follow-up if a model needs it.

### Decision 5: TLB invalidation is a yield point

SMMU TLB invalidation involves writing to TLBIALLNSNH-equivalent registers and polling a status register for completion. Worst case is multiple microseconds on Tegra234. SmallAIOS uses cooperative async scheduling that yields at ONNX operator boundaries (`docs/scheduling-model.md`); SMMU TLB invalidation gets the same treatment — the syscall handler issues the invalidate, then `yield_now().await` while polling status, then resumes when the invalidation completes.

This matters because tensor map/unmap calls happen inside the inference loop, not just at model load.

### Decision 6: No userspace IOMMU API

VFIO-style userspace IOMMU access is rejected. SmallAIOS is a unikernel — there is no userspace in the Linux sense. The "tasks" that call syscalls are still part of the kernel address space (`docs/architecture.md` AMP unikernel model). Exposing SMMU programming directly would let a buggy task install a mapping that lets a peripheral read another task's tensors. The kernel-only model keeps the trusted computing base small and makes the capability checks the sole authorization decision.

## Alternatives considered

### Alt A: Use UEFI's existing SMMU passthrough setup, do nothing in the kernel

**Rejected.** UEFI programs the SMMU permissively during Boot Services so that block I/O and console output work. As soon as we call `ExitBootServices` (which we do in `unikernel-orin-bringup-v1`'s Phase 2e), that state is *retained* — the SMMU continues to permit DMA, and we inherit whatever stream-table state UEFI left. Inheriting permissive state is the worst of all worlds: we look isolated on paper, but a malicious peripheral can still hit any DRAM region. We have to take over SMMU programming explicitly.

### Alt B: Software-only DMA bounce buffers (no SMMU)

**Rejected.** Bounce buffering routes every DMA through a kernel-managed staging buffer, then copies in/out via the CPU. It avoids the SMMU entirely but doubles memory bandwidth and adds memcpy latency on every tensor map. Orin NX has ~100 GB/s LPDDR5 bandwidth and ML workloads saturate it; halving it via bounce buffers is a non-starter for an inference-focused OS. (Linux uses bounce buffers as a fallback on hardware without an IOMMU; we have an IOMMU.)

### Alt C: Per-task SMMU contexts (one context bank per inference task)

**Considered, deferred.** SMMU500 has 16 context banks (CB0-CB15). One per inference task would give task-level DMA isolation in addition to peripheral-level isolation. Deferred because (a) the unikernel-task model is cooperative — tasks already trust each other within the address space — and (b) 16 context banks is not enough for any future scale-out scenario where dozens of inference tasks coexist. We use one context bank per *stream ID*, which is enough for peripheral isolation today; per-task partitioning is a separate follow-up if and when SmallAIOS grows a tenant-isolation model.

### Alt D: ARM SMMU upstream Rust driver

**Considered, no upstream exists.** As of this proposal, there is no `arm-smmu` crate in `std`-or-`no_std` form on crates.io. The TockOS project has an experimental arm-smmu driver but it's MMU400 focused and not in a state to vendor. We port the Linux driver shape to `#![no_std]` Rust, similar to how the existing virtio drivers in `arch/aarch64/src/virtio_blk.rs` were written.

## Risks

### Risk 1: UEFI hand-off race

UEFI may still have outstanding DMA in flight at `ExitBootServices` (e.g., a console flush). If we tear down SMMU passthrough immediately, those in-flight DMAs fault and we see spurious early-boot fault interrupts. Mitigation: the SMMU `init` call quiesces all stream IDs (deny-all, but with a 100ms grace period during which faults are logged as warnings, not errors). After the grace period the policy is enforced.

### Risk 2: Stream-ID assignment drift between L4T / mainline / OpenSUSE Tegra234 device-trees

Different distributions of the Tegra234 device tree (NVIDIA L4T R36.4 vs. upstream Linux 6.6+ vs. OE/Yocto meta-tegra) assign slightly different stream IDs to host1x sub-clients. Mitigation: the `stream_id.rs` constants reference the **Tegra234 TRM** (NVIDIA's authoritative reference), not a particular Linux DTS. Where the TRM is ambiguous, we test on actual hardware (Orin NX 16 GB, the same `nx` host used by `unikernel-orin-bringup-v1`).

### Risk 3: Fault interrupt routing

SMMU500 raises fault interrupts via GICv3 as SPI (shared peripheral interrupts). The GICv3 driver currently in `unikernel-orin-bringup-v1` Phase 2e is bring-up-quality, not fully featured. Mitigation: Phase 4 of this change (fault handler) is sequenced *after* the parent change lands the full GICv3 dispatch. If the parent change's GICv3 driver is incomplete by the time we need it, we add the missing SPI dispatch as a small Phase 4 sub-task.

### Risk 4: Performance regression

SMMU page-table walks are hardware-accelerated but every cold TLB miss costs cycles. Inference workloads that re-map tensors on every batch could see throughput drops. Mitigation: (a) keep IOVA mappings long-lived (allocate-once-per-model-load, not per-batch); (b) benchmark `bench/` cuDNN throughput on Orin before/after SMMU enablement and gate the merge on <5% regression in steady-state; (c) document a `smmu-disable` Cargo feature for non-safety-critical builds that want the raw passthrough performance.

### Risk 5: SMMU500 register-bank discovery

Tegra234's SMMU500 registers sit at MMIO bases that differ between platform variants (Orin Nano / NX / AGX). We read them from the DTB at runtime rather than hard-coding. The DTB parser in `kernel/src/mem/phys.rs` already handles `reg = <...>` properties; we extend it to walk the `arm,mmu-500` compatible node.

## Build/CI surface

- New module `arch/aarch64/src/smmu/{mod.rs, smmu_v2.rs, stream_id.rs, fault.rs}`.
- New Cargo feature `smmu` on `smallaios-arch-aarch64`, off by default initially, default-on for `--features tegra234` builds once Phase 4 lands.
- New CI advisory job `smmu-on-orin-smoke` (self-hosted Orin runner; gated behind the same runner-availability flag as the parent change's Phase 2 promotion).
- New `just smmu-fault-test` recipe that boots the kernel under QEMU with a virt-smmu-v2 model and validates fault handling against a deliberately-broken test driver.
- Module-level acyclicity (the standing `just arch-check` gate) must still pass — `arch/aarch64/src/smmu/` depends only on `kernel/` foundation services (allocator, syscall types), never the other way.

## What this change explicitly does NOT do

- Does not modify any x86_64 IOMMU code (there is none today; that's a separate change).
- Does not change syscall ABI numbers — `sys_tensor_map_gpu` keeps its current number; only the meaning of the returned `GpuPtr` changes from "physical" to "IOVA".
- Does not touch userspace ONNX-rt code paths — the SMMU is invisible above the kernel/HAL boundary.
- Does not add new capabilities beyond `kernel-iommu`; `TensorBuffer:WRITE` + `GpuDevice:EXECUTE` remain the authorization gates.
