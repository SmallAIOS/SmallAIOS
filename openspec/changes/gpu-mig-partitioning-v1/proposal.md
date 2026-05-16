# gpu-mig-partitioning-v1

## Summary

NVIDIA A100 (Ampere, GA100), H100 (Hopper, GH100), and H200 (Hopper-refresh, GH100) datacenter GPUs support **Multi-Instance GPU (MIG)**: a single physical GPU is hardware-partitioned into up to 7 fully isolated GPU instances, each with dedicated SMs, L2 cache slices, DRAM partition, and memory-bandwidth quota. For multi-tenant datacenter inference, this is the foundational primitive that lets multiple small models share a single H100 without interference. This change adds a `arch-nvidia-mig` capability to SmallAIOS that lets a unikernel instance run on a **single MIG partition** rather than the full GPU, gated behind a new Cargo feature on `smallaios-arch-nvidia`.

**Jetson Orin (Ampere GA10B / cc 8.7) does NOT support MIG.** This is explicitly out of scope on the Jetson production target. The MIG capability is purely a datacenter scale-out feature for A100 / H100 / H200 / B100-class hardware. The proposal documents the per-SKU support matrix and adds a runtime check that fails fast with a clear error message when MIG is requested on a non-MIG-capable GPU.

## Why

- **Multi-tenant inference is the datacenter inference shape.** A single H100 80GB has more compute and memory than most inference workloads need. In a datacenter, that GPU is shared — either via time-slicing (MPS), via SR-IOV-style virtualization (vGPU), or via MIG. MIG is the only one of these that provides **hardware-level isolation** — DRAM, L2 cache, and memory bandwidth are physically partitioned, so a misbehaving tenant cannot starve another. For inference serving with strict SLO requirements, MIG is the option.
- **SmallAIOS isolation maps cleanly to MIG.** Each SmallAIOS unikernel instance owns its address space, its tensor pool, its scheduler. Pinning a unikernel instance to a MIG slice gives end-to-end hardware-and-software isolation in a way that a Linux container on a shared GPU cannot. This is the strongest possible isolation story for a multi-tenant inference platform and aligns with the DO-178C / safety-critical positioning of the project.
- **Jetson does not need this.** Jetson Orin's Ampere GA10B is a single-tenant edge accelerator. There is no MIG support in the silicon (NVIDIA explicitly documents this), and a Jetson is rarely deployed multi-tenant. The Jetson container and unikernel paths use the full GPU directly. Adding MIG support to the Jetson code path would be wasted work.
- **The work is bounded and feature-gated.** MIG configuration is done by the system administrator outside SmallAIOS (via `nvidia-smi mig -cgi` on the host before the unikernel runs); SmallAIOS only needs to (a) detect that it has been given a MIG slice rather than a full device, (b) honor the slice's resource limits, (c) report MIG-specific telemetry. The CUDA runtime already transparently respects MIG when given a MIG device UUID. The new code is mostly detection + telemetry + documentation.

## Hardware prerequisites — MIG support matrix

| GPU | Architecture | MIG support | Max instances | Notes |
|-----|--------------|-------------|---------------|-------|
| NVIDIA A100 40GB / 80GB | Ampere GA100 (cc 8.0) | YES | 7 | Original MIG silicon; canonical reference |
| NVIDIA A30 24GB | Ampere GA100 (cc 8.0) | YES | 4 | Same silicon family, half the partitions |
| NVIDIA H100 80GB SXM5 / PCIe | Hopper GH100 (cc 9.0) | YES | 7 | Enhanced MIG (per-instance compute pipelines, isolated NVDEC/NVJPG) |
| NVIDIA H100 NVL (94GB) | Hopper GH100 (cc 9.0) | YES | 7 | Same as H100 |
| NVIDIA H200 141GB | Hopper-refresh GH100 (cc 9.0) | YES | 7 | Same MIG capability as H100 |
| NVIDIA B100 / B200 | Blackwell GB100 (cc 10.0) | YES | 7 (expected) | Pre-production; confirm at silicon release |
| NVIDIA L4 24GB | Ada Lovelace AD104 (cc 8.9) | NO | — | Edge / inference card; no MIG |
| NVIDIA L40 / L40S 48GB | Ada Lovelace AD102 (cc 8.9) | NO | — | Datacenter Ada; no MIG (despite the L-prefix) |
| NVIDIA RTX 6000 Ada | Ada Lovelace AD102 (cc 8.9) | NO | — | Workstation card |
| NVIDIA Jetson Orin (Nano / NX / AGX) | Ampere GA10B (cc 8.7) | **NO** | — | **Out of scope for this change** |
| NVIDIA Jetson Thor | Blackwell GB10B (cc 10.0) | NO (expected) | — | Edge Blackwell; same single-tenant shape as Orin |
| NVIDIA Tesla T4, V100, P-series | Pre-Ampere | NO | — | Pre-MIG silicon |
| NVIDIA GeForce RTX consumer | Various | NO | — | Consumer cards; no MIG firmware path |

**Datacenter Linux prerequisites:**
- NVIDIA driver R450+ for A100, R470+ for A30, R535+ for H100, R550+ for H200.
- `nvidia-smi mig -cgi` (create GPU instance) + `-cci` (create compute instance) configured **before** the SmallAIOS container or VM starts.
- For container deployments: NVIDIA CDI runtime configured to pass a specific MIG device UUID (`MIG-GPU-...`) rather than the full GPU.

## What changes

- **New capability `arch-nvidia-mig`.** Owned by the `smallaios-arch-nvidia` crate, gated behind the new `mig` Cargo feature. When the feature is off (the Jetson default), the MIG code path does not compile in — zero footprint.
- **Runtime detection.** On CUDA context initialization, query `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MIG_MODE)` and `cuDeviceGetUuid` against the device assigned to the unikernel. If the UUID prefix is `MIG-`, classify the device as a MIG slice and store the slice profile (compute capability, SM count, DRAM partition size, L2 size) in the `NvidiaDevice` descriptor.
- **Slice-aware resource budgeting.** The tensor pool's GPU-side allocator honors the slice's DRAM partition size as the hard ceiling. The scheduler's batch-size heuristic respects the slice's SM count. The hybrid CPU-GPU executor's stream count is bounded by the slice's per-instance compute pipeline count (1 for A100 slices, more for H100).
- **MIG-aware telemetry.** The existing `gpu-profile` telemetry path adds `mig.profile`, `mig.gpu_instance_id`, `mig.compute_instance_id`, and `mig.dram_partition_bytes` fields. Per-MIG resource utilization is reported in addition to per-process.
- **Fail-fast on misconfiguration.** If the operator builds with `--features mig` but the assigned device is a non-MIG-capable GPU (Jetson, L4, L40), the unikernel SHALL refuse to start with a clear error pointing at the MIG support matrix. If the operator builds without `--features mig` but the assigned device is a MIG slice, the unikernel runs but logs a warning that it is not making use of MIG-specific telemetry.
- **Documentation** `docs/gpu-mig.md`: support matrix, host-side configuration steps (`nvidia-smi mig` workflow), container-side env-var requirements, troubleshooting, Jetson-explicit-out-of-scope callout.

## Out of scope

- **Multi-MIG-slice within one unikernel instance.** A unikernel instance is bound to a single MIG slice. Spanning multiple slices is not supported — that defeats the isolation purpose of MIG. If a tenant needs more compute than a single MIG slice provides, they should use a larger slice profile (e.g., MIG 4g.40gb instead of 1g.10gb), or the full GPU.
- **Configuring MIG from inside the unikernel.** MIG instance creation requires host root privileges and the NVIDIA driver. SmallAIOS does not invoke `nvidia-smi mig` itself; the operator does, before the unikernel starts. Documenting the workflow is in scope; automating it inside the kernel is not.
- **MPS (Multi-Process Service).** MPS is a software time-slicing approach with weaker isolation. It is orthogonal to MIG and is its own deferred topic (`gpu-mps-coexistence-v1` if/when needed).
- **vGPU (NVIDIA GRID virtualization).** A licensed virtualization product with its own driver stack. Not applicable to the bare-metal / container deployment shapes SmallAIOS targets.
- **Jetson MIG support.** The Jetson silicon does not support MIG. This proposal makes Jetson out-of-scope explicit and adds a runtime check to fail fast if someone enables `--features mig` on a Jetson build.
- **AMD MxGPU / Intel Flex partitioning.** AMD and Intel have analogous partitioning concepts (SR-IOV-based on AMD, "Flex" tiles on Intel data-center GPUs). These would be separate capabilities (`arch-amd-mxgpu-v1`, `arch-intel-flex-v1`) under their respective architecture crates. The MIG capability covers only NVIDIA datacenter GPUs.
- **Dynamic MIG reconfiguration at runtime.** MIG profiles can be reshaped without rebooting on H100+ silicon. SmallAIOS treats the MIG slice as static at boot — reshape requires unikernel restart. Dynamic reshape is a future concern.

## When this becomes important

- **Now (deferred):** Jetson Orin sweet spot — no MIG silicon. Treat as roadmap documentation; do not allocate engineering capacity.
- **Trigger event:** First SmallAIOS deployment on a multi-tenant H100 or A100 cluster where two or more SmallAIOS instances should share a single physical GPU with hardware-level isolation. This is the canonical datacenter scale-out shape; if SmallAIOS sees adoption in datacenter inference services, this becomes an early request.
- **Likely horizon:** 6-18 months out, depending on datacenter scale-out demand. If only single-tenant Jetson and full-GPU x86 deployments materialize, this change can be archived without implementation.

## Effort estimate

| Sub-phase | Scope | Estimate |
|-----------|-------|----------|
| 1 | `mig` feature flag + runtime detection (UUID-prefix + attribute query) | ~0.5 week |
| 2 | Slice-aware resource budgeting (tensor pool DRAM ceiling, scheduler SM count) | ~1 week |
| 3 | MIG-aware telemetry + per-instance utilization metrics | ~1 week |
| 4 | Fail-fast error paths + comprehensive unit tests against a mocked MIG topology | ~0.5 week |
| 5 | Integration test on real A100 or H100 hardware (loaner / cloud rental) + docs | ~1-2 weeks |
| **Total** | | **~4-5 weeks** |
