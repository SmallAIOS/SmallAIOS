# persistent-memory-v1

## Summary

Add a memory-tier abstraction to the SmallAIOS tensor pool so that large model weights, KV caches, and warm inference state can be backed by **persistent memory** — concretely, **CXL.mem** devices and **CXL 3.0 pooled / shared memory fabrics** — in addition to the existing DRAM-only path. The pool gains a tier concept (`Hot = DRAM`, `Warm = CXL.mem` / persistent memory, `Cold = NVMe-mapped`), allocation gains an optional `MemTier` preference, and the kernel grows a discovery shim that enumerates persistent-memory regions via ACPI HMAT (Heterogeneous Memory Attribute Table) on x86-64 and via CXL CDAT (Coherent Device Attribute Table) walk on both x86-64 and ARM64. Datacenter scale-out deployments running models that exceed local DRAM, or sharing weight tensors across multiple SmallAIOS instances on a CXL 3.0 fabric, become possible without rebuilding the whole memory subsystem.

This is a Tier 3 scale-out feature. The Jetson Orin production target has no CXL.mem and no persistent-memory devices; the tiering API is a no-op fallback there.

## Why

- **Intel Optane is dead; CXL.mem is the successor.** Intel ended Optane DC Persistent Memory production in 2022. The 2024-2026 industry direction is **CXL.mem** (CXL 1.1/2.0 type-3 devices — memory expanders) and **CXL 3.0** (pooled / shared memory fabrics across multiple hosts). All near-term persistent-memory designs ride the CXL bus, share PCIe physical layer, and present as `system-ram` to the OS with NUMA-distance-style affinity attributes. SmallAIOS skipping Optane entirely and going directly to CXL.mem is the right design call.
- **Large-model deployments need tiered memory.** A 405B-parameter LLM in INT8 is ~400 GB. A 2P server can ship with 1-2 TB of DRAM, but DRAM is expensive ($/GB) and power-hungry (W/GB). CXL.mem expanders deliver 1-4 TB at half the $/GB and a fraction of the power, with latency 1.5-3x local-DRAM (similar to remote-socket access on a 2P system). For inference where the working set is reuse-heavy (model weights stream through compute, KV cache is read-mostly), the latency penalty is hidden by compute time and the cost/density win is large.
- **CXL 3.0 pooled memory enables multi-instance weight sharing.** When two or more SmallAIOS unikernel instances on a single CXL 3.0 fabric share access to a pooled memory region, they can map the same model weights once into pooled CXL and reference them read-only from every instance. This is the industry-standard datacenter scale-out story for inference: weight tensors are amortized across N replicas, KV cache and activations are per-instance and stay in local DRAM. SmallAIOS as a unikernel is uniquely well-positioned for this — no Linux page-cache layer, no userspace mmap dance, just a kernel-level tiered allocator.
- **Tier 3 — not urgent for Jetson.** Jetson Orin has no CXL controller and never will (Tegra234 fixed-function memory controller). Adding tiering to the production critical path while the production target is Orin pays cost with zero benefit. This proposal exists so the design is captured, the spec is reviewed, and the implementation is ready when a CXL-equipped target enters the roadmap.

## Hardware prerequisites

| Hardware class | Examples | What we get |
|----------------|----------|-------------|
| **CXL 1.1 type-3 (host attached memory expander)** | Samsung CMM-D, SK hynix CXL Memory Module, Micron CZ120 | DDR5-class memory at PCIe-bus-attached latencies; presented as a separate NUMA node with a "slow tier" attribute via HMAT/CDAT |
| **CXL 2.0 type-3 (switch attached, single host)** | Astera Labs Leo controller, MemVerge solutions | Same as 1.1, plus hot-add/remove and richer telemetry |
| **CXL 3.0 type-3 (pooled / shared)** | Pre-production silicon (Astera Aries, Marvell Structera) targeting 2025-2027 datacenter rollout | Multi-host shared memory; SmallAIOS instances on the same fabric can map the same region |
| **Host CPU support** | Intel Xeon Sapphire Rapids / Emerald Rapids / Granite Rapids (CXL 1.1/2.0), Intel Xeon Granite Rapids-AP (CXL 3.0), AMD EPYC Genoa / Bergamo (CXL 1.1/2.0), AMD EPYC Turin (CXL 2.0/3.0), Ampere AmpereOne (CXL 1.1+), NVIDIA Grace (CXL 1.1) | Required — CXL controller must be present in the host root complex |
| **Discontinued / out of scope** | Intel Optane DC PMEM (discontinued 2022), NVDIMM-N | We do not target legacy persistent memory; CXL.mem is the forward-looking design |

**Not supported:** Jetson Orin (Tegra234 — no CXL controller), Apple Silicon (proprietary memory architecture, no CXL public path), Raspberry Pi-class SBCs.

## What changes

- **Capability `kernel-mem-tiering` (new):**
  - `MemTier` enum — `Hot`, `Warm`, `Cold`.
  - `MemTierPolicy` — per-allocation: `Required(tier)`, `Preferred(tier)`, `Any` (default).
  - `TensorPool::alloc_with_tier(size, dtype, tier_policy)` — new API. Existing `alloc()` defaults to `Any`.
- **Discovery shim** `kernel/src/mem/tier.rs`:
  - x86-64: ACPI HMAT — System Locality Latency / Bandwidth tables identify persistent-memory devices.
  - CXL CDAT — common path on both x86-64 and ARM64; walks CXL device CDAT structures for memory attribute info.
  - Single-tier fallback: when no HMAT / CDAT is present, every node reports `MemTier::Hot` and the API is a no-op (Jetson Orin path).
- **Tier integration with NUMA:** Tiered memory regions appear as additional NUMA nodes with a tier tag. The pool's free-list becomes `[FreeList; MAX_NUMA_NODES]` indexed by node, with each node carrying its `tier` attribute. The two changes (`numa-aware-tensor-alloc-v1` and this one) compose: NUMA hint says "which node"; tier policy says "which tier of node".
- **CXL 3.0 shared-region support (Phase 2 deliverable):** A `SharedRegion` abstraction representing pooled CXL memory accessible to multiple hosts. Read-only mapping for model weights; explicit coordination protocol for any writable region (no implicit cache coherence assumptions across hosts — CXL 3.0 spec leaves coherence semantics implementation-defined).
- **Documentation**: `docs/persistent-memory.md` covering the tier model, the discovery path, CXL.mem provisioning on a host, and the CXL 3.0 shared-region usage pattern.

## Out of scope

- **Persistence semantics (crash consistency).** Despite the name "persistent memory", we treat the warm tier as a **slow DRAM**, not as a crash-consistent store. We do not implement durable-write barriers, persistence-aware allocator transactions, or anything from the SNIA NVM Programming Model. If a SmallAIOS instance crashes, the warm-tier contents are abandoned just like DRAM. Crash-consistent persistent state belongs in storage (NVMe), not tiered memory.
- **Optane DC PMEM support.** Intel discontinued Optane in 2022. No engineering will be invested in App Direct Mode, Memory Mode, or `pmem` namespace management. The forward-looking design is CXL.mem.
- **Page migration between tiers.** Linux's `damon` / `numa_balancing` migrates hot pages from slow tiers to fast tiers based on access patterns. Out of scope for v1 — we use explicit `tier_policy` hints from the inference scheduler (which knows weight tensors are read-heavy and belong in warm tier; KV cache is read-write and belongs in hot tier). Revisit if profiling shows mis-placement.
- **CXL.cache and CXL.io.** CXL has three protocols on the same wire: `.io` (PCIe-equivalent), `.cache` (accelerator caches host memory), `.mem` (host accesses device memory). Only `.mem` is in scope for this change. `.cache` belongs in a future GPU-accelerator change if/when an accelerator vendor ships CXL-attached silicon.
- **DCD (Dynamic Capacity Device) provisioning.** CXL 3.0 DCD lets the fabric dynamically add/remove memory regions to a host. We treat the topology as static at boot. Hot-add of a tier-1 device is a v2 concern.

## When this becomes important

- **Now (deferred):** Jetson Orin — no CXL, no benefit. Treat as roadmap documentation only.
- **Trigger event 1:** SmallAIOS deployment on a single-host CXL.mem-equipped server running a model whose total weights exceed local DRAM (canonical example: 400B-class LLM, 1 TB DRAM + 2 TB CXL.mem). This unlocks Phase 1 (single-host tiered allocation).
- **Trigger event 2:** SmallAIOS deployment on a CXL 3.0 fabric with multiple unikernel instances sharing weights. This unlocks Phase 2 (`SharedRegion` API). Likely 24-36 months out given CXL 3.0 production silicon timing.
- **Likely horizon for Phase 1:** 12-24 months out, contingent on CXL.mem-equipped server availability in our deployment matrix.

## Effort estimate

| Sub-phase | Scope | Estimate |
|-----------|-------|----------|
| 1 | HMAT + CDAT discovery shim, MemTier types | ~1 week |
| 2 | TensorPool tier-aware alloc API + per-tier counters | ~1 week |
| 3 | Scheduler integration (weight tensors → Warm; KV cache → Hot) | ~1 week |
| 4 | Single-host CXL.mem benchmark + docs | ~1 week |
| 5 | CXL 3.0 `SharedRegion` API + cross-host coordination protocol | ~2 weeks |
| **Total** | | **~5-6 weeks** (Phase 1 alone: ~3-4 weeks) |

Phase 5 only proceeds when CXL 3.0 silicon and a test harness exist.
