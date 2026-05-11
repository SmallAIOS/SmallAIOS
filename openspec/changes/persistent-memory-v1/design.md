# Design — persistent-memory-v1

## Goal

Add a memory-tier abstraction on top of the existing tensor pool so that **CXL.mem** (host-attached memory expanders, CXL 1.1+) and **CXL 3.0 pooled / shared memory** can back large model weights and warm caches, while preserving zero overhead on the Jetson Orin single-tier production target.

Success: (a) on a CXL.mem-equipped reference host (e.g., Sapphire Rapids + Astera Leo, or EPYC Genoa + Samsung CMM-D), the kernel discovers the warm-tier node via HMAT/CDAT and the inference scheduler places weight tensors on the warm tier and KV caches on the hot tier; (b) on a CXL 3.0 fabric reference host (when silicon is available), two SmallAIOS instances on the same fabric map the same `SharedRegion` for weights read-only; (c) on Jetson Orin (no CXL), the same code path falls through to single-tier behavior with zero measurable regression on the existing tensor-pool benchmarks.

## Alternatives considered

### A1. Treat persistent memory as a block device, not as memory

**Rejected.** The pmem-as-block-device path (with a filesystem on top, e.g., ext4-DAX or XFS-DAX) is a Linux legacy from the Optane era. CXL.mem is presented to the OS as system RAM, not as a block device — the controller emits `EFI_MEMORY_SP` (Special-Purpose Memory) attributes that tell the OS "this is memory, but a different tier". Block-device emulation throws away the memory-mapped programming model that CXL.mem is designed around, and the inference fast path is already memory-mapped tensor access — exposing it via `read()` / `write()` would defeat the purpose.

### A2. Use Linux's `memhotplug` / `kmem` driver model

**Not applicable.** SmallAIOS is a unikernel. There is no Linux `kmem` driver or hotplug subsystem. We must own the discovery and integration ourselves. The Linux model does, however, validate the design point: tiered memory appears as additional NUMA nodes with a `target` attribute. We follow the same shape in the kernel-side `NumaTopology`.

### A3. Implement persistent / crash-consistent semantics now

**Rejected.** Crash consistency in the SNIA NVM Programming Model sense (`pmem_persist`, `pmem_flush`, transactional allocators à la libpmemobj) is a deep rabbit hole that adds significant complexity to the allocator and assumes a use case (in-memory databases) we are not targeting. SmallAIOS uses CXL.mem as **slow DRAM, not as durable store**. If a crash-consistent state is needed, it goes in NVMe storage via the existing filesystem path. This decision is reversible: if a future workload demands persistence, we can layer a transactional allocator on top of the tier API; the API itself does not preclude it.

### A4. Build a single, unified "any memory" allocator with no tier concept

**Rejected.** Without tiering, the allocator has two options: (a) treat all memory the same (puts model weights in DRAM, defeats the purpose of CXL.mem); (b) auto-migrate based on access patterns (Linux's `numa_balance` — adds non-trivial complexity, requires page-fault-driven migration which the unikernel does not have today). Explicit tier hints are cheaper and more predictable. The inference scheduler already knows which tensors are weights (read-only after load, large, reuse-heavy) and which are activations / KV cache (read-write, small per-op, latency-sensitive) — passing that signal to the allocator is a 1-line change at each call site.

### A5. CXL 3.0 shared memory as a transparent shared-page-table feature

**Rejected.** CXL 3.0 leaves coherence semantics implementation-defined; some fabrics expose hardware-coherent shared regions, others require software-managed coherence. SmallAIOS cannot assume the strong model. We expose `SharedRegion` as **explicit, advisory** — read-only model-weight sharing is the only blessed use case; writable shared state requires application-level coordination (which is out of scope for this change).

## Discovery shim

### x86-64 — ACPI HMAT

The Heterogeneous Memory Attribute Table (ACPI 6.3+) is the canonical CXL.mem discovery source on x86-64. Key sub-structures:

- **Memory Proximity Domain Attributes** (type 0): flags indicate whether a proximity domain is "memory side cache" or "memory only", plus reservation hints.
- **System Locality Latency and Bandwidth Information** (type 1): per-(initiator, target) latency / bandwidth matrix. Higher latency + lower bandwidth = warm tier.
- **Memory Side Cache Information** (type 2): not directly relevant for tier classification but useful for future hardware acceleration.

Classification rule for v1: a memory proximity domain is **Warm tier** if its access latency from any local CPU exceeds 2x the lowest-latency domain's reciprocal access latency. This is the same heuristic Linux uses for `tier_higher_rank` ordering in `mm/memory-tiers.c`. ~200 LOC of new code in `kernel/src/acpi/hmat.rs`.

### CXL CDAT (both x86-64 and ARM64)

The Coherent Device Attribute Table is published by each CXL device via the DOE (Data Object Exchange) mailbox. For CXL.mem devices, CDAT carries:

- **Device Scoped Memory Affinity Structure (DSMAS):** range + proximity domain.
- **Device Scoped Latency and Bandwidth Information Structure (DSLBIS):** per-range latency / bandwidth.

We walk the PCI / CXL bus at boot, query CDAT via DOE, and merge the data with HMAT (on x86-64) or use CDAT alone (on ARM64 where HMAT may be absent). ~250 LOC in `kernel/src/cxl/cdat.rs`. The CXL bus walker reuses the existing PCIe enumeration code in the `bus` crate.

### Single-tier fallback

When neither HMAT nor CDAT yields any tier data — Jetson Orin, single-socket x86-64 desktop without CXL, ARM64 servers without `numa-node-id` for CXL devices — the discovery shim synthesizes a single-tier topology where every node reports `MemTier::Hot`. The tier API becomes a no-op; `Preferred(Warm)` resolves to "any node" since no warm node exists.

## Allocation API

```rust
pub enum MemTier {
    Hot,    // Local DRAM
    Warm,   // CXL.mem / persistent memory
    Cold,   // NVMe-mapped (future)
}

pub enum MemTierPolicy {
    Required(MemTier),  // Fail if no node of that tier has capacity
    Preferred(MemTier), // Try that tier; fall back to any tier
    Any,                // Default; identical to v0 behavior
}

// New API
pub fn alloc_with_tier(
    size: usize,
    dtype: DType,
    tier_policy: MemTierPolicy,
) -> Result<TensorBuf, MemError>;

// Compose with NUMA hint from numa-aware-tensor-alloc-v1
pub fn alloc_full(
    size: usize,
    dtype: DType,
    numa_hint: Option<NumaNodeId>,
    tier_policy: MemTierPolicy,
) -> Result<TensorBuf, MemError>;
```

The two-knob API (NUMA hint + tier policy) is the composition of this change with `numa-aware-tensor-alloc-v1`. The NUMA hint says "which node"; the tier policy says "which tier of node". They are orthogonal: a `Preferred(Warm) + Some(node=3)` allocation tries node 3 first if it is a warm tier, otherwise tries any warm node, otherwise any node.

## Scheduler integration

The inference scheduler knows the lifetime and access pattern of each tensor at op-dispatch time. Default placement policy:

- **Model weights** (loaded once at boot, read-only thereafter): `Preferred(Warm)`. Large, reuse-heavy, latency-tolerant because compute time hides the access cost.
- **KV cache** (read-write per token, latency-critical): `Required(Hot)`. Falls back to alloc failure if hot tier is exhausted; do not silently put KV cache in CXL.mem.
- **Activations / intermediates** (short-lived, latency-critical): `Required(Hot)`.
- **Diagnostic buffers, logs, telemetry**: `Preferred(Warm)`. Same rationale as weights — large, read-mostly.

These defaults are documented and configurable via `SMALLAIOS_TIER_POLICY` environment variable or a kernel boot argument.

## CXL 3.0 SharedRegion (Phase 2)

```rust
pub struct SharedRegion {
    fabric_id: CxlFabricId,
    region_id: u64,
    base: PhysAddr,
    len: usize,
    access: AccessMode,  // ReadOnly or ReadWrite (with caveats)
}

impl SharedRegion {
    pub fn map_read_only(spec: SharedRegionSpec) -> Result<Self, CxlError>;
    // No map_writable in v1; writable shared regions require
    // application-level coordination protocols out of scope here.
}
```

The model-loading code looks up a shared-region spec from a pre-provisioned config (CXL fabric manager publishes region ids; the SmallAIOS instance reads the id from its env / config). On `map_read_only`, the kernel verifies (a) the region exists on the fabric, (b) the local node has read permission, (c) the region size matches the expected model size, and maps it into the tensor-pool address space tagged `MemTier::Warm` + `shared = true`. Reads from this region behave like reads from any warm-tier address.

Cache coherence across hosts is **explicitly not assumed**. A v1 read-only region is safe because no host writes after the model-load epoch. Future writable shared regions will need application-managed coherence (e.g., epoch-based reload signaling).

## Observability

Telemetry endpoint extends `/proc/smallaios/numa` (from `numa-aware-tensor-alloc-v1`) with a `tier` column per node and per-tier counters:

```
tiers: hot warm
tier hot:  total=64GB used=42GB alloc_required=12345 alloc_preferred=89
tier warm: total=512GB used=380GB alloc_required=0 alloc_preferred=7654

node 0: tier=hot cpus=0,1,2,3 mem=32GB alloc_local=8000 alloc_remote=12
node 1: tier=hot cpus=4,5,6,7 mem=32GB alloc_local=7800 alloc_remote=15
node 2: tier=warm cxl_dev=cxl0 mem=256GB alloc_local=3800
node 3: tier=warm cxl_dev=cxl1 mem=256GB alloc_local=3854
```

## What this change explicitly does NOT do

- Does not implement persistent / crash-consistent semantics.
- Does not target Optane DC PMEM (discontinued hardware).
- Does not implement page migration between tiers.
- Does not assume CXL 3.0 hardware coherence for writable shared regions.
- Does not enable hot-add / hot-remove of CXL devices (static topology at boot for v1).
- Does not modify the existing single-tier path's behavior on Jetson Orin or other CXL-less hosts.
- Does not extend the `net` crate, the `ipc` crate, or any non-`kernel-mem` capability with tier awareness in this change.
