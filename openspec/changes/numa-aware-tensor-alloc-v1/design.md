# Design — numa-aware-tensor-alloc-v1

## Goal

Make the `kernel/src/mem/tensor.rs` allocator NUMA-aware on multi-socket x86-64 and multi-die ARM64 servers, while keeping the Jetson Orin single-domain path a zero-cost fall-through. Success is measured as: (a) NUMA topology is correctly discovered and reported on at least one 2P EPYC and one Ampere Altra Max reference host; (b) under a synthetic two-node tensor-allocation workload, ≥90% of allocations land on the requesting thread's home node; (c) the existing Orin single-node benchmark suite shows no regression (within ±2% of the v0 baseline).

## Alternatives considered

### A1. Do nothing — let Linux-style userspace NUMA libraries handle it

**Rejected.** SmallAIOS is a unikernel; there is no userspace `libnuma` to fall back to. The tensor pool is the only allocator on the inference fast path. If the pool ignores NUMA, every inference op pays the cross-socket tax.

### A2. Implement first-touch placement only (default Linux behavior)

**Considered, partially adopted as fallback.** First-touch (memory is allocated on the node where it is first written) is the cheapest NUMA-aware policy and matches Linux's default. We adopt first-touch as the **fallback when no hint is supplied**, but we add the explicit `numa_hint` API because the inference scheduler knows the consumer node up front — it can give a better placement decision than first-touch's lazy heuristic. First-touch alone leaves performance on the table when the producing thread (e.g., a model-loader) writes the weight tensor on node 0 but a different node's inference thread is going to consume it.

### A3. Build a richer "NUMA policy" object (MPOL_BIND, MPOL_PREFERRED, MPOL_INTERLEAVE)

**Rejected for v1.** Linux's `mbind` policy framework is general but adds API surface and code paths we cannot validate against the Jetson Orin target. We ship the minimal `Option<NumaNodeId>` hint (which maps to "preferred" semantics — try the hinted node, fall back to any node under pressure). If interleave or strict-bind becomes necessary, extend the API in a follow-up change.

### A4. Make NUMA awareness a compile-time feature (`--features numa`)

**Rejected.** Single-socket builds would have to know which feature flags are appropriate; multi-socket production builds would have to recompile. The runtime cost of a `Option<NumaNodeId>` check that resolves to "always node 0 because single-node topology" is one branch on a path that already dispatches on size class — well below the noise floor. Runtime detection with single-node fallback is the right shape.

## Topology discovery

### x86-64 — ACPI SRAT

The System Resource Affinity Table (SRAT) is the canonical NUMA topology source on x86-64. SmallAIOS already parses the RSDP / RSDT at boot for early hardware enumeration; SRAT parsing adds:

- **Memory Affinity structures** (type 1): map physical address ranges to proximity domains.
- **Processor Local APIC/SAPIC Affinity structures** (type 0): map APIC IDs to proximity domains.

The proximity domain → `NumaNodeId` mapping is a `BTreeMap` populated at boot. ~150 LOC of new code in `kernel/src/acpi/srat.rs`. We do not parse SLIT (System Locality Information Table) in v1 — the distance matrix is interesting but not required for the "prefer my home node" policy.

### ARM64 — Device Tree

The Linux device-tree binding `numa-node-id` on `cpu@N` and `memory@N` nodes is the standard ARM64 NUMA source. Ampere Altra and Grace both expose this when configured for multi-node operation. We walk the DTB once at boot, populate the same `NumaNodeId` mapping. ~80 LOC in `kernel/src/dt/numa.rs`.

### Single-node fallback

When neither path yields useful data (the Jetson Orin case — no SRAT, no `numa-node-id` properties), the allocator constructs `NumaTopology::single_node()` containing one node spanning all RAM, all CPUs. Every `numa_hint` resolves trivially to node 0. This is the **default** path and the only path exercised by the existing CI matrix.

## Tensor pool extension

```rust
// Existing API (kept verbatim, calls through to alloc_with_hint(.., None))
pub fn alloc(size: usize, dtype: DType) -> Result<TensorBuf, MemError>;

// New API
pub fn alloc_with_hint(
    size: usize,
    dtype: DType,
    numa_hint: Option<NumaNodeId>,
) -> Result<TensorBuf, MemError>;

// Topology accessor
pub fn topology() -> &'static NumaTopology;
```

Internally, the pool's free-list becomes `[FreeList; MAX_NUMA_NODES]` where `MAX_NUMA_NODES` is a `const` chosen at workspace level (`16` for v1 — covers every system in the hardware-prerequisite table). Allocation flow:

1. If `numa_hint = Some(n)`, try `nodes[n].free_list` first.
2. On miss, try any node with free capacity in this size class.
3. On total miss, fall back to the underlying page allocator (existing path).

The page allocator itself is **not** modified in v1. We accept that newly-allocated pages may not be on the hinted node — they are allocated wherever the kernel's page allocator finds free memory, and the NUMA hint affects only the free-list bucket choice. This is a known limitation; a future change can extend the page allocator with a per-node freelist if profiling shows the limitation matters.

### Per-node accounting

The pool exports atomic counters per node:

- `numa.node[i].alloc_local` — alloc on hinted node, hit in local free-list.
- `numa.node[i].alloc_remote` — alloc on hinted node, satisfied from a different node's free-list.
- `numa.node[i].alloc_unhinted` — alloc with no hint.

These feed the same kernel telemetry path that exports cooperative-scheduler stats today.

## Scheduler integration

The cooperative scheduler runs inference ops on cores 1..N. Each core is statically assigned to a NUMA node at boot via the topology discovery shim. The scheduler exposes a per-thread `home_node: NumaNodeId` and passes it as the hint on every tensor allocation made on behalf of that thread.

Core 0 (System/IPC) is bound to node 0 by convention. Cores assigned to a non-zero node use that node's id; on single-node systems every core gets node 0.

## Observability

A new file `/proc/smallaios/numa` (or the unikernel-equivalent telemetry endpoint) reports:

```
nodes: 2
node 0: cpus=0,1,2,3 mem=64GB alloc_local=12345 alloc_remote=42 alloc_unhinted=8
node 1: cpus=4,5,6,7 mem=64GB alloc_local=11987 alloc_remote=35 alloc_unhinted=4
```

Local/remote ratio is the headline metric for tuning.

## What this change explicitly does NOT do

- Does not modify the page allocator (only the tensor free-list layer).
- Does not add NUMA awareness to the `net` crate's packet pool, the `ipc` crate's ring buffers, or the cooperative scheduler's task queue. Those are separate follow-ups if profiling shows them mattering.
- Does not implement page migration, interleave, or strict-bind policies.
- Does not change behavior on single-node systems beyond the addition of zero-cost dispatch through `Option::None`.
- Does not require new CI hardware. The single-node CI path continues to validate the default; multi-socket validation is a manual / scheduled-runner concern.
