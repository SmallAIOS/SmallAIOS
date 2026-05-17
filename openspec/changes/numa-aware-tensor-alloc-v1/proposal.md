# numa-aware-tensor-alloc-v1

## Summary

SmallAIOS today assumes a single memory domain. The Jetson Orin sweet spot (Orin NX 16 GB, single-die Tegra234, single LPDDR5 controller) is a uniform memory machine and the existing tensor pool at `kernel/src/mem/tensor.rs` correctly ignores NUMA topology. This change introduces NUMA-aware tensor allocation as a Tier 3 scale-out feature for **multi-socket x86-64** (dual/quad EPYC Genoa, dual Xeon Sapphire Rapids) and **multi-die ARM64 server** (Ampere Altra Max, NVIDIA Grace, Grace-Hopper Superchip) deployments running models that exceed a single NUMA node's local DRAM (>100 GB working set is the canonical case). Tensor allocations SHOULD prefer the NUMA node whose CPU cores or PCIe root complex consume them; cross-socket DRAM traffic measurably costs latency (UPI / Infinity Fabric / NVLink-C2C round-trip) and energy.

The kernel-side change is bounded: extend the tensor pool API with an optional `numa_hint: Option<NumaNodeId>` parameter, surface a per-node free-list, add a topology-discovery shim that reads ACPI SRAT / `nodeX/cpulist` on x86-64 and `numa-node-id` device-tree properties on ARM64, and wire a thread-locality bias into the inference scheduler. No ABI break — existing callers pass `None` and get the current behavior. Capability `kernel-mem` is extended (not replaced).

## Why

- **Today's target is uniform.** Tegra Orin is a single-socket SoC; there is no NUMA topology to honor. Adding NUMA awareness to the tensor pool while the production target is a single-die system pays development cost with zero observable benefit on the production hardware. This proposal explicitly defers the kernel work until a multi-socket runner enters the test matrix.
- **Scale-out demand is the gate.** When SmallAIOS is deployed as a unikernel on a dual-socket EPYC server hosting a 70B-parameter LLM split across both NUMA nodes' DRAM, cross-socket KV-cache traffic dominates the latency budget — measured on Linux at 1.5-3x local-DRAM latency for naive allocations. At that point, NUMA awareness is no longer cosmetic; it is the difference between fitting in a service-level objective and missing it. Until that deployment shape appears in the roadmap, this is a future-facing investment.
- **The change is bounded, not architectural.** The tensor pool already separates allocation from consumption (the cooperative scheduler picks ops, then asks the pool for buffers). Adding a node-affinity hint is a parameter addition, not a redesign. The existing single-domain code path remains the default; the new path is opt-in via a hint argument.
- **It unlocks future GPU-aware work.** Multi-GPU systems (8x H100 in a single chassis) already have an analogous problem: each GPU's PCIe root complex lives on one socket, and CPU-side memory used for staging that GPU's batches SHOULD be on the matching node. The `numa_hint` API generalizes to this case — a "preferred socket for this GPU's host buffers" is just another NUMA hint. The MIG change (`gpu-mig-partitioning-v1`) and this change compose cleanly without coupling.

## Hardware prerequisites

| Class | Examples | NUMA topology |
|-------|----------|----------------|
| Multi-socket x86-64 | 2P AMD EPYC Genoa/Bergamo, 2P/4P Intel Xeon Sapphire Rapids / Emerald Rapids | 1 node per socket (or per CCD for "NPS4" mode on EPYC) |
| Multi-die ARM64 | Ampere Altra Max (128c, 1-2 sockets), NVIDIA Grace (72c, single CPU but with multi-die LPDDR5X memory controllers), Grace-Hopper GH200 | 1-2 nodes per socket when firmware exposes them; some Grace-class systems may appear as a single UMA node to the OS despite internal locality, so this proposal follows discovered topology (SRAT/DT) and uses single-node fallback when only one node is visible |
| Single-socket x86-64 / single-die ARM64 | Jetson Orin, Apple M-series, Ryzen 7000, Xeon Bronze single-socket | 1 node — NUMA hint is a no-op |

The Jetson Orin production target falls in the last row. NUMA is observable but not exploitable on it.

## What changes

- **Capability `kernel-mem`** extends with NUMA-aware allocation:
  - `TensorPool::alloc_with_hint(size, dtype, numa_hint: Option<NumaNodeId>)` — new API; `numa_hint = None` matches today's behavior.
  - `TensorPool::topology() -> &NumaTopology` — accessor for discovered topology (read-only).
  - Per-node free-lists inside the pool; cross-node steal allowed under memory pressure.
- **Topology discovery shim** lives in `kernel/src/mem/numa.rs`:
  - x86-64: parse ACPI SRAT (System Resource Affinity Table) at boot.
  - ARM64: walk the device tree for `numa-node-id` properties on memory and CPU nodes.
  - Single-node fallback: synthesize `NumaTopology::single_node()` when neither path yields useful data (this is the Jetson Orin path).
- **Scheduler integration**:
  - Inference threads gain a "home node" — Core 0 (System/IPC) on node 0; data-parallel cores 1..N bound to their physical node.
  - The cooperative scheduler passes the current thread's home node as a hint into the tensor pool on every allocation.
- **Documentation**: `docs/numa-tensor-alloc.md` covering the topology, when to opt in, and observability (per-node hit/miss counters in `/proc/smallaios/numa` analog).

## Out of scope

- **Memory migration / page migration on demand.** Linux's `numa_balance` migrates pages between nodes based on access patterns. SmallAIOS will not implement this — we believe explicit hints are sufficient for the data-parallel inference workload (ops have known input/output shapes; we know up front where they should land). Revisit if profiling shows hot-page mis-placement after this lands.
- **Interleave policy.** Linux's `MPOL_INTERLEAVE` round-robins pages across nodes for bandwidth-bound workloads. Out of scope for the v1; if interleave benefits emerge for specific GEMM sizes, add it as a future change.
- **CPU pinning APIs.** Thread-to-core binding is already handled by the AMP scheduler; this change only adds the home-node bias, not a full `sched_setaffinity` equivalent.
- **NUMA across the QUIC stack.** The networking crate's buffer pool ignores NUMA. Adding NIC-RX-queue → home-node affinity is a sensible follow-up but is its own change (`net-numa-affinity-v1` if/when it lands).
- **HMM / heterogeneous memory.** Tier-aware allocation across DRAM and PMem is covered by the sibling `persistent-memory-v1` change. The two compose: PMem appears as additional NUMA nodes with a "slow tier" attribute.

## When this becomes important

- **Now (deferred):** Jetson Orin single-socket — zero benefit. Document the API surface and reference design so the implementation is ready when needed; do not allocate engineering capacity.
- **Trigger event:** First scale-out customer / internal deployment that uses a 2P EPYC / Xeon SP server with a model >50 GB. At that point, schedule the implementation: ~3-4 weeks for the kernel work, plus a benchmark harness to quantify the cross-socket latency reduction.
- **Likely horizon:** 12-18 months out, contingent on SmallAIOS adoption on multi-socket hosts. If only Jetson + single-socket cloud (e.g., Graviton, c7g) deployments materialize, this change can be archived without implementation and reopened later.

## Effort estimate

| Sub-phase | Scope | Estimate |
|-----------|-------|----------|
| 1 | Topology discovery shim (x86 SRAT + ARM DT walk + single-node fallback) | ~1 week |
| 2 | Tensor pool API extension + per-node free-lists | ~1 week |
| 3 | Scheduler home-node bias + observability counters | ~1 week |
| 4 | Multi-socket benchmark + docs + CI matrix expansion | ~0.5-1 week |
| **Total** | | **~3-4 weeks** |
