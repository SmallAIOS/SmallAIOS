# Tasks — numa-aware-tensor-alloc-v1

> **Status: Future-facing.** No work has started. This change is a roadmap document for scale-out / multi-socket deployments; the Jetson Orin production target does not exercise NUMA. Tasks are sequenced for when scale-out demand triggers implementation.

## 0. Trigger conditions (review before starting)

- [ ] 0.1 Confirm at least one production / customer deployment target is multi-socket (2P+ x86-64 or multi-die ARM64 server). If only single-socket targets exist, defer this change indefinitely.
- [ ] 0.2 Confirm CI matrix can be extended with at least one multi-socket runner (self-hosted; GitHub-hosted runners are single-socket).
- [ ] 0.3 Capture a baseline on the Jetson Orin single-node benchmark (current tensor-pool alloc/free latency + per-op tensor throughput). This is the regression guard for the implementation phase.

## 1. Phase 1 — Topology discovery shim

### 1a. x86-64 SRAT parser

- [ ] 1.1 Add `kernel/src/acpi/srat.rs` parsing SRAT memory-affinity (type 1) and processor local APIC/SAPIC affinity (type 0) structures. Use the existing ACPI table-walker.
- [ ] 1.2 Construct a `NumaTopology` containing `nodes: [NumaNode; N]` where each node holds `cpu_ids: BitSet<256>`, `mem_ranges: Vec<PhysAddrRange>`.
- [ ] 1.3 Unit-test the parser against captured SRAT blobs from 2P EPYC Genoa and 2P Xeon Sapphire Rapids (committed under `kernel/src/acpi/test-data/`).

### 1b. ARM64 device-tree walk

- [ ] 1.4 Add `kernel/src/dt/numa.rs` walking the device tree for `numa-node-id` properties on `cpu@*` and `memory@*` nodes.
- [ ] 1.5 Unit-test against captured DTBs from Ampere Altra Max and NVIDIA Grace (committed under `kernel/src/dt/test-data/`).

### 1c. Fallback

- [ ] 1.6 Implement `NumaTopology::single_node()` synthesizer for the no-NUMA case. Verify against the Jetson Orin DTB — single node, all RAM, all cores.
- [ ] 1.7 Boot sequence: try SRAT (x86), then DT (ARM), then fall back to single-node. Log the chosen path at info level.

## 2. Phase 2 — Tensor pool API extension

- [ ] 2.1 Add `alloc_with_hint(size, dtype, numa_hint: Option<NumaNodeId>)` to `kernel/src/mem/tensor.rs`. Keep the existing `alloc(size, dtype)` as `alloc_with_hint(size, dtype, None)`.
- [ ] 2.2 Restructure the pool's free-list from `FreeList` to `[FreeList; MAX_NUMA_NODES]` (const `MAX_NUMA_NODES = 16`).
- [ ] 2.3 Implement the three-tier allocation flow: try hinted node → try any node → fall back to page allocator.
- [ ] 2.4 Add atomic per-node counters (`alloc_local`, `alloc_remote`, `alloc_unhinted`).
- [ ] 2.5 Expose `TensorPool::topology()` accessor returning `&'static NumaTopology`.

## 3. Phase 3 — Scheduler home-node bias

- [ ] 3.1 Extend the AMP scheduler with a `home_node: NumaNodeId` per thread, populated from the topology at thread creation.
- [ ] 3.2 Inference ops invoked via the scheduler pick up the current thread's `home_node` and pass it as the hint into tensor allocation.
- [ ] 3.3 Document the rule: Core 0 (System/IPC) is always `home_node = 0`; data-parallel cores 1..N use their physical node.

## 4. Phase 4 — Observability and benchmarks

- [ ] 4.1 Add a `/proc/smallaios/numa`-equivalent telemetry endpoint (or extend the existing telemetry-otel-export path) reporting per-node counters + topology.
- [ ] 4.2 Write a synthetic two-node tensor-allocation benchmark in `bench/`: spawn N threads, each pinned to its own node, allocate / free tensors in a loop, report `alloc_local / (alloc_local + alloc_remote)` ratio.
- [ ] 4.3 Run the benchmark on a 2P EPYC Genoa host (loaner / customer hardware) and capture: (a) ratio ≥90%; (b) p99 alloc latency vs single-socket Jetson baseline.
- [ ] 4.4 Run the existing Jetson Orin tensor-pool benchmark on develop and on this branch; confirm ≤2% regression on the single-node path.

## 5. Phase 5 — Documentation

- [ ] 5.1 Create `docs/numa-tensor-alloc.md` covering: when NUMA matters, how to read the topology endpoint, how to interpret the local/remote ratio, troubleshooting (zero local hits = scheduler not setting home_node correctly).
- [ ] 5.2 Update `docs/architecture.md` to note the per-node free-list and the topology discovery layer.
- [ ] 5.3 Add a row to the README hardware matrix for "multi-socket x86-64" and "multi-die ARM64" with a link to the NUMA doc.

## 6. Phase 6 — CI matrix expansion

- [ ] 6.1 Add a scheduled job `numa-multi-socket-smoke` that runs the benchmark on a self-hosted 2P EPYC runner. Advisory initially.
- [ ] 6.2 Confirm the existing single-socket gate jobs continue to pass (no regression).
- [ ] 6.3 Promote `numa-multi-socket-smoke` to `change-gates` when self-hosted runner availability is reliable (separate change).

## 7. Close-out

- [ ] 7.1 PR title: `feat(kernel): numa-aware-tensor-alloc-v1 — multi-socket NUMA topology + per-node tensor free-lists`.
- [ ] 7.2 Reviewer sign-off + green CI + manual 2P EPYC benchmark evidence pasted in the PR description.
- [ ] 7.3 Update CLAUDE.md "Current state" to mention NUMA-aware tensor allocation on multi-socket hosts.
