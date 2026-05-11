## ADDED Requirements

### Requirement: NUMA topology discovery at boot

The kernel SHALL discover the host's NUMA topology at boot and expose it via a read-only `TensorPool::topology()` accessor. The topology SHALL include, for each NUMA node, the set of CPU IDs local to that node and the set of physical memory ranges owned by that node.

#### Scenario: x86-64 SRAT discovery

- **GIVEN** a multi-socket x86-64 host (e.g., 2P AMD EPYC Genoa, 2P Intel Xeon Sapphire Rapids) that publishes an ACPI SRAT
- **WHEN** the kernel boots
- **THEN** the kernel SHALL parse SRAT Memory Affinity (type 1) structures to map physical address ranges to proximity domains
- **AND** the kernel SHALL parse SRAT Processor Local APIC/SAPIC Affinity (type 0) structures to map APIC IDs to proximity domains
- **AND** the resulting `NumaTopology` SHALL report at least one node per proximity domain
- **AND** boot logs SHALL include a line like `numa: discovered N nodes via SRAT` at info level

#### Scenario: ARM64 device-tree discovery

- **GIVEN** a multi-die ARM64 server (e.g., Ampere Altra Max, NVIDIA Grace) whose device tree publishes `numa-node-id` properties on `cpu@*` and `memory@*` nodes
- **WHEN** the kernel boots
- **THEN** the kernel SHALL walk the device tree and read every `numa-node-id` property on CPU and memory nodes
- **AND** the resulting `NumaTopology` SHALL group CPUs and memory ranges by node id
- **AND** boot logs SHALL include `numa: discovered N nodes via device-tree` at info level

#### Scenario: Single-node fallback (Jetson Orin and other UMA hosts)

- **GIVEN** a host that publishes neither SRAT NUMA data nor `numa-node-id` device-tree properties (e.g., Jetson Orin NX, single-socket x86-64 desktop, Apple silicon)
- **WHEN** the kernel boots
- **THEN** the kernel SHALL synthesize a `NumaTopology::single_node()` containing exactly one node spanning all RAM and all CPUs
- **AND** boot logs SHALL include `numa: single-node fallback` at info level
- **AND** subsequent calls to `alloc_with_hint(.., Some(0))` and `alloc_with_hint(.., None)` SHALL behave identically

### Requirement: NUMA-aware tensor allocation hint

The `TensorPool` SHALL accept an optional NUMA node hint on allocation and SHALL prefer the hinted node's per-node free-list when satisfying the allocation.

#### Scenario: Hinted allocation on a multi-node host hits the local free-list

- **GIVEN** a multi-socket host with discovered topology `N >= 2`
- **GIVEN** the per-node free-list for node 1 has at least one buffer of the requested size class
- **WHEN** a caller invokes `TensorPool::alloc_with_hint(size, dtype, Some(1))`
- **THEN** the returned `TensorBuf` SHALL be drawn from node 1's free-list
- **AND** the `numa.node[1].alloc_local` counter SHALL increment by one
- **AND** the `numa.node[1].alloc_remote` counter SHALL NOT increment

#### Scenario: Hinted allocation falls back to another node under pressure

- **GIVEN** the per-node free-list for the hinted node has no buffers in the requested size class
- **GIVEN** another node's free-list has a buffer in that size class
- **WHEN** a caller invokes `TensorPool::alloc_with_hint(size, dtype, Some(hinted))`
- **THEN** the pool SHALL satisfy the allocation from any other node's free-list
- **AND** the `numa.node[hinted].alloc_remote` counter SHALL increment by one
- **AND** the allocation SHALL NOT fail solely because the hinted node is empty

#### Scenario: Unhinted allocation works on every system

- **GIVEN** any topology (single-node or multi-node)
- **WHEN** a caller invokes `TensorPool::alloc_with_hint(size, dtype, None)` or the legacy `TensorPool::alloc(size, dtype)`
- **THEN** the pool SHALL satisfy the allocation from any available node's free-list
- **AND** the corresponding node's `alloc_unhinted` counter SHALL increment by one
- **AND** the call SHALL NOT panic, error, or behave differently from the v0 (pre-NUMA) implementation on a single-node host

### Requirement: Cooperative scheduler propagates home-node hint

The AMP cooperative scheduler SHALL associate each thread with a `home_node: NumaNodeId` and SHALL pass that node id as the NUMA hint on every tensor-pool allocation made on behalf of that thread.

#### Scenario: Data-parallel inference threads run on their physical node

- **GIVEN** a multi-socket host with the inference scheduler running data-parallel ops on cores 1..N
- **GIVEN** core K is physically located on NUMA node M (per `NumaTopology`)
- **WHEN** the scheduler dispatches an inference op to core K
- **AND** the op's kernel allocates a tensor via the scheduler-provided allocator handle
- **THEN** the allocation SHALL be issued as `alloc_with_hint(.., Some(M))`
- **AND** on a 2P EPYC host, the steady-state ratio `alloc_local / (alloc_local + alloc_remote)` SHALL be ≥ 0.90 under a synthetic two-node benchmark

#### Scenario: Core 0 (System/IPC) is bound to node 0 by convention

- **GIVEN** any host topology
- **WHEN** the scheduler dispatches System/IPC work on core 0
- **THEN** core 0's `home_node` SHALL be 0
- **AND** allocations made by core 0 SHALL be hinted to node 0

### Requirement: Single-node hosts pay zero NUMA overhead

On single-node hosts (the Jetson Orin production target), the NUMA-aware allocation path SHALL impose no measurable performance regression on the existing tensor-pool benchmark suite.

#### Scenario: Jetson Orin regression guard

- **GIVEN** the existing tensor-pool alloc/free benchmark suite captured on the Jetson Orin NX 16 GB baseline
- **WHEN** the benchmark is re-run on this change's branch on the same Jetson Orin host
- **THEN** the median and p99 alloc latency SHALL be within ±2% of the v0 baseline
- **AND** the per-op tensor throughput SHALL be within ±2% of the v0 baseline
- **AND** the boot log SHALL show `numa: single-node fallback`, confirming the no-cost path is taken

### Requirement: Per-node allocation counters are exposed via telemetry

The kernel SHALL expose per-node tensor-allocation counters via the existing telemetry endpoint, suitable for observability and tuning.

#### Scenario: Counters reflect alloc paths

- **GIVEN** the telemetry endpoint `/proc/smallaios/numa` (or the unikernel-equivalent path under the OTel exporter)
- **WHEN** the operator reads the endpoint
- **THEN** the endpoint SHALL list every discovered node with: cpu set, memory size, `alloc_local`, `alloc_remote`, `alloc_unhinted`
- **AND** the sum of all three counters across all nodes SHALL equal the total number of `TensorPool` allocations since boot
- **AND** the endpoint SHALL be read-only and SHALL NOT modify allocator state
