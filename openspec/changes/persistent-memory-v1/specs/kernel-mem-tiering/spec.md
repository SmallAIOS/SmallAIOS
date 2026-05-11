## ADDED Requirements

### Requirement: Memory tier discovery at boot

The kernel SHALL discover the host's memory-tier topology at boot and SHALL classify each memory region as `Hot`, `Warm`, or `Cold` tier based on standard ACPI / CXL attribute tables.

#### Scenario: x86-64 HMAT discovery on a CXL.mem-equipped host

- **GIVEN** a Sapphire Rapids / Granite Rapids / EPYC Genoa host with a CXL 1.1+ type-3 memory expander attached, publishing ACPI HMAT
- **WHEN** the kernel boots
- **THEN** the kernel SHALL parse HMAT Memory Proximity Domain Attributes (type 0) to enumerate memory proximity domains
- **AND** the kernel SHALL parse HMAT System Locality Latency and Bandwidth Information (type 1) to derive per-(initiator, target) access latencies
- **AND** the kernel SHALL classify a proximity domain as `MemTier::Warm` when its access latency from any local CPU exceeds 2x the lowest-latency domain's access latency
- **AND** boot logs SHALL include `mem-tier: discovered N tiers via HMAT` at info level

#### Scenario: CDAT discovery on any architecture with CXL devices

- **GIVEN** a host with CXL type-3 devices on the PCI bus, each publishing CDAT via DOE
- **WHEN** the kernel boots
- **THEN** the kernel SHALL walk the PCI / CXL bus and query each CXL device's CDAT via the DOE mailbox
- **AND** the kernel SHALL parse Device Scoped Memory Affinity (DSMAS) and Device Scoped Latency and Bandwidth (DSLBIS) structures
- **AND** the resulting tier classification SHALL merge cleanly with HMAT on x86-64 hosts
- **AND** on ARM64 hosts without HMAT, CDAT alone SHALL drive tier classification
- **AND** boot logs SHALL include `mem-tier: discovered N CXL devices via CDAT` at info level

#### Scenario: Single-tier fallback on Jetson Orin and other CXL-less hosts

- **GIVEN** a host that publishes neither HMAT nor CXL CDAT (e.g., Jetson Orin NX, single-socket x86-64 desktop without CXL, Apple silicon)
- **WHEN** the kernel boots
- **THEN** the kernel SHALL synthesize a single-tier topology where every memory node reports `MemTier::Hot`
- **AND** boot logs SHALL include `mem-tier: single-tier fallback` at info level
- **AND** subsequent calls to `alloc_with_tier(.., MemTierPolicy::Preferred(MemTier::Warm))` SHALL transparently fall back to `Hot` (the only tier present)

### Requirement: Tier-aware tensor allocation API

The `TensorPool` SHALL accept an optional `MemTierPolicy` argument controlling whether allocations are restricted to a tier, prefer a tier, or accept any tier.

#### Scenario: Required(Hot) succeeds when hot tier has capacity

- **GIVEN** a CXL.mem-equipped host with discovered tiers `Hot` and `Warm`
- **GIVEN** the hot tier has free buffers in the requested size class
- **WHEN** a caller invokes `TensorPool::alloc_with_tier(size, dtype, MemTierPolicy::Required(MemTier::Hot))`
- **THEN** the returned `TensorBuf` SHALL be drawn from a hot-tier node's free-list
- **AND** the `alloc_required` counter on the hot tier SHALL increment by one
- **AND** the buffer SHALL NOT be drawn from any warm-tier or cold-tier node

#### Scenario: Required(Hot) fails when hot tier is exhausted

- **GIVEN** the hot tier has no free buffers in the requested size class and the page allocator cannot grow it
- **WHEN** a caller invokes `TensorPool::alloc_with_tier(size, dtype, MemTierPolicy::Required(MemTier::Hot))`
- **THEN** the call SHALL return `Err(MemError::TierExhausted(MemTier::Hot))`
- **AND** the pool SHALL NOT silently satisfy the allocation from a warm-tier node
- **AND** the failure SHALL propagate to the calling op (e.g., KV-cache allocation failure aborts the inference step rather than placing KV cache in CXL.mem)

#### Scenario: Preferred(Warm) prefers warm but falls back

- **GIVEN** a CXL.mem-equipped host where the warm tier has free buffers
- **WHEN** a caller invokes `TensorPool::alloc_with_tier(size, dtype, MemTierPolicy::Preferred(MemTier::Warm))`
- **THEN** the returned `TensorBuf` SHALL be drawn from a warm-tier node
- **AND** the warm tier's `alloc_preferred` counter SHALL increment by one
- **GIVEN** the warm tier is exhausted
- **WHEN** the same call is made
- **THEN** the allocation SHALL succeed against a hot-tier node
- **AND** the warm tier's `alloc_fallback` counter SHALL increment by one

#### Scenario: Any policy is the v0-compatible default

- **GIVEN** any host topology
- **WHEN** a caller invokes `TensorPool::alloc_with_tier(size, dtype, MemTierPolicy::Any)` or the legacy `TensorPool::alloc(size, dtype)`
- **THEN** the pool SHALL satisfy the allocation from any tier with capacity
- **AND** the call SHALL NOT panic or behave differently from the v0 (pre-tiering) implementation on a single-tier host

### Requirement: Inference scheduler applies default tier policies

The inference scheduler SHALL classify each tensor allocation by role and SHALL apply tier policies aligned with that role.

#### Scenario: Model weights prefer the warm tier

- **GIVEN** a CXL.mem-equipped host
- **WHEN** the model loader allocates a buffer for a model-weight tensor
- **THEN** the allocation SHALL use `MemTierPolicy::Preferred(MemTier::Warm)`
- **AND** the `alloc_preferred` counter on the warm tier SHALL increment

#### Scenario: KV cache requires the hot tier

- **GIVEN** any host
- **WHEN** the inference scheduler allocates a buffer for KV cache state
- **THEN** the allocation SHALL use `MemTierPolicy::Required(MemTier::Hot)`
- **AND** if hot-tier capacity is unavailable, the inference step SHALL fail with a `TierExhausted` error rather than degrading silently

#### Scenario: Policy override via environment / boot argument

- **GIVEN** the operator sets `SMALLAIOS_TIER_POLICY=weights:any,kv:hot` (container path) or the equivalent kernel boot argument
- **WHEN** the inference scheduler dispatches an op
- **THEN** the scheduler SHALL honor the override, using `MemTierPolicy::Any` for weight allocations instead of the `Preferred(Warm)` default
- **AND** the chosen policy SHALL be logged at info level on first use

### Requirement: CXL 3.0 shared regions support read-only weight sharing across instances

When CXL 3.0 pooled / shared memory is available, the kernel SHALL provide a `SharedRegion` abstraction allowing multiple SmallAIOS instances on the same CXL fabric to map the same physical memory region read-only.

#### Scenario: Read-only mapping of a pre-provisioned shared region

- **GIVEN** a CXL 3.0 fabric with a pre-provisioned pooled memory region containing a model's weight tensors
- **GIVEN** the SmallAIOS instance has been granted read access to the region by the fabric manager
- **WHEN** the model loader invokes `SharedRegion::map_read_only(spec)`
- **THEN** the kernel SHALL verify the region exists, verify size and permissions match the spec, and map the region into the tensor-pool address space
- **AND** the mapped region SHALL be tagged with `MemTier::Warm` and the `shared = true` flag
- **AND** subsequent tensor allocations SHALL be able to reference the shared region as backing storage for read-only weight buffers

#### Scenario: Two SmallAIOS instances share weights without duplication

- **GIVEN** two SmallAIOS unikernel instances on the same CXL 3.0 fabric
- **GIVEN** both instances have read access to a pre-provisioned shared region containing model weights
- **WHEN** both instances start and load the same model
- **THEN** both instances SHALL map the same `SharedRegion` read-only
- **AND** the total fabric memory consumed by weight tensors SHALL be the size of one copy (not two)
- **AND** each instance SHALL produce identical inference outputs for the same input

#### Scenario: Writable shared regions are explicitly unsupported in v1

- **GIVEN** a caller attempting to use a writable shared region in v1
- **WHEN** the caller invokes a writable shared-region API
- **THEN** no such API SHALL exist; only `map_read_only` is provided
- **AND** documentation SHALL note that writable shared regions require application-level coherence coordination that is out of scope for v1

### Requirement: Single-tier hosts pay zero tiering overhead

On hosts with a single memory tier (the Jetson Orin production target), the tier-aware allocation path SHALL impose no measurable performance regression on the existing tensor-pool benchmark suite.

#### Scenario: Jetson Orin regression guard

- **GIVEN** the existing tensor-pool alloc/free benchmark suite captured on the Jetson Orin NX 16 GB baseline
- **WHEN** the benchmark is re-run on this change's branch on the same Jetson Orin host
- **THEN** the median and p99 alloc latency SHALL be within ±2% of the v0 baseline
- **AND** the boot log SHALL show `mem-tier: single-tier fallback`, confirming the no-cost path is taken
