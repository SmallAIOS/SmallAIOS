# Tasks — persistent-memory-v1

> **Status: Future-facing.** No work has started. This change is a roadmap document for CXL.mem / CXL 3.0 datacenter deployments. The Jetson Orin production target has no CXL controller. Phase 5 (CXL 3.0 shared regions) is contingent on CXL 3.0 production silicon availability (likely 2026-2028).

## 0. Trigger conditions (review before starting)

- [ ] 0.1 Confirm at least one production / customer target ships with CXL.mem-equipped hardware (Sapphire Rapids+ / Granite Rapids / EPYC Genoa+ / Ampere AmpereOne / Grace) and a CXL type-3 memory expander.
- [ ] 0.2 Capture HMAT and CDAT blobs from a reference CXL.mem host for unit-test fixtures. Without this, the discovery shim is unverifiable.
- [ ] 0.3 Establish a CXL.mem-equipped scheduled CI runner (self-hosted; no GitHub-hosted runner has CXL.mem).
- [ ] 0.4 (Phase 5 gate) Confirm CXL 3.0 pooled-memory silicon and fabric manager are available before starting `SharedRegion` work.

## 1. Phase 1 — Discovery shim

### 1a. HMAT parser

- [ ] 1.1 Add `kernel/src/acpi/hmat.rs` parsing Memory Proximity Domain Attributes (type 0), System Locality Latency and Bandwidth (type 1), and Memory Side Cache Information (type 2).
- [ ] 1.2 Classify each proximity domain as `MemTier::Hot` or `MemTier::Warm` based on the >2x access-latency heuristic (consistent with Linux `mm/memory-tiers.c` ordering).
- [ ] 1.3 Unit-test the parser against captured HMAT blobs from Sapphire Rapids + Astera Leo and EPYC Genoa + Samsung CMM-D fixtures (committed under `kernel/src/acpi/test-data/`).

### 1b. CDAT walker

- [ ] 1.4 Add `kernel/src/cxl/cdat.rs` walking CXL devices on the PCI bus, querying CDAT via DOE mailbox.
- [ ] 1.5 Parse Device Scoped Memory Affinity (DSMAS) and Device Scoped Latency/Bandwidth (DSLBIS) structures.
- [ ] 1.6 Merge CDAT data with HMAT (x86-64) or use standalone (ARM64).
- [ ] 1.7 Unit-test the walker against captured DOE / CDAT byte streams (committed under `kernel/src/cxl/test-data/`).

### 1c. Single-tier fallback

- [ ] 1.8 Implement `MemTopology::single_tier()` synthesizer for hosts without HMAT or CDAT. Verify against Jetson Orin DTB — all nodes report `MemTier::Hot`.
- [ ] 1.9 Boot sequence: try HMAT → CDAT → single-tier fallback. Log the chosen path at info level.

## 2. Phase 2 — TensorPool tier-aware API

- [ ] 2.1 Add `MemTier` enum and `MemTierPolicy` types to `kernel/src/mem/tier.rs`.
- [ ] 2.2 Add `TensorPool::alloc_with_tier(size, dtype, tier_policy)` to `kernel/src/mem/tensor.rs`. Keep `alloc()` as a thin wrapper passing `MemTierPolicy::Any`.
- [ ] 2.3 Compose with NUMA hint from `numa-aware-tensor-alloc-v1` via `alloc_full(size, dtype, numa_hint, tier_policy)`.
- [ ] 2.4 Implement allocation flow:
  - `Required(tier)`: scan only nodes of that tier; fail if none have capacity.
  - `Preferred(tier)`: scan that tier first; fall back to any tier.
  - `Any`: existing v0 behavior.
- [ ] 2.5 Add per-tier and per-(node,tier) counters: `alloc_required`, `alloc_preferred`, `alloc_fallback`, `alloc_any`.

## 3. Phase 3 — Scheduler integration

- [ ] 3.1 Define default tier policies for the inference scheduler:
  - Model weights → `Preferred(Warm)`.
  - KV cache → `Required(Hot)`.
  - Activations / intermediates → `Required(Hot)`.
  - Diagnostics / logs → `Preferred(Warm)`.
- [ ] 3.2 Allow override via `SMALLAIOS_TIER_POLICY` env var (container path) and kernel boot argument (unikernel path).
- [ ] 3.3 Document the failure mode: if `Required(Hot)` allocation fails because hot tier is exhausted, the op fails (we do not silently promote KV cache to warm tier).

## 4. Phase 4 — Single-host CXL.mem benchmark + docs

- [ ] 4.1 Write a benchmark in `bench/` that loads a 100-GB-class weight tensor with `Preferred(Warm)` and measures effective end-to-end inference latency vs an all-DRAM baseline.
- [ ] 4.2 Run the benchmark on a CXL.mem-equipped reference host (Sapphire Rapids + Astera Leo or equivalent). Capture: (a) `alloc_preferred` hits on warm tier, (b) inference p50 / p99 vs DRAM baseline, (c) cost-per-token estimate using nominal $/GB DRAM vs $/GB CXL.mem.
- [ ] 4.3 Run the existing Jetson Orin benchmark on this branch vs develop. Confirm ≤2% regression on the single-tier fallback path.
- [ ] 4.4 Create `docs/persistent-memory.md` covering: when tiering matters, hardware prerequisites, how to provision CXL.mem on a host, the default tier policies, the telemetry endpoint, troubleshooting.
- [ ] 4.5 Add a row to the README hardware matrix for "CXL.mem-equipped server".

## 5. Phase 5 — CXL 3.0 SharedRegion API

> **Phase 5 gate:** CXL 3.0 production silicon, fabric manager software, and at least two SmallAIOS instances on the same fabric are prerequisites. If unavailable, defer Phase 5 indefinitely; Phase 1-4 are independently valuable on single-host CXL.mem.

- [ ] 5.1 Add `SharedRegion` and `SharedRegionSpec` types to `kernel/src/cxl/shared.rs`.
- [ ] 5.2 Implement `map_read_only(spec)` — verify region exists on fabric, verify size and permissions match spec, map into tensor-pool address space tagged `MemTier::Warm` + `shared = true`.
- [ ] 5.3 Document the coherence model: writable shared regions are out of scope for v1; v1 supports read-only weight sharing only.
- [ ] 5.4 Add a multi-instance test scenario: two SmallAIOS unikernels on the same CXL 3.0 fabric, both map the same weight region read-only, run inference, confirm matching outputs.
- [ ] 5.5 Update `docs/persistent-memory.md` with a CXL 3.0 shared-region section + provisioning workflow.

## 6. Phase 6 — CI integration

- [ ] 6.1 Add a scheduled job `cxl-mem-smoke` running the Phase 4 benchmark on a self-hosted CXL.mem runner. Advisory initially.
- [ ] 6.2 Confirm the existing single-tier gate jobs continue to pass on the Jetson Orin path.
- [ ] 6.3 Promote `cxl-mem-smoke` to `change-gates` when self-hosted runner availability is reliable (separate change).

## 7. Close-out

- [ ] 7.1 PR title: `feat(kernel): persistent-memory-v1 — CXL.mem tiered tensor allocation + CXL 3.0 shared regions`.
- [ ] 7.2 Reviewer sign-off + green CI + CXL.mem benchmark evidence pasted in the PR description.
- [ ] 7.3 Update CLAUDE.md "Current state" to mention tiered memory allocation for CXL-equipped hosts.
