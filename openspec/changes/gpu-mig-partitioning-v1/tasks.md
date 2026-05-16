# Tasks — gpu-mig-partitioning-v1

> **Status: Future-facing.** No work has started. This change is a roadmap document for multi-tenant datacenter inference on NVIDIA A100 / H100 / H200 / B100-class GPUs. **Jetson Orin does not support MIG and is explicitly out of scope.**

## 0. Trigger conditions (review before starting)

- [ ] 0.1 Confirm at least one production / customer target uses MIG-capable hardware (A100, A30, H100, H100 NVL, H200, or pre-production Blackwell). If only Jetson or single-tenant full-GPU deployments exist, defer this change.
- [ ] 0.2 Secure access to an A100 or H100 host for integration testing (loaner hardware, cloud rental, or partner lab). Without this, the implementation cannot be validated.
- [ ] 0.3 Capture NVML query outputs from a MIG-configured reference host for unit-test fixtures (UUIDs, MIG profile metadata, attribute dumps).

## 1. Phase 1 — Feature flag + detection

- [ ] 1.1 Add `mig` Cargo feature to `arch/nvidia/Cargo.toml` with a clear doc-comment describing the support matrix and the Jetson out-of-scope callout.
- [ ] 1.2 Declare `mig` and `tegra-orin` mutually exclusive via a `compile_error!` in `arch/nvidia/src/lib.rs` (`#[cfg(all(feature = "mig", feature = "tegra-orin"))]`). The error message SHALL point at the support matrix in `docs/gpu-mig.md`.
- [ ] 1.3 Add `is_mig_slice()` detection in `arch/nvidia/src/mig.rs` that combines `cuDeviceGetUuid` UUID-prefix check with `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MIG_MODE)`.
- [ ] 1.4 Extend the `NvidiaDevice` descriptor with the MIG fields (`mig_profile`, `mig_gpu_instance_id`, `mig_compute_instance_id`, `is_mig_slice`).
- [ ] 1.5 On a CXL-less, MIG-less reference machine (Jetson Orin or x86 with a non-MIG GPU like L4), confirm building **without** `--features mig` is unchanged from develop.

## 2. Phase 2 — Slice-aware resource budgeting

- [ ] 2.1 Update the GPU tensor pool's DRAM ceiling to use `device.dram_bytes` (slice partition size on MIG, full GPU memory otherwise). On MIG, attempt to allocate beyond the partition SHALL return `Err(MemError::OutOfDeviceMemory)` not a CUDA error from deep inside the allocator.
- [ ] 2.2 Update the scheduler's stream count heuristic to bound on `device.sm_count`. On a 1g.10gb A100 slice (14 SMs), stream count SHALL NOT exceed the slice's compute pipeline count.
- [ ] 2.3 Update the GEMM tile-size heuristic to honor `device.l2_cache_bytes` (slice's L2 partition).
- [ ] 2.4 Add unit tests against a mocked MIG topology (simulated `NvidiaDevice` with MIG fields) covering: alloc within budget, alloc exceeds budget, scheduler stream count caps.

## 3. Phase 3 — Telemetry

- [ ] 3.1 Extend the `gpu-profile` telemetry path with the `device.uuid`, `device.is_mig`, `device.mig.*` fields described in `design.md`.
- [ ] 3.2 Sample peak memory and peak SM utilization at unikernel exit (NVML where available; CUDA driver attribute where not).
- [ ] 3.3 Document the telemetry schema in `docs/gpu-mig.md` alongside the existing `gpu-profile` documentation.

## 4. Phase 4 — Fail-fast paths

- [ ] 4.1 Implement the runtime check: built with `--features mig` AND assigned device is NOT a MIG slice → refuse to start with the structured error from `design.md`.
- [ ] 4.2 Implement the runtime warning: built WITHOUT `--features mig` AND assigned device IS a MIG slice → log a warning at boot.
- [ ] 4.3 Test the runtime check on a Jetson Orin: rebuild with `--features mig` (after temporarily removing the compile-time exclusion to exercise the runtime path) and confirm the runtime check fires correctly with the documented error message.

## 5. Phase 5 — Integration test on real hardware

- [ ] 5.1 Provision an A100 or H100 host with at least two MIG slices (e.g., `7x 1g.10gb` on A100, or `2x 3g.40gb` on H100 80GB).
- [ ] 5.2 Deploy two SmallAIOS container instances, each pinned to a separate MIG slice via `NVIDIA_VISIBLE_DEVICES=MIG-GPU-...`.
- [ ] 5.3 Run a benchmark workload concurrently on both instances. Confirm: (a) both instances report their slice profile correctly in telemetry; (b) p99 inference latency on instance A is unaffected by load on instance B (the hardware-isolation claim); (c) the sum of per-slice memory utilization equals the GPU's total allocated memory.
- [ ] 5.4 Capture the benchmark results and paste in the PR description as the acceptance evidence.

## 6. Phase 6 — Documentation

- [ ] 6.1 Create `docs/gpu-mig.md` covering: support matrix, host-side `nvidia-smi mig` configuration, container deployment, build instructions, telemetry, troubleshooting, **Jetson out-of-scope callout in bold**.
- [ ] 6.2 Add a row to the README hardware matrix for "Datacenter MIG-capable GPUs" with a link to `docs/gpu-mig.md`.
- [ ] 6.3 Update `Dockerfile` (or add `Dockerfile.mig` if needed) to be MIG-compatible — verify CDI runtime configuration works when a MIG device UUID is passed.

## 7. Phase 7 — CI

- [ ] 7.1 Add a build matrix entry that compiles `--features mig` (host-build only; CI runners don't have MIG hardware). Validates compile-time correctness.
- [ ] 7.2 Add a scheduled integration job `mig-h100-smoke` that runs against a self-hosted H100 runner (when available) with a MIG profile configured. Advisory initially.
- [ ] 7.3 Promote `mig-h100-smoke` to `change-gates` when self-hosted MIG runner availability is reliable (separate change).

## 8. Close-out

- [ ] 8.1 PR title: `feat(arch/nvidia): gpu-mig-partitioning-v1 — datacenter MIG slice support for A100/H100/H200`.
- [ ] 8.2 Reviewer sign-off + green CI + real-MIG-hardware integration test evidence in the PR description.
- [ ] 8.3 Update CLAUDE.md "Current state" to mention MIG slice support for A100/H100/H200, and that Jetson Orin is explicitly out of scope for MIG.
