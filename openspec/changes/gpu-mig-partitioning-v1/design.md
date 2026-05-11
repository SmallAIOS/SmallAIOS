# Design — gpu-mig-partitioning-v1

## Goal

Let a SmallAIOS unikernel instance run on a single NVIDIA MIG (Multi-Instance GPU) slice of an A100 / H100 / H200 / B100-class datacenter GPU, with hardware-isolation guarantees, MIG-aware resource budgeting, and MIG-aware telemetry — without adding any code path to the Jetson Orin production build.

Success: (a) on an H100 host configured with MIG and at least two 1g.20gb slices, two SmallAIOS instances each pinned to a different slice run inference workloads concurrently with no observable performance interference; (b) the unikernel correctly reports the slice's DRAM partition size, compute pipeline count, and per-slice utilization in telemetry; (c) building `--features mig` on a Jetson Orin target either fails at compile-time (preferred) or fails fast at runtime with a clear error message; (d) building without `--features mig` on a non-MIG host or with the full GPU continues to work exactly as today.

## Alternatives considered

### A1. Treat MIG as transparent — let CUDA handle it, do nothing in SmallAIOS

**Rejected as incomplete.** The CUDA runtime does honor MIG transparently — `cudaMalloc` will succeed up to the slice's partition limit, `cudaGetDeviceProperties` reports the slice's SM count. So functionally, SmallAIOS works today on a MIG slice with no code changes (the existing Jetson container path likely runs fine on an A100 MIG slice with no modification). The reasons we still want explicit MIG support:

1. **Resource budgeting.** Without explicit MIG knowledge, the inference scheduler heuristics (batch size, stream count, KV cache size) target the full-GPU profile and over-commit on a MIG slice — leading to allocation failures or thrashing. Explicit MIG awareness lets the scheduler tune to the slice profile.
2. **Telemetry.** The `gpu-profile` path reports memory and compute usage as if the slice were a full GPU, hiding the fact that the unikernel only has 1/7th of the silicon. Operators monitoring a multi-tenant cluster need per-slice metrics.
3. **Fail-fast guarantees.** Today a misconfigured deployment (asking for MIG but getting a full GPU, or asking for full GPU but getting MIG) just silently produces wrong-shaped resource budgets. A clear error early is better than degraded behavior late.

So A1 is rejected — explicit MIG awareness adds value beyond what CUDA's transparency provides.

### A2. Make MIG always-on (no Cargo feature)

**Rejected.** The detection code adds non-trivial CUDA-attribute queries to the boot path. On the Jetson production target where MIG is not supported, those queries are wasted work and may return error values that complicate error handling. A Cargo feature lets us compile-out the entire MIG path on Jetson builds, keeping the Jetson critical path minimal.

### A3. Build "vGPU" / time-slicing support alongside MIG

**Rejected for v1.** vGPU (NVIDIA GRID) is a separate licensed product with its own driver stack and is orthogonal to MIG. Time-slicing via MPS is software-only and has weaker isolation. Both are valid follow-up topics but mixing them into a single change inflates the scope without commensurate benefit.

### A4. Support multi-MIG-slice within one unikernel instance

**Rejected.** The whole point of MIG is hardware isolation — one slice per tenant. Spanning two slices from one process re-introduces the cross-tenant interference MIG was designed to prevent (the SmallAIOS process would be a single noisy neighbor across multiple slices). If a tenant needs more compute, the right answer is a larger slice profile, not multiple slices.

## MIG slice detection

NVIDIA exposes MIG slices via two mechanisms:

1. **Device UUID prefix.** A full GPU has a UUID like `GPU-12345678-1234-1234-1234-1234567890ab`. A MIG slice has a UUID like `MIG-GPU-12345678-...` with additional slice identifier suffix. The UUID is returned by `cuDeviceGetUuid` and `nvmlDeviceGetUUID`.
2. **CUDA device attribute.** `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MIG_MODE)` returns `1` on a MIG slice of a MIG-enabled GPU, `0` otherwise.

Detection sequence at CUDA context init:

```
1. Enumerate CUDA devices (cuDeviceGetCount + cuDeviceGet for each).
2. For each device, query UUID and MIG_MODE attribute.
3. If UUID starts with "MIG-" or MIG_MODE == 1:
     classify as MIG slice; record slice profile via NVML
     (nvmlDeviceGetMigDeviceHandleByIndex chain + nvmlDeviceGetAttributes)
4. Else:
     classify as full GPU; existing code path
```

The NVML calls require the NVIDIA Management Library to be available. On the container path (CDI runtime), NVML is part of the standard image. On the bare-metal / unikernel path (future), NVML is not available — instead we use the CUDA driver API attributes directly, which carry enough information for resource budgeting (SM count, memory size) even without NVML.

## Slice-aware resource budgeting

The unikernel maintains a `NvidiaDevice` descriptor per device. On a MIG slice, the descriptor populates:

| Field | Source | Used by |
|-------|--------|---------|
| `compute_capability` | `cuDeviceGetAttribute(CC_MAJOR/MINOR)` | Kernel selection (cuBLAS / cuDNN dispatch) |
| `sm_count` | `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` | Scheduler stream count and batch-size heuristic |
| `dram_bytes` | `cuDeviceTotalMem` | Tensor pool DRAM ceiling |
| `l2_cache_bytes` | `cuDeviceGetAttribute(L2_CACHE_SIZE)` | Tile-size heuristic in GEMM |
| `mig_profile` | NVML or attribute parse | Telemetry |
| `mig_gpu_instance_id` | NVML | Telemetry |
| `mig_compute_instance_id` | NVML | Telemetry |
| `is_mig_slice` | Detection result | Branch points in budgeting code |

The tensor pool's GPU side is unchanged in structure — it already honors `dram_bytes` as a hard ceiling. The change is that `dram_bytes` now reports the slice partition (e.g., 10 GB for a 1g.10gb slice on A100 80GB) rather than the full GPU's memory.

## Telemetry extension

The existing `gpu-profile` feature (per-op timing + memcpy byte counters dumped at `CudaRuntime::drop`) extends with:

- `device.uuid` (string)
- `device.is_mig` (bool)
- `device.mig.profile` (string, e.g., `"1g.10gb"`)
- `device.mig.gpu_instance_id` (u32)
- `device.mig.compute_instance_id` (u32)
- `device.mig.dram_partition_bytes` (u64)
- `device.mig.peak_memory_used_bytes` (u64, sampled at unikernel exit)
- `device.mig.peak_sm_utilization_pct` (f32, sampled from NVML if available, else 0)

These fields appear alongside the existing per-op profile data in the dump. On non-MIG devices, the `mig.*` fields are absent.

## Fail-fast paths

### Compile-time

The `mig` feature is mutually exclusive with the `tegra-orin` userspace CUDA feature on `smallaios-arch-nvidia`. The `Cargo.toml` declares this:

```toml
[features]
mig = []
tegra-orin = []
# Other features omitted.

# Mutual-exclusion enforced at build time via a compile_error! in lib.rs:
# #[cfg(all(feature = "mig", feature = "tegra-orin"))]
# compile_error!("The 'mig' feature is for datacenter GPUs (A100/H100/H200/B100); 'tegra-orin' is for Jetson Orin (Ampere GA10B) which does not support MIG. Enable one or the other, not both.");
```

This catches the misconfiguration "Jetson user enables --features mig" at build time with a clear pointer to the support matrix.

### Runtime

When `--features mig` is built but the assigned device at boot is NOT a MIG slice (full GPU, non-MIG GPU like L4, or no GPU), the unikernel emits a structured error and refuses to start:

```
[error] arch-nvidia: built with --features mig but assigned device is not a MIG slice
  device: GPU-12345678-... (NVIDIA L4)
  hint:   either rebuild without --features mig, or assign a MIG slice
          (set NVIDIA_VISIBLE_DEVICES=MIG-GPU-..., or use --gpus '"device=0:1"' for the container path)
  matrix: see docs/gpu-mig.md for supported GPUs and MIG profiles
```

When `--features mig` is NOT built but the assigned device IS a MIG slice, the unikernel runs but logs a warning at boot:

```
[warn] arch-nvidia: assigned device is a MIG slice but --features mig was not enabled at build time
       MIG-specific telemetry will be unavailable. Rebuild with --features mig for full integration.
```

## Container-path integration

The container path passes the GPU device via the NVIDIA CDI runtime. The operator's `docker run` (or Kubernetes pod spec) selects a MIG slice via:

```bash
docker run --gpus '"device=0:1"' ...  # GPU 0, MIG instance 1
# or equivalently
docker run -e NVIDIA_VISIBLE_DEVICES=MIG-GPU-12345678-... ...
```

SmallAIOS-side, the existing CUDA initialization code picks up the assigned device. The MIG-detection code path runs unconditionally when `--features mig` is on; no Docker-side changes are required beyond using a MIG-aware deployment manifest.

## Documentation

`docs/gpu-mig.md` covers:

1. Support matrix (replicates the table in `proposal.md`).
2. Host-side MIG configuration with `nvidia-smi mig`.
3. Container-side deployment with MIG device assignment.
4. SmallAIOS-side build with `--features mig`.
5. Telemetry / observability — how to read the `mig.*` fields.
6. Troubleshooting — common error messages and their fixes.
7. **Jetson out-of-scope callout** — bold and unmissable, with a link back to `Dockerfile.jetson` for the supported Jetson workflow.

## What this change explicitly does NOT do

- Does not implement MIG configuration from inside the unikernel (host admin task).
- Does not add MIG support to the Jetson code path.
- Does not span multiple MIG slices from a single unikernel instance.
- Does not implement vGPU, MPS, or any other software time-slicing approach.
- Does not implement AMD MxGPU or Intel Flex partitioning analogs (separate future changes).
- Does not implement dynamic MIG reshape at runtime (reshape requires unikernel restart).
- Does not change the existing full-GPU code path when `--features mig` is off.
