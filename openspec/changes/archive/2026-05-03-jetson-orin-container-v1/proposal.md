## Why

Today the smallaios container only builds and runs with CUDA on x86-64 + discrete-GPU hosts. On NVIDIA Jetson (Orin / NX / Nano-Super) — the canonical ARM64 + integrated-Tegra-GPU target — `Dockerfile.cuda` cannot run at all and the underlying Rust code does not compile.

Two distinct failures were observed when trying to validate `smallaios:gpu` on a Jetson Orin NX (L4T R36.4.7, JetPack 6, CUDA 12.6, cuDNN 9.3, driver 540.4.0):

1. **CUDA build is broken on every platform (regression).** `cargo build --features cuda,nvidia_gpu` fails with seven `E0277` errors. The recently-merged CUDA Graphs (#118) and async-multistream (#120) changes added `RefCell<Option<CudaGraphCache>>`, `RefCell<Option<StreamPool>>`, `RefCell<Option<Arc<BTreeMap<String, Arc<DeviceTensor>>>>>`, and a raw `*mut c_void` (cudaGraphExec_t) into `Session`. None of these are `Send + Sync`, but `Session` is captured by HTTP handler closures (`HttpServer::route_fn` requires `Send + 'static`) and by `std::thread::spawn` calls in `container/src/main.rs:319,348` and `ipc/src/dataflow_runner.rs:75`. There is no CI gate building the `cuda` feature, so this regression slipped through.

2. **`Dockerfile.cuda` cannot run on Jetson.** The image base `nvcr.io/nvidia/cuda:13.0.0-{devel,runtime}-ubuntu24.04` declares a CDI `requirements` clause of `cuda>=13.0 || driver>=535&&driver<536 || ... || driver>=575&&driver<576`. The Jetson driver is 540.4.0 with CUDA 12.6 capability and is rejected by `nvidia-container-runtime` *before the container even starts*, regardless of `--gpus`. JetPack 6's user-mode CUDA stack is fixed at 12.6 — there is no 13.0 path. The correct base image family for Jetson is `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (or `nvcr.io/nvidia/l4t-cuda:12.6.x-devel/runtime`).

Once both are fixed, smallaios's CUDA executor (Conv via cuDNN, GEMM via cuBLAS, fused ops) gets validated end-to-end on real Tegra hardware for the first time. The `tegra` feature flag in `arch/nvidia` currently selects only `cc_53` (Tegra X1 / Jetson Nano), which is wrong for every shipping Jetson Orin SKU; we extend it to cover Orin (cc_87) as well. CLAUDE.md's project state — "GPU crates are architectural stubs with HAL interfaces but no hardware interaction" — moves to "validated on Jetson Orin NX with cuDNN+cuBLAS dispatch."

## What Changes

### Phase 1 — fix the cross-platform CUDA build regression

- Wrap the GPU-side caches and raw CUDA handles in `Session` so the type is `Send + Sync` again. Concretely:
  - Replace `RefCell<Option<CudaGraphCache>>`, `RefCell<Option<StreamPool>>`, and `RefCell<Option<Arc<BTreeMap<String, Arc<DeviceTensor>>>>>` with `Mutex<...>` (or wrap the inner cache in a `SendCudaState` newtype with documented `unsafe impl Send + Sync`, since the underlying CUDA contexts are accessed under the existing executor lock).
  - For `*mut c_void` carried in `cuda::graph::CudaGraphExec` and `cuda::graph_cache::CudaGraphCacheEntry` (per the `--> graph.rs:24` and `graph_cache.rs:98` error sites), introduce a `#[repr(transparent)] struct ExecHandle(*mut c_void); unsafe impl Send for ExecHandle {}; unsafe impl Sync for ExecHandle {}` with a `// SAFETY:` justification covering CUDA's API contract (graph execs may be launched from any thread that has the matching CUDA context current, and Session pins itself to a single device).
- Add a CI gate that runs `cargo check --features cuda,nvidia_gpu` (CPU-host check is sufficient — link step needs CUDA dev libs but the `cargo check` does not). This prevents future regressions from re-breaking the GPU build.

### Phase 2 — Jetson Orin container path

- Add **two** Jetson Dockerfiles for different size/feature trade-offs:
  - `Dockerfile.jetson` (full JetPack base, ~10 GB) — `nvcr.io/nvidia/l4t-jetpack:r36.4.0` for both stages. Convenient for first-time bring-up and for users who want the rest of JetPack (TensorRT, VPI, NPP, multimedia, samples, nvcc) preinstalled.
  - `Dockerfile.jetson.slim` (recommended, ~4 GB) — JetPack builder + `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime` runtime, with the cuDNN .so files copied across. SmallAIOS only links against CUDA runtime + cuBLAS + cuDNN, so dropping the rest of JetPack is safe; the slim variant passes the same end-to-end smoke test as the full one.
- Build with `aarch64-unknown-linux-gnu` (glibc, not musl) in both, so the binary dynamically links against the host-driver-provided `libcuda.so.1` cleanly.
- Pass `--features cuda,nvidia_gpu,smallaios-arch-nvidia/cc_87` (or equivalently `smallaios-arch-nvidia/tegra-orin`) to capture Orin's Ampere compute capability.
- Add a new `tegra-orin` feature on `arch/nvidia` that selects `cc_87` (Orin family — Nano Super / NX / AGX all share cc 8.7). This is purely container-side: the Orin path uses the userspace CUDA runtime mounted by the NVIDIA Container Toolkit, not the X1-specific bare-metal HAL under `arch/nvidia/src/tegra/`. The existing `tegra` feature continues to mean "Tegra X1 bare-metal HAL" and remains unchanged so the `smallaios-arch-aarch64` Jetson Nano boot path is not disturbed.
- Add `smallaios-jetson` (full JetPack) and `smallaios-jetson-slim` (recommended) services to `docker-compose.yml` under `jetson` and `jetson-slim` profiles respectively (both separate from `gpu` so x86 GPU users aren't accidentally targeting L4T).
- Verify the binary boots on Jetson with `runtime: nvidia` (default on the test box), runs `cudaGetDeviceCount`, reports `compute 8.7`, loads the SqueezeNet fixture, and serves a successful inference round-trip.

### Phase 3 — documentation + tests

- Add `docs/jetson-quickstart.md` with: hardware requirements (JetPack 6 / L4T R36.4+, NVIDIA Container Toolkit), the `docker compose --profile jetson up` workflow, a one-liner `curl` check against `/v1/inference`, and a troubleshooting section covering the CDI driver-mismatch error pattern.
- Update `README.md` to add a Jetson row to the deployment matrix (alongside x86 CPU and x86 GPU container).
- Update `CLAUDE.md` "Current state" to reflect Jetson Orin GPU validation; add `Dockerfile.jetson` to the Build Configuration section.
- Add a smoke test script `scripts/test-jetson-gpu.sh` that builds the image, runs it, hits `/healthz` + `/readyz` + `/v1/inference` against SqueezeNet, asserts the GPU init log line shows `compute 8.7`, and exits non-zero on any failure. Wire it into `Justfile` as `just test-jetson-gpu`.
- Add a CI advisory job (initially `continue-on-error: true` since GitHub-hosted runners can't host Jetson) that builds `Dockerfile.jetson` under QEMU emulation to catch Dockerfile rot.
- Update `bench/configs/jetson.env` (already exists for the upstream Orin Nano profile) with a comment pointing to `docker-compose --profile jetson` as the recommended invocation.

## Capabilities

### New Capabilities

- `jetson-orin-container`: Jetson-specific Dockerfile, compose profile, smoke test, and quickstart doc validated on L4T R36.4+ with Orin Nano / NX / AGX.

### Modified Capabilities

- `cuda-container-runtime`: extend `Session` thread-safety contract to require `Send + Sync` so CUDA-feature builds can be served from multi-threaded HTTP handlers, and to forbid future raw-pointer cache regressions.
- `docker-gpu-runtime`: extend with the Jetson L4T compose service and the runtime-detection rule that the container must report `compute 8.7` on Orin hardware before claiming GPU readiness.
- `arch/nvidia` feature surface: existing `tegra` feature semantics unchanged (Tegra X1 / cc 5.3 bare-metal HAL — preserves the `smallaios-arch-aarch64` Jetson Nano boot caller). New `tegra-orin` feature added for the Orin-family container path (cc 8.7).

## Impact

- **Code:**
  - `onnx-rt/src/cuda/graph.rs`, `graph_cache.rs`: newtype wrappers + `unsafe impl Send + Sync` with safety justification.
  - `onnx-rt/src/session.rs`: replace `RefCell` with `Mutex` (or move to the wrapped types). Verify `Session: Send + Sync`.
  - `arch/nvidia/Cargo.toml`: add `tegra-orin = ["cc_87"]` for the Orin-family container target. `tegra` (cc 5.3 / Tegra X1 bare-metal HAL) is unchanged.
  - `Dockerfile.jetson` (full JetPack base, convenience target): new file.
  - `Dockerfile.jetson.slim` (recommended runtime base, ~6 GB smaller): new file.
  - `docker-compose.yml`: new `smallaios-jetson` and `smallaios-jetson-slim` services under `jetson` and `jetson-slim` profiles.
- **Docs:** `docs/jetson-quickstart.md` (new), `README.md`, `CLAUDE.md`.
- **Tests:** `scripts/test-jetson-gpu.sh` (new), `Justfile` `test-jetson-gpu` recipe.
- **CI:** new advisory build job for `Dockerfile.jetson` under QEMU; new gate job for `cargo check --features cuda,nvidia_gpu`.
- **Container size:** `Dockerfile.jetson.slim` lands at 4.09 GB on the dev box (l4t-cuda runtime + 956 MB cuDNN stack). `Dockerfile.jetson` (full JetPack) is 9.83 GB on the dev box. For comparison, NVIDIA's `tritonserver:24.10-py3-igpu` is 9.04 GB on the same host — the slim variant is **55% smaller than Triton's published Jetson image**. The slim CPU image and the existing x86 Dockerfile.cuda remain unchanged.

- **Cold-start latency (measured on Jetson Orin NX, 5 runs, median, all `--runtime=nvidia`, SqueezeNet ONNX loaded GPU-side):**
  - smallaios slim → /healthz 200: **762 ms**
  - smallaios fat → /healthz 200: 901 ms
  - NVIDIA Triton → /v2/health/ready 200 (gRPC + metrics disabled, same model): **3,300 ms**
  - Docker baseline (`l4t-cuda` + `echo ok`, control): 769 ms
  - python:3.11-slim + stdlib `http.server` (no GPU work): 862 ms

  smallaios slim is **4.33× faster cold-start than Triton** while loading the same model on the same hardware. SmallAIOS's own init time is statistically indistinguishable from `docker run echo` on the same NVIDIA-runtime base — the smallaios contribution is bounded above by ~10 ms within measurement noise. Triton's 3.3 s comes from its ORT backend plugin load, server framework boot, and model repository scan, none of which smallaios incurs.

  Caveat: this measures cold-start, not steady-state inference throughput. Both stacks are GPU-bound by the same cuDNN kernels at runtime; the gap is at boot, which matters for FaaS / per-request-cold-start workloads and matters less for long-lived services that warm up once.
- **No new Rust crate dependencies.** All CUDA libs continue to come from the L4T base image at runtime via dynamic linking against `libcudart.so.12 / libcublas.so.12 / libcudnn.so.9`.
- **No bare-metal `arch/aarch64` Tegra driver changes.** This change is container-only. The bare-metal Tegra HAL (`arch/aarch64/src/tegra_*.rs`, `arch/nvidia/src/tegra/`) stays as-is.
