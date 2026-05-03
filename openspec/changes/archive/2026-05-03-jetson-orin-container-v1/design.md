## Context

`Session` is the central object that owns model state, optimized graphs, and (with `feature = "cuda"`) GPU-side caches. The `arm64-gpu-container-v1` and `cuda-graphs-v1` changes both touched this type without exercising the multi-threaded code path that `container/src/main.rs` uses.

Concretely, `container/src/main.rs:125` constructs a per-model `Session` and stores it inside the route closure passed to `HttpServer::route_fn`. The route function takes `Fn + Send + Sync + 'static` because `HttpServer` worker threads dispatch requests from a thread pool. `cuda-graphs-v1` then added `RefCell<Option<CudaGraphCache>>` and a raw `*mut c_void` (cudaGraphExec_t handle) into `Session`. Once `--features cuda` is enabled, those fields propagate `!Send` / `!Sync` upward — the closure is rejected at compile time. The error is invisible without the feature, which is why the `develop` branch is green while `cargo build --features cuda` is broken.

On the deployment side, `Dockerfile.cuda` uses `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`. NVIDIA's CDI runtime evaluates the image's `com.nvidia.cuda.requirements` label against the host driver before mounting GPU files; CUDA 13 requires driver ≥ 535 in one of several version bands (535.x, 550.x, 565.x, 570.x, 575.x). Jetson's L4T R36.4 ships driver 540.4.0 — outside every band — so the runtime errors out: *"requirements not met: cuda>=13.0||..."*. There is no workaround at the container layer; JetPack 6 is fixed at CUDA 12.6 user-mode and an L4T-derived base image is the only correct choice.

## Goals / Non-Goals

**Goals**

- Make `cargo build --features cuda,nvidia_gpu` succeed on every supported target (x86_64-linux-{musl,gnu}, aarch64-linux-{musl,gnu}). This is a compile-only fix; no runtime behavior changes for existing GPU users.
- Provide a working `Dockerfile.jetson` that boots smallaios end-to-end on Jetson Orin (Nano / NX / AGX — they share cc 8.7), running real ONNX inference (SqueezeNet at minimum) on the integrated Tegra GPU.
- Catch future `cuda`-feature regressions in CI without requiring a GPU runner.

**Non-Goals**

- TensorRT integration. JetPack ships TensorRT, but smallaios's GPU executor uses cuDNN+cuBLAS directly. TRT is a separate, future change.
- Tegra X1 (Jetson Nano original) container support. cc 5.3 is end-of-life on JetPack 4.x and we're not investing in the legacy path.
- Bare-metal Tegra GPU bring-up. The bare-metal Tegra HAL is its own track.
- Replacing `Dockerfile.cuda`. The x86 + discrete-GPU image continues to use CUDA 13.0 NGC images for DGX Spark / standard server hosts.
- Running TensorRT-format engines. SmallAIOS loads ONNX directly.

## Decisions

### Decision 1 — `Mutex<Option<T>>` over `unsafe impl Send` for higher-level caches

`StreamPool`, `CudaGraphCache`, and the `Arc<BTreeMap<String, Arc<DeviceTensor>>>` weight cache are all *Rust types* whose unsoundness comes from interior `RefCell`. Switching to `Mutex<Option<T>>` is mechanical: the same get-or-init pattern keeps working, lock contention is negligible (these are one-shot inits and per-request reads), and we avoid sprinkling `unsafe` for what is fundamentally a data-race protection problem.

For the raw CUDA handles (`*mut c_void` cudaGraphExec_t in `cuda::graph::CudaGraphExec`), `Mutex` would not change anything — the pointer remains `!Send` at the type-system level regardless of locking. Here we **do** need `unsafe impl Send + Sync` on a newtype wrapper, with a justification that captures CUDA's actual contract:

> CUDA primary contexts are bound to a process and become "current" on whichever thread last set them. `cudaGraphLaunch` is documented as thread-safe with respect to the same graph handle as long as the calling thread has a current context for the same device. `Session` pins itself to a single device at construction (cudaSetDevice), and our HTTP request handler always re-asserts `cudaSetDevice(self.device)` before launching. Therefore the `cudaGraphExec_t` pointer can be transferred between worker threads safely.

### Decision 2 — Ship both `Dockerfile.jetson` (full) and `Dockerfile.jetson.slim`

Initial impulse was to ship only the full JetPack image because it's
mechanically simpler — no cuDNN install step, no repo-key risk, just one
`FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0`. After validating end-to-end on
the dev box, that image came in at **9.83 GB**. SmallAIOS dynamically
links against six CUDA-side .so files; everything else in JetPack 6
(TensorRT, VPI, NPP, multimedia API, DLA compiler, CUDA samples, nvcc)
is dead weight.

The slim variant uses `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime` as the
runtime base (2 GB) and copies just the cuDNN .so files from the
JetPack builder stage. We don't try to apt-install cuDNN at build time
— NVIDIA's L4T cuDNN apt package isn't pre-configured on l4t-cuda
images and pinning the repo / GPG key adds churn for marginal layer
benefit. Copying the files we need is reliable, version-locked to the
builder, and ~956 MB total. The slim image lands at **4.09 GB** on the
dev box and passes the identical smoke test (compute 8.7 boot,
SqueezeNet `[GPU]`, /v1/inference reachable).

We ship both rather than only the slim because:

- `Dockerfile.jetson` is a one-line template anyone can extend without
  thinking about cuDNN file lists. It's a useful "kitchen sink"
  starting point for users who plan to adopt TensorRT or VPI later.
- `Dockerfile.jetson.slim` is what production deployments should use,
  and it's the variant the README points to as recommended.

The README, CLAUDE.md, and `docs/jetson-quickstart.md` all surface the
slim variant first and label it "recommended."

### Decision 3 — `aarch64-unknown-linux-gnu`, not -musl

The existing `Dockerfile` and `Dockerfile.cuda` target `aarch64-unknown-linux-musl` because the runtime is `FROM scratch`. On Jetson, the runtime base is L4T (Ubuntu 22.04) which is glibc, *and* `libcuda.so.1` mounted by NVIDIA Container Toolkit is glibc-linked. Mixing musl + glibc dynamic loaders works only by accident and would break the moment cuDNN's static initializer touches anything pthread-related (TLS layouts differ).

We add `aarch64-unknown-linux-gnu` to `rust-toolchain.toml`'s `targets` and use it in `Dockerfile.jetson`. The slim x86 / ARM64 musl path is unaffected.

### Decision 4 — Add `tegra-orin`, leave `tegra` alone

The first draft of this change rewrote `tegra = ["cc_53"]` to `tegra = ["cc_87"]` and renamed the legacy flag to `tegra-x1`. That broke the existing `smallaios-arch-aarch64 = { ..., features = ["tegra"] }` dependency, which uses the X1-specific bare-metal HAL under `arch/nvidia/src/tegra/` (Falcon ucode / GM20B GR / FIFO / GMMU register drivers — none of that applies to Orin).

We instead add a new `tegra-orin = ["cc_87"]` feature and leave `tegra = ["cc_53"]` untouched. The Orin container path is purely userspace CUDA (cuDNN + cuBLAS via the NVIDIA Container Toolkit's CDI mount); it does not touch the bare-metal HAL. Decoupling the two paths keeps the existing Jetson Nano boot caller working without migration churn and matches the intuition that "Tegra HAL" and "Tegra container target" are independent concerns.

### Decision 5 — Compose profile separation

`docker-compose.yml` already has a `gpu` profile that targets the x86 + discrete-GPU `Dockerfile.cuda` flow. Adding the Jetson service under the same profile would silently target the wrong base image on x86 hosts and the wrong CDI label on Jetson hosts. We add a distinct `jetson` profile so the user picks the deployment surface explicitly.

### Decision 6 — CI gate strategy

Two new jobs:

- **Gate job** (blocks PR merge): `cargo check --features cuda,nvidia_gpu` on a default GitHub-hosted runner. `cargo check` does not link, so it does not need CUDA libraries — only the headers, which `bindgen`-style FFI does not consume in this codebase (it uses hand-rolled `extern "C"` declarations). This catches all the `Send`/`Sync` regressions cheaply.
- **Advisory job**: `docker buildx build -f Dockerfile.jetson --platform linux/arm64` under QEMU emulation. Slow (~10 min) but catches Dockerfile syntactic rot. `continue-on-error: true` because no GitHub-hosted runner has a Jetson, and we cannot do a runtime smoke test there.

Real Jetson smoke testing happens on the user's hardware via `just test-jetson-gpu` and is documented in `docs/jetson-quickstart.md`. Once a self-hosted Jetson runner exists, the advisory job promotes to a gate.

## Risks / Trade-offs

- **3 GB image weight is uncomfortable** but matches NVIDIA's own published Jetson containers (l4t-pytorch, l4t-tensorflow, l4t-tensorrt are all 3-7 GB). Customers running the slim 15 MB CPU container continue to do so unchanged.
- **`unsafe impl Send + Sync` is a real safety claim.** If we get the safety reasoning wrong (e.g. someone later spawns a worker that tries to launch a graph against a stale CUDA context), we get a use-after-free, not a panic. Mitigation: the `unsafe impl` lives on a tiny newtype with a doc comment that names the invariant; a `// TODO(jetson): consider Mutex` marker would invite a future tightening.
- **Aarch64-gnu adds a fourth Rust target** the workspace has to keep installable. Risk is mainly CI minute cost on Linux distros where `rustup target add aarch64-unknown-linux-gnu` is a fast no-op.
- **No breaking change for Tegra X1 callers.** `tegra` continues to mean Tegra X1 / cc 5.3. The Orin container target uses the new `tegra-orin` flag instead. Cost: one extra feature name to remember; benefit: zero migration burden on the Jetson Nano boot path.

## Migration Plan

1. Land Phase 1 first as a self-contained PR — Send/Sync fixes plus the `cargo check --features cuda` gate. This unbreaks any future GPU work on `develop`.
2. Land Phase 2 + 3 as a follow-up PR. The two phases share the worktree and the OpenSpec change but split cleanly at the code-vs-deployment boundary.
3. Run `just test-jetson-gpu` on the Jetson Orin NX dev box to confirm end-to-end inference. Capture the actual `cudaGetDeviceCount` + `compute 8.7` lines into the PR description as evidence.
4. After merge, archive the change as `2026-05-02-jetson-orin-container-v1`.

## Open Questions

- Should the bare-metal Tegra X1 driver path under `arch/nvidia/src/tegra/` migrate behind the new `tegra-x1` feature too, for symmetry? (Probably yes; out of scope for this change but worth a follow-up note.)
- Do we want to publish `smallaios:jetson` to GHCR as a prebuilt image? Skipped here to keep the change focused on bring-up; revisit when a self-hosted Jetson runner exists.
