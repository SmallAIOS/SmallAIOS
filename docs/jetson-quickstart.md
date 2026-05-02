# Jetson Orin Quickstart

This walks you through building and running SmallAIOS as a GPU-accelerated
container on an NVIDIA Jetson Orin (Nano Super, NX, or AGX).

## Hardware support matrix

| Board                 | SoC      | GPU        | CC  | Tested |
|-----------------------|----------|------------|-----|--------|
| Jetson Orin Nano Super (8 GB) | Orin     | Ampere | 8.7 | yes (NX dev box) |
| Jetson Orin NX (8 / 16 GB)    | Orin     | Ampere | 8.7 | yes |
| Jetson AGX Orin (32 / 64 GB)  | Orin     | Ampere | 8.7 | architectural — same image |
| Jetson Nano (original)        | Tegra X1 | Maxwell | 5.3 | NO — see "Why not Nano original?" below |

The Orin family ships with JetPack 6 / L4T R36.4+ and a fixed CUDA 12.6
user-mode stack. SmallAIOS's container build links against
`libcudart.so.12 / libcublas.so.12 / libcudnn.so.9` from that base image
and dynamically picks up the host driver shim (`libcuda.so.1`) at runtime.

## Prerequisites

- JetPack 6 (≥ L4T R36.4) — confirm with `cat /etc/nv_tegra_release`. The
  first line should look like `# R36 (release), REVISION: 4.7, ...`.
- Docker Engine (≥ 24).
- NVIDIA Container Toolkit (`nvidia-container-toolkit`). Confirm with
  `docker info | grep nvidia` — you should see `nvidia` listed under
  `Runtimes:` and ideally as `Default Runtime:` for `--runtime=nvidia`
  to be implicit.
- ~10 GB free disk space (the L4T base image is ~3 GB; build artifacts
  add another few GB).

## Two image variants

There are two Jetson Dockerfiles. Pick based on whether you value image
size or a full JetPack toolbox:

| Variant | Dockerfile | Compose profile | Image size | Includes |
|---------|-----------|-----------------|-----------|----------|
| **Slim (recommended)** | `Dockerfile.jetson.slim` | `jetson-slim` | **~4 GB** | CUDA 12.6 runtime + cuDNN 9.3 only |
| Full JetPack | `Dockerfile.jetson` | `jetson` | ~10 GB | CUDA + cuDNN + TensorRT + VPI + NPP + multimedia API + DLA compiler |

Both run the same SmallAIOS binary on the same Tegra Orin GPU and
report identical `compute 8.7` boot lines. The slim variant copies just
the cuDNN `.so` files from the JetPack builder stage onto an
`l4t-cuda:12.6.11-runtime` base; it has full SmallAIOS Conv / fused-op
coverage. The full variant exists for users who want the rest of
JetPack pre-installed (e.g. for ad-hoc tooling, samples, or future
TensorRT integration). **For most production deployments, use the
slim variant.**

## One-command run

```bash
# Slim (recommended)
docker compose --profile jetson-slim up --build

# Full JetPack base
docker compose --profile jetson up --build
```

After ~30s the service serves on port 8080.

To smoke-test end-to-end:

```bash
just test-jetson-gpu          # full JetPack variant
just test-jetson-gpu slim     # slim variant
# or directly:
./scripts/test-jetson-gpu.sh slim
```

The smoke test downloads SqueezeNet, builds the image, asserts the
container reports `compute 8.7` (proving the integrated Tegra GPU was
probed, not a silent CPU fallback), and confirms `/v1/inference` is
reachable.

## Manual flow

```bash
# 1. Get a model. Anything in models/ is loaded at boot.
mkdir -p models
curl -L \
  -o models/squeezenet.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx

# 2. Build the Jetson-specific image.
just docker-build-jetson
# or: docker build -f Dockerfile.jetson -t smallaios:jetson .

# 3. Run it. --runtime=nvidia is implicit on Jetson where the default
#    Docker runtime is already nvidia, but pass it explicitly to be safe.
docker run --runtime=nvidia --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e SMALLAIOS_GPU_BACKEND=cuda \
  smallaios:jetson

# 4. Verify GPU init from the logs (look for `compute 8.7`):
#    SmallAIOS 0.2.1
#    Container mode: ...
#    Config: model_dir=/models, port=8080, gpu=cuda, ...
#    GPU precision mode: TF32
#    CUDA initialized: Orin (compute 8.7, 8192 MB VRAM, CUDA 12.6)
#    Loading models from '/models'...
#      Model 'squeezenet': 4956208 bytes [onnx] OK
#    HTTP inference sessions ready: 1
#    Ready. Listening on 0.0.0.0:8080

# 5. Health checks.
curl http://localhost:8080/healthz   # {"status":"healthy"}
curl http://localhost:8080/readyz    # {"status":"ready"}
curl http://localhost:8080/v1/models # [{"name":"squeezenet","file_size":4956208,"loaded":true}]
```

## Troubleshooting

### "requirements not met: cuda>=13.0..."

```
docker: Error response from daemon: failed to create task for container:
... failed to construct OCI spec modifier: requirements not met:
cuda>=13.0||brand=...&&driver>=535... not met
```

You're trying to run an x86 + discrete-GPU image (`Dockerfile.cuda` / NGC
`nvidia/cuda:13.0`) on a Jetson. The NVIDIA Container Toolkit's CDI
runtime checks the image's `com.nvidia.cuda.requirements` label against
the Jetson's driver (540.x for L4T R36.4) and rejects 13.0. Fix: use
`Dockerfile.jetson` (this guide) instead.

### "compute 8.7" missing — silent CPU fallback

If the logs show `WARNING: GPU backend=cuda requested but CUDA init failed`
and falling back to CPU, the most common causes are:

- The container was started without `--runtime=nvidia` (or the default
  runtime isn't `nvidia` and the `runtime: nvidia` in `docker-compose.yml`
  was lost). Fix: `docker compose --profile jetson up` always sets
  `runtime: nvidia`; verify the override didn't get clobbered.
- `nvidia-container-toolkit` isn't installed. Fix:
  `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker`.
- The Jetson is running in low-power mode and `nvgpu` isn't loaded. Fix:
  `sudo nvpmodel -m 0 && sudo jetson_clocks` (production-mode clocks).

### Image is huge

If you built the full JetPack image (`Dockerfile.jetson`, profile
`jetson`) you got ~10 GB. About 95% of that is unused JetPack 6 stack
(TensorRT, VPI, NPP, multimedia, samples, nvcc) — SmallAIOS only links
against CUDA runtime + cuBLAS + cuDNN.

**Just use the slim variant** (`Dockerfile.jetson.slim`, profile
`jetson-slim`). It comes in at ~4 GB, passes the same smoke test, and
keeps full Conv / fused-op coverage. The size difference is purely the
JetPack toolbox you weren't using.

If 4 GB is still too big you can drop further by editing
`Dockerfile.jetson.slim`:

- Remove `libcudnn_engines_precompiled.so.9*` (510 MB) — at the cost of
  cold-start JIT compilation for cuDNN kernels on the first inference.
- Remove `libcudnn_adv.so.9*` (288 MB) — only safe if your workload
  has no RNN, LSTM, or multi-head attention ops.

Both are documented in the comments at the top of the slim Dockerfile.

### Why not Jetson Nano (original)?

The original Jetson Nano runs a Tegra X1 (Maxwell, cc 5.3) on JetPack 4.x
with CUDA 10.2. The L4T base images for Tegra X1 are end-of-life as of
JetPack 4.6.4, and SmallAIOS's CUDA path uses cuDNN 9 / cuBLAS APIs that
are not available on CUDA 10.2. We do not support Tegra X1 in the
container path; the bare-metal HAL under `arch/nvidia/src/tegra/` is
present for the X1 boot path (see `arch/aarch64/Cargo.toml` —
`smallaios-arch-nvidia/tegra` feature) and is independent of this
container target.

## Performance — measured cold-start

Benchmarked on a Jetson Orin NX (16 GB), JetPack 6 / L4T R36.4.7,
NVIDIA Container Runtime as default Docker runtime. Metric: wall-clock
time from `docker run` to the first 200 response from the readiness
endpoint, with the SqueezeNet 1.1 ONNX model loaded GPU-side. 5 runs
each, median reported.

| Container | Work done at ready signal | Median | × slim | Image |
|-----------|--------------------------|-------:|-------:|------:|
| `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime` + `echo ok` (control — pure Docker overhead) | nothing | 769 ms | 1.01× | 2.09 GB |
| **smallaios slim** (`Dockerfile.jetson.slim`) | CUDA init + SqueezeNet → GPU session + HTTP listen | **762 ms** | **1.00×** | **4.09 GB** |
| smallaios fat (`Dockerfile.jetson`) | same as slim | 901 ms | 1.18× | 9.83 GB |
| `python:3.11-slim` + stdlib `http.server` (no GPU) | interpreter + HTTP listen | 862 ms | 1.13× | 0.13 GB |
| **NVIDIA Triton** (`tritonserver:24.10-py3-igpu`) + SqueezeNet, gRPC + metrics disabled | ORT backend + SqueezeNet → /v2/health/ready | **3,300 ms** | **4.33×** | **9.04 GB** |

**Takeaway:** smallaios slim cold-starts ~4.3× faster than NVIDIA Triton
on the same hardware with the same model loaded, in an image 55%
smaller. The smallaios init time is at the floor that the NVIDIA
Container Runtime hook allows on Jetson — there is no measurable
difference between "smallaios is GPU-ready and serving SqueezeNet" and
"the L4T base image's `echo ok` exited."

**Caveats:**
- Triton supports many more backends (TensorRT, PyTorch, custom Python),
  dynamic batching, ensembles, gRPC, model versioning, auth. SmallAIOS
  does ONNX-only with a fixed operator set.
- Steady-state inference latency is GPU-bound by the same cuDNN kernels
  in both — the gap is at cold-start, not throughput.
- The 4.3× number is for SqueezeNet specifically; larger models shift
  both numbers up but not necessarily proportionally.
- Run on a single dev box; numbers will vary on other Orin SKUs and
  with different CUDA / JetPack versions.

To reproduce on your own hardware, see the bench scripts in this PR's
description (or equivalent: `time` a `docker run -d` followed by
polling `/healthz` / `/v2/health/ready` until 200).

## Running the SmallAIOS smoke test

```bash
just test-jetson-gpu
```

That script:

1. Confirms `nvidia-container-toolkit` is installed.
2. Downloads SqueezeNet to `./models` if absent.
3. Builds and starts the Jetson service.
4. Polls `/healthz` + `/readyz` until ready (≤ 120 s).
5. Asserts `compute 8.7` is in the logs (catches silent CPU fallback).
6. Hits `POST /v1/inference` against SqueezeNet.
7. Tears the service down on exit.

Exit codes are documented in the script header.

## Related docs

- [Inference bus](./inference-bus.md) — pub/sub topic conventions when
  pairing Jetson with Zenoh / DDS dataflow.
- [CAN inference](./can-inference.md) — CAN bus inference bridge for
  vehicle / industrial integrations.
- [Architecture](./architecture.md) — workspace layering (the Jetson
  container path lives entirely in Layer 3 + Layer 1 userspace; no
  Layer 2 bare-metal HAL involvement).
- [Local testing](./local-testing.md) — general developer workflow.
