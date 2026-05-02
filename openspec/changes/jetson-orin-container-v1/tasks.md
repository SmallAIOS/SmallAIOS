## 1. Reproduce + diagnose the CUDA build regression

- [x] 1.1 Confirm `cargo build --features cuda,nvidia_gpu` fails on aarch64 with seven `E0277` errors (captured in proposal)
- [x] 1.2 Map each error to its source: `RefCell<Option<CudaGraphCache>>`, `RefCell<Option<StreamPool>>`, `RefCell<Option<Arc<BTreeMap<String, Arc<DeviceTensor>>>>>`, raw `*mut c_void` cudaGraphExec_t in `onnx-rt/src/cuda/graph.rs:24` and `onnx-rt/src/cuda/graph_cache.rs:98`

## 2. Fix Send + Sync on Session GPU caches (Phase 1)

- [x] 2.1 In `onnx-rt/src/cuda/graph.rs`, replace the bare `*mut c_void` (cudaGraphExec_t) field with `pub(crate) struct ExecHandle(*mut core::ffi::c_void); unsafe impl Send for ExecHandle {} unsafe impl Sync for ExecHandle {}` and a `// SAFETY:` block citing CUDA's same-context-current contract
- [x] 2.2 Apply the same wrapper to `onnx-rt/src/cuda/graph_cache.rs:98` cached graph pointer
- [x] 2.3 In `onnx-rt/src/session.rs:312`, replace `RefCell<Option<CudaGraphCache>>` with `Mutex<Option<CudaGraphCache>>`. Update all call sites to `.lock().unwrap()` on the get-or-init path
- [x] 2.4 Same conversion for `RefCell<Option<StreamPool>>` and `RefCell<Option<Arc<BTreeMap<String, Arc<DeviceTensor>>>>>` (the device weight cache)
- [x] 2.5 Add a static assertion in `onnx-rt/src/session.rs`: `const _: fn() = || { fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<Session>(); };` — guarantees the regression cannot recur silently
- [x] 2.6 Run `cargo build --features cuda,nvidia_gpu` natively (host CPU, no link to libcudart needed via `cargo check`) — must pass
- [x] 2.7 Run `cargo test --features cuda` for any tests that exercise the cache types — verify no behavioral change

## 3. CI gate for the cuda feature

- [x] 3.1 Add a job `cuda-check` to `.github/workflows/ci.yml` that runs `cargo check --workspace --features cuda,nvidia_gpu` on `ubuntu-latest`
- [x] 3.2 Wire `cuda-check` into the `change-gates` meta-job so it blocks merge
- [x] 3.3 Verify on this PR that the new gate catches a deliberately-reverted-Send change (test-time only — revert local change after green run)

## 4. arch/nvidia tegra-orin feature

- [x] 4.1 In `arch/nvidia/Cargo.toml`, add `tegra-orin = ["cc_87"]` for the Orin-family container target. Leave `tegra = ["cc_53"]` (X1 bare-metal HAL) unchanged so the `smallaios-arch-aarch64` Jetson Nano boot path is not disturbed
- [x] 4.2 Doc-comment both features explaining when each one applies (HAL bare-metal vs. container userspace CUDA)
- [x] 4.3 Reference `tegra-orin` from `Dockerfile.jetson` build command

## 5. Dockerfile.jetson (Phase 2)

- [x] 5.1 Create `Dockerfile.jetson` based on `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (full JetPack convenience target, ~10 GB)
- [x] 5.2 Builder stage: install Rust nightly-2026-02-01 with `aarch64-unknown-linux-gnu` target
- [x] 5.3 Build command: `cargo build --release --target aarch64-unknown-linux-gnu -p smallaios-container --features cuda,nvidia_gpu,smallaios-arch-nvidia/tegra-orin`
- [x] 5.4 Runtime stage: same L4T base (cuDNN + CUDA already present), copy binary, set `SMALLAIOS_GPU_BACKEND=cuda` default
- [x] 5.5 Add `EXPOSE 8080`, `VOLUME /models`, `ENTRYPOINT ["/smallaios"]`
- [x] 5.6 Add `aarch64-unknown-linux-gnu` to `rust-toolchain.toml` `targets` so local devs can `rustup target add` it cleanly
- [x] 5.7 Add `RUSTFLAGS="-L /usr/local/cuda/lib64 -L /usr/local/cuda/lib64/stubs -L /usr/lib/aarch64-linux-gnu"` so the linker can find libcudart / libcublas / libcudnn at link time

## 5b. Dockerfile.jetson.slim (Phase 2, recommended variant)

- [x] 5b.1 Create `Dockerfile.jetson.slim` — JetPack builder, `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime` runtime
- [x] 5b.2 Copy `libcudnn*.so.9` and `*.so.9.3.0` files from the JetPack builder stage into `/usr/lib/aarch64-linux-gnu/` of the runtime stage; run `ldconfig`
- [x] 5b.3 Document the per-cuDNN-sub-library trim options (drop `libcudnn_engines_precompiled.so` for −510 MB or `libcudnn_adv.so` for −288 MB) in the Dockerfile header
- [x] 5b.4 Verify slim variant passes the same smoke test (compute 8.7, SqueezeNet `[GPU]`, /v1/inference 200/400) on Jetson Orin NX

## 6. docker-compose Jetson profiles

- [x] 6.1 Add `smallaios-jetson` service under `profiles: [jetson]` (Dockerfile.jetson)
- [x] 6.2 Add `smallaios-jetson-slim` service under `profiles: [jetson-slim]` (Dockerfile.jetson.slim)
- [x] 6.3 Use `runtime: nvidia` and `NVIDIA_VISIBLE_DEVICES=all` on both
- [x] 6.4 Use `/healthz` for the healthcheck (CPU service uses `/health`; the Jetson and GPU services correctly use `/healthz` per existing convention)
- [x] 6.5 Mount `./models:/models:ro`
- [x] 6.6 Update the comment block at the top of `docker-compose.yml` to list the new profile and image-size delta

## 7. Smoke test script

- [x] 7.1 Create `scripts/test-jetson-gpu.sh` (chmod +x) that:
  - takes an optional `slim` argument to switch to the slim variant; default = full JetPack
  - downloads SqueezeNet to `models/squeezenet.onnx` if absent
  - runs `docker compose --profile jetson{,-slim} up -d --build smallaios-jetson{,-slim}`
  - polls `/healthz` and `/readyz` until ready or 120 s timeout
  - asserts logs include `compute 8.7` (with retry loop + `--no-color` to handle slow log flush on the slim variant)
  - posts a minimal payload to `/v1/inference` and asserts 200 or 400 (probe — proves the endpoint is reachable; full numerics are covered by onnx-rt integration tests)
  - tears the service down on success or failure (always cleans up)
- [x] 7.2 Add `test-jetson-gpu [variant]` recipe to `Justfile` calling the script
- [x] 7.3 Document expected vs failure exit codes inside the script header

## 8. Documentation (Phase 3)

- [x] 8.1 Create `docs/jetson-quickstart.md` covering: hardware support matrix (Orin Nano/NX/AGX), required JetPack version (≥ 6 / L4T R36.4), full vs slim image variants table, one-command run via `docker compose --profile jetson-slim up`, sample `/v1/inference` curl, troubleshooting CDI driver-mismatch errors, deeper-trim options (drop libcudnn_engines_precompiled / libcudnn_adv)
- [x] 8.2 Add a Jetson row to `README.md`'s deployment matrix with a link to the quickstart
- [x] 8.3 Update `CLAUDE.md` "Current state" to note Jetson Orin GPU validation, and add `Dockerfile.jetson` to the Build Configuration / Container Environment Variables sections as appropriate
- [ ] 8.4 Cross-link `docs/jetson-quickstart.md` from `docs/safetensors-integration.md` and `docs/inference-bus.md` if the Jetson story changes anything for those subsystems (likely a single sentence each) — DEFERRED: nothing in the Jetson path changes either subsystem's contract
- [ ] 8.5 Add a CHANGELOG entry under the next release describing both the Send/Sync fix and the new Jetson container path — DEFERRED: CHANGELOG is auto-generated by `git-cliff` from conventional commits per `docs/release-runbook.md`

## 9. CI advisory build

- [x] 9.1 Add a `jetson-image-build` job to `.github/workflows/ci.yml` running `docker buildx build -f Dockerfile.jetson --platform linux/arm64 --load .` with `continue-on-error: true`
- [x] 9.2 Cache the L4T base layer with `cache-from`/`cache-to: type=gha`
- [x] 9.3 Add a comment block in the workflow file noting "promote to gate when self-hosted Jetson runner is available"

## 10. End-to-end validation on real hardware

- [x] 10.1 Run `just test-jetson-gpu` on the dev Jetson Orin NX (L4T R36.4.7, JetPack 6, driver 540.4.0) — passes for both fat and slim variants
- [x] 10.2 Capture the boot log with `compute 8.7` and the successful `/v1/inference` response into the PR description — recorded: `CUDA initialized: Orin (compute 8.7, 15655 MB VRAM, CUDA 12)`, `Session ready: squeezenet [GPU]`, `/v1/inference` returns 200 with [1, 1000] f32 logits
- [x] 10.3 Run `nvidia-smi` inside the container to confirm GPU visibility (or note the equivalent `tegrastats` workflow if `nvidia-smi` is not present) — used `tegrastats` (host-side, since Jetson `nvidia-smi` is limited): GR3D_FREQ baseline 0% → bursts up to 23% during sustained inference → 0% after; 305/2822 samples non-zero
- [x] 10.4 Compare the GPU inference output against CPU-mode output for SqueezeNet within 1e-3 relative tolerance — log result in PR. Result: max |gpu - cpu| = 5.07e-4, mean 1.62e-4, well within tolerance

## 11. Verify + archive

- [x] 11.1 Run `openspec validate jetson-orin-container-v1 --strict`
- [ ] 11.2 Open PR against `develop` with proposal/design/tasks/specs and the implementation
- [ ] 11.3 After merge, run `/opsx:archive jetson-orin-container-v1` (will move to `openspec/changes/archive/2026-05-02-jetson-orin-container-v1`)
