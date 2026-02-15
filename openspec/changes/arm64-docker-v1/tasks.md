## Phase 1: Dockerfile Multi-Arch Support

- [ ] T1: Add TARGETARCH argument and architecture mapping to Dockerfile
  Add `ARG TARGETARCH` to the builder stage. Add a `RUN` step that maps `TARGETARCH` values (`amd64` -> `x86_64-unknown-linux-musl`, `arm64` -> `aarch64-unknown-linux-musl`) to a Rust target triple stored in `/tmp/rust-target`. Fail with a clear error for unsupported architectures.

- [ ] T2: Update rustup target add to use dynamic target triple
  Replace the hardcoded `rustup target add x86_64-unknown-linux-musl` with `rustup target add $(cat /tmp/rust-target)` so the correct musl target is installed for the build architecture.

- [ ] T3: Update cargo build to use dynamic target triple
  Replace the hardcoded `--target x86_64-unknown-linux-musl` in the `cargo build` command with `--target "$RUST_TARGET"` where `RUST_TARGET` is read from `/tmp/rust-target`. Preserve the existing `ENABLE_GPU` conditional logic.

- [ ] T4: Update COPY instruction for architecture-independent binary path
  Replace the hardcoded `COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/smallaios-container` with a pattern that works for both architectures, such as using a glob or a second `ARG TARGETARCH` in the runtime stage to compute the correct path.

- [ ] T5: Test Dockerfile builds for linux/amd64
  Run `docker buildx build --platform linux/amd64 -t smallaios:test-amd64 .` and verify the image contains a working x86_64 binary and is under 15 MB.

- [ ] T6: Test Dockerfile builds for linux/arm64
  Run `docker buildx build --platform linux/arm64 -t smallaios:test-arm64 .` and verify the image contains a working aarch64 binary and is under 15 MB. (Requires QEMU user-mode emulation or an arm64 host.)

- [ ] T7: Test multi-arch manifest build
  Run `docker buildx build --platform linux/amd64,linux/arm64 -t smallaios:test-multi .` and verify a multi-arch manifest is created with both architectures.

- [ ] T8: Test GPU build arg works on both architectures
  Run `docker buildx build --platform linux/amd64 --build-arg ENABLE_GPU=1 .` and `docker buildx build --platform linux/arm64 --build-arg ENABLE_GPU=1 .` and verify both compile with the `nvidia_gpu` feature.

## Phase 2: Makefile Updates

- [ ] T9: Verify existing docker-build target works with updated Dockerfile
  Run `make docker-build` and verify it produces a multi-arch build using the updated Dockerfile. No Makefile changes should be needed since it already uses `--platform linux/amd64,linux/arm64`.

- [ ] T10: Verify existing docker-push target works with updated Dockerfile
  Run `make docker-push` and verify it pushes a multi-arch manifest. (Requires registry credentials.)

## Phase 3: CI/Release Workflow

- [ ] T11: Add docker-publish job to release workflow
  Add a `docker-publish` job to `.github/workflows/release.yml` that: sets up QEMU emulation, sets up Docker Buildx, logs in to GHCR using `GITHUB_TOKEN`, builds for `linux/amd64,linux/arm64`, and pushes to `ghcr.io/smallaios/smallaios` with version and latest tags.

- [ ] T12: Add QEMU and Buildx setup actions
  Use `docker/setup-qemu-action@v3` and `docker/setup-buildx-action@v3` in the docker-publish job to enable cross-architecture builds on the amd64 CI runner.

- [ ] T13: Add GHCR login step
  Use `docker/login-action@v3` with `registry: ghcr.io`, `username: ${{ github.actor }}`, `password: ${{ secrets.GITHUB_TOKEN }}`. Add `packages: write` to the job permissions.

- [ ] T14: Add build-push step with version tagging
  Use `docker/build-push-action@v6` to build and push. Tag with `ghcr.io/smallaios/smallaios:${{ github.ref_name }}`. Conditionally add `latest` tag only for non-pre-release versions.

- [ ] T15: Add image size check to docker-publish job
  After the build-push step, pull each per-architecture image and verify its size is under 15 MB. Fail the job if either exceeds the limit.

- [ ] T16: Verify CI docker-build-local job still passes
  Ensure the existing `docker-build-local` job in `ci.yml` continues to build the native (amd64) image and check its size. No changes to this job are expected.

## Phase 4: Testing and Documentation

- [ ] T17: End-to-end test: build and run arm64 image on ARM64 hardware
  Pull or load the arm64 image on an ARM64 device (e.g., Jetson Nano) and verify the container starts and the health check endpoint responds. (Requires ARM64 hardware.)

- [ ] T18: Verify image sizes for both architectures
  Inspect the final images for both `linux/amd64` and `linux/arm64` and record their sizes. Both must be under 15 MB.

- [ ] T19: Run make clippy and make fmt-check
  Verify no Rust lint warnings or formatting issues are introduced. (This change should not affect Rust code, but verify.)

- [ ] T20: Verify existing docker-compose local development still works
  Run `make docker-local` and verify the docker-compose workflow is unaffected by the Dockerfile changes.
