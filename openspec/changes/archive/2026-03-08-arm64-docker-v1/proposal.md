## Why

The SmallAIOS Dockerfile hardcodes `x86_64-unknown-linux-musl` as the build target, making it impossible to produce ARM64 container images. This blocks deployment to ARM64 hardware such as Jetson Nano (JetPack R32.7.6), AWS Graviton instances, and Apple Silicon Macs. The Jetson Nano has only ~1 GB free eMMC, so on-device builds are impractical -- images must be pulled from a container registry.

SmallAIOS already supports `aarch64-unknown-linux-musl` as a build target (`make build-container-arm`) and the Makefile `docker-build` target already invokes `docker buildx` with `--platform linux/amd64,linux/arm64`, but the Dockerfile itself does not handle the architecture mapping. The buildx invocation silently produces x86-only images.

**GitHub Issue:** #19

## What Changes

- **Dockerfile**: Use `TARGETARCH` (injected by `docker buildx`) to select the correct Rust target triple (`amd64` -> `x86_64-unknown-linux-musl`, `arm64` -> `aarch64-unknown-linux-musl`). Install the correct musl toolchain per architecture. Conditionally copy the binary from the architecture-specific output path.
- **CI workflow**: Add a job that builds multi-arch images on release tags and pushes them to GHCR (`ghcr.io/smallaios/smallaios`). The existing `docker-build-local` CI job validates the Dockerfile still builds on every PR.
- **Release workflow**: Add a `docker-publish` job that builds and pushes multi-arch images to GHCR when a version tag is pushed.
- **Makefile**: The `docker-build` and `docker-push` targets already use buildx; no changes needed since the Dockerfile will now handle architecture selection.

## Capabilities

### New Capabilities
- `docker-multiarch`: Dockerfile supports building for both `linux/amd64` and `linux/arm64` platforms via `docker buildx`, producing architecture-specific container images under 15 MB each.
- `ci-release-push`: CI/release workflow builds multi-arch images and pushes them to GHCR on version tags, producing a multi-arch manifest at `ghcr.io/smallaios/smallaios:<version>`.

### Modified Capabilities
- The existing `docker-build-local` CI job continues to validate x86-only builds on PRs (no buildx/QEMU required in CI for PR checks).

## Impact

- **`Dockerfile`**: Rewritten to use `ARG TARGETARCH` for architecture selection while preserving the existing two-stage build structure and GPU feature flag
- **`.github/workflows/release.yml`**: New `docker-publish` job added
- **`.github/workflows/ci.yml`**: `docker-build-local` job updated to also validate arm64 build via buildx (optional, depends on QEMU availability in CI)
- **No crate code changes** -- this change is Dockerfile, CI, and build tooling only
- **Image size constraint**: Both amd64 and arm64 images must remain under 15 MB
