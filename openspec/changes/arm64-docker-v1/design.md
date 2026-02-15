## Context

SmallAIOS container mode produces a statically-linked musl binary targeting either `x86_64-unknown-linux-musl` or `aarch64-unknown-linux-musl`. The current Dockerfile hardcodes the x86_64 target at three points: `rustup target add`, `cargo build --target`, and `COPY --from=builder`. The Makefile already has `docker buildx build --platform linux/amd64,linux/arm64` but the Dockerfile ignores the platform, producing broken arm64 images.

Docker BuildKit automatically injects `TARGETARCH` (values: `amd64`, `arm64`) and `TARGETPLATFORM` (values: `linux/amd64`, `linux/arm64`) as build args when using `docker buildx build --platform`. The Dockerfile must map these to Rust target triples.

The container binary is ~594 KB (x86_64), well under the 15 MB limit. The arm64 binary is expected to be similar in size.

## Goals / Non-Goals

**Goals:**
- `docker buildx build --platform linux/arm64 -t smallaios .` produces a working aarch64 container image
- `docker buildx build --platform linux/amd64,linux/arm64` produces a multi-arch manifest
- Multi-arch images are pushed to GHCR on release tags via CI
- Both architecture images remain under 15 MB
- Existing GPU feature flag (`ENABLE_GPU`) continues to work for both architectures
- Existing `docker-build-local` CI job continues to pass

**Non-Goals:**
- RISC-V container images (no musl target readily available for riscv64 in Docker)
- Building on ARM64 CI runners (use QEMU emulation via buildx)
- On-device builds for Jetson Nano (image is pulled from registry)
- Modifying any Rust crate code
- Multi-arch builds for the bare-metal kernel (separate from container images)

## Decisions

### 1. Dockerfile architecture mapping via TARGETARCH

**Decision:** Use `ARG TARGETARCH` with a shell case statement to map Docker architecture names to Rust target triples and musl toolchain package names.

Mapping:
| TARGETARCH | Rust target | musl package |
|------------|------------|--------------|
| `amd64` | `x86_64-unknown-linux-musl` | `musl-tools` (provides `musl-gcc` for x86_64) |
| `arm64` | `aarch64-unknown-linux-musl` | `gcc-aarch64-linux-gnu musl-tools` |

**Dockerfile structure:**
```dockerfile
FROM rustlang/rust:nightly-slim AS builder

ARG TARGETARCH
ARG ENABLE_GPU=0

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*

# Map Docker arch to Rust target triple
RUN case "$TARGETARCH" in \
      amd64) echo "x86_64-unknown-linux-musl" > /tmp/rust-target ;; \
      arm64) echo "aarch64-unknown-linux-musl" > /tmp/rust-target ;; \
      *) echo "Unsupported architecture: $TARGETARCH" && exit 1 ;; \
    esac && \
    rustup target add $(cat /tmp/rust-target)

WORKDIR /app
COPY . .

RUN RUST_TARGET=$(cat /tmp/rust-target) && \
    if [ "$ENABLE_GPU" = "1" ]; then \
      cargo build --release --target "$RUST_TARGET" \
        -p smallaios-container --features nvidia_gpu; \
    else \
      cargo build --release --target "$RUST_TARGET" \
        -p smallaios-container; \
    fi

FROM scratch
ARG TARGETARCH
COPY --from=builder /app/target/*/release/smallaios-container /smallaios
ENTRYPOINT ["/smallaios"]
```

**Why `TARGETARCH` instead of `TARGETPLATFORM`?** `TARGETARCH` gives just the architecture string (`amd64`/`arm64`), which is cleaner to map to Rust targets than `linux/amd64`.

**Why a file (`/tmp/rust-target`) instead of an ENV?** Docker `ARG` values are scoped per stage and cannot be used across `RUN` instructions as dynamic variables without either a file or `ENV`. A file avoids polluting the environment and works cleanly with `RUN` instructions that need the value.

**Cross-compilation note:** When building arm64 on an amd64 host via buildx, Docker uses QEMU user-mode emulation. The Rust compiler runs under QEMU, so no cross-compilation toolchain is needed -- `cargo build` runs natively (emulated) for the target architecture. This is slower than cross-compilation but simpler and avoids linker configuration issues.

### 2. COPY path for multi-arch binary

**Decision:** Use a glob pattern in the COPY instruction to avoid hardcoding the architecture in the path:

```dockerfile
COPY --from=builder /app/target/*/release/smallaios-container /smallaios
```

This works because only one target directory exists in the builder stage (either `x86_64-unknown-linux-musl` or `aarch64-unknown-linux-musl`).

**Alternative considered:** Using a second `ARG TARGETARCH` in the runtime stage and computing the path. This is more explicit but adds complexity. The glob approach is simpler and safe because the builder stage only builds for one target.

### 3. CI: Multi-arch build validation on PRs

**Decision:** Keep the existing `docker-build-local` CI job building only the native (amd64) image for PR validation. Multi-arch builds with QEMU emulation are slow (~10-15 min for arm64) and not worth running on every PR.

The multi-arch build is validated in the release workflow, which runs only on version tags. This is acceptable because the Dockerfile logic is straightforward (a case statement) and rarely changes.

### 4. Release workflow: GHCR push

**Decision:** Add a `docker-publish` job to `.github/workflows/release.yml` that:
1. Sets up Docker Buildx with QEMU emulation
2. Logs in to GHCR using `GITHUB_TOKEN`
3. Builds for `linux/amd64,linux/arm64`
4. Tags with both the version number and `latest`
5. Pushes the multi-arch manifest to `ghcr.io/smallaios/smallaios`

**Workflow steps:**
```yaml
docker-publish:
  name: Publish Multi-Arch Docker Image
  needs: [test]
  runs-on: ubuntu-latest
  permissions:
    contents: read
    packages: write
  steps:
    - uses: actions/checkout@v4
    - uses: docker/setup-qemu-action@v3
    - uses: docker/setup-buildx-action@v3
    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - uses: docker/build-push-action@v6
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: |
          ghcr.io/smallaios/smallaios:${{ github.ref_name }}
          ghcr.io/smallaios/smallaios:latest
```

**Why GHCR over Docker Hub?** GHCR is integrated with GitHub (same org, same auth), free for public repos, and does not require a separate Docker Hub account/token.

**Why `GITHUB_TOKEN` instead of a PAT?** `GITHUB_TOKEN` has automatic write access to the repository's packages when granted `packages: write` permission. No additional secrets needed.

**Tagging strategy:** Each release tag (e.g., `v0.1.0`) produces images tagged as both `ghcr.io/smallaios/smallaios:v0.1.0` and `ghcr.io/smallaios/smallaios:latest`. Pre-release tags (alpha/beta/rc) are tagged with the version only, not `latest`.

### 5. Image size verification

**Decision:** Add an image size check to the release docker-publish job. After building, inspect the per-architecture image sizes and fail if either exceeds 15 MB.

The existing `docker-build-local` CI job already checks the x86_64 image size. The release workflow adds the arm64 check.

### 6. Makefile: No changes needed

**Decision:** The existing Makefile targets already use buildx:

```makefile
docker-build:
    $(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
        -t smallaios/runtime:latest .

docker-push:
    $(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
        -t smallaios/runtime:latest --push .
```

These will work correctly once the Dockerfile handles `TARGETARCH`. No Makefile changes are required.

## Risks / Trade-offs

**[QEMU emulation speed]** Building arm64 on amd64 runners via QEMU emulation is slow (~10-15 minutes vs ~2 minutes native). This only affects the release workflow (not PR CI), so the impact is limited to release cadence.

**[arm64 binary size variance]** The arm64 binary may differ slightly in size from x86_64 due to instruction encoding differences. Both should be well under 15 MB given the x86_64 binary is ~594 KB.

**[GHCR rate limits]** GHCR has rate limits for unauthenticated pulls (public repos). For Jetson Nano deployment, the device must authenticate to GHCR or the image must be small enough to pull within rate limits. At <1 MB compressed, this is not a concern.

**[musl-tools on arm64]** The `musl-tools` package in Debian provides the x86_64 musl-gcc by default. When building under arm64 (QEMU emulation), `musl-tools` provides the native aarch64 musl-gcc. No cross-compilation packages are needed because buildx runs the entire build under emulation for the target architecture.

## Open Questions

None -- the approach is straightforward and follows standard Docker buildx patterns.
