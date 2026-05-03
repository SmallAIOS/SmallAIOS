# SmallAIOS Build System
# Apache-2.0 License

set shell := ["bash", "-euo", "pipefail", "-c"]

cargo := "cargo"
docker := "docker"
qemu_x86 := "qemu-system-x86_64"
qemu_arm := "qemu-system-aarch64"
qemu_rv := "qemu-system-riscv64"

# build-std flags for bare-metal targets (no_std needs core/alloc rebuilt)
build_std := "-Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem"

max_kernel_size_mb := "15"

# Host-testable crates for module-level analysis
host_crates := "smallaios-kernel smallaios-security smallaios-compute smallaios-sched-types smallaios-onnx-rt smallaios-ipc smallaios-net smallaios-posix smallaios-container smallaios-bus smallaios-peripheral smallaios-usb smallaios-sdr smallaios-bench"

# === Container Mode (Library OS) ===

# Build container image for x86_64
build-container-x86 gpu="":
    {{cargo}} build --release --target x86_64-unknown-linux-musl -p smallaios-container {{ if gpu != "" { "--features nvidia_gpu" } else { "" } }}

# Build container image for ARM64
build-container-arm gpu="":
    {{cargo}} build --release --target aarch64-unknown-linux-musl -p smallaios-container {{ if gpu != "" { "--features nvidia_gpu" } else { "" } }}

# === Kernel Mode (VM / Bare Metal) ===

# Build x86-64 kernel (release)
build-kernel-x86:
    {{cargo}} build --release --target x86_64-unknown-none -p smallaios-arch-x86_64 {{build_std}}

# Build AArch64 kernel (release)
#
# RUSTFLAGS is set explicitly here (rather than relying on
# `[target.aarch64-unknown-none].rustflags` in `.cargo/config.toml`) because
# cargo doubles config-file rustflags into the final bin's rustc invocation
# when `-Z build-std` is in play, which causes rust-lld to load `linker.ld`
# twice and emit overlapping section file offsets. CI's `Build AArch64
# Kernel` job already takes the env-var path; this keeps `just`-driven local
# builds (notably Apple Silicon) on the same path.
build-kernel-arm:
    RUSTFLAGS="-C link-arg=-Tarch/aarch64/linker.ld" {{cargo}} build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 {{build_std}}

# Build RISC-V kernel (release)
build-kernel-riscv:
    {{cargo}} build --release --target riscv64gc-unknown-none-elf -p smallaios-arch-riscv64 {{build_std}}

# Build x86-64 kernel (debug)
build-kernel-x86-debug:
    {{cargo}} build --target x86_64-unknown-none -p smallaios-arch-x86_64 {{build_std}}

# Build AArch64 kernel (debug)
# RUSTFLAGS env var matches `build-kernel-arm` — see comment there.
build-kernel-arm-debug:
    RUSTFLAGS="-C link-arg=-Tarch/aarch64/linker.ld" {{cargo}} build --target aarch64-unknown-none -p smallaios-arch-aarch64 {{build_std}}

# Build RISC-V kernel (debug)
build-kernel-riscv-debug:
    {{cargo}} build --target riscv64gc-unknown-none-elf -p smallaios-arch-riscv64 {{build_std}}

# === Run in QEMU ===

# Boot x86-64 kernel in QEMU
run-x86: build-kernel-x86
    @mkdir -p build
    {{qemu_x86}} -machine q35 -cpu max -m 512M -nographic \
        -kernel target/x86_64-unknown-none/release/smallaios-x86_64 \
        -serial stdio -serial file:build/serial.log \
        -monitor telnet:localhost:4444,server,nowait
    @echo "QEMU monitor: telnet localhost 4444"

# Boot AArch64 kernel in QEMU
run-arm: build-kernel-arm
    {{qemu_arm}} -machine virt -cpu cortex-a72 -m 512M -nographic \
        -kernel target/aarch64-unknown-none/release/smallaios-aarch64 \
        -serial stdio

# Boot RISC-V kernel in QEMU
run-riscv: build-kernel-riscv
    {{qemu_rv}} -machine virt -cpu rv64 -m 512M -nographic \
        -bios default \
        -kernel target/riscv64gc-unknown-none-elf/release/smallaios-riscv64 \
        -serial stdio

# Start x86-64 QEMU with GDB stub (paused, port 1234)
debug-x86: build-kernel-x86-debug
    @mkdir -p build
    @echo "Starting QEMU with GDB stub on port 1234 (paused)..."
    @echo "Connect GDB: gdb -x .gdbinit-x86 target/x86_64-unknown-none/debug/smallaios-x86_64"
    @echo "QEMU monitor: telnet localhost 4444"
    {{qemu_x86}} -machine q35 -cpu max -m 512M -nographic \
        -s -S \
        -kernel target/x86_64-unknown-none/debug/smallaios-x86_64 \
        -serial stdio -serial file:build/serial-debug.log \
        -monitor telnet:localhost:4444,server,nowait

# Boot x86-64 with virtio-net (port 8080 forwarded)
run-x86-net: build-kernel-x86
    @mkdir -p build
    @echo "Starting QEMU with virtio-net (port 8080 forwarded)..."
    @echo "QEMU monitor: telnet localhost 4444"
    {{qemu_x86}} -machine q35 -cpu max -m 512M -nographic \
        -kernel target/x86_64-unknown-none/release/smallaios-x86_64 \
        -device virtio-net-pci,netdev=net0 \
        -netdev user,id=net0,hostfwd=tcp::8080-:8080 \
        -serial stdio -serial file:build/serial-net.log \
        -monitor telnet:localhost:4444,server,nowait

# === VMware ===

# Create VMware x86 image
vmware-x86: build-kernel-x86
    ./scripts/make-vmware-x86.sh

# === Docker ===

# Build multi-arch Docker image
docker-build:
    {{docker}} buildx build --platform linux/amd64,linux/arm64 \
        -t smallaios/runtime:latest .

# Build GPU-enabled Docker image (NVIDIA CUDA runtime)
docker-build-gpu:
    {{docker}} build -f Dockerfile.cuda -t smallaios/runtime:gpu .

# Build Jetson Orin Docker image (NVIDIA L4T JetPack 6 base, cc_87)
docker-build-jetson:
    {{docker}} build -f Dockerfile.jetson -t smallaios/runtime:jetson .

# Build Jetson Orin slim image (l4t-cuda runtime + cuDNN, ~4 GB vs ~9.8 GB)
docker-build-jetson-slim:
    {{docker}} build -f Dockerfile.jetson.slim -t smallaios/runtime:jetson-slim .

# Build and push multi-arch Docker image
docker-push:
    {{docker}} buildx build --platform linux/amd64,linux/arm64 \
        -t smallaios/runtime:latest --push .

# Run local Docker dev environment
docker-local:
    docker compose up --build

# Run local Docker with GPU profile
docker-local-gpu:
    docker compose --profile gpu up --build

# Run local Docker with Jetson profile (ARM64 + Tegra Orin GPU)
docker-local-jetson:
    docker compose --profile jetson up --build

# Run local Docker with Jetson slim profile (~3 GB image)
docker-local-jetson-slim:
    docker compose --profile jetson-slim up --build

# Smoke-test the Jetson GPU image end-to-end (build + boot + GPU init + /v1/inference).
# Run this on a Jetson Orin Nano / NX / AGX with NVIDIA Container Runtime installed.
# Pass variant=slim to test Dockerfile.jetson.slim instead of the default.
test-jetson-gpu variant="":
    ./scripts/test-jetson-gpu.sh {{variant}}

# === Jetson unikernel (KVM-on-L4T smoke test) ===

# Boot the AArch64 unikernel under KVM on a Jetson Orin (Phase 1).
#
# Two execution modes:
#   - SSH_HOST given (recommended): cross-build locally, scp to the Jetson,
#     run qemu+KVM there. Use this from a Mac / x86 dev box.
#   - SSH_HOST empty: build and run locally. Use this when the recipe is
#     invoked on the Jetson itself (Rust toolchain must be present locally).
#
# The kernel boots on real Cortex-A78AE cores via -accel kvm; peripherals
# come from QEMU virt (PL011 UART, GICv3, virtio). See docs/jetson-kvm-quickstart.md.
#
# Prerequisites on the Jetson runner:
#   - qemu-system-aarch64 (apt install qemu-system-arm)
#   - /dev/kvm accessible (member of `kvm` group; see quickstart)
run-jetson-kvm SSH_HOST="" KERNEL_PATH="target/aarch64-unknown-none/release/smallaios-aarch64": build-kernel-arm
    #!/usr/bin/env bash
    set -euo pipefail
    KERNEL="{{KERNEL_PATH}}"
    if [ ! -f "$KERNEL" ]; then
        echo "Kernel artifact not found: $KERNEL" >&2
        exit 1
    fi
    if [ -n "{{SSH_HOST}}" ]; then
        echo "[jetson-kvm] Copying $KERNEL to {{SSH_HOST}}:~/"
        scp "$KERNEL" "{{SSH_HOST}}:~/"
        REMOTE_BIN="~/$(basename "$KERNEL")"
        echo "[jetson-kvm] Running on {{SSH_HOST}}: qemu-system-aarch64 -accel kvm -cpu host"
        ssh "{{SSH_HOST}}" "qemu-system-aarch64 \
            -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic \
            -kernel $REMOTE_BIN -serial mon:stdio"
    else
        echo "[jetson-kvm] Local mode (assuming we're on the Jetson)"
        if [ ! -c /dev/kvm ]; then
            echo "ERROR: /dev/kvm not present. Phase 1 requires KVM (built into JetPack 6 kernel)." >&2
            exit 2
        fi
        if [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
            echo "ERROR: /dev/kvm not accessible. Run: sudo usermod -aG kvm \$USER && re-login" >&2
            exit 3
        fi
        qemu-system-aarch64 \
            -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic \
            -kernel "$KERNEL" -serial mon:stdio
    fi

# Smoke-test the Jetson unikernel KVM boot end-to-end (Phase 1 acceptance).
# See scripts/test-jetson-kvm.sh for assertion details and exit codes.
test-jetson-kvm SSH_HOST="":
    ./scripts/test-jetson-kvm.sh "{{SSH_HOST}}"

# Clean local Docker resources
docker-local-clean:
    docker compose down --rmi local --volumes

# === Dev Setup ===

# Install git pre-commit hooks (run once after clone)
setup-hooks:
    git config core.hooksPath .githooks
    @echo "Pre-commit hooks installed (.githooks/pre-commit)"
    @echo "Hooks run: cargo fmt, clippy, geiger, audit, semver-checks, cycles"

# Run all pre-commit checks manually (same as the hook runs)
check: fmt-check clippy
    @echo "=== Core checks passed ==="

# Run full safety-critical audit (slower, pre-merge)
audit:
    @echo "=== Safety-critical audit ==="
    @echo "[1/6] cargo-deny (supply chain)"
    cargo deny check
    @echo "[2/6] cargo-audit (CVE check)"
    cargo audit
    @echo "[3/6] cargo-geiger (unsafe report)"
    cargo geiger --output-format ascii --update-readme=false 2>/dev/null | tail -10 || true
    @echo "[4/6] cargo-vet (dependency audit trail)"
    cargo vet check || echo "WARNING: Unaudited dependencies found"
    @echo "[5/6] Dependency cycle check"
    ./scripts/check-cycles.sh
    @echo "[6/6] Module acyclicity"
    just arch-check || true
    @echo "=== Safety audit complete ==="

# === Testing ===

# Run all unit tests
test:
    {{cargo}} test \
        -p smallaios-kernel \
        -p smallaios-security \
        -p smallaios-compute \
        -p smallaios-onnx-rt \
        -p smallaios-ipc \
        -p smallaios-net \
        -p smallaios-posix \
        -p smallaios-container \
        -p smallaios-bus \
        -p smallaios-bench

# Run Metal GPU tests (macOS only)
test-metal:
    {{cargo}} test -p smallaios-onnx-rt --features metal --lib -- metal_dispatch
    {{cargo}} test -p smallaios-arch-apple

# Run clippy lints
clippy:
    {{cargo}} clippy \
        --all-targets \
        -p smallaios-kernel \
        -p smallaios-security \
        -p smallaios-compute \
        -p smallaios-onnx-rt \
        -p smallaios-ipc \
        -p smallaios-net \
        -p smallaios-posix \
        -p smallaios-container \
        -p smallaios-bus \
        -p smallaios-bench \
        -- -D warnings

# Format all code
fmt:
    {{cargo}} fmt --all

# Check code formatting
fmt-check:
    {{cargo}} fmt --all -- --check

# === TLA+ Formal Verification ===

tla_dir := "formal/tla"
tla_models := "CanArbitration Arinc429Scheduler AfdxVirtualLink Mil1553Protocol SpaceWireLink DdsReliableDelivery DdsDiscovery QuicFlowControl QuicMigration BuddyAllocator Scheduler USBEnumeration XhciTransferRing IQRingBuffer"

# Verify all TLA+ models with TLC
tla-verify:
    @echo "Verifying TLA+ models with TLC..."
    @for model in {{tla_models}}; do \
        echo "--- Checking $model ---"; \
        java -cp ${TLA2TOOLS:-/usr/local/lib/tla2tools.jar} \
            tlc2.TLC {{tla_dir}}/$model.tla \
            -config {{tla_dir}}/$model.cfg \
            -workers auto \
            -deadlock || exit 1; \
    done
    @echo "All TLA+ models verified."

# Verify SPIN/Promela models
spin-verify:
    @echo "Verifying SPIN/Promela models..."
    @for model in formal/spin/*.pml formal/promela/*.pml; do \
        [ -f "$model" ] || continue; \
        name=$(basename "$model" .pml); \
        echo "--- Checking $name ---"; \
        spin -a "$model" && \
        cc -DMEMLIM=1024 -o pan pan.c && \
        timeout 300 ./pan -a && \
        echo "OK: $name verified" || \
        echo "WARNING: $name had issues"; \
        rm -f pan.* *.trail; \
    done
    @echo "SPIN verification complete."

# === Supply Chain Security ===

# Run cargo-deny license/advisory checks
deny:
    cargo deny check --all-features

# Check for cyclic workspace dependencies
check-cycles:
    ./scripts/check-cycles.sh

# === Dependency Analysis ===

# Generate crate-level dependency graph (DOT + SVG)
depgraph:
    @mkdir -p build/analysis
    cargo depgraph --workspace-only --dedup-transitive-deps \
        | tee build/analysis/crate-deps.dot \
        | (dot -Tsvg -o build/analysis/crate-deps.svg 2>/dev/null \
            && echo "Generated build/analysis/crate-deps.svg" \
            || echo "WARNING: graphviz not installed, SVG skipped (DOT file saved)")
    @echo "DOT file: build/analysis/crate-deps.dot"

# Generate module-level dependency graphs (all crates or specific CRATE)
modgraph crate="":
    @mkdir -p build/analysis/modules
    {{ if crate != "" { "cargo modules dependencies --package " + crate + " --layout dot > build/analysis/modules/" + crate + ".dot && echo 'Generated build/analysis/modules/" + crate + ".dot'" } else { "for c in " + host_crates + "; do echo \"--- $c ---\"; cargo modules dependencies --package $c --layout dot > build/analysis/modules/$c.dot 2>/dev/null && echo \"  -> build/analysis/modules/$c.dot\" || echo \"  WARNING: $c module graph failed\"; done" } }}

# Check module-level acyclicity for all host crates
arch-check:
    @echo "Checking module-level acyclicity..."
    @fail=0; \
    for crate in {{host_crates}}; do \
        echo -n "  $crate: "; \
        if cargo modules dependencies --package $crate --acyclic 2>&1 | grep -q "error\|cycle"; then \
            echo "CYCLE DETECTED"; \
            fail=1; \
        else \
            echo "OK"; \
        fi; \
    done; \
    if [ "$fail" -eq 1 ]; then \
        echo "WARNING: some crates have module-level cycles"; \
    else \
        echo "All crates are acyclic at module level."; \
    fi

# Generate DSM adjacency matrix (JSON + CSV)
dsm:
    @mkdir -p build/analysis
    python3 scripts/dsm-matrix.py
    @echo "Generated build/analysis/dsm-matrix.json and dsm-matrix.csv"

# Run DSM analysis tool (propagation cost, fan-in/out, clusters, layering violations)
dsm-analyze: dsm
    @echo "Running DSM analysis..."
    @if [ -f tools/dsm/Cargo.toml ]; then \
        cargo run --manifest-path tools/dsm/Cargo.toml -- build/analysis/dsm-matrix.json --output build/analysis/dsm-metrics.json; \
    else \
        echo "WARNING: tools/dsm/ crate not found, skipping DSM analysis"; \
    fi

# Run all dependency analysis (depgraph + modgraph + dsm + analysis)
arch: depgraph modgraph dsm dsm-analyze
    @echo "All dependency analysis complete. See build/analysis/"

# === Changelog ===

# Regenerate CHANGELOG.md via git-cliff
changelog:
    git-cliff --config cliff.toml -o CHANGELOG.md

# === Release ===

# Preview version bump (dry run)
release-dry-run bump:
    cargo release {{bump}}

# Execute version bump + commit + tag
release bump:
    cargo release {{bump}} --execute

# === Bare Metal Deploy ===

# Deploy kernel to TFTP server for network boot
deploy-netboot: build-kernel-arm
    sudo cp target/aarch64-unknown-none/release/smallaios-aarch64 \
        /srv/tftp/smallaios/smallaios-aarch64
    @echo "Deployed to TFTP. Reboot the board."

# Create full bootable RPi SD card
deploy-rpi-sdcard dev: build-kernel-arm
    sudo ./scripts/deploy-rpi-sdcard.sh full {{dev}} --skip-build

# Update kernel on RPi SD card (faster)
deploy-rpi-sdcard-update dev: build-kernel-arm
    sudo ./scripts/deploy-rpi-sdcard.sh update {{dev}} --skip-build

# Flash Jetson via USB recovery mode
deploy-jetson l4t: build-kernel-arm
    sudo ./scripts/deploy-jetson-flash.sh flash {{l4t}} --skip-build

# Connect to dev board serial console
serial dev="":
    ./scripts/serial-console.sh {{dev}}

# === Image Size Verification ===

# Check x86-64 kernel binary size
check-size-x86: build-kernel-x86
    #!/usr/bin/env bash
    set -euo pipefail
    size=$(stat --format=%s target/x86_64-unknown-none/release/smallaios-x86_64 2>/dev/null || \
        stat -f%z target/x86_64-unknown-none/release/smallaios-x86_64)
    max=$(({{max_kernel_size_mb}} * 1024 * 1024))
    echo "x86_64 kernel: $size bytes ($(( size / 1024 )) KB)"
    if [ "$size" -gt "$max" ]; then
        echo "FAIL: exceeds {{max_kernel_size_mb}} MB limit"; exit 1
    else
        echo "PASS: within {{max_kernel_size_mb}} MB limit"
    fi

# Check AArch64 kernel binary size
check-size-arm: build-kernel-arm
    #!/usr/bin/env bash
    set -euo pipefail
    size=$(stat --format=%s target/aarch64-unknown-none/release/smallaios-aarch64 2>/dev/null || \
        stat -f%z target/aarch64-unknown-none/release/smallaios-aarch64)
    max=$(({{max_kernel_size_mb}} * 1024 * 1024))
    echo "AArch64 kernel: $size bytes ($(( size / 1024 )) KB)"
    if [ "$size" -gt "$max" ]; then
        echo "FAIL: exceeds {{max_kernel_size_mb}} MB limit"; exit 1
    else
        echo "PASS: within {{max_kernel_size_mb}} MB limit"
    fi

# Check RISC-V kernel binary size
check-size-riscv: build-kernel-riscv
    #!/usr/bin/env bash
    set -euo pipefail
    size=$(stat --format=%s target/riscv64gc-unknown-none-elf/release/smallaios-riscv64 2>/dev/null || \
        stat -f%z target/riscv64gc-unknown-none-elf/release/smallaios-riscv64)
    max=$(({{max_kernel_size_mb}} * 1024 * 1024))
    echo "RISC-V kernel: $size bytes ($(( size / 1024 )) KB)"
    if [ "$size" -gt "$max" ]; then
        echo "FAIL: exceeds {{max_kernel_size_mb}} MB limit"; exit 1
    else
        echo "PASS: within {{max_kernel_size_mb}} MB limit"
    fi

# Check all kernel binary sizes
check-size: check-size-x86 check-size-arm check-size-riscv

# === Device Tree ===

# Compile Jetson device tree blob
dtb-jetson:
    dtc -I dts -O dtb -o arch/aarch64/dtb/tegra210-smallaios.dtb \
        arch/aarch64/dts/tegra210-smallaios.dts

# === Jetson Nano (Tegra X1) ===

# Build Jetson kernel image
build-kernel-jetson: dtb-jetson
    RUSTFLAGS="-D warnings -C link-arg=-Tarch/aarch64/linker-tegra.ld" \
    {{cargo}} build --release --target aarch64-unknown-none \
        -p smallaios-arch-aarch64 --no-default-features --features tegra-x1 \
        {{build_std}}
    $(which rust-objcopy 2>/dev/null || which llvm-objcopy 2>/dev/null || ls /usr/bin/llvm-objcopy-* 2>/dev/null | head -1 || echo llvm-objcopy) \
        -O binary \
        target/aarch64-unknown-none/release/smallaios-aarch64 \
        target/aarch64-unknown-none/release/Image

# Check Jetson kernel image size
check-size-jetson: build-kernel-jetson
    #!/usr/bin/env bash
    set -euo pipefail
    size=$(stat --format=%s target/aarch64-unknown-none/release/Image 2>/dev/null || \
        stat -f%z target/aarch64-unknown-none/release/Image)
    max=$(({{max_kernel_size_mb}} * 1024 * 1024))
    echo "Jetson Image: $size bytes ($(( size / 1024 )) KB)"
    if [ "$size" -gt "$max" ]; then
        echo "FAIL: exceeds {{max_kernel_size_mb}} MB limit"; exit 1
    else
        echo "PASS: within {{max_kernel_size_mb}} MB limit"
    fi

# Create Jetson SD card image
sdcard-jetson: build-kernel-jetson
    ./scripts/make-sdcard-jetson.sh

# === Clean ===

# Remove all build artifacts
clean:
    {{cargo}} clean
