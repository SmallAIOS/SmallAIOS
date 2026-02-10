# SmallAIOS Build System
# Apache-2.0 License

CARGO = cargo
DOCKER = docker
QEMU_X86 = qemu-system-x86_64
QEMU_ARM = qemu-system-aarch64

# build-std flags for bare-metal targets (no_std needs core/alloc rebuilt)
BUILD_STD = -Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem

# Feature flags
ifdef GPU
  FEATURES += --features nvidia_gpu
endif

# === Container Mode (Library OS) ===

.PHONY: build-container-x86
build-container-x86:
	$(CARGO) build --release --target x86_64-unknown-linux-musl $(FEATURES)

.PHONY: build-container-arm
build-container-arm:
	$(CARGO) build --release --target aarch64-unknown-linux-musl $(FEATURES)

# === Kernel Mode (VM / Bare Metal) ===

.PHONY: build-kernel-x86
build-kernel-x86:
	$(CARGO) build --release --target x86_64-unknown-none -p smallaios-arch-x86_64 $(BUILD_STD) $(FEATURES)

.PHONY: build-kernel-arm
build-kernel-arm:
	$(CARGO) build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 $(BUILD_STD) $(FEATURES)

.PHONY: build-kernel-x86-debug
build-kernel-x86-debug:
	$(CARGO) build --target x86_64-unknown-none -p smallaios-arch-x86_64 $(BUILD_STD) $(FEATURES)

.PHONY: build-kernel-arm-debug
build-kernel-arm-debug:
	$(CARGO) build --target aarch64-unknown-none -p smallaios-arch-aarch64 $(BUILD_STD) $(FEATURES)

# === Run in QEMU ===

.PHONY: run-x86
run-x86: build-kernel-x86
	$(QEMU_X86) -machine q35 -cpu max -m 512M -nographic \
		-kernel target/x86_64-unknown-none/release/smallaios-x86_64 \
		-serial stdio

.PHONY: run-arm
run-arm: build-kernel-arm
	$(QEMU_ARM) -machine virt -cpu cortex-a72 -m 512M -nographic \
		-kernel target/aarch64-unknown-none/release/smallaios-aarch64 \
		-serial stdio

# === Docker ===

.PHONY: docker-build
docker-build:
	$(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
		-t smallaios/runtime:latest .

.PHONY: docker-push
docker-push:
	$(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
		-t smallaios/runtime:latest --push .

# === Testing ===

.PHONY: test
test:
	$(CARGO) test -p smallaios-kernel

.PHONY: clippy
clippy:
	$(CARGO) clippy -p smallaios-kernel -- -D warnings

.PHONY: fmt
fmt:
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check:
	$(CARGO) fmt --all -- --check

# === Clean ===

.PHONY: clean
clean:
	$(CARGO) clean
