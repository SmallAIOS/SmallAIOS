# SmallAIOS Build System
# Apache-2.0 License

CARGO = cargo
DOCKER = docker
QEMU_X86 = qemu-system-x86_64
QEMU_ARM = qemu-system-aarch64
QEMU_RV = qemu-system-riscv64

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

.PHONY: build-kernel-riscv
build-kernel-riscv:
	$(CARGO) build --release --target riscv64gc-unknown-none-elf -p smallaios-arch-riscv64 $(BUILD_STD) $(FEATURES)

.PHONY: build-kernel-riscv-debug
build-kernel-riscv-debug:
	$(CARGO) build --target riscv64gc-unknown-none-elf -p smallaios-arch-riscv64 $(BUILD_STD) $(FEATURES)

# === Run in QEMU ===

.PHONY: run-x86
run-x86: build-kernel-x86
	@mkdir -p build
	$(QEMU_X86) -machine q35 -cpu max -m 512M -nographic \
		-kernel target/x86_64-unknown-none/release/smallaios-x86_64 \
		-serial stdio -serial file:build/serial.log \
		-monitor telnet:localhost:4444,server,nowait
	@echo "QEMU monitor: telnet localhost 4444"

.PHONY: run-arm
run-arm: build-kernel-arm
	$(QEMU_ARM) -machine virt -cpu cortex-a72 -m 512M -nographic \
		-kernel target/aarch64-unknown-none/release/smallaios-aarch64 \
		-serial stdio

.PHONY: run-riscv
run-riscv: build-kernel-riscv
	$(QEMU_RV) -machine virt -cpu rv64 -m 512M -nographic \
		-bios default \
		-kernel target/riscv64gc-unknown-none-elf/release/smallaios-riscv64 \
		-serial stdio

# === QEMU Development Targets ===

.PHONY: debug-x86
debug-x86: build-kernel-x86-debug
	@mkdir -p build
	@echo "Starting QEMU with GDB stub on port 1234 (paused)..."
	@echo "Connect GDB: gdb -x .gdbinit-x86 target/x86_64-unknown-none/debug/smallaios-x86_64"
	@echo "QEMU monitor: telnet localhost 4444"
	$(QEMU_X86) -machine q35 -cpu max -m 512M -nographic \
		-s -S \
		-kernel target/x86_64-unknown-none/debug/smallaios-x86_64 \
		-serial stdio -serial file:build/serial-debug.log \
		-monitor telnet:localhost:4444,server,nowait

.PHONY: run-x86-net
run-x86-net: build-kernel-x86
	@mkdir -p build
	@echo "Starting QEMU with virtio-net (port 8080 forwarded)..."
	@echo "QEMU monitor: telnet localhost 4444"
	$(QEMU_X86) -machine q35 -cpu max -m 512M -nographic \
		-kernel target/x86_64-unknown-none/release/smallaios-x86_64 \
		-device virtio-net-pci,netdev=net0 \
		-netdev user,id=net0,hostfwd=tcp::8080-:8080 \
		-serial stdio -serial file:build/serial-net.log \
		-monitor telnet:localhost:4444,server,nowait

# === VMware ===

.PHONY: vmware-x86
vmware-x86: build-kernel-x86
	./scripts/make-vmware-x86.sh

# === Docker ===

.PHONY: docker-build
docker-build:
	$(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
		-t smallaios/runtime:latest .

.PHONY: docker-push
docker-push:
	$(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
		-t smallaios/runtime:latest --push .

# === Docker Local Development ===

.PHONY: docker-local
docker-local:
	docker compose up --build

.PHONY: docker-local-gpu
docker-local-gpu:
	docker compose --profile gpu up --build

.PHONY: docker-local-clean
docker-local-clean:
	docker compose down --rmi local --volumes

# === Testing ===

.PHONY: test
test:
	$(CARGO) test \
		-p smallaios-kernel \
		-p smallaios-security \
		-p smallaios-onnx-rt \
		-p smallaios-ipc \
		-p smallaios-net \
		-p smallaios-posix \
		-p smallaios-container \
		-p smallaios-bus \
		-p smallaios-bench

.PHONY: clippy
clippy:
	$(CARGO) clippy \
		-p smallaios-kernel \
		-p smallaios-security \
		-p smallaios-onnx-rt \
		-p smallaios-ipc \
		-p smallaios-net \
		-p smallaios-posix \
		-p smallaios-container \
		-p smallaios-bus \
		-p smallaios-bench \
		-- -D warnings

# === TLA+ Formal Verification ===

TLA_DIR = formal/tla
TLA_MODELS = CanArbitration Arinc429Scheduler AfdxVirtualLink Mil1553Protocol \
             SpaceWireLink DdsReliableDelivery DdsDiscovery QuicFlowControl \
             QuicMigration BuddyAllocator Scheduler USBEnumeration \
             XhciTransferRing IQRingBuffer

.PHONY: tla-verify
tla-verify:
	@echo "Verifying TLA+ models with TLC..."
	@for model in $(TLA_MODELS); do \
		echo "--- Checking $$model ---"; \
		java -cp $${TLA2TOOLS:-/usr/local/lib/tla2tools.jar} \
			tlc2.TLC $(TLA_DIR)/$$model.tla \
			-config $(TLA_DIR)/$$model.cfg \
			-workers auto \
			-deadlock || exit 1; \
	done
	@echo "All TLA+ models verified."

.PHONY: fmt
fmt:
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check:
	$(CARGO) fmt --all -- --check

# === Bare Metal Deploy ===

# Network boot: build + copy kernel to TFTP server
# One-time setup: sudo ./scripts/deploy-netboot.sh setup --server-ip <ip>
.PHONY: deploy-netboot
deploy-netboot: build-kernel-arm
	sudo cp target/aarch64-unknown-none/release/smallaios-aarch64 \
		/srv/tftp/smallaios/smallaios-aarch64
	@echo "Deployed to TFTP. Reboot the board."

# RPi SD card: build + create full bootable SD card
# Usage: make deploy-rpi-sdcard DEV=/dev/sdX
.PHONY: deploy-rpi-sdcard
deploy-rpi-sdcard: build-kernel-arm
	sudo ./scripts/deploy-rpi-sdcard.sh full $(DEV) --skip-build

# RPi SD card: update kernel only (faster)
# Usage: make deploy-rpi-sdcard-update DEV=/dev/sdX
.PHONY: deploy-rpi-sdcard-update
deploy-rpi-sdcard-update: build-kernel-arm
	sudo ./scripts/deploy-rpi-sdcard.sh update $(DEV) --skip-build

# Jetson: flash via USB recovery mode
# Usage: make deploy-jetson L4T=~/nvidia/Linux_for_Tegra
.PHONY: deploy-jetson
deploy-jetson: build-kernel-arm
	sudo ./scripts/deploy-jetson-flash.sh flash $(L4T) --skip-build

# Serial console: connect to dev board
# Usage: make serial DEV=/dev/ttyUSB0  (or auto-detect if DEV omitted)
.PHONY: serial
serial:
	./scripts/serial-console.sh $(DEV)

# === Image Size Verification ===

MAX_KERNEL_SIZE_MB = 15

.PHONY: check-size-x86
check-size-x86: build-kernel-x86
	@size=$$(stat --format=%s target/x86_64-unknown-none/release/smallaios-x86_64 2>/dev/null || \
		stat -f%z target/x86_64-unknown-none/release/smallaios-x86_64); \
	max=$$(($(MAX_KERNEL_SIZE_MB) * 1024 * 1024)); \
	echo "x86_64 kernel: $$size bytes ($$(( size / 1024 )) KB)"; \
	if [ "$$size" -gt "$$max" ]; then \
		echo "FAIL: exceeds $(MAX_KERNEL_SIZE_MB) MB limit"; exit 1; \
	else \
		echo "PASS: within $(MAX_KERNEL_SIZE_MB) MB limit"; \
	fi

.PHONY: check-size-arm
check-size-arm: build-kernel-arm
	@size=$$(stat --format=%s target/aarch64-unknown-none/release/smallaios-aarch64 2>/dev/null || \
		stat -f%z target/aarch64-unknown-none/release/smallaios-aarch64); \
	max=$$(($(MAX_KERNEL_SIZE_MB) * 1024 * 1024)); \
	echo "AArch64 kernel: $$size bytes ($$(( size / 1024 )) KB)"; \
	if [ "$$size" -gt "$$max" ]; then \
		echo "FAIL: exceeds $(MAX_KERNEL_SIZE_MB) MB limit"; exit 1; \
	else \
		echo "PASS: within $(MAX_KERNEL_SIZE_MB) MB limit"; \
	fi

.PHONY: check-size-riscv
check-size-riscv: build-kernel-riscv
	@size=$$(stat --format=%s target/riscv64gc-unknown-none-elf/release/smallaios-riscv64 2>/dev/null || \
		stat -f%z target/riscv64gc-unknown-none-elf/release/smallaios-riscv64); \
	max=$$(($(MAX_KERNEL_SIZE_MB) * 1024 * 1024)); \
	echo "RISC-V kernel: $$size bytes ($$(( size / 1024 )) KB)"; \
	if [ "$$size" -gt "$$max" ]; then \
		echo "FAIL: exceeds $(MAX_KERNEL_SIZE_MB) MB limit"; exit 1; \
	else \
		echo "PASS: within $(MAX_KERNEL_SIZE_MB) MB limit"; \
	fi

.PHONY: check-size
check-size: check-size-x86 check-size-arm check-size-riscv

# === Jetson Nano (Tegra X1) ===

.PHONY: build-kernel-jetson
build-kernel-jetson:
	RUSTFLAGS="-D warnings -C link-arg=-Tarch/aarch64/linker-tegra.ld" \
	$(CARGO) build --release --target aarch64-unknown-none \
		-p smallaios-arch-aarch64 --no-default-features --features tegra-x1 \
		$(BUILD_STD)
	$(or $(shell which rust-objcopy 2>/dev/null),$(shell which llvm-objcopy 2>/dev/null),$(shell ls /usr/bin/llvm-objcopy-* 2>/dev/null | head -1),llvm-objcopy) \
		-O binary \
		target/aarch64-unknown-none/release/smallaios-aarch64 \
		target/aarch64-unknown-none/release/Image

.PHONY: check-size-jetson
check-size-jetson: build-kernel-jetson
	@size=$$(stat --format=%s target/aarch64-unknown-none/release/Image 2>/dev/null || \
		stat -f%z target/aarch64-unknown-none/release/Image); \
	max=$$(($(MAX_KERNEL_SIZE_MB) * 1024 * 1024)); \
	echo "Jetson Image: $$size bytes ($$(( size / 1024 )) KB)"; \
	if [ "$$size" -gt "$$max" ]; then \
		echo "FAIL: exceeds $(MAX_KERNEL_SIZE_MB) MB limit"; exit 1; \
	else \
		echo "PASS: within $(MAX_KERNEL_SIZE_MB) MB limit"; \
	fi

.PHONY: sdcard-jetson
sdcard-jetson: build-kernel-jetson
	./scripts/make-sdcard-jetson.sh

# === Clean ===

.PHONY: clean
clean:
	$(CARGO) clean
