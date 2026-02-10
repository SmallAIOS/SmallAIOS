# Delta for Target Hardware Platforms

## ADDED Requirements

### Requirement: RISC-V Tier 2 Platforms

The following RISC-V platforms SHALL be added to SmallAIOS Tier 2 (supported, best-effort optimization): Microchip PolarFire SoC (RISC-V RV64GC + FPGA fabric, radiation-tolerant), SiFive HiFive Unmatched (SiFive U74 quad-core RV64GC), and QEMU `virt` machine for riscv64. SmallAIOS SHALL support CPU inference on all RISC-V platforms. GPU inference is not applicable to RISC-V targets. PolarFire SoC SHALL be the primary RISC-V hardware validation platform due to its combined RISC-V + FPGA fabric and radiation tolerance for space applications.

#### Scenario: Boot SmallAIOS on PolarFire SoC Icicle Kit

- WHEN SmallAIOS is loaded on a Microchip PolarFire SoC Icicle Kit via HSS (Hart Software Services) boot flow
- THEN SmallAIOS MUST boot on the primary U54 application hart in S-mode with OpenSBI providing M-mode services
- AND MUST detect 4 U54 application harts and 1 E51 monitor hart via DTB
- AND MUST initialize SV48 paging and PLIC interrupt controller
- AND MUST reach ready state and accept inference requests over the on-chip Ethernet interface
- AND the E51 monitor hart MUST NOT be used for inference scheduling (reserved for platform management)

#### Scenario: Boot SmallAIOS on SiFive HiFive Unmatched

- WHEN SmallAIOS is loaded on a SiFive HiFive Unmatched board via U-Boot/OpenSBI
- THEN SmallAIOS MUST boot on the 4 U74 harts in S-mode
- AND MUST detect and use the PCIe interface for any supported peripherals
- AND MUST use the on-board Gigabit Ethernet for IPC TCP transport
- AND CPU inference of MobileNetV2-class models MUST complete successfully

#### Scenario: Boot SmallAIOS on QEMU riscv64 virt machine

- WHEN SmallAIOS is launched in QEMU with `qemu-system-riscv64 -machine virt -cpu rv64 -bios opensbi-riscv64-generic-fw_jump.bin -kernel smallaios-riscv64.elf`
- THEN SmallAIOS MUST boot successfully in S-mode
- AND MUST detect the virtio-net and virtio-blk devices from the QEMU-generated DTB
- AND MUST pass all kernel unit tests (memory, scheduler, IPC) in the QEMU environment
- AND this target MUST be used as the primary CI test platform for RISC-V

#### Scenario: RISC-V CPU inference with RV64GC baseline

- WHEN an ONNX inference request for MobileNetV2 is submitted on a RISC-V platform
- THEN the ONNX runtime MUST execute using scalar RV64GC instructions (no vector extension dependency)
- AND MUST use the hardware floating-point unit (F+D extensions) for FP32 operations
- AND MUST complete inference correctly, producing results matching the x86-64 and ARM64 reference outputs within floating-point tolerance (1e-5 relative error)

### Requirement: SoC FPGA Tier 2 Platforms

The following SoC FPGA platforms SHALL be added to SmallAIOS Tier 2: Xilinx/AMD Zynq UltraScale+ (ARM64 Cortex-A53 quad-core + FPGA fabric) and Microchip PolarFire SoC (RISC-V RV64GC + FPGA fabric). On SoC FPGA platforms, SmallAIOS SHALL run on the embedded CPU cores while FPGA fabric provides bus peripherals (CAN controllers, ARINC transceivers, SpaceWire links, MIL-STD-1553 interfaces) as memory-mapped devices discovered via device tree. SmallAIOS SHALL NOT program or manage FPGA bitstreams; the FPGA fabric is configured by external tooling before SmallAIOS boots.

#### Scenario: Boot SmallAIOS on Zynq UltraScale+ with FPGA peripherals

- WHEN SmallAIOS boots on a Zynq UltraScale+ platform where the FPGA fabric has been pre-configured with an AXI CAN controller and an AXI DMA engine
- THEN SmallAIOS MUST discover the FPGA peripherals by parsing the DTB provided by the platform firmware (FSBL/ATF/U-Boot)
- AND MUST initialize the AXI CAN controller via the `BusController` HAL trait
- AND MUST initialize the AXI DMA engine via the `FpgaFabric` HAL trait
- AND MUST register the CAN transport adapter with the Zenoh router

#### Scenario: Boot SmallAIOS on PolarFire SoC with FPGA peripherals

- WHEN SmallAIOS boots on a PolarFire SoC platform where the FPGA fabric provides a SpaceWire link interface and a MIL-STD-1553 controller as AXI-mapped soft-IP
- THEN SmallAIOS MUST discover the FPGA peripherals via DTB nodes under the platform bus
- AND MUST initialize each peripheral using the appropriate `BusController` HAL implementation
- AND MUST register the corresponding transport adapters with the Zenoh router
- AND all FPGA peripheral access MUST go through the `FpgaFabric` HAL trait for AXI register reads/writes

#### Scenario: FPGA peripheral not present in DTB is not initialized

- WHEN SmallAIOS boots on a Zynq UltraScale+ platform and the DTB does not contain a node for any bus peripheral (e.g., the FPGA bitstream only includes custom accelerators, not bus controllers)
- THEN SmallAIOS MUST NOT attempt to initialize any bus peripheral drivers
- AND MUST boot and operate normally using only the PS-side (Processing System) peripherals (GigE, UART, SD/MMC)

#### Scenario: Zynq UltraScale+ CPU inference on Cortex-A53 cores

- WHEN an ONNX inference request is submitted on a Zynq UltraScale+ platform
- THEN SmallAIOS MUST execute inference on the quad-core Cortex-A53 cluster using NEON SIMD
- AND inference results MUST match the reference ARM64 outputs within floating-point tolerance
- AND the FPGA fabric MUST NOT be required for inference execution (CPU-only inference is the baseline)

### Requirement: RISC-V Build Target

The SmallAIOS build system SHALL add `riscv64gc-unknown-none-elf` as a bare-metal build target for RISC-V platforms. The `smallaios-arch-riscv64` crate SHALL be compiled with this target using `-Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem` flags. The RISC-V build MUST produce a bootable ELF binary compatible with OpenSBI `fw_jump` or `fw_payload` firmware loading.

#### Scenario: Build SmallAIOS for RISC-V bare metal

- WHEN a developer runs `cargo build --target riscv64gc-unknown-none-elf -p smallaios-arch-riscv64 -Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem`
- THEN the build MUST succeed and produce a valid ELF binary
- AND the binary MUST have the ELF machine type set to EM_RISCV (243)
- AND the binary entry point MUST be compatible with the OpenSBI jump address convention

#### Scenario: RISC-V build includes all kernel crates

- WHEN the full SmallAIOS kernel is built for `riscv64gc-unknown-none-elf`
- THEN the build MUST compile `smallaios-kernel`, `smallaios-arch-riscv64`, `smallaios-onnx-rt`, `smallaios-ipc`, and `smallaios-bus` (with selected features)
- AND all `#![no_std]` crate constraints MUST be satisfied
- AND the resulting binary MUST link without undefined symbol errors

#### Scenario: RISC-V host tests run on native or cross-compiled target

- WHEN a developer runs `cargo test -p smallaios-arch-riscv64` on an x86-64 host
- THEN architecture-independent unit tests MUST compile and run on the host target
- AND architecture-specific tests (requiring RISC-V instructions) MUST be gated with `#[cfg(target_arch = "riscv64")]`
- AND RISC-V-specific integration tests MUST be runnable via QEMU user-mode emulation or QEMU system-mode with the test harness
