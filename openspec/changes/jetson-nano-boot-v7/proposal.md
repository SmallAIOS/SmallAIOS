## Why

SmallAIOS currently boots only in QEMU emulation. To validate the OS concept on real hardware, we need to boot on an actual device. The NVIDIA Jetson Nano (Tegra X1, Cortex-A57 AArch64) is the target: it's affordable, has the GPU compute we need for ONNX inference, and the existing AArch64 boot code is 90% compatible. The user has a J10 Nano x1 with micro-USB power, USB 3.0/2.0 ports, and a microSD slot. The first milestone is UART "Hello World" from the kernel on real silicon — no networking, no display, just proof that SmallAIOS boots bare-metal.

## What Changes

- **Tegra X1 platform support** in `arch/aarch64`: feature-gated board config for Jetson Nano (load address, UART base, GIC version, memory map)
- **GICv2 interrupt controller driver**: the Tegra X1 uses GICv2 (not GICv3 like QEMU virt), with GICD at `0x50041000` and GICC at `0x50042000`
- **Tegra UART early-boot driver**: wire the existing NS16550A driver (`peripheral/src/uart/ns16550a.rs`) for Tegra UART-A at `0x70006000` with reg-shift=2 and 408 MHz clock; U-Boot pre-initializes it so only TX polling is needed initially
- **Tegra-specific linker script**: load address `0x80080000` (vs QEMU's `0x40080000`), with ARM64 Image header so U-Boot's `booti` can load it
- **ARM64 Image header**: prepend the 64-byte Linux ARM64 boot protocol header so U-Boot recognizes the binary
- **SD card image build tooling**: `make sdcard-jetson` script/Makefile target that creates a GPT-partitioned ext4 image with `/boot/Image`, device tree blob, and `extlinux.conf` ready to `dd` onto a microSD
- **DHCP client**: minimal DHCPv4 client in the `net` crate for automatic IP assignment over Ethernet
- **Static IP configuration**: ability to set a static IPv4/IPv6 address, gateway, and DNS at boot (via DTB, kernel command line, or compiled-in config)
- **RTL8168/8169 Ethernet driver**: the Jetson Nano's built-in Gigabit Ethernet is a Realtek RTL8168 on the PCIe bus; requires Tegra PCIe root complex init + RTL8169-family NIC driver
- **Tegra PCIe controller init**: initialize the Tegra X1 PCIe root complex at `0x01003000` to enumerate the Ethernet controller

## Capabilities

### New Capabilities
- `tegra-x1-platform`: Tegra X1 SoC platform support — clock addresses, memory map, UART base, GIC base, PCIe base, feature flags for `arch/aarch64`
- `gicv2-interrupt-controller`: ARM GICv2 distributor + CPU interface driver (distinct from existing GICv3)
- `arm64-image-header`: ARM64 Linux boot protocol Image header generation for U-Boot compatibility
- `sdcard-image-builder`: Build tooling to create bootable microSD card images (GPT, ext4, extlinux.conf)
- `dhcp-client`: DHCPv4 client (RFC 2131) for automatic network configuration — discover, offer, request, ack
- `static-ip-config`: Static IPv4/IPv6 network configuration at boot time
- `rtl8169-ethernet`: Realtek RTL8169/8168/8111 family NIC driver for built-in Gigabit Ethernet
- `tegra-pcie-controller`: Tegra X1 PCIe root complex initialization and device enumeration

### Modified Capabilities
<!-- No existing spec-level requirements change; this is all additive -->

## Impact

- **`arch/aarch64`**: Add feature flags (`tegra-x1` vs `qemu-virt`), second linker script, GICv2 module, platform config module
- **`peripheral`**: NS16550A driver already supports reg-shift=2; just needs Tegra-specific instantiation at boot
- **`net`**: New DHCP client module, static IP config; existing IPv4/TCP/ARP stack used as-is
- **New driver code**: RTL8169 Ethernet (~500-800 lines), Tegra PCIe init (~300 lines), GICv2 (~200 lines)
- **Build system**: New Makefile targets for Jetson SD card image creation, new CI job for Tegra kernel build
- **CI**: Add `Build Jetson Nano Kernel` job (cross-compile only, no hardware-in-loop yet)
- **Hardware required**: Jetson Nano, 3.3V USB-to-UART adapter (FTDI/CP2102), microSD card, 5V power supply
