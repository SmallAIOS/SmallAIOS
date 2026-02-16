# Tegra X1 Device Tree Blobs

Compiled DTB output for Jetson Nano SD card image builds.

## Building

The DTB is compiled from the SmallAIOS-authored DTS source:

```bash
make dtb-jetson
# or directly:
dtc -I dts -O dtb -o arch/aarch64/dtb/tegra210-smallaios.dtb \
    arch/aarch64/dts/tegra210-smallaios.dts
```

The `build-kernel-jetson` and `sdcard-jetson` targets run this automatically.

## Source

The DTS source is at `arch/aarch64/dts/tegra210-smallaios.dts`. It is a minimal,
Apache 2.0 licensed device tree written independently from public hardware
documentation. It is **not** derived from GPL-licensed Linux kernel or NVIDIA L4T
BSP DTS files.

SmallAIOS only reads `/memory` nodes from the DTB (base address + size). All other
hardware — GIC, GPU, PCIe, display — uses hardcoded addresses in `platform.rs`.
The DTS also includes `chosen`, `aliases`, and UART nodes needed by U-Boot for
console output before kernel handoff.

## Output file

- `tegra210-smallaios.dtb` — compiled DTB (git-ignored, built by `make dtb-jetson`)

## License

The DTS source and compiled DTB are licensed under Apache-2.0, consistent with the
rest of the SmallAIOS project.
