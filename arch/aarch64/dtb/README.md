# Tegra X1 Device Tree Blobs

Place the Jetson Nano DTB here for SD card image builds.

## Source

Extract from NVIDIA JetPack L4T BSP (Linux_for_Tegra):

```
Linux_for_Tegra/kernel/dtb/tegra210-p3450-0000.dtb
```

Download from: https://developer.nvidia.com/embedded/linux-tegra

## Required file

- `tegra210-p3450-0000.dtb` — Jetson Nano Developer Kit (P3450-0000)

## License

The DTB is derived from NVIDIA's L4T BSP and is subject to NVIDIA's license terms.
It is not included in this repository — you must extract it from the JetPack BSP.

## Alternative

U-Boot on the Jetson Nano has a built-in DTB. If you omit the DTB from the SD card,
U-Boot will use its internal copy. The extlinux.conf FDT line can be removed.
