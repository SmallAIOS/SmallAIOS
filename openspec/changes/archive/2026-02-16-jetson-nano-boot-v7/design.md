## Context

SmallAIOS boots and runs on QEMU virt (AArch64, x86-64, RISC-V). No real hardware has been validated yet. The Jetson Nano (Tegra X1 SoC, 4× Cortex-A57 @ 1.43 GHz, 4 GB LPDDR4, 128 Maxwell CUDA cores) is the first hardware target.

The existing AArch64 crate (`arch/aarch64`) is hardcoded for QEMU virt: PL011 UART at `0x0900_0000`, GICv3 at `0x0800_0000`, linker base `0x40080000`. The Tegra X1 uses different peripherals (NS16550A UART, GICv2) at different addresses, with a different memory map.

The Jetson Nano boot chain is: BootROM → TegraBoot → CBoot → U-Boot → kernel. U-Boot is already flashed on the QSPI-NOR from stock JetPack. It loads a kernel `Image` from `/boot/` on the microSD card's ext4 partition, passing a DTB pointer in x0 and leaving the CPU at EL1/EL2 with MMU off and UART-A already initialized at 115200 8N1.

The user has a J10 Nano x1 with micro-USB power, USB 3.0, USB 2.0, and microSD. The device has built-in Gigabit Ethernet (Realtek RTL8168 on PCIe). For development, a 3.3V USB-to-UART adapter on the J44 debug header provides serial console.

## Goals / Non-Goals

**Goals:**
- Boot SmallAIOS on real Jetson Nano hardware and print to UART (phase 1)
- Create a microSD card image build pipeline the user can `dd` to a card and boot (phase 1)
- Feature-gate Tegra X1 support so QEMU virt builds remain the default (phase 1)
- Add DHCP and static IP networking over built-in Ethernet (phase 2)
- Add RTL8169 NIC driver and Tegra PCIe controller init (phase 2)

**Non-Goals:**
- HDMI/display output — unnecessary attack surface, UART is sufficient for validation
- GPU compute on Tegra X1 — the 128 Maxwell cores are too weak for useful inference; GPU comes later on discrete cards
- WiFi/Bluetooth — not present on the Nano carrier board without USB adapter
- USB device mode (micro-USB serial) — requires USB gadget stack, not needed when J44 UART is available
- Audio, camera, or other peripheral bringup — boot validation only
- Flashing the QSPI bootloader chain — use stock JetPack U-Boot as-is

## Decisions

### 1. Feature-gated platform support in `arch/aarch64`

**Decision:** Add `tegra-x1` and `qemu-virt` (default) feature flags to `arch/aarch64/Cargo.toml`. Platform-specific constants (UART base, GIC base, load address) live in a `platform` module selected at compile time.

**Why not DTB parsing at boot?** DTB parsing adds ~500 lines of code and complexity. For a first boot validation, compile-time selection is simpler, faster, and the DTB can be consumed later for device enumeration. The boot code already receives DTB in x0 — we just don't parse it yet.

**Why not a separate `arch/tegra` crate?** The Tegra X1 CPU is a standard Cortex-A57. All the AArch64 boot code, paging, exception vectors, timer, and PSCI code is identical. Only peripheral addresses differ. A feature flag in the existing crate avoids duplicating ~1,000 lines of shared code.

### 2. Separate linker script for Tegra

**Decision:** Create `arch/aarch64/linker-tegra.ld` with base address `0x80080000`. The Makefile/build system selects the linker script based on the target feature.

**Alternatives considered:**
- Single linker script with a symbol override — linker scripts don't support conditional logic cleanly
- Position-independent binary — would work but adds complexity and slight overhead; not worth it for a first boot

**Tegra memory map:**
| Region | Address |
|--------|---------|
| DRAM base | `0x80000000` |
| Kernel load (by U-Boot) | `0x80080000` |
| Tegra UART-A | `0x70006000` |
| GIC Distributor | `0x50041000` |
| GIC CPU Interface | `0x50042000` |
| PCIe root complex | `0x01003000` |

### 3. Early UART: inline NS16550A, not the peripheral crate

**Decision:** For the earliest boot output (before BSS is cleared or allocator is up), use a minimal inline NS16550A write function directly in `arch/aarch64/src/uart.rs` behind the `tegra-x1` feature. After full kernel init, the `peripheral::uart::Ns16550a` driver takes over.

**Why not use the peripheral crate directly?** The peripheral crate's `Ns16550a` struct requires initialization (`.new()`, `.configure()`). During early boot (`_start` → `kernel_main` first lines), we need a function that can write a character with zero setup — because U-Boot already initialized UART-A. A 10-line polled write function is all we need:

```rust
#[cfg(feature = "tegra-x1")]
const UART_BASE: usize = 0x7000_6000;
const THR: usize = 0x00;
const LSR: usize = 0x14; // offset 5 × 4 (reg-shift=2)
const LSR_THRE: u32 = 1 << 5;

pub fn putc(byte: u8) {
    unsafe {
        while (read_volatile((UART_BASE + LSR) as *const u32) & LSR_THRE) == 0 {
            core::hint::spin_loop();
        }
        write_volatile((UART_BASE + THR) as *mut u32, byte as u32);
    }
}
```

This is the same pattern used by ARM Trusted Firmware and TRENTOS for Tegra X1 early boot.

### 4. ARM64 Image header for U-Boot compatibility

**Decision:** Prepend the standard 64-byte ARM64 Image header in the linker script's `.text.boot` section. This lets U-Boot's `booti` command load and verify the binary without manual address hacking.

The header format (from Linux ARM64 boot protocol):
```
Offset  Size  Field
0x00    4     branch to _start (or MZ for PE compat)
0x04    4     reserved
0x08    8     text_offset
0x10    8     image_size (0 = unknown OK)
0x18    8     flags (little-endian, 4K pages, anywhere)
0x20-0x34     reserved
0x38    4     magic: 0x644D5241 ("ARM\x64")
0x3C    4     PE offset (0)
```

**Implementation:** Add a `__image_header` assembly block at the very start of `.text.boot`, before `_start`. The branch instruction at offset 0 jumps over the header to `_start`.

### 5. GICv2 driver as a separate module

**Decision:** Create `arch/aarch64/src/gicv2.rs` with the same public API as the GICv3 code (`init_gicd`, `init_cpu_interface`, `icc_iar`, `icc_eoir`, etc.). The `lib.rs` conditionally exports either `gicv2` or `gicv3` (renamed as `interrupts`) based on the feature flag.

**GICv2 vs GICv3 differences:**
- GICv2 has no Redistributor — CPU interface is memory-mapped (GICC at `0x50042000`)
- GICv2 IAR/EOIR are MMIO registers, not system registers
- GICv2 has no affinity routing (ARE)
- Timer IRQ 30 (physical timer PPI) is the same

The GICv2 driver is ~150 lines (simpler than GICv3). The key registers:
- GICD_CTLR at GICD + 0x000 (enable)
- GICD_ISENABLER at GICD + 0x100 (set-enable)
- GICC_CTLR at GICC + 0x000 (CPU interface enable)
- GICC_PMR at GICC + 0x004 (priority mask)
- GICC_IAR at GICC + 0x00C (acknowledge)
- GICC_EOIR at GICC + 0x010 (end of interrupt)

### 6. SD card image: shell script, not Rust

**Decision:** The SD card image builder is a shell script (`scripts/make-sdcard-jetson.sh`) invoked by `make sdcard-jetson`. It uses standard tools (`dd`, `sgdisk`/`gdisk`, `mkfs.ext4`, `mount`) to create a GPT image with one ext4 partition containing:
```
/boot/Image                        (SmallAIOS kernel binary)
/boot/tegra210-p3450-0000.dtb      (device tree blob from JetPack)
/boot/extlinux/extlinux.conf       (U-Boot boot config)
```

**Why a shell script and not a Rust build tool?** Image creation requires root (for mount/loop devices) or `mtools`/`guestfish`. This is a host-side build step, not target code. Shell + standard Linux tools is simpler and more portable than a Rust binary for this purpose.

**extlinux.conf:**
```
DEFAULT primary
MENU TITLE SmallAIOS Boot
TIMEOUT 30

LABEL primary
    MENU LABEL SmallAIOS v0.1.0
    LINUX /boot/Image
    FDT /boot/tegra210-p3450-0000.dtb
    APPEND console=ttyS0,115200n8
```

The DTB (`tegra210-p3450-0000.dtb`) is extracted from the stock JetPack L4T package. We vendor it in `arch/aarch64/dtb/` for reproducibility.

### 7. Two-phase networking

**Decision:** Networking is phase 2, cleanly separated from the boot validation in phase 1.

**Phase 2 dependency chain:**
1. Tegra PCIe root complex init (`0x01003000`) — enables PCIe bus scanning
2. RTL8169 NIC driver — standard register-compatible driver for the RTL8168
3. Wire NIC into existing `net` crate's Ethernet/ARP/IPv4/TCP stack
4. DHCP client (RFC 2131) — 4-message exchange: DISCOVER → OFFER → REQUEST → ACK
5. Static IP config — fallback/alternative to DHCP, set via compiled-in constants or DTB

**RTL8169 driver approach:** The RTL8169 family is well-documented with Linux, NetBSD, and DPDK open-source implementations as references. Key registers:
- MAC address at offset 0x00
- TX/RX descriptor rings (physically contiguous DMA buffers)
- Command register at 0x37 (TX enable, RX enable, reset)
- Interrupt status/mask at 0x3C/0x3E

The driver is ~600-800 lines. DMA buffers must be in the first 4 GB of physical memory (32-bit addressable) for the non-64-bit variants; the RTL8168 supports 64-bit descriptors.

**DHCP client:** Minimal implementation (~300 lines) using the existing UDP stack. Only implements the 4-message DHCPv4 exchange with lease renewal. No DHCP options beyond subnet mask, gateway, DNS, and lease time.

### 8. CI: cross-compile only, no hardware-in-loop

**Decision:** Add a `Build Jetson Nano Kernel` CI job that cross-compiles with `--features tegra-x1`. No QEMU smoke test (QEMU doesn't emulate Tegra X1). Hardware validation is manual until a Jetson Nano is connected to CI infrastructure.

## Risks / Trade-offs

**[U-Boot incompatibility]** → The ARM64 Image header must exactly match what U-Boot expects. If the header is malformed, U-Boot will refuse to boot. Mitigation: validate the header bytes with a unit test comparing against a known-good Linux Image header.

**[EL2 vs EL1 entry]** → U-Boot may hand off at EL2 (hypervisor) or EL1 (kernel) depending on ATF/CBoot configuration. The boot assembly must handle both. Mitigation: check CurrentEL in `_start` and drop to EL1 if at EL2 (standard ARM64 bootstrap pattern).

**[Tegra clock gating]** → If a peripheral's clock is gated by the Clock and Reset controller (CAR at `0x60006000`), MMIO writes silently fail. U-Boot enables UART-A clocks, but other peripherals (PCIe, additional UARTs) may need explicit CAR programming. Mitigation: defer non-UART peripherals to phase 2 where we implement CAR init.

**[RTL8168 variant differences]** → The RTL8168 has many silicon revisions (8168B, 8168C, 8168D, etc.) with slightly different initialization sequences. Mitigation: start with the common RTL8169 register set, test on actual hardware, and add variant-specific quirks as needed.

**[DMA buffer placement]** → The RTL8168 requires physically contiguous DMA buffers. SmallAIOS's buddy allocator provides physically contiguous pages, but we must ensure the buffers are in the DMA-able region (below 4 GB, which is all of Tegra's DRAM). Mitigation: the entire 4 GB DRAM is below `0x180000000`, within 32-bit+1 DMA range.

**[No QEMU testing for Tegra]** → QEMU doesn't emulate the Tegra X1 SoC. We can only unit-test driver logic and validate on real hardware. Mitigation: maximum unit test coverage for register calculations, DMA descriptor layout, DHCP state machine; manual hardware testing for integration.

## Open Questions

1. **DTB source:** Should we vendor the `tegra210-p3450-0000.dtb` from JetPack, or expect the user to provide it? Vendoring is simpler but may have licensing implications (NVIDIA proprietary modifications to the upstream DT).

2. **EL2 handling:** Does the stock JetPack CBoot/ATF chain hand off at EL2 or EL1 on the Nano? Need to check on actual hardware. The boot code should handle both regardless.

3. **PCIe initialization complexity:** How much Tegra-specific clock/reset/PHY initialization is needed for PCIe? The Linux driver (`pci-tegra.c`) is ~2,000 lines. We may be able to rely on U-Boot having partially initialized PCIe, reducing the work.
