## 1. Platform Feature Flags and Constants

- [x] 1.1 Add `tegra-x1` and `qemu-virt` (default) feature flags to `arch/aarch64/Cargo.toml`; add `compile_error!` enforcing mutual exclusivity
- [x] 1.2 Create `arch/aarch64/src/platform.rs` with `cfg`-gated constants: UART_BASE, GICD_BASE, GICC_BASE/GICR_BASE, DRAM_BASE, KERNEL_LOAD_ADDR, PCIE_BASE, CAR_BASE for both tegra-x1 and qemu-virt
- [x] 1.3 Register `pub mod platform;` in `arch/aarch64/src/lib.rs`
- [x] 1.4 Refactor `uart.rs` and `interrupts.rs` to import base addresses from `platform` instead of hardcoding
- [x] 1.5 Add unit tests in `platform.rs` verifying correct addresses for each feature configuration
- [x] 1.6 Verify existing QEMU build still works with default features

## 2. Boot Assembly and EL2-to-EL1 Transition

- [x] 2.1 Extend `_start` in `arch/aarch64/src/boot.rs` to read `CurrentEL`: if EL2, configure HCR_EL2 (RW=1), SCTLR_EL1, SPSR_EL2 (EL1h + DAIF masked = `0x3C5`), ELR_EL2, then `eret` to EL1; if EL1, fall through to BSS clear
- [x] 2.2 Add `#[cfg(feature = "tegra-x1")]` guard to skip `uart::init()` in `kernel_main` (U-Boot pre-inits UART-A); keep PL011 init under `qemu-virt`
- [x] 2.3 Update `kernel_main` to print platform-specific boot banner: `"[SmallAIOS] booting on Tegra X1 (Jetson Nano)"` under tegra-x1, with DTB address via `uart::put_hex`
- [x] 2.4 Write unit tests for EL2-to-EL1 register constants: HCR_EL2 bit 31 set, SPSR_EL2 = `0x3C5`

## 3. Tegra UART Early Boot

- [x] 3.1 Add NS16550A polled TX path to `arch/aarch64/src/uart.rs` behind `#[cfg(feature = "tegra-x1")]`: THR at `0x00`, LSR at `0x14` (reg-shift=2), LSR_THRE = `1 << 5`; `putc()` spin-waits on THRE then writes to THR
- [x] 3.2 Gate existing PL011 code (`init`, `putc`, register constants) on `#[cfg(not(feature = "tegra-x1"))]`; keep `puts`, `put_hex`, `put_dec` shared (they call the platform-specific `putc`)
- [x] 3.3 Add `#[cfg(feature = "tegra-x1")]` no-op `init()` so callers don't need conditional compilation
- [x] 3.4 Write unit tests for NS16550A register offset calculations (LSR = 5x4, THR = 0, LSR_THRE = 0x20)

## 4. GICv2 Interrupt Controller

- [x] 4.1 Create `arch/aarch64/src/gicv2.rs` with register definitions: GICD_CTLR, GICD_ISENABLER, GICD_ICENABLER, GICD_IPRIORITYR, GICC_CTLR, GICC_PMR, GICC_IAR, GICC_EOIR; import bases from `platform`
- [x] 4.2 Implement `init_gicd()`: enable distributor and timer PPI (IRQ 30)
- [x] 4.3 Implement `init_cpu_interface()`: enable GICC_CTLR, set GICC_PMR = `0xFF`
- [x] 4.4 Implement `iar() -> u32` (read GICC_IAR) and `eoir(irq_id)` (write GICC_EOIR)
- [x] 4.5 Implement `enable_irq(irq_id)` and `disable_irq(irq_id)` for SPIs (IRQ 32+)
- [x] 4.6 Update `lib.rs` to conditionally export `gicv2` under tegra-x1 and existing GICv3 module under qemu-virt
- [x] 4.7 Ensure timer functions (`init_timer`, `timer_reload`, `timer_disable`, `timer_status`) remain unconditional in `interrupts.rs`
- [x] 4.8 Write unit tests: ISENABLER index/bit calculations (IRQ 30 = reg 0 bit 30, IRQ 45 = reg 1 bit 13), GICC_PMR value

## 5. ARM64 Image Header and Linker Script

- [x] 5.1 Create `arch/aarch64/linker-tegra.ld` with base `0x80080000`; place `.text.image_header` before `.text.boot`
- [x] 5.2 Create `arch/aarch64/src/image_header.rs` with `#[link_section = ".text.image_header"]` 64-byte header: branch over header, text_offset, image_size, flags (`0x02`), magic (`0x644D5241`)
- [x] 5.3 Register `#[cfg(feature = "tegra-x1")] pub mod image_header;` in `lib.rs`
- [x] 5.4 Write unit tests: magic `0x644D5241` == b"ARM\x64" LE, flags `0x02` = LE + 4K pages
- [x] 5.5 Add `build-kernel-jetson` Makefile target: build with `--features tegra-x1`, `linker-tegra.ld`, and `llvm-objcopy -O binary` to produce raw Image
- [x] 5.6 Add `check-size-jetson` Makefile target verifying Image < 15 MB

## 6. SD Card Image Builder

- [x] 6.1 Vendor DTB at `arch/aarch64/dtb/tegra210-p3450-0000.dtb` from JetPack L4T BSP; add README noting source and license
- [x] 6.2 Create `scripts/make-sdcard-jetson.sh`: 64 MB sparse file, GPT + ext4, mount via loop, copy Image + DTB + extlinux.conf, unmount; output to `build/sdcard-jetson.img`
- [x] 6.3 Write extlinux.conf template: DEFAULT primary, LINUX /boot/Image, FDT /boot/tegra210-p3450-0000.dtb, APPEND console=ttyS0,115200n8
- [x] 6.4 Add `make sdcard-jetson` target depending on `build-kernel-jetson`
- [x] 6.5 Add 64 MB size-check assertion in script; print final image size
- [x] 6.6 Test end-to-end: run `make sdcard-jetson`, verify GPT image, mount ext4, confirm /boot contents (requires root for losetup/mount)

## 7. CI Integration

- [x] 7.1 Add `build-jetson` job to `.github/workflows/ci.yml`: cross-compile AArch64 with `--features tegra-x1` and `linker-tegra.ld`
- [x] 7.2 Add Tegra binary to image-size job's size check (< 15 MB)
- [x] 7.3 Add `build-jetson` to change-gates job's `needs` array
- [x] 7.4 Verify CI: Tegra build passes, existing AArch64 QEMU build unaffected, clippy + fmt clean
- [x] 7.5 Update crate description to `"SmallAIOS AArch64 architecture HAL: boot, GICv2/v3, paging, SVE, Tegra X1"`

## 8. Tegra PCIe Controller

- [x] 8.1 Create `arch/aarch64/src/tegra_pcie.rs` with register definitions: AFI base, pads base, RP config base, CAR offsets
- [x] 8.2 Implement CAR helpers: `enable_pcie_clocks()` and `deassert_pcie_resets()` for AFI/PCIe/CML clocks
- [x] 8.3 Implement `tegra_pcie::init()`: CAR clock enable → pad/PHY init → AFI config → root port link training
- [x] 8.4 Implement ECAM-style config space access: `config_read(bus, dev, func, offset) -> u32` and `config_write(...)` via RP config window
- [x] 8.5 Implement PCIe bus enumeration: scan bus 0, build `Vec<PciDevice>` capped at 32
- [x] 8.6 Define `PciDevice` struct: bus/device/function, vendor/device ID, class/subclass, BARs
- [x] 8.7 Implement BAR decoding and assignment: detect 32/64-bit MMIO, compute size via write-all-ones, assign within PCIe MMIO window
- [x] 8.8 Implement `enable_bus_mastering(dev)`: set bit 2 of PCI Command register
- [x] 8.9 Gate module export behind `#[cfg(feature = "tegra-x1")]` in `lib.rs`
- [x] 8.10 Write unit tests: PciAddress encoding, BAR size calculation, config offset validation
- [x] 8.11 Write unit tests: mock enumeration detecting RTL8168 (vendor `0x10EC`, device `0x8168`), BAR assignment
- [x] 8.12 Write unit tests: bus mastering bit set, clock enable sequencing

## 9. RTL8169 Ethernet Driver

- [x] 9.1 Create `net/src/rtl8169.rs` with register offsets: MAC (IDR0-IDR5), Command, TxPoll, IntrMask/Status, TxConfig, RxConfig, TNPDS, RDSAR, PHYStatus
- [x] 9.2 Define TX/RX DMA descriptor structs (16 bytes, 64-bit mode): opts1 (OWN/EOR/FS/LS/length), opts2, addr_low, addr_high
- [x] 9.3 Define `TxDescriptorRing` and `RxDescriptorRing` with 64 entries each, head/tail indices
- [x] 9.4 Define `Rtl8169Device` struct: base_addr, mac, tx_ring, rx_ring, is_initialized
- [x] 9.5 Implement `new(base_addr)`: read MAC from IDR0-IDR5 via MMIO
- [x] 9.6 Implement `reset()`: set Command register reset bit, poll until clear
- [x] 9.7 Implement `init()`: reset → allocate DMA rings → program TNPDS/RDSAR → configure TX/RX → enable
- [x] 9.8 Implement `send(frame)`: write to TX descriptor, set OWN/FS/LS, write TxPoll
- [x] 9.9 Implement `tx_complete()`: scan TX ring for completed descriptors, reclaim
- [x] 9.10 Implement `receive(buf) -> usize`: check RX descriptors, copy frame, replenish buffer
- [x] 9.11 Implement `link_status() -> Option<LinkSpeed>`: read PHYStatus for 10/100/1000 Mbps
- [x] 9.12 Implement `mac_address() -> MacAddress`
- [x] 9.13 Add `pub mod rtl8169;` to `net/src/lib.rs` behind `#[cfg(feature = "rtl8169")]`; add feature to `net/Cargo.toml`
- [x] 9.14 Write unit tests: descriptor struct layout, OWN/EOR/FS/LS bits, 64-bit address split
- [x] 9.15 Write unit tests: ring index wraparound, ring-full detection, reclaim + re-enqueue
- [x] 9.16 Write unit tests: mock MMIO for MAC read, reset poll, TxPoll write
- [x] 9.17 Write unit tests: PHYStatus parsing for link speeds and link-down

## 10. DHCP Client

- [x] 10.1 Create `net/src/dhcp.rs` with message type constants (DISCOVER/OFFER/REQUEST/ACK/NAK) and option codes (subnet=1, router=3, DNS=6, lease=51, type=53, server=54)
- [x] 10.2 Define `DhcpMessage` struct with all fixed fields + options; implement `serialize()` and `parse()`
- [x] 10.3 Implement DHCP options TLV parser: decode after 236-byte header + magic cookie `0x63825363`, stop at option 255
- [x] 10.4 Define `DhcpState` enum (Init/Selecting/Requesting/Bound/Renewing/Expired) and `DhcpLease` struct
- [x] 10.5 Implement `DhcpClient::start_discover()`: build DISCOVER as UDP broadcast (0.0.0.0:68 → 255.255.255.255:67)
- [x] 10.6 Implement `handle_offer()`: validate, extract IP/options, transition to Requesting, return REQUEST
- [x] 10.7 Implement `handle_ack() -> DhcpLease`: parse ACK options (mask, gateway, DNS, lease), transition to Bound
- [x] 10.8 Implement retry logic: 3 retries at 2s/4s/8s exponential backoff; timeout after 10s total
- [x] 10.9 Implement lease renewal: `check_renewal(elapsed_secs)` triggers REQUEST at T1 (50% lease); expire → restart DISCOVER
- [x] 10.10 Add `pub mod dhcp;` to `net/src/lib.rs` behind `#[cfg(feature = "dhcp")]`; add feature to `net/Cargo.toml`
- [x] 10.11 Write unit tests: `DhcpMessage` serialize/parse roundtrip
- [x] 10.12 Write unit tests: options parser with TLV buffer containing options 1, 3, 6, 51, 53, 54, 255
- [x] 10.13 Write unit tests: full DISCOVER→OFFER→REQUEST→ACK state machine with correct DhcpLease output
- [x] 10.14 Write unit tests: retry/timeout behavior (3 retries, exponential backoff, Init on timeout)
- [x] 10.15 Write unit tests: lease renewal at T1, lease expiry triggers DISCOVER restart

## 11. Static IP Configuration

- [x] 11.1 Create `net/src/static_ip.rs` with `StaticIpConfig` struct: ipv4_addr, subnet_mask, gateway, dns, ipv6_addr
- [x] 11.2 Define compile-time config constants behind `#[cfg(feature = "static-ip")]`; implement `StaticIpConfig::from_compiled()`
- [x] 11.3 Implement `InterfaceConfig` enum (Static/Dhcp) and `resolve_config()`: static takes precedence over DHCP
- [x] 11.4 Implement `apply_static_config()`: set IPv4 address, subnet, default route, DNS
- [x] 11.5 Implement `apply_static_ipv6()`: configure IPv6 address for dual-stack
- [x] 11.6 Add `pub mod static_ip;` to `net/src/lib.rs` behind `#[cfg(feature = "static-ip")]`; add feature to `net/Cargo.toml`
- [x] 11.7 Write unit tests: `from_compiled()` returns Some when constants set, None otherwise
- [x] 11.8 Write unit tests: `resolve_config()` priority — static overrides DHCP
- [x] 11.9 Write unit tests: `apply_static_config` stores correct IPv4/mask/gateway/DNS; dual-stack with IPv6

## 12. Network Integration and Testing

- [x] 12.1 Define `NetworkDevice` trait in `net/src/lib.rs`: `send()`, `receive()`, `mac_address()`, `link_status()`; implement for `Rtl8169Device`
- [x] 12.2 Implement end-to-end TX path: NetworkDevice → Ethernet framing → ARP resolution → RTL8169 TX DMA
- [x] 12.3 Implement end-to-end RX path: RTL8169 RX → Ethernet parse → dispatch by EtherType to ARP/IPv4/IPv6
- [x] 12.4 Wire PCIe init into Tegra boot sequence (behind `tegra-x1`): `tegra_pcie::init()` → enumerate → init RTL8169
- [x] 12.5 Wire network config into boot: `resolve_config()` → static or DHCP → configure interface
- [x] 12.6 Add `tegra-net` convenience feature to `net/Cargo.toml` enabling `rtl8169` + `dhcp` + `static-ip`
- [x] 12.7 Write integration test: mock PCIe → RTL8169 init → MAC read → ARP TX/RX via descriptor rings
- [x] 12.8 Write integration test: full DHCP flow over RTL8169
- [x] 12.9 Write integration test: static IP config → ARP table → ICMP echo TX
- [x] 12.10 Write integration test: DHCP fallback behavior (static takes precedence, DHCP when no static)
- [x] 12.11 Run `cargo test -p smallaios-net --features rtl8169,dhcp,static-ip` and clippy; verify zero warnings
- [x] 12.12 Add tegra-x1 feature builds to CI for both `arch/aarch64` and `net` crates
