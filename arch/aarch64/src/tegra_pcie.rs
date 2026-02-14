// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra X1 PCIe root complex controller driver.
//!
//! Implements PCIe initialization and enumeration for the NVIDIA Tegra X1
//! (Jetson Nano) SoC. The Tegra X1 has a single-segment PCIe root complex
//! with one root port (x4 Gen2), controlled by:
//!
//! - **AFI** (AXI-to-FIF bridge) at `0x0100_3000`: Controls PCIe-to-AXI
//!   mapping, MMIO windows, and bus mastering.
//! - **Root Port** config space at `0x0100_0000`: ECAM-style configuration
//!   space window for Type 0/1 config transactions.
//! - **CAR** (Clock And Reset) at `0x6000_6000`: Provides clock enables
//!   and reset de-assertion for the PCIe block.
//!
//! Initialization sequence:
//! 1. Enable PCIe clocks via CAR
//! 2. De-assert PCIe resets via CAR
//! 3. Configure AFI (MMIO window, bus mastering)
//! 4. Initialize pads/PHY
//! 5. Train root port link
//! 6. Enumerate bus 0 for connected devices
//!
//! # No-Allocator Design
//!
//! All data structures use fixed-size arrays since this runs during
//! bare-metal boot before any allocator is available.

#![allow(dead_code)]

use crate::platform;

// ─── Platform base addresses ────────────────────────────────────────────────

/// AFI controller base (PCIe AXI-to-FIFO bridge).
const AFI_BASE: usize = platform::PCIE_AFI_BASE;

/// Root port configuration space window base.
const RP_BASE: usize = platform::PCIE_RP_BASE;

/// Clock and Reset controller base.
const CAR_BASE: usize = platform::CAR_BASE;

// ─── AFI register offsets ───────────────────────────────────────────────────

/// AFI configuration register (master enables, MMIO window config).
const AFI_AXI_BAR0_START: usize = 0x000;
/// AFI AXI BAR0 size register.
const AFI_AXI_BAR0_SZ: usize = 0x004;
/// AFI AXI BAR1 start (MMIO window for downstream devices).
const AFI_AXI_BAR1_START: usize = 0x008;
/// AFI AXI BAR1 size register.
const AFI_AXI_BAR1_SZ: usize = 0x00C;
/// AFI FPCI BAR0 register (maps AXI address to PCIe address).
const AFI_FPCI_BAR0: usize = 0x010;
/// AFI FPCI BAR1 register.
const AFI_FPCI_BAR1: usize = 0x014;
/// AFI configuration register (global enables).
const AFI_CONFIGURATION: usize = 0x0AC;
/// AFI PCIe configuration register (link enables, power control).
const AFI_PCIE_CONFIG: usize = 0x0F8;
/// AFI cache control / MCCIF control.
const AFI_CACHE_CTRL: usize = 0x0FC;
/// AFI pad control register (PLLE, calibration).
const AFI_PLLE_CONTROL: usize = 0x100;
/// AFI power management register.
const AFI_PEXBIAS_CTRL: usize = 0x168;

// ─── AFI configuration bits ─────────────────────────────────────────────────

/// Enable PCIe fabric in AFI_CONFIGURATION.
const AFI_CFG_EN_FPCI: u32 = 1 << 0;
/// Enable bus mastering through AFI.
const AFI_CFG_BUS_MASTER: u32 = 1 << 1;
/// Enable memory space decoding.
const AFI_CFG_MEM_SPACE: u32 = 1 << 2;
/// Enable I/O space decoding.
const AFI_CFG_IO_SPACE: u32 = 1 << 3;

/// AFI_PCIE_CONFIG: disable clock-gating of root port 0.
const AFI_PCIE_CONFIG_SM2TMS0_XBAR_CONFIG_SINGLE: u32 = 0x1 << 20;
/// AFI_PCIE_CONFIG: enable PCIe root port 0.
const AFI_PCIE_CONFIG_PCIE_DISABLE_RP0: u32 = 1 << 5;

// ─── CAR register offsets (PCIe-relevant subset) ────────────────────────────

/// Clock enable L register (contains PCIe, AFI, PEX USB/PHY clocks).
const CAR_CLK_OUT_ENB_U: usize = 0x018;
/// Reset de-assertion register for PCIe block.
const CAR_RST_DEV_U_CLR: usize = 0x314;
/// Clock enable set register (set-only, no RMW needed).
const CAR_CLK_ENB_U_SET: usize = 0x330;

/// Bit position: PCIe controller clock in CLK_OUT_ENB_U / RST_DEV_U.
const CAR_CLK_PCIE: u32 = 1 << 6;
/// Bit position: AFI clock in CLK_OUT_ENB_U / RST_DEV_U.
const CAR_CLK_AFI: u32 = 1 << 8;
/// Bit position: CML (Current Mode Logic, PCIe PHY) clock.
const CAR_CLK_CML: u32 = 1 << 9;
/// Bit position: PCIe XCLK (transaction layer clock).
const CAR_CLK_PCIEX: u32 = 1 << 10;

/// All PCIe-related clocks ORed together.
const CAR_CLK_PCIE_ALL: u32 = CAR_CLK_PCIE | CAR_CLK_AFI | CAR_CLK_CML | CAR_CLK_PCIEX;

// ─── PCI Configuration Space offsets ────────────────────────────────────────

/// Vendor ID (16-bit) + Device ID (16-bit).
const PCI_CFG_VENDOR_DEVICE: usize = 0x00;
/// Command (16-bit) + Status (16-bit).
const PCI_CFG_COMMAND_STATUS: usize = 0x04;
/// Revision (8-bit) + Class code (24-bit: prog-if, subclass, class).
const PCI_CFG_REVISION_CLASS: usize = 0x08;
/// BAR0.
const PCI_CFG_BAR0: usize = 0x10;
/// Number of BAR registers in Type 0 header.
const PCI_CFG_BAR_COUNT: usize = 6;

/// PCI Command register: Bus Master Enable (bit 2).
const PCI_CMD_BUS_MASTER: u32 = 1 << 2;
/// PCI Command register: Memory Space Enable (bit 1).
const PCI_CMD_MEMORY_SPACE: u32 = 1 << 1;

// ─── ECAM-style config address encoding ─────────────────────────────────────

/// PCIe MMIO window base for downstream devices.
/// On Tegra X1, AFI maps bus 1+ config space starting at RP_BASE + 0x1000.
const ECAM_OFFSET: usize = 0x1000;

/// Tegra X1 PCIe MMIO window for BAR assignments (1 GiB window).
const PCIE_MMIO_BASE: u64 = 0x1200_0000;
/// Size of the MMIO window available for BAR assignment.
const PCIE_MMIO_SIZE: u64 = 0x2000_0000; // 512 MiB

// ─── Link training constants ────────────────────────────────────────────────

/// Root Port link status register offset (within RP config space).
const RP_LINK_STATUS_CTRL: usize = 0x090;
/// Link training complete bit in link status register.
const LINK_STATUS_DL_LINK_ACTIVE: u32 = 1 << 29;
/// Maximum microsecond polls waiting for link training.
const LINK_TRAIN_TIMEOUT_US: u32 = 200_000;

// ─── Limits ─────────────────────────────────────────────────────────────────

/// Maximum number of PCI devices we can discover (no allocator).
const MAX_PCI_DEVICES: usize = 32;

/// Maximum devices per bus (PCI spec: 5-bit device number).
const MAX_DEVICES_PER_BUS: u8 = 32;

/// Maximum functions per device (PCI spec: 3-bit function number).
const MAX_FUNCTIONS_PER_DEVICE: u8 = 8;

/// Invalid vendor ID (no device present).
const VENDOR_ID_INVALID: u16 = 0xFFFF;

// ─── MMIO helpers ───────────────────────────────────────────────────────────

/// Read a 32-bit value from an MMIO register.
///
/// # Safety
/// `addr` must be a valid, mapped MMIO address aligned to 4 bytes.
#[inline(always)]
unsafe fn mmio_read32(addr: usize) -> u32 {
    let ptr = addr as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit value to an MMIO register.
///
/// # Safety
/// `addr` must be a valid, mapped MMIO address aligned to 4 bytes.
#[inline(always)]
unsafe fn mmio_write32(addr: usize, value: u32) {
    let ptr = addr as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

// ─── AFI helpers ────────────────────────────────────────────────────────────

/// Read an AFI register.
///
/// # Safety
/// Must only be called after PCIe clocks are enabled.
unsafe fn afi_read(offset: usize) -> u32 {
    mmio_read32(AFI_BASE + offset)
}

/// Write an AFI register.
///
/// # Safety
/// Must only be called after PCIe clocks are enabled.
unsafe fn afi_write(offset: usize, value: u32) {
    mmio_write32(AFI_BASE + offset, value);
}

// ─── CAR helpers ────────────────────────────────────────────────────────────

/// Read a CAR register.
///
/// # Safety
/// CAR registers are always accessible (system-level clock controller).
unsafe fn car_read(offset: usize) -> u32 {
    mmio_read32(CAR_BASE + offset)
}

/// Write a CAR register.
///
/// # Safety
/// Incorrect CAR writes can hang the SoC. Only write documented values.
unsafe fn car_write(offset: usize, value: u32) {
    mmio_write32(CAR_BASE + offset, value);
}

/// Enable PCIe, AFI, CML, and PCIEX clocks via the CAR set register.
///
/// Uses the set-only register (`CLK_ENB_U_SET`) so no read-modify-write
/// is required — writing a 1 enables the clock, writing 0 has no effect.
///
/// # Safety
/// Must be called before any PCIe/AFI register access.
pub unsafe fn enable_pcie_clocks() {
    car_write(CAR_CLK_ENB_U_SET, CAR_CLK_PCIE_ALL);
}

/// De-assert resets for the PCIe controller block via CAR.
///
/// Writes to the clear register (`RST_DEV_U_CLR`) — writing a 1 clears
/// the corresponding reset bit, bringing the block out of reset.
///
/// # Safety
/// Must be called after `enable_pcie_clocks()`.
pub unsafe fn deassert_pcie_resets() {
    car_write(CAR_RST_DEV_U_CLR, CAR_CLK_PCIE_ALL);
}

// ─── PciDevice ──────────────────────────────────────────────────────────────

/// A single PCI device discovered during bus enumeration.
///
/// Uses fixed-size arrays (no allocator required). BAR values are stored
/// as raw 64-bit addresses; size is computed via the write-all-ones method
/// during enumeration.
#[derive(Clone, Debug)]
pub struct PciDevice {
    /// PCI bus number (0 for root port downstream).
    pub bus: u8,
    /// Device number on the bus (0-31).
    pub device: u8,
    /// Function number (0-7).
    pub function: u8,
    /// PCI vendor ID (0xFFFF = no device).
    pub vendor_id: u16,
    /// PCI device ID.
    pub device_id: u16,
    /// PCI class code (bits [31:24] of offset 0x08).
    pub class_code: u8,
    /// PCI subclass (bits [23:16] of offset 0x08).
    pub subclass: u8,
    /// Base Address Register values (up to 6 BARs, 0 = unused).
    pub bars: [u64; 6],
    /// BAR sizes in bytes (0 = unused or unimplemented BAR).
    pub bar_sizes: [u64; 6],
}

impl PciDevice {
    /// Create a zeroed/empty `PciDevice`.
    pub const fn empty() -> Self {
        Self {
            bus: 0,
            device: 0,
            function: 0,
            vendor_id: VENDOR_ID_INVALID,
            device_id: 0,
            class_code: 0,
            subclass: 0,
            bars: [0; 6],
            bar_sizes: [0; 6],
        }
    }

    /// Returns `true` if this slot contains a valid device.
    pub fn is_valid(&self) -> bool {
        self.vendor_id != VENDOR_ID_INVALID && self.vendor_id != 0
    }
}

// ─── PciDeviceList ──────────────────────────────────────────────────────────

/// Fixed-size list of discovered PCI devices (no allocator).
pub struct PciDeviceList {
    devices: [PciDevice; MAX_PCI_DEVICES],
    count: usize,
}

impl PciDeviceList {
    /// Create an empty device list.
    pub const fn new() -> Self {
        Self {
            devices: [const { PciDevice::empty() }; MAX_PCI_DEVICES],
            count: 0,
        }
    }

    /// Add a device to the list. Returns `false` if full.
    pub fn push(&mut self, device: PciDevice) -> bool {
        if self.count >= MAX_PCI_DEVICES {
            return false;
        }
        self.devices[self.count] = device;
        self.count += 1;
        true
    }

    /// Number of devices discovered.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get a reference to a device by index.
    pub fn get(&self, index: usize) -> Option<&PciDevice> {
        if index < self.count {
            Some(&self.devices[index])
        } else {
            None
        }
    }

    /// Return a slice of all discovered devices.
    pub fn devices(&self) -> &[PciDevice] {
        &self.devices[..self.count]
    }
}

impl Default for PciDeviceList {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ECAM config space access ───────────────────────────────────────────────

/// Compute the ECAM-style MMIO address for a PCI config register.
///
/// Tegra X1 config space encoding for MMIO-based access:
///
/// ```text
/// address = RP_BASE + ECAM_OFFSET + (bus << 20) | (dev << 15) | (func << 12) | offset
/// ```
///
/// For bus 0 (root port itself), the address is `RP_BASE + offset`.
pub const fn ecam_address(bus: u8, device: u8, function: u8, offset: usize) -> usize {
    if bus == 0 {
        (RP_BASE + ((device as usize & 0x1F) << 15))
            | ((function as usize & 0x07) << 12)
            | (offset & 0xFFC)
    } else {
        (RP_BASE + ECAM_OFFSET + ((bus as usize) << 20))
            | ((device as usize & 0x1F) << 15)
            | ((function as usize & 0x07) << 12)
            | (offset & 0xFFC)
    }
}

/// Encode a PCI config address for offset calculation (pure logic, no MMIO).
///
/// ```text
/// (bus << 20) | (dev << 15) | (func << 12) | (offset & 0xFFC)
/// ```
pub const fn pci_config_offset(bus: u8, device: u8, function: u8, offset: usize) -> usize {
    ((bus as usize) << 20)
        | ((device as usize & 0x1F) << 15)
        | ((function as usize & 0x07) << 12)
        | (offset & 0xFFC)
}

/// Read a 32-bit PCI configuration register.
///
/// # Safety
/// - PCIe must be initialized (clocks, AFI, link trained)
/// - `offset` must be 4-byte aligned and < 4096
pub unsafe fn config_read(bus: u8, device: u8, function: u8, offset: usize) -> u32 {
    let addr = ecam_address(bus, device, function, offset);
    mmio_read32(addr)
}

/// Write a 32-bit PCI configuration register.
///
/// # Safety
/// - PCIe must be initialized (clocks, AFI, link trained)
/// - `offset` must be 4-byte aligned and < 4096
pub unsafe fn config_write(bus: u8, device: u8, function: u8, offset: usize, value: u32) {
    let addr = ecam_address(bus, device, function, offset);
    mmio_write32(addr, value);
}

// ─── BAR decoding ───────────────────────────────────────────────────────────

/// Determine whether a BAR raw value indicates a 64-bit memory BAR.
///
/// PCI BAR layout for memory:
/// - Bit 0: 0 = memory, 1 = I/O
/// - Bits 2:1: 00 = 32-bit, 10 = 64-bit
pub const fn bar_is_64bit(bar_value: u32) -> bool {
    // Bit 0 must be 0 (memory) and bits 2:1 must be 10 (64-bit)
    (bar_value & 0x01) == 0 && ((bar_value >> 1) & 0x03) == 0x02
}

/// Determine whether a BAR raw value indicates an I/O BAR.
pub const fn bar_is_io(bar_value: u32) -> bool {
    (bar_value & 0x01) != 0
}

/// Determine whether a memory BAR is prefetchable.
pub const fn bar_is_prefetchable(bar_value: u32) -> bool {
    (bar_value & 0x01) == 0 && (bar_value & 0x08) != 0
}

/// Calculate BAR size from the write-all-ones / read-back method result.
///
/// Given the read-back value after writing `0xFFFF_FFFF` to a BAR:
/// 1. Mask off the type/attribute bits (lowest 4 bits for memory BARs)
/// 2. Invert the result
/// 3. Add 1 to get the size
///
/// Returns 0 if the BAR is unimplemented (read-back is 0).
pub const fn bar_size_from_mask(readback: u32, is_io: bool) -> u64 {
    if readback == 0 {
        return 0;
    }
    let mask = if is_io { 0xFFFF_FFFC } else { 0xFFFF_FFF0 };
    let masked = readback & mask;
    if masked == 0 {
        return 0;
    }
    let inverted = !masked;
    (inverted as u64) + 1
}

/// Decode all 6 BARs for a PCI device.
///
/// Performs the standard write-all-ones/read-back procedure to determine
/// BAR sizes, then restores original values.
///
/// # Safety
/// - PCIe config space must be accessible
/// - This temporarily modifies BAR registers (restores afterwards)
unsafe fn decode_bars(
    bus: u8,
    device: u8,
    function: u8,
    bars: &mut [u64; 6],
    bar_sizes: &mut [u64; 6],
) {
    let mut i = 0;
    while i < PCI_CFG_BAR_COUNT {
        let bar_offset = PCI_CFG_BAR0 + (i * 4);

        // Read original BAR value
        let original = config_read(bus, device, function, bar_offset);

        if original == 0 {
            // Unimplemented BAR
            bars[i] = 0;
            bar_sizes[i] = 0;
            i += 1;
            continue;
        }

        let is_io = bar_is_io(original);

        // Write all-ones to determine size
        config_write(bus, device, function, bar_offset, 0xFFFF_FFFF);
        let readback = config_read(bus, device, function, bar_offset);

        // Restore original value
        config_write(bus, device, function, bar_offset, original);

        if bar_is_64bit(original) && i + 1 < PCI_CFG_BAR_COUNT {
            // 64-bit BAR: read high 32 bits from next BAR slot
            let hi_offset = PCI_CFG_BAR0 + ((i + 1) * 4);
            let original_hi = config_read(bus, device, function, hi_offset);

            bars[i] = ((original_hi as u64) << 32) | (original as u64 & 0xFFFF_FFF0);
            bar_sizes[i] = bar_size_from_mask(readback, false);
            bars[i + 1] = 0; // Consumed by 64-bit BAR
            bar_sizes[i + 1] = 0;
            i += 2;
        } else {
            // 32-bit BAR or I/O BAR
            let addr_mask: u32 = if is_io { 0xFFFF_FFFC } else { 0xFFFF_FFF0 };
            bars[i] = (original & addr_mask) as u64;
            bar_sizes[i] = bar_size_from_mask(readback, is_io);
            i += 1;
        }
    }
}

// ─── Bus mastering ──────────────────────────────────────────────────────────

/// Enable bus mastering for a PCI device.
///
/// Sets bit 2 (Bus Master Enable) in the PCI Command register at
/// config offset 0x04. Also enables Memory Space (bit 1).
///
/// # Safety
/// PCIe config space must be accessible and the device must be valid.
pub unsafe fn enable_bus_mastering(dev: &PciDevice) {
    let cmd = config_read(dev.bus, dev.device, dev.function, PCI_CFG_COMMAND_STATUS);
    let new_cmd = cmd | PCI_CMD_BUS_MASTER | PCI_CMD_MEMORY_SPACE;
    config_write(
        dev.bus,
        dev.device,
        dev.function,
        PCI_CFG_COMMAND_STATUS,
        new_cmd,
    );
}

// ─── AFI initialization ────────────────────────────────────────────────────

/// Configure the AFI controller.
///
/// Sets up the AXI-to-PCIe MMIO window and enables the PCIe fabric.
///
/// # Safety
/// Must be called after clocks are enabled and resets de-asserted.
unsafe fn init_afi() {
    // Configure AXI BAR0: maps CPU MMIO region to PCIe config space
    afi_write(AFI_AXI_BAR0_START, RP_BASE as u32);
    afi_write(AFI_AXI_BAR0_SZ, 0x1000); // 4 KiB for root port config

    // Configure AXI BAR1: downstream device MMIO window
    afi_write(AFI_AXI_BAR1_START, PCIE_MMIO_BASE as u32);
    afi_write(AFI_AXI_BAR1_SZ, PCIE_MMIO_SIZE as u32);

    // Map FPCI addresses
    afi_write(AFI_FPCI_BAR0, (RP_BASE >> 12) as u32);
    afi_write(AFI_FPCI_BAR1, (PCIE_MMIO_BASE >> 12) as u32);

    // Enable PCIe root port 0 (clear the disable bit)
    let pcie_cfg = afi_read(AFI_PCIE_CONFIG);
    let pcie_cfg = pcie_cfg & !AFI_PCIE_CONFIG_PCIE_DISABLE_RP0;
    let pcie_cfg = pcie_cfg | AFI_PCIE_CONFIG_SM2TMS0_XBAR_CONFIG_SINGLE;
    afi_write(AFI_PCIE_CONFIG, pcie_cfg);

    // Enable FPCI fabric, bus mastering, memory space
    afi_write(
        AFI_CONFIGURATION,
        AFI_CFG_EN_FPCI | AFI_CFG_BUS_MASTER | AFI_CFG_MEM_SPACE,
    );
}

/// Initialize pads and PHY for PCIe link.
///
/// Configures the PLLE (PLL for PCIe) and pad calibration via AFI
/// control registers.
///
/// # Safety
/// Must be called after clocks are enabled and resets de-asserted.
unsafe fn init_pads() {
    // Enable PLLE for PCIe reference clock
    let plle = afi_read(AFI_PLLE_CONTROL);
    afi_write(AFI_PLLE_CONTROL, plle | (1 << 0)); // PLLE enable

    // Enable PEX bias pad
    afi_write(AFI_PEXBIAS_CTRL, 0);

    // Basic pad calibration delay (hardware needs time to stabilize)
    // In a real driver we'd poll a status register; here we rely on
    // the subsequent link training timeout to cover this.
}

/// Wait for PCIe link training to complete on root port 0.
///
/// Polls the link status register for the Data Link Layer active bit.
/// Returns `true` if the link trained successfully, `false` on timeout.
///
/// # Safety
/// Must be called after AFI and pad initialization.
unsafe fn wait_for_link() -> bool {
    // In bare-metal context we would use a timer; here we use a
    // simple poll loop with a generous iteration count.
    for _ in 0..LINK_TRAIN_TIMEOUT_US {
        let status = mmio_read32(RP_BASE + RP_LINK_STATUS_CTRL);
        if status & LINK_STATUS_DL_LINK_ACTIVE != 0 {
            return true;
        }
        // Tiny spin delay (placeholder for proper timer-based delay)
        for _ in 0..100 {
            core::hint::spin_loop();
        }
    }
    false
}

// ─── Top-level init ─────────────────────────────────────────────────────────

/// Initialize the Tegra X1 PCIe root complex.
///
/// Performs the full initialization sequence:
/// 1. Enable PCIe clocks via CAR
/// 2. De-assert PCIe resets
/// 3. Configure AFI (MMIO windows, fabric enable)
/// 4. Initialize pads/PHY
/// 5. Train root port link
///
/// Returns `true` if the PCIe link trained successfully.
///
/// # Safety
/// Must be called exactly once during boot, before any PCIe device access.
/// Requires that the UART is initialized for debug output.
pub unsafe fn init() -> bool {
    // Step 1: Enable clocks
    enable_pcie_clocks();

    // Step 2: De-assert resets
    deassert_pcie_resets();

    // Step 3: Configure AFI
    init_afi();

    // Step 4: Initialize pads and PHY
    init_pads();

    // Step 5: Train root port link
    wait_for_link()
}

// ─── Bus enumeration ────────────────────────────────────────────────────────

/// Enumerate PCI bus 0 and return discovered devices.
///
/// Scans all 32 device slots on bus 0, reading vendor/device IDs. For
/// each valid device, reads class code, subclass, and decodes all BARs.
///
/// # Safety
/// PCIe must be fully initialized (`init()` returned `true`).
pub unsafe fn enumerate_bus(bus: u8) -> PciDeviceList {
    let mut list = PciDeviceList::new();

    for dev in 0..MAX_DEVICES_PER_BUS {
        // Read vendor/device ID from function 0
        let id_reg = config_read(bus, dev, 0, PCI_CFG_VENDOR_DEVICE);
        let vendor_id = (id_reg & 0xFFFF) as u16;
        let device_id = ((id_reg >> 16) & 0xFFFF) as u16;

        if vendor_id == VENDOR_ID_INVALID || vendor_id == 0 {
            continue;
        }

        // Read class code and subclass
        let class_reg = config_read(bus, dev, 0, PCI_CFG_REVISION_CLASS);
        let class_code = ((class_reg >> 24) & 0xFF) as u8;
        let subclass = ((class_reg >> 16) & 0xFF) as u8;

        // Decode BARs
        let mut bars = [0u64; 6];
        let mut bar_sizes = [0u64; 6];
        decode_bars(bus, dev, 0, &mut bars, &mut bar_sizes);

        let device = PciDevice {
            bus,
            device: dev,
            function: 0,
            vendor_id,
            device_id,
            class_code,
            subclass,
            bars,
            bar_sizes,
        };

        if !list.push(device) {
            break; // List full
        }
    }

    list
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Register offset constants ──────────────────────────────────────

    #[test]
    fn test_afi_register_offsets() {
        assert_eq!(AFI_AXI_BAR0_START, 0x000);
        assert_eq!(AFI_AXI_BAR0_SZ, 0x004);
        assert_eq!(AFI_AXI_BAR1_START, 0x008);
        assert_eq!(AFI_AXI_BAR1_SZ, 0x00C);
        assert_eq!(AFI_FPCI_BAR0, 0x010);
        assert_eq!(AFI_FPCI_BAR1, 0x014);
        assert_eq!(AFI_CONFIGURATION, 0x0AC);
        assert_eq!(AFI_PCIE_CONFIG, 0x0F8);
        assert_eq!(AFI_PLLE_CONTROL, 0x100);
        assert_eq!(AFI_PEXBIAS_CTRL, 0x168);
    }

    #[test]
    fn test_car_register_offsets() {
        assert_eq!(CAR_CLK_OUT_ENB_U, 0x018);
        assert_eq!(CAR_RST_DEV_U_CLR, 0x314);
        assert_eq!(CAR_CLK_ENB_U_SET, 0x330);
    }

    #[test]
    fn test_car_clock_bits() {
        assert_eq!(CAR_CLK_PCIE, 1 << 6);
        assert_eq!(CAR_CLK_AFI, 1 << 8);
        assert_eq!(CAR_CLK_CML, 1 << 9);
        assert_eq!(CAR_CLK_PCIEX, 1 << 10);
        // All clock bits ORed
        assert_eq!(CAR_CLK_PCIE_ALL, (1 << 6) | (1 << 8) | (1 << 9) | (1 << 10));
    }

    #[test]
    fn test_pci_cfg_offsets() {
        assert_eq!(PCI_CFG_VENDOR_DEVICE, 0x00);
        assert_eq!(PCI_CFG_COMMAND_STATUS, 0x04);
        assert_eq!(PCI_CFG_REVISION_CLASS, 0x08);
        assert_eq!(PCI_CFG_BAR0, 0x10);
    }

    // ─── PCI config address encoding ────────────────────────────────────

    #[test]
    fn test_pci_config_offset_bus0_dev0() {
        let offset = pci_config_offset(0, 0, 0, 0);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_pci_config_offset_bus_shift() {
        // bus=1 should shift left by 20 bits
        let offset = pci_config_offset(1, 0, 0, 0);
        assert_eq!(offset, 1 << 20);
    }

    #[test]
    fn test_pci_config_offset_device_shift() {
        // device=1 should shift left by 15 bits
        let offset = pci_config_offset(0, 1, 0, 0);
        assert_eq!(offset, 1 << 15);
    }

    #[test]
    fn test_pci_config_offset_function_shift() {
        // function=1 should shift left by 12 bits
        let offset = pci_config_offset(0, 0, 1, 0);
        assert_eq!(offset, 1 << 12);
    }

    #[test]
    fn test_pci_config_offset_register() {
        // Register offset masked to 12-bit (4K config space)
        let offset = pci_config_offset(0, 0, 0, 0x100);
        assert_eq!(offset, 0x100);
    }

    #[test]
    fn test_pci_config_offset_full_encoding() {
        // bus=2, dev=3, func=1, offset=0x10
        let offset = pci_config_offset(2, 3, 1, 0x10);
        let expected = (2 << 20) | (3 << 15) | (1 << 12) | 0x10;
        assert_eq!(offset, expected);
    }

    #[test]
    fn test_pci_config_offset_device_masked_to_5bits() {
        // Device number > 31 should be masked to 5 bits
        let offset = pci_config_offset(0, 0x3F, 0, 0);
        let dev_field = (offset >> 15) & 0x1F;
        assert_eq!(dev_field, 0x1F); // 0x3F & 0x1F = 0x1F
    }

    #[test]
    fn test_pci_config_offset_function_masked_to_3bits() {
        // Function number > 7 should be masked to 3 bits
        let offset = pci_config_offset(0, 0, 0xFF, 0);
        let func_field = (offset >> 12) & 0x07;
        assert_eq!(func_field, 0x07); // 0xFF & 0x07 = 0x07
    }

    #[test]
    fn test_pci_config_offset_register_aligned() {
        // Register offset is masked to 12 bits (0xFFC)
        let offset = pci_config_offset(0, 0, 0, 0xFFC);
        assert_eq!(offset & 0xFFF, 0xFFC);
    }

    #[test]
    fn test_pci_config_offset_max_values() {
        // All fields at maximum valid values
        let offset = pci_config_offset(255, 31, 7, 0xFFC);
        assert_eq!((offset >> 20) & 0xFF, 255);
        assert_eq!((offset >> 15) & 0x1F, 31);
        assert_eq!((offset >> 12) & 0x07, 7);
        assert_eq!(offset & 0xFFF, 0xFFC);
    }

    // ─── BAR decoding logic ─────────────────────────────────────────────

    #[test]
    fn test_bar_is_64bit_memory() {
        // Bit 0 = 0 (memory), bits 2:1 = 10 (64-bit)
        let bar_val: u32 = 0xF000_0004; // type bits = 0b100
        assert!(bar_is_64bit(bar_val));
    }

    #[test]
    fn test_bar_is_32bit_memory() {
        // Bit 0 = 0 (memory), bits 2:1 = 00 (32-bit)
        let bar_val: u32 = 0xF000_0000;
        assert!(!bar_is_64bit(bar_val));
    }

    #[test]
    fn test_bar_is_io() {
        // Bit 0 = 1 (I/O)
        let bar_val: u32 = 0x0000_1001;
        assert!(bar_is_io(bar_val));
    }

    #[test]
    fn test_bar_is_not_io() {
        let bar_val: u32 = 0xF000_0000;
        assert!(!bar_is_io(bar_val));
    }

    #[test]
    fn test_bar_is_prefetchable() {
        // Bit 3 = 1, bit 0 = 0 (memory)
        let bar_val: u32 = 0xF000_0008;
        assert!(bar_is_prefetchable(bar_val));
    }

    #[test]
    fn test_bar_is_not_prefetchable() {
        let bar_val: u32 = 0xF000_0000;
        assert!(!bar_is_prefetchable(bar_val));
    }

    #[test]
    fn test_bar_io_not_prefetchable() {
        // I/O BARs are never prefetchable (bit 0 = 1)
        let bar_val: u32 = 0x0000_1009; // I/O with bit 3 set
        assert!(!bar_is_prefetchable(bar_val));
    }

    #[test]
    fn test_bar_size_from_mask_1mb() {
        // 1 MiB BAR: after writing 0xFFFFFFFF, readback has low 20 bits
        // set to 0 (size indicator), with attribute bits.
        // readback = 0xFFF0_0000 (memory BAR, 1 MiB)
        let readback: u32 = 0xFFF0_0000;
        let size = bar_size_from_mask(readback, false);
        assert_eq!(size, 0x0010_0000); // 1 MiB
    }

    #[test]
    fn test_bar_size_from_mask_16mb() {
        // 16 MiB BAR
        let readback: u32 = 0xFF00_0000;
        let size = bar_size_from_mask(readback, false);
        assert_eq!(size, 0x0100_0000); // 16 MiB
    }

    #[test]
    fn test_bar_size_from_mask_256mb() {
        // 256 MiB BAR
        let readback: u32 = 0xF000_0000;
        let size = bar_size_from_mask(readback, false);
        assert_eq!(size, 0x1000_0000); // 256 MiB
    }

    #[test]
    fn test_bar_size_from_mask_4kb() {
        // Minimum memory BAR: 4 KiB
        let readback: u32 = 0xFFFF_F000;
        let size = bar_size_from_mask(readback, false);
        assert_eq!(size, 0x1000); // 4 KiB
    }

    #[test]
    fn test_bar_size_from_mask_io_256() {
        // I/O BAR, 256 bytes: readback = 0xFFFF_FF00
        let readback: u32 = 0xFFFF_FF00;
        let size = bar_size_from_mask(readback, true);
        assert_eq!(size, 256);
    }

    #[test]
    fn test_bar_size_from_mask_unimplemented() {
        // Unimplemented BAR reads back as 0
        let size = bar_size_from_mask(0, false);
        assert_eq!(size, 0);
    }

    #[test]
    fn test_bar_size_from_mask_all_attribute_bits() {
        // Memory BAR with prefetchable + 64-bit: low 4 bits = 0b1100
        // readback = 0xFFF0_000C -> masked to 0xFFF0_0000 -> size = 1 MiB
        let readback: u32 = 0xFFF0_000C;
        let size = bar_size_from_mask(readback, false);
        assert_eq!(size, 0x0010_0000); // 1 MiB
    }

    // ─── AFI configuration bits ─────────────────────────────────────────

    #[test]
    fn test_afi_cfg_bits() {
        assert_eq!(AFI_CFG_EN_FPCI, 1 << 0);
        assert_eq!(AFI_CFG_BUS_MASTER, 1 << 1);
        assert_eq!(AFI_CFG_MEM_SPACE, 1 << 2);
        assert_eq!(AFI_CFG_IO_SPACE, 1 << 3);
    }

    #[test]
    fn test_afi_pcie_config_bits() {
        assert_eq!(AFI_PCIE_CONFIG_SM2TMS0_XBAR_CONFIG_SINGLE, 0x1 << 20);
        assert_eq!(AFI_PCIE_CONFIG_PCIE_DISABLE_RP0, 1 << 5);
    }

    // ─── PCI Command register bits ──────────────────────────────────────

    #[test]
    fn test_pci_cmd_bus_master_bit() {
        assert_eq!(PCI_CMD_BUS_MASTER, 0x04);
        assert_eq!(PCI_CMD_BUS_MASTER, 1 << 2);
    }

    #[test]
    fn test_pci_cmd_memory_space_bit() {
        assert_eq!(PCI_CMD_MEMORY_SPACE, 0x02);
        assert_eq!(PCI_CMD_MEMORY_SPACE, 1 << 1);
    }

    #[test]
    fn test_bus_master_enable_ors_correctly() {
        let cmd: u32 = 0x0000_0000;
        let new_cmd = cmd | PCI_CMD_BUS_MASTER | PCI_CMD_MEMORY_SPACE;
        assert_eq!(new_cmd & PCI_CMD_BUS_MASTER, PCI_CMD_BUS_MASTER);
        assert_eq!(new_cmd & PCI_CMD_MEMORY_SPACE, PCI_CMD_MEMORY_SPACE);
        assert_eq!(new_cmd, 0x06);
    }

    #[test]
    fn test_bus_master_preserves_existing_bits() {
        let cmd: u32 = 0x0000_0100; // Some existing bits set
        let new_cmd = cmd | PCI_CMD_BUS_MASTER | PCI_CMD_MEMORY_SPACE;
        assert_eq!(new_cmd, 0x0000_0106);
    }

    // ─── PciDevice ──────────────────────────────────────────────────────

    #[test]
    fn test_pci_device_empty() {
        let dev = PciDevice::empty();
        assert_eq!(dev.vendor_id, VENDOR_ID_INVALID);
        assert_eq!(dev.device_id, 0);
        assert_eq!(dev.bus, 0);
        assert_eq!(dev.device, 0);
        assert_eq!(dev.function, 0);
        assert!(!dev.is_valid());
    }

    #[test]
    fn test_pci_device_valid() {
        let dev = PciDevice {
            bus: 0,
            device: 1,
            function: 0,
            vendor_id: 0x10EC,
            device_id: 0x8168,
            class_code: 0x02,
            subclass: 0x00,
            bars: [0; 6],
            bar_sizes: [0; 6],
        };
        assert!(dev.is_valid());
    }

    #[test]
    fn test_pci_device_invalid_zero_vendor() {
        let dev = PciDevice {
            bus: 0,
            device: 0,
            function: 0,
            vendor_id: 0,
            device_id: 0,
            class_code: 0,
            subclass: 0,
            bars: [0; 6],
            bar_sizes: [0; 6],
        };
        assert!(!dev.is_valid());
    }

    // ─── PciDeviceList ──────────────────────────────────────────────────

    #[test]
    fn test_device_list_new_empty() {
        let list = PciDeviceList::new();
        assert_eq!(list.count(), 0);
        assert!(list.devices().is_empty());
    }

    #[test]
    fn test_device_list_push_and_count() {
        let mut list = PciDeviceList::new();
        let dev = PciDevice {
            bus: 0,
            device: 1,
            function: 0,
            vendor_id: 0x10EC,
            device_id: 0x8168,
            class_code: 0x02,
            subclass: 0x00,
            bars: [0; 6],
            bar_sizes: [0; 6],
        };
        assert!(list.push(dev));
        assert_eq!(list.count(), 1);
    }

    #[test]
    fn test_device_list_get() {
        let mut list = PciDeviceList::new();
        let dev = PciDevice {
            bus: 0,
            device: 3,
            function: 0,
            vendor_id: 0x10EC,
            device_id: 0x8168,
            class_code: 0x02,
            subclass: 0x00,
            bars: [0; 6],
            bar_sizes: [0; 6],
        };
        list.push(dev);
        let retrieved = list.get(0).unwrap();
        assert_eq!(retrieved.vendor_id, 0x10EC);
        assert_eq!(retrieved.device_id, 0x8168);
        assert_eq!(retrieved.device, 3);
    }

    #[test]
    fn test_device_list_get_out_of_bounds() {
        let list = PciDeviceList::new();
        assert!(list.get(0).is_none());
        assert!(list.get(100).is_none());
    }

    #[test]
    fn test_device_list_max_capacity() {
        let mut list = PciDeviceList::new();
        for i in 0..MAX_PCI_DEVICES {
            let dev = PciDevice {
                bus: 0,
                device: (i % 32) as u8,
                function: 0,
                vendor_id: 0x10EC,
                device_id: 0x8168,
                class_code: 0x02,
                subclass: 0x00,
                bars: [0; 6],
                bar_sizes: [0; 6],
            };
            assert!(list.push(dev), "push #{i} should succeed");
        }
        assert_eq!(list.count(), MAX_PCI_DEVICES);

        // Next push should fail
        let overflow = PciDevice::empty();
        assert!(
            !list.push(overflow),
            "push beyond capacity should return false"
        );
        assert_eq!(list.count(), MAX_PCI_DEVICES);
    }

    #[test]
    fn test_device_list_devices_slice() {
        let mut list = PciDeviceList::new();
        for i in 0..5u8 {
            let dev = PciDevice {
                bus: 0,
                device: i,
                function: 0,
                vendor_id: 0x10EC,
                device_id: 0x8168 + i as u16,
                class_code: 0x02,
                subclass: 0x00,
                bars: [0; 6],
                bar_sizes: [0; 6],
            };
            list.push(dev);
        }
        let slice = list.devices();
        assert_eq!(slice.len(), 5);
        for (i, dev) in slice.iter().enumerate() {
            assert_eq!(dev.device, i as u8);
        }
    }

    // ─── Mock enumeration (RTL8168) ─────────────────────────────────────

    #[test]
    fn test_mock_rtl8168_detection() {
        // Simulate what enumeration would produce for an RTL8168
        let dev = PciDevice {
            bus: 1,
            device: 0,
            function: 0,
            vendor_id: 0x10EC, // Realtek
            device_id: 0x8168, // RTL8168
            class_code: 0x02,  // Network controller
            subclass: 0x00,    // Ethernet controller
            bars: [
                0xDF10_0000, // BAR0: 32-bit MMIO
                0x0000_E001, // BAR1: I/O
                0,
                0,
                0,
                0,
            ],
            bar_sizes: [
                0x100, // BAR0: 256 bytes
                0x100, // BAR1: 256 bytes
                0, 0, 0, 0,
            ],
        };

        assert!(dev.is_valid());
        assert_eq!(dev.vendor_id, 0x10EC);
        assert_eq!(dev.device_id, 0x8168);
        assert_eq!(dev.class_code, 0x02);
        assert_eq!(dev.subclass, 0x00);
    }

    #[test]
    fn test_mock_enumeration_with_rtl8168() {
        let mut list = PciDeviceList::new();

        // Simulate enumeration finding an RTL8168 on bus 1
        let rtl8168 = PciDevice {
            bus: 1,
            device: 0,
            function: 0,
            vendor_id: 0x10EC,
            device_id: 0x8168,
            class_code: 0x02,
            subclass: 0x00,
            bars: [0xDF10_0000, 0x0000_E001, 0, 0, 0, 0],
            bar_sizes: [0x100, 0x100, 0, 0, 0, 0],
        };
        list.push(rtl8168);

        assert_eq!(list.count(), 1);
        let dev = list.get(0).unwrap();
        assert_eq!(dev.vendor_id, 0x10EC);
        assert_eq!(dev.device_id, 0x8168);
        assert_eq!(dev.class_code, 0x02);
    }

    #[test]
    fn test_mock_enumeration_multiple_devices() {
        let mut list = PciDeviceList::new();

        // RTL8168 Ethernet
        list.push(PciDevice {
            bus: 1,
            device: 0,
            function: 0,
            vendor_id: 0x10EC,
            device_id: 0x8168,
            class_code: 0x02,
            subclass: 0x00,
            bars: [0xDF10_0000, 0, 0, 0, 0, 0],
            bar_sizes: [0x100, 0, 0, 0, 0, 0],
        });

        // Some USB controller
        list.push(PciDevice {
            bus: 1,
            device: 1,
            function: 0,
            vendor_id: 0x1B21,
            device_id: 0x1142,
            class_code: 0x0C,
            subclass: 0x03,
            bars: [0xDF00_0000, 0, 0, 0, 0, 0],
            bar_sizes: [0x1000, 0, 0, 0, 0, 0],
        });

        assert_eq!(list.count(), 2);

        // Find the RTL8168
        let ethernet = list
            .devices()
            .iter()
            .find(|d| d.vendor_id == 0x10EC && d.device_id == 0x8168);
        assert!(ethernet.is_some());

        // All devices should be valid
        for dev in list.devices() {
            assert!(dev.is_valid());
        }
    }

    // ─── ECAM address encoding ──────────────────────────────────────────

    #[test]
    fn test_ecam_address_bus0_dev0() {
        // Bus 0 uses RP_BASE directly
        let addr = ecam_address(0, 0, 0, 0);
        assert_eq!(addr, RP_BASE);
    }

    #[test]
    fn test_ecam_address_bus0_with_offset() {
        let addr = ecam_address(0, 0, 0, 0x10);
        assert_eq!(addr, RP_BASE | 0x10);
    }

    #[test]
    fn test_ecam_address_bus1_base() {
        // Bus 1+ uses RP_BASE + ECAM_OFFSET + encoding
        let addr = ecam_address(1, 0, 0, 0);
        let expected = RP_BASE + ECAM_OFFSET + (1 << 20);
        assert_eq!(addr, expected);
    }

    #[test]
    fn test_ecam_address_preserves_offset() {
        let addr = ecam_address(1, 0, 0, 0x10);
        // Verify the offset component
        assert_eq!(addr & 0xFFF, 0x10);
    }

    // ─── Link status bits ───────────────────────────────────────────────

    #[test]
    fn test_link_status_dl_active_bit() {
        assert_eq!(LINK_STATUS_DL_LINK_ACTIVE, 1 << 29);
        // Verify the check logic
        let status_active: u32 = 0x2000_0000; // bit 29 set
        assert!(status_active & LINK_STATUS_DL_LINK_ACTIVE != 0);

        let status_inactive: u32 = 0x0000_0000;
        assert!(status_inactive & LINK_STATUS_DL_LINK_ACTIVE == 0);
    }

    #[test]
    fn test_link_status_register_offset() {
        assert_eq!(RP_LINK_STATUS_CTRL, 0x090);
    }

    // ─── MMIO window constants ──────────────────────────────────────────

    #[test]
    fn test_pcie_mmio_window() {
        assert_eq!(PCIE_MMIO_BASE, 0x1200_0000);
        assert_eq!(PCIE_MMIO_SIZE, 0x2000_0000); // 512 MiB
    }

    #[test]
    fn test_pcie_mmio_window_no_overlap_with_rp() {
        // MMIO window should not overlap with the root port config space
        assert!(PCIE_MMIO_BASE as usize > RP_BASE + ECAM_OFFSET);
    }

    // ─── Vendor ID constants ────────────────────────────────────────────

    #[test]
    fn test_vendor_id_invalid() {
        assert_eq!(VENDOR_ID_INVALID, 0xFFFF);
    }

    #[test]
    fn test_max_pci_devices() {
        assert_eq!(MAX_PCI_DEVICES, 32);
    }

    #[test]
    fn test_max_devices_per_bus() {
        assert_eq!(MAX_DEVICES_PER_BUS, 32);
    }

    #[test]
    fn test_max_functions_per_device() {
        assert_eq!(MAX_FUNCTIONS_PER_DEVICE, 8);
    }
}
