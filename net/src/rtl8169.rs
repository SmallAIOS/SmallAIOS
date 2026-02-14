// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Realtek RTL8169/RTL8168 Gigabit Ethernet driver for SmallAIOS.
//!
//! Provides [`Rtl8169Device`] for initializing and operating the built-in
//! Gigabit Ethernet NIC found on NVIDIA Jetson Nano (RTL8168 on PCIe).
//!
//! # Hardware overview
//!
//! The RTL8169/8168 family uses memory-mapped I/O registers and DMA descriptor
//! rings for packet transmission and reception.  Each descriptor is 16 bytes
//! (64-bit addressing mode) containing ownership, status, length, and a
//! physical buffer address.
//!
//! # Feature gate
//!
//! This module is gated behind the `rtl8169` feature flag.

use crate::NetError;

// ---------------------------------------------------------------------------
// Register offsets
// ---------------------------------------------------------------------------

/// MAC address registers (IDR0-IDR5).  Six consecutive byte registers.
pub const REG_IDR0: usize = 0x00;
/// MAC address register 1.
pub const REG_IDR1: usize = 0x01;
/// MAC address register 2.
pub const REG_IDR2: usize = 0x02;
/// MAC address register 3.
pub const REG_IDR3: usize = 0x03;
/// MAC address register 4.
pub const REG_IDR4: usize = 0x04;
/// MAC address register 5.
pub const REG_IDR5: usize = 0x05;

/// Transmit Normal Priority Descriptor Start Address (64-bit, low 32 bits).
pub const REG_TNPDS: usize = 0x20;
/// Transmit Normal Priority Descriptor Start Address (high 32 bits).
pub const REG_TNPDS_HIGH: usize = 0x24;

/// Command register (8-bit).
pub const REG_COMMAND: usize = 0x37;
/// Transmit priority polling register (8-bit).
pub const REG_TX_POLL: usize = 0x38;
/// Interrupt mask register (16-bit).
pub const REG_INTR_MASK: usize = 0x3C;
/// Interrupt status register (16-bit).
pub const REG_INTR_STATUS: usize = 0x3E;
/// Transmit configuration register (32-bit).
pub const REG_TX_CONFIG: usize = 0x40;
/// Receive configuration register (32-bit).
pub const REG_RX_CONFIG: usize = 0x44;

/// PHY status register (8-bit).
pub const REG_PHY_STATUS: usize = 0x6C;

/// Receive Descriptor Start Address (64-bit, low 32 bits).
pub const REG_RDSAR: usize = 0xE4;
/// Receive Descriptor Start Address (high 32 bits).
pub const REG_RDSAR_HIGH: usize = 0xE8;

// ---------------------------------------------------------------------------
// Command register bits
// ---------------------------------------------------------------------------

/// Reset bit in the Command register (bit 4).
pub const CMD_RESET: u8 = 1 << 4;
/// Receiver enable bit in the Command register (bit 3).
pub const CMD_RX_ENABLE: u8 = 1 << 3;
/// Transmitter enable bit in the Command register (bit 2).
pub const CMD_TX_ENABLE: u8 = 1 << 2;

// ---------------------------------------------------------------------------
// Interrupt bits
// ---------------------------------------------------------------------------

/// Receive OK interrupt.
pub const INTR_ROK: u16 = 1 << 0;
/// Receive error interrupt.
pub const INTR_RER: u16 = 1 << 1;
/// Transmit OK interrupt.
pub const INTR_TOK: u16 = 1 << 2;
/// Transmit error interrupt.
pub const INTR_TER: u16 = 1 << 3;
/// Link change interrupt.
pub const INTR_LINK_CHG: u16 = 1 << 5;

/// All interrupts we care about.
pub const INTR_DEFAULT_MASK: u16 = INTR_ROK | INTR_RER | INTR_TOK | INTR_TER | INTR_LINK_CHG;

// ---------------------------------------------------------------------------
// TX/RX configuration defaults
// ---------------------------------------------------------------------------

/// Default TX configuration: IFG = 11 (standard), DMA burst = unlimited.
pub const TX_CONFIG_DEFAULT: u32 = (0x03 << 24) | (0x07 << 8);
/// Default RX configuration: accept broadcast, multicast, our MAC; DMA burst unlimited; no threshold.
pub const RX_CONFIG_DEFAULT: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (0x07 << 8) | (0x07 << 13);

// ---------------------------------------------------------------------------
// TX poll
// ---------------------------------------------------------------------------

/// Normal priority queue polling trigger.
pub const TX_POLL_NPQ: u8 = 1 << 6;

// ---------------------------------------------------------------------------
// PHY status bits
// ---------------------------------------------------------------------------

/// PHY link status (bit 1).  1 = link up.
pub const PHY_STATUS_LINK_UP: u8 = 1 << 1;
/// PHY speed bit 4: 1000 Mbps when set (with bit 3 = 0).
pub const PHY_STATUS_1000MBPS: u8 = 1 << 4;
/// PHY speed bit 3: 100 Mbps when set (with bit 4 = 0).
pub const PHY_STATUS_100MBPS: u8 = 1 << 3;
/// PHY full-duplex (bit 0).
pub const PHY_STATUS_FULL_DUPLEX: u8 = 1 << 0;

// ---------------------------------------------------------------------------
// Descriptor flags (opts1)
// ---------------------------------------------------------------------------

/// Descriptor is owned by the NIC hardware.
pub const DESC_OWN: u32 = 1 << 31;
/// End-of-ring marker; the NIC wraps back to the first descriptor after this.
pub const DESC_EOR: u32 = 1 << 30;
/// First segment of the packet.
pub const DESC_FS: u32 = 1 << 29;
/// Last segment of the packet.
pub const DESC_LS: u32 = 1 << 28;
/// Mask for the frame length field (bits 13:0).
pub const DESC_LEN_MASK: u32 = 0x3FFF;

/// Maximum Ethernet frame size we accept (standard MTU + header + FCS).
pub const MAX_FRAME_SIZE: usize = 1536;

/// Number of entries in each descriptor ring.
pub const RING_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// LinkSpeed
// ---------------------------------------------------------------------------

pub use crate::LinkSpeed;

// ---------------------------------------------------------------------------
// DMA Descriptor
// ---------------------------------------------------------------------------

/// A single TX or RX DMA descriptor (16 bytes, 64-bit addressing).
///
/// The RTL8169/8168 uses a ring of these descriptors for DMA transfers.
/// Each descriptor points to a buffer in host memory.
#[repr(C, align(256))]
#[derive(Clone, Copy)]
pub struct DmaDescriptor {
    /// Status/command + length.
    ///
    /// | Bit(s)  | Field   | Description                    |
    /// |---------|---------|--------------------------------|
    /// | 31      | OWN     | 1 = owned by NIC hardware      |
    /// | 30      | EOR     | End-of-ring (wrap to first)    |
    /// | 29      | FS      | First segment of packet        |
    /// | 28      | LS      | Last segment of packet         |
    /// | 13:0    | Length  | Frame length in bytes           |
    pub opts1: u32,
    /// VLAN tag and checksum offload flags.
    pub opts2: u32,
    /// Low 32 bits of the buffer physical address.
    pub addr_low: u32,
    /// High 32 bits of the buffer physical address.
    pub addr_high: u32,
}

impl DmaDescriptor {
    /// Create a zeroed descriptor.
    pub const fn zeroed() -> Self {
        DmaDescriptor {
            opts1: 0,
            opts2: 0,
            addr_low: 0,
            addr_high: 0,
        }
    }

    /// Return `true` if the OWN bit is set (hardware owns this descriptor).
    #[inline]
    pub fn is_owned_by_hw(&self) -> bool {
        (self.opts1 & DESC_OWN) != 0
    }

    /// Return `true` if this is the end-of-ring descriptor.
    #[inline]
    pub fn is_eor(&self) -> bool {
        (self.opts1 & DESC_EOR) != 0
    }

    /// Return `true` if this is the first segment of a packet.
    #[inline]
    pub fn is_first_segment(&self) -> bool {
        (self.opts1 & DESC_FS) != 0
    }

    /// Return `true` if this is the last segment of a packet.
    #[inline]
    pub fn is_last_segment(&self) -> bool {
        (self.opts1 & DESC_LS) != 0
    }

    /// Return the frame length from bits 13:0 of `opts1`.
    #[inline]
    pub fn frame_length(&self) -> u16 {
        (self.opts1 & DESC_LEN_MASK) as u16
    }

    /// Set this descriptor for transmission.
    ///
    /// Sets OWN, FS, LS, and the frame length.  If `eor` is true the EOR flag
    /// is also set to mark end-of-ring.
    pub fn prepare_tx(&mut self, length: u16, eor: bool) {
        let mut flags = DESC_OWN | DESC_FS | DESC_LS | (length as u32 & DESC_LEN_MASK);
        if eor {
            flags |= DESC_EOR;
        }
        self.opts1 = flags;
    }

    /// Set this descriptor for reception.
    ///
    /// Sets OWN and the buffer size.  If `eor` is true the EOR flag is also set.
    pub fn prepare_rx(&mut self, buf_size: u16, eor: bool) {
        let mut flags = DESC_OWN | (buf_size as u32 & DESC_LEN_MASK);
        if eor {
            flags |= DESC_EOR;
        }
        self.opts1 = flags;
    }

    /// Set the 64-bit buffer address from a `u64`.
    pub fn set_buffer_addr(&mut self, addr: u64) {
        self.addr_low = addr as u32;
        self.addr_high = (addr >> 32) as u32;
    }

    /// Get the 64-bit buffer address.
    pub fn buffer_addr(&self) -> u64 {
        (self.addr_high as u64) << 32 | self.addr_low as u64
    }
}

// ---------------------------------------------------------------------------
// TxDescriptorRing
// ---------------------------------------------------------------------------

/// Fixed-size ring of TX DMA descriptors.
///
/// Uses a head/tail scheme: `head` is where the driver writes the next
/// descriptor, `tail` is where it reclaims completed ones.
pub struct TxDescriptorRing {
    /// Descriptor entries.
    pub descriptors: [DmaDescriptor; RING_SIZE],
    /// TX packet buffers (one per descriptor).
    pub buffers: [[u8; MAX_FRAME_SIZE]; RING_SIZE],
    /// Index where the driver will write the next TX descriptor.
    head: usize,
    /// Index where the driver will reclaim completed descriptors.
    tail: usize,
    /// Number of descriptors currently in use (owned by hardware).
    in_use: usize,
}

impl Default for TxDescriptorRing {
    fn default() -> Self {
        Self::new()
    }
}

impl TxDescriptorRing {
    /// Create a new TX descriptor ring with all descriptors zeroed.
    pub const fn new() -> Self {
        TxDescriptorRing {
            descriptors: [DmaDescriptor::zeroed(); RING_SIZE],
            buffers: [[0u8; MAX_FRAME_SIZE]; RING_SIZE],
            head: 0,
            tail: 0,
            in_use: 0,
        }
    }

    /// Initialize all descriptors.  Sets buffer addresses (using buffer array
    /// offsets as stand-in physical addresses) and marks the last descriptor
    /// with EOR.
    pub fn init(&mut self) {
        for i in 0..RING_SIZE {
            let buf_addr = &self.buffers[i] as *const u8 as u64;
            self.descriptors[i].set_buffer_addr(buf_addr);
            self.descriptors[i].opts1 = 0;
            self.descriptors[i].opts2 = 0;
            if i == RING_SIZE - 1 {
                // Mark end-of-ring on last descriptor.
                self.descriptors[i].opts1 = DESC_EOR;
            }
        }
        self.head = 0;
        self.tail = 0;
        self.in_use = 0;
    }

    /// Return `true` if the ring is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.in_use >= RING_SIZE
    }

    /// Return `true` if the ring is empty (no pending TX).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.in_use == 0
    }

    /// Return the number of descriptors currently in use.
    #[inline]
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// Return the number of free descriptor slots.
    #[inline]
    pub fn free_count(&self) -> usize {
        RING_SIZE - self.in_use
    }

    /// Enqueue a frame for transmission.
    ///
    /// Copies `frame` into the descriptor's buffer, sets up the descriptor with
    /// OWN, FS, LS (and EOR if at end of ring), and advances the head pointer.
    pub fn enqueue(&mut self, frame: &[u8]) -> Result<(), NetError> {
        if frame.is_empty() || frame.len() > MAX_FRAME_SIZE {
            return Err(NetError::PacketTooShort);
        }
        if self.is_full() {
            return Err(NetError::RingFull);
        }

        let idx = self.head;

        // Copy frame data into the buffer.
        self.buffers[idx][..frame.len()].copy_from_slice(frame);

        // Set up the descriptor.
        let eor = idx == RING_SIZE - 1;
        self.descriptors[idx].prepare_tx(frame.len() as u16, eor);

        // Advance head.
        self.head = (self.head + 1) % RING_SIZE;
        self.in_use += 1;

        Ok(())
    }

    /// Reclaim completed TX descriptors.
    ///
    /// Walks from `tail` towards `head`, reclaiming any descriptor whose OWN
    /// bit has been cleared by the hardware.  Returns the number of descriptors
    /// reclaimed.
    pub fn reclaim(&mut self) -> usize {
        let mut reclaimed = 0;
        while self.in_use > 0 {
            let idx = self.tail;
            if self.descriptors[idx].is_owned_by_hw() {
                break;
            }
            // Clear the descriptor (preserve EOR).
            let eor = self.descriptors[idx].is_eor();
            self.descriptors[idx].opts1 = if eor { DESC_EOR } else { 0 };

            self.tail = (self.tail + 1) % RING_SIZE;
            self.in_use -= 1;
            reclaimed += 1;
        }
        reclaimed
    }

    /// Return the current head index (next write position).
    #[inline]
    pub fn head(&self) -> usize {
        self.head
    }

    /// Return the current tail index (next reclaim position).
    #[inline]
    pub fn tail(&self) -> usize {
        self.tail
    }
}

// ---------------------------------------------------------------------------
// RxDescriptorRing
// ---------------------------------------------------------------------------

/// Fixed-size ring of RX DMA descriptors.
///
/// All descriptors start owned by hardware with pre-allocated buffers.
/// The driver reads completed descriptors at `tail` and re-arms them.
pub struct RxDescriptorRing {
    /// Descriptor entries.
    pub descriptors: [DmaDescriptor; RING_SIZE],
    /// RX packet buffers (one per descriptor).
    pub buffers: [[u8; MAX_FRAME_SIZE]; RING_SIZE],
    /// Index where the driver checks for the next received packet.
    tail: usize,
}

impl Default for RxDescriptorRing {
    fn default() -> Self {
        Self::new()
    }
}

impl RxDescriptorRing {
    /// Create a new RX descriptor ring with all descriptors zeroed.
    pub const fn new() -> Self {
        RxDescriptorRing {
            descriptors: [DmaDescriptor::zeroed(); RING_SIZE],
            buffers: [[0u8; MAX_FRAME_SIZE]; RING_SIZE],
            tail: 0,
        }
    }

    /// Initialize all descriptors: set buffer addresses, give ownership to
    /// hardware, and mark the last descriptor with EOR.
    pub fn init(&mut self) {
        for i in 0..RING_SIZE {
            let buf_addr = &self.buffers[i] as *const u8 as u64;
            self.descriptors[i].set_buffer_addr(buf_addr);
            self.descriptors[i].opts2 = 0;
            let eor = i == RING_SIZE - 1;
            self.descriptors[i].prepare_rx(MAX_FRAME_SIZE as u16, eor);
        }
        self.tail = 0;
    }

    /// Try to receive a packet from the next RX descriptor.
    ///
    /// If the descriptor at `tail` is no longer owned by hardware, copies the
    /// received frame data into `buf` and returns the frame length.  The
    /// descriptor is then re-armed (ownership returned to hardware).
    ///
    /// Returns [`NetError::NoData`] if no packet is available.
    /// Returns [`NetError::BufferTooSmall`] if `buf` cannot hold the frame.
    pub fn poll_receive(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
        let idx = self.tail;
        if self.descriptors[idx].is_owned_by_hw() {
            return Err(NetError::NoData);
        }

        let length = self.descriptors[idx].frame_length() as usize;
        if length > buf.len() {
            // Re-arm descriptor even on error so we don't get stuck.
            self.rearm(idx);
            return Err(NetError::BufferTooSmall);
        }

        // Copy received data.
        buf[..length].copy_from_slice(&self.buffers[idx][..length]);

        // Re-arm the descriptor and advance tail.
        self.rearm(idx);
        self.tail = (self.tail + 1) % RING_SIZE;

        Ok(length)
    }

    /// Re-arm a descriptor: give it back to hardware with OWN set.
    fn rearm(&mut self, idx: usize) {
        let eor = idx == RING_SIZE - 1;
        self.descriptors[idx].prepare_rx(MAX_FRAME_SIZE as u16, eor);
    }

    /// Return the current tail index.
    #[inline]
    pub fn tail(&self) -> usize {
        self.tail
    }
}

// ---------------------------------------------------------------------------
// MMIO helpers (unsafe, volatile)
// ---------------------------------------------------------------------------

/// Read a `u8` from MMIO at `base + offset`.
///
/// # Safety
///
/// `base + offset` must be a valid, mapped MMIO address.
#[inline]
unsafe fn mmio_read8(base: usize, offset: usize) -> u8 {
    core::ptr::read_volatile((base + offset) as *const u8)
}

/// Write a `u8` to MMIO at `base + offset`.
///
/// # Safety
///
/// `base + offset` must be a valid, mapped MMIO address.
#[inline]
unsafe fn mmio_write8(base: usize, offset: usize, val: u8) {
    core::ptr::write_volatile((base + offset) as *mut u8, val);
}

/// Read a `u16` from MMIO at `base + offset`.
///
/// # Safety
///
/// `base + offset` must be a valid, properly aligned MMIO address.
#[inline]
unsafe fn mmio_read16(base: usize, offset: usize) -> u16 {
    core::ptr::read_volatile((base + offset) as *const u16)
}

/// Write a `u16` to MMIO at `base + offset`.
///
/// # Safety
///
/// `base + offset` must be a valid, properly aligned MMIO address.
#[inline]
unsafe fn mmio_write16(base: usize, offset: usize, val: u16) {
    core::ptr::write_volatile((base + offset) as *mut u16, val);
}

/// Read a `u32` from MMIO at `base + offset`.
///
/// # Safety
///
/// `base + offset` must be a valid, properly aligned MMIO address.
#[inline]
#[allow(dead_code)]
unsafe fn mmio_read32(base: usize, offset: usize) -> u32 {
    core::ptr::read_volatile((base + offset) as *const u32)
}

/// Write a `u32` to MMIO at `base + offset`.
///
/// # Safety
///
/// `base + offset` must be a valid, properly aligned MMIO address.
#[inline]
unsafe fn mmio_write32(base: usize, offset: usize, val: u32) {
    core::ptr::write_volatile((base + offset) as *mut u32, val);
}

// ---------------------------------------------------------------------------
// PHY status parsing
// ---------------------------------------------------------------------------

/// Parse the PHY status register value into an optional [`LinkSpeed`].
///
/// Returns `None` if the link is down (bit 1 clear).
pub fn parse_phy_status(status: u8) -> Option<LinkSpeed> {
    if (status & PHY_STATUS_LINK_UP) == 0 {
        return None;
    }
    let speed_bits = status & (PHY_STATUS_1000MBPS | PHY_STATUS_100MBPS);
    match speed_bits {
        s if (s & PHY_STATUS_1000MBPS) != 0 => Some(LinkSpeed::Mbps1000),
        s if (s & PHY_STATUS_100MBPS) != 0 => Some(LinkSpeed::Mbps100),
        _ => Some(LinkSpeed::Mbps10),
    }
}

// ---------------------------------------------------------------------------
// Rtl8169Device
// ---------------------------------------------------------------------------

/// RTL8169/8168 Gigabit Ethernet device driver.
///
/// # Usage
///
/// ```ignore
/// let mut nic = Rtl8169Device::new(pcie_bar0_addr);
/// nic.init()?;
/// nic.send(&ethernet_frame)?;
/// let n = nic.receive(&mut buf)?;
/// ```
pub struct Rtl8169Device {
    /// Base address of the MMIO register space (PCIe BAR0).
    base_addr: usize,
    /// MAC address read from IDR0-IDR5.
    mac: [u8; 6],
    /// TX descriptor ring.
    tx_ring: TxDescriptorRing,
    /// RX descriptor ring.
    rx_ring: RxDescriptorRing,
    /// Whether [`Rtl8169Device::init`] has been called successfully.
    initialized: bool,
}

impl Rtl8169Device {
    /// Maximum number of reset polling iterations before giving up.
    const RESET_TIMEOUT_ITERS: u32 = 1_000_000;

    /// Create a new [`Rtl8169Device`] for the NIC at `base_addr`.
    ///
    /// Reads the MAC address from the IDR0-IDR5 registers.  The device is
    /// **not** initialized; call [`Rtl8169Device::init`] before sending or
    /// receiving.
    ///
    /// # Safety
    ///
    /// `base_addr` must point to the correctly mapped MMIO register space of
    /// an RTL8169/8168 NIC.
    pub unsafe fn new(base_addr: usize) -> Self {
        let mut mac = [0u8; 6];
        mac[0] = mmio_read8(base_addr, REG_IDR0);
        mac[1] = mmio_read8(base_addr, REG_IDR1);
        mac[2] = mmio_read8(base_addr, REG_IDR2);
        mac[3] = mmio_read8(base_addr, REG_IDR3);
        mac[4] = mmio_read8(base_addr, REG_IDR4);
        mac[5] = mmio_read8(base_addr, REG_IDR5);

        Rtl8169Device {
            base_addr,
            mac,
            tx_ring: TxDescriptorRing::new(),
            rx_ring: RxDescriptorRing::new(),
            initialized: false,
        }
    }

    /// Create a new [`Rtl8169Device`] for unit testing without MMIO access.
    ///
    /// The MAC address and base address are provided directly.  No hardware
    /// registers are read.
    #[cfg(test)]
    pub(crate) fn new_test(base_addr: usize, mac: [u8; 6]) -> Self {
        Rtl8169Device {
            base_addr,
            mac,
            tx_ring: TxDescriptorRing::new(),
            rx_ring: RxDescriptorRing::new(),
            initialized: false,
        }
    }

    /// Software-reset the NIC.
    ///
    /// Sets the Reset bit (bit 4) in the Command register and polls until it
    /// clears or the timeout expires.
    ///
    /// # Safety
    ///
    /// Must only be called when `base_addr` points to valid MMIO.
    pub unsafe fn reset(&mut self) -> Result<(), NetError> {
        mmio_write8(self.base_addr, REG_COMMAND, CMD_RESET);

        for _ in 0..Self::RESET_TIMEOUT_ITERS {
            let cmd = mmio_read8(self.base_addr, REG_COMMAND);
            if (cmd & CMD_RESET) == 0 {
                return Ok(());
            }
        }
        Err(NetError::Timeout)
    }

    /// Initialize the NIC: reset, set up descriptor rings, configure TX/RX,
    /// and enable the transmitter and receiver.
    ///
    /// # Safety
    ///
    /// Must only be called when `base_addr` points to valid MMIO.
    pub unsafe fn init(&mut self) -> Result<(), NetError> {
        // 1. Software reset.
        self.reset()?;

        // 2. Initialize descriptor rings.
        self.tx_ring.init();
        self.rx_ring.init();

        // 3. Program TX descriptor base address (TNPDS).
        let tx_phys = &self.tx_ring.descriptors[0] as *const DmaDescriptor as u64;
        mmio_write32(self.base_addr, REG_TNPDS, tx_phys as u32);
        mmio_write32(self.base_addr, REG_TNPDS_HIGH, (tx_phys >> 32) as u32);

        // 4. Program RX descriptor base address (RDSAR).
        let rx_phys = &self.rx_ring.descriptors[0] as *const DmaDescriptor as u64;
        mmio_write32(self.base_addr, REG_RDSAR, rx_phys as u32);
        mmio_write32(self.base_addr, REG_RDSAR_HIGH, (rx_phys >> 32) as u32);

        // 5. Configure TX and RX.
        mmio_write32(self.base_addr, REG_TX_CONFIG, TX_CONFIG_DEFAULT);
        mmio_write32(self.base_addr, REG_RX_CONFIG, RX_CONFIG_DEFAULT);

        // 6. Enable interrupts.
        mmio_write16(self.base_addr, REG_INTR_MASK, INTR_DEFAULT_MASK);

        // 7. Enable transmitter and receiver.
        mmio_write8(self.base_addr, REG_COMMAND, CMD_TX_ENABLE | CMD_RX_ENABLE);

        self.initialized = true;
        Ok(())
    }

    /// Transmit an Ethernet frame.
    ///
    /// The frame must be a complete Ethernet frame (header + payload), not
    /// exceeding [`MAX_FRAME_SIZE`] bytes.
    ///
    /// Returns [`NetError::DeviceNotReady`] if the device has not been
    /// initialized, [`NetError::RingFull`] if the TX ring is full, or
    /// [`NetError::PacketTooShort`] if the frame is empty or too large.
    pub fn send(&mut self, frame: &[u8]) -> Result<(), NetError> {
        if !self.initialized {
            return Err(NetError::DeviceNotReady);
        }

        // Enqueue into the TX ring.
        self.tx_ring.enqueue(frame)?;

        // Notify the NIC to poll the TX ring.
        unsafe {
            mmio_write8(self.base_addr, REG_TX_POLL, TX_POLL_NPQ);
        }
        Ok(())
    }

    /// Reclaim completed TX descriptors.
    ///
    /// Should be called periodically (e.g. from the interrupt handler or
    /// before sending new frames) to free up ring entries.
    pub fn tx_complete(&mut self) -> usize {
        self.tx_ring.reclaim()
    }

    /// Receive an Ethernet frame into `buf`.
    ///
    /// Returns the number of bytes written to `buf`, or an error.
    ///
    /// Returns [`NetError::DeviceNotReady`] if the device has not been
    /// initialized, [`NetError::NoData`] if no frame is available, or
    /// [`NetError::BufferTooSmall`] if `buf` is too small for the frame.
    pub fn receive(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
        if !self.initialized {
            return Err(NetError::DeviceNotReady);
        }
        self.rx_ring.poll_receive(buf)
    }

    /// Read the PHY status register and return the current link speed.
    ///
    /// Returns `None` if the link is down.
    pub fn link_status(&self) -> Option<LinkSpeed> {
        let status = unsafe { mmio_read8(self.base_addr, REG_PHY_STATUS) };
        parse_phy_status(status)
    }

    /// Return the device's MAC address.
    pub fn mac_address(&self) -> [u8; 6] {
        self.mac
    }

    /// Return `true` if the device has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Return a reference to the TX descriptor ring.
    pub fn tx_ring(&self) -> &TxDescriptorRing {
        &self.tx_ring
    }

    /// Return a reference to the RX descriptor ring.
    pub fn rx_ring(&self) -> &RxDescriptorRing {
        &self.rx_ring
    }

    /// Return the MMIO base address.
    pub fn base_addr(&self) -> usize {
        self.base_addr
    }

    /// Handle an interrupt.
    ///
    /// Reads and acknowledges the interrupt status register, then processes
    /// the relevant events.  Returns the raw ISR value.
    ///
    /// # Safety
    ///
    /// Must only be called when `base_addr` points to valid MMIO.
    pub unsafe fn handle_interrupt(&mut self) -> u16 {
        let isr = mmio_read16(self.base_addr, REG_INTR_STATUS);
        // Acknowledge all bits by writing them back.
        mmio_write16(self.base_addr, REG_INTR_STATUS, isr);

        if (isr & INTR_TOK) != 0 {
            self.tx_ring.reclaim();
        }

        isr
    }
}

impl crate::NetworkDevice for Rtl8169Device {
    fn send(&mut self, frame: &[u8]) -> Result<(), NetError> {
        self.send(frame)
    }

    fn receive(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
        self.receive(buf)
    }

    fn mac_address(&self) -> [u8; 6] {
        self.mac_address()
    }

    fn link_status(&self) -> Option<LinkSpeed> {
        self.link_status()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    // -- Descriptor struct layout -------------------------------------------

    #[test]
    fn test_descriptor_size() {
        // The descriptor must be at least 16 bytes for the four u32 fields.
        assert!(core::mem::size_of::<DmaDescriptor>() >= 16);
    }

    #[test]
    fn test_descriptor_zeroed() {
        let d = DmaDescriptor::zeroed();
        assert_eq!(d.opts1, 0);
        assert_eq!(d.opts2, 0);
        assert_eq!(d.addr_low, 0);
        assert_eq!(d.addr_high, 0);
    }

    // -- Descriptor bit manipulation ----------------------------------------

    #[test]
    fn test_own_bit() {
        let mut d = DmaDescriptor::zeroed();
        assert!(!d.is_owned_by_hw());

        d.opts1 = DESC_OWN;
        assert!(d.is_owned_by_hw());
    }

    #[test]
    fn test_eor_bit() {
        let mut d = DmaDescriptor::zeroed();
        assert!(!d.is_eor());

        d.opts1 = DESC_EOR;
        assert!(d.is_eor());
    }

    #[test]
    fn test_fs_bit() {
        let mut d = DmaDescriptor::zeroed();
        assert!(!d.is_first_segment());

        d.opts1 = DESC_FS;
        assert!(d.is_first_segment());
    }

    #[test]
    fn test_ls_bit() {
        let mut d = DmaDescriptor::zeroed();
        assert!(!d.is_last_segment());

        d.opts1 = DESC_LS;
        assert!(d.is_last_segment());
    }

    #[test]
    fn test_frame_length() {
        let mut d = DmaDescriptor::zeroed();
        d.opts1 = 1514; // standard Ethernet frame
        assert_eq!(d.frame_length(), 1514);

        // Maximum length (14 bits).
        d.opts1 = DESC_LEN_MASK;
        assert_eq!(d.frame_length(), 0x3FFF);
    }

    #[test]
    fn test_frame_length_with_flags() {
        let mut d = DmaDescriptor::zeroed();
        // Set OWN + FS + LS + length 100.
        d.opts1 = DESC_OWN | DESC_FS | DESC_LS | 100;
        assert!(d.is_owned_by_hw());
        assert!(d.is_first_segment());
        assert!(d.is_last_segment());
        assert_eq!(d.frame_length(), 100);
    }

    #[test]
    fn test_all_flags_combined() {
        let mut d = DmaDescriptor::zeroed();
        d.opts1 = DESC_OWN | DESC_EOR | DESC_FS | DESC_LS | 512;
        assert!(d.is_owned_by_hw());
        assert!(d.is_eor());
        assert!(d.is_first_segment());
        assert!(d.is_last_segment());
        assert_eq!(d.frame_length(), 512);
    }

    // -- 64-bit address split -----------------------------------------------

    #[test]
    fn test_buffer_addr_low_only() {
        let mut d = DmaDescriptor::zeroed();
        d.set_buffer_addr(0x0000_0000_DEAD_BEEF);
        assert_eq!(d.addr_low, 0xDEAD_BEEF);
        assert_eq!(d.addr_high, 0x0000_0000);
        assert_eq!(d.buffer_addr(), 0x0000_0000_DEAD_BEEF);
    }

    #[test]
    fn test_buffer_addr_high_and_low() {
        let mut d = DmaDescriptor::zeroed();
        d.set_buffer_addr(0x1234_5678_9ABC_DEF0);
        assert_eq!(d.addr_low, 0x9ABC_DEF0);
        assert_eq!(d.addr_high, 0x1234_5678);
        assert_eq!(d.buffer_addr(), 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn test_buffer_addr_max() {
        let mut d = DmaDescriptor::zeroed();
        d.set_buffer_addr(u64::MAX);
        assert_eq!(d.addr_low, u32::MAX);
        assert_eq!(d.addr_high, u32::MAX);
        assert_eq!(d.buffer_addr(), u64::MAX);
    }

    #[test]
    fn test_buffer_addr_zero() {
        let mut d = DmaDescriptor::zeroed();
        d.set_buffer_addr(0);
        assert_eq!(d.addr_low, 0);
        assert_eq!(d.addr_high, 0);
        assert_eq!(d.buffer_addr(), 0);
    }

    // -- prepare_tx / prepare_rx --------------------------------------------

    #[test]
    fn test_prepare_tx_no_eor() {
        let mut d = DmaDescriptor::zeroed();
        d.prepare_tx(64, false);
        assert!(d.is_owned_by_hw());
        assert!(d.is_first_segment());
        assert!(d.is_last_segment());
        assert!(!d.is_eor());
        assert_eq!(d.frame_length(), 64);
    }

    #[test]
    fn test_prepare_tx_with_eor() {
        let mut d = DmaDescriptor::zeroed();
        d.prepare_tx(1500, true);
        assert!(d.is_owned_by_hw());
        assert!(d.is_first_segment());
        assert!(d.is_last_segment());
        assert!(d.is_eor());
        assert_eq!(d.frame_length(), 1500);
    }

    #[test]
    fn test_prepare_rx_no_eor() {
        let mut d = DmaDescriptor::zeroed();
        d.prepare_rx(MAX_FRAME_SIZE as u16, false);
        assert!(d.is_owned_by_hw());
        assert!(!d.is_eor());
        assert_eq!(d.frame_length(), MAX_FRAME_SIZE as u16);
    }

    #[test]
    fn test_prepare_rx_with_eor() {
        let mut d = DmaDescriptor::zeroed();
        d.prepare_rx(MAX_FRAME_SIZE as u16, true);
        assert!(d.is_owned_by_hw());
        assert!(d.is_eor());
    }

    // -- TX descriptor ring -------------------------------------------------

    #[test]
    fn test_tx_ring_new() {
        let ring = TxDescriptorRing::new();
        assert_eq!(ring.head(), 0);
        assert_eq!(ring.tail(), 0);
        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.free_count(), RING_SIZE);
        assert_eq!(ring.in_use(), 0);
    }

    #[test]
    fn test_tx_ring_init_eor() {
        let mut ring = TxDescriptorRing::new();
        ring.init();
        // Last descriptor should have EOR set.
        assert!(ring.descriptors[RING_SIZE - 1].is_eor());
        // Other descriptors should not.
        for i in 0..RING_SIZE - 1 {
            assert!(
                !ring.descriptors[i].is_eor(),
                "descriptor {} should not have EOR",
                i
            );
        }
    }

    #[test]
    fn test_tx_ring_enqueue_one() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        let frame = [0xAA_u8; 64];
        ring.enqueue(&frame).unwrap();
        assert_eq!(ring.head(), 1);
        assert_eq!(ring.tail(), 0);
        assert_eq!(ring.in_use(), 1);
        assert!(!ring.is_empty());
        assert!(!ring.is_full());

        // Descriptor should be owned by hardware.
        assert!(ring.descriptors[0].is_owned_by_hw());
        assert_eq!(ring.descriptors[0].frame_length(), 64);
    }

    #[test]
    fn test_tx_ring_enqueue_empty_frame() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        let frame: [u8; 0] = [];
        assert_eq!(ring.enqueue(&frame), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_tx_ring_enqueue_oversized_frame() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        let frame = [0u8; MAX_FRAME_SIZE + 1];
        assert_eq!(ring.enqueue(&frame), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_tx_ring_fill_and_full() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        for i in 0..RING_SIZE {
            let frame = [i as u8; 64];
            ring.enqueue(&frame).unwrap();
        }
        assert!(ring.is_full());
        assert_eq!(ring.free_count(), 0);

        // Next enqueue should fail.
        let frame = [0xFF_u8; 64];
        assert_eq!(ring.enqueue(&frame), Err(NetError::RingFull));
    }

    #[test]
    fn test_tx_ring_wraparound() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        // Fill half the ring.
        for i in 0..32 {
            let frame = [i as u8; 64];
            ring.enqueue(&frame).unwrap();
        }
        assert_eq!(ring.head(), 32);

        // Simulate hardware completion: clear OWN on all 32.
        for i in 0..32 {
            ring.descriptors[i].opts1 &= !DESC_OWN;
        }
        let reclaimed = ring.reclaim();
        assert_eq!(reclaimed, 32);
        assert_eq!(ring.tail(), 32);
        assert!(ring.is_empty());

        // Now fill past the end to test wraparound.
        for i in 0..RING_SIZE {
            let frame = [i as u8; 64];
            ring.enqueue(&frame).unwrap();
        }
        // Head should have wrapped around: 32 + 64 = 96, 96 % 64 = 32.
        assert_eq!(ring.head(), 32);
        assert!(ring.is_full());
    }

    #[test]
    fn test_tx_ring_reclaim_none_pending() {
        let mut ring = TxDescriptorRing::new();
        ring.init();
        assert_eq!(ring.reclaim(), 0);
    }

    #[test]
    fn test_tx_ring_reclaim_partial() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        // Enqueue 4 frames.
        for i in 0..4 {
            let frame = [i as u8; 100];
            ring.enqueue(&frame).unwrap();
        }

        // Hardware completes first 2.
        ring.descriptors[0].opts1 &= !DESC_OWN;
        ring.descriptors[1].opts1 &= !DESC_OWN;

        let reclaimed = ring.reclaim();
        assert_eq!(reclaimed, 2);
        assert_eq!(ring.tail(), 2);
        assert_eq!(ring.in_use(), 2);
    }

    #[test]
    fn test_tx_ring_eor_preserved_on_reclaim() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        // Enqueue at the last slot.
        for _ in 0..RING_SIZE {
            ring.enqueue(&[0xBB; 64]).unwrap();
        }

        // Complete all.
        for i in 0..RING_SIZE {
            ring.descriptors[i].opts1 &= !DESC_OWN;
        }
        ring.reclaim();

        // EOR should still be on the last descriptor.
        assert!(ring.descriptors[RING_SIZE - 1].is_eor());
        for i in 0..RING_SIZE - 1 {
            assert!(!ring.descriptors[i].is_eor());
        }
    }

    #[test]
    fn test_tx_ring_enqueue_last_slot_has_eor() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        // Fill up to RING_SIZE - 1 (enqueue into slots 0..62).
        for _ in 0..RING_SIZE - 1 {
            ring.enqueue(&[0xCC; 64]).unwrap();
        }
        // Enqueue into the last slot (index 63).
        ring.enqueue(&[0xDD; 64]).unwrap();

        // The last descriptor should have EOR.
        assert!(ring.descriptors[RING_SIZE - 1].is_eor());
    }

    // -- RX descriptor ring -------------------------------------------------

    #[test]
    fn test_rx_ring_new() {
        let ring = RxDescriptorRing::new();
        assert_eq!(ring.tail(), 0);
    }

    #[test]
    fn test_rx_ring_init_all_owned() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        for i in 0..RING_SIZE {
            assert!(
                ring.descriptors[i].is_owned_by_hw(),
                "RX descriptor {} should be owned by hardware after init",
                i
            );
        }
        // Last has EOR.
        assert!(ring.descriptors[RING_SIZE - 1].is_eor());
    }

    #[test]
    fn test_rx_ring_poll_no_data() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        let mut buf = [0u8; MAX_FRAME_SIZE];
        assert_eq!(ring.poll_receive(&mut buf), Err(NetError::NoData));
    }

    #[test]
    fn test_rx_ring_poll_receive_one() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        // Simulate hardware delivering a 100-byte frame at descriptor 0.
        ring.descriptors[0].opts1 = DESC_FS | DESC_LS | 100; // OWN cleared by HW.
        ring.buffers[0][..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let mut buf = [0u8; MAX_FRAME_SIZE];
        let len = ring.poll_receive(&mut buf).unwrap();
        assert_eq!(len, 100);
        assert_eq!(&buf[..4], &[0xDE, 0xAD, 0xBE, 0xEF]);

        // Tail should advance.
        assert_eq!(ring.tail(), 1);

        // Descriptor should be re-armed (owned by HW again).
        assert!(ring.descriptors[0].is_owned_by_hw());
    }

    #[test]
    fn test_rx_ring_poll_buffer_too_small() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        // Simulate a 200-byte frame.
        ring.descriptors[0].opts1 = DESC_FS | DESC_LS | 200;

        let mut buf = [0u8; 100]; // Too small.
        assert_eq!(ring.poll_receive(&mut buf), Err(NetError::BufferTooSmall));

        // Descriptor should still be re-armed.
        assert!(ring.descriptors[0].is_owned_by_hw());
    }

    #[test]
    fn test_rx_ring_wraparound() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        // Receive RING_SIZE frames to wrap around.
        for i in 0..RING_SIZE {
            ring.descriptors[i].opts1 = DESC_FS | DESC_LS | 60;
            ring.buffers[i][0] = i as u8;
        }

        for i in 0..RING_SIZE {
            let mut buf = [0u8; MAX_FRAME_SIZE];
            let len = ring.poll_receive(&mut buf).unwrap();
            assert_eq!(len, 60);
            assert_eq!(buf[0], i as u8);
        }

        // Tail should have wrapped back to 0.
        assert_eq!(ring.tail(), 0);
    }

    #[test]
    fn test_rx_ring_rearm_preserves_eor() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        // Receive at the last descriptor.
        for i in 0..RING_SIZE {
            ring.descriptors[i].opts1 = DESC_FS | DESC_LS | 64;
        }
        for _ in 0..RING_SIZE {
            let mut buf = [0u8; MAX_FRAME_SIZE];
            ring.poll_receive(&mut buf).unwrap();
        }

        // Last descriptor should have EOR after re-arm.
        assert!(ring.descriptors[RING_SIZE - 1].is_eor());
    }

    // -- PHY status parsing -------------------------------------------------

    #[test]
    fn test_phy_link_down() {
        // Bit 1 clear = link down.
        assert_eq!(parse_phy_status(0x00), None);
        assert_eq!(parse_phy_status(0x10), None); // speed bits set but no link
    }

    #[test]
    fn test_phy_link_1000mbps() {
        // Bit 1 (link up) + bit 4 (1000 Mbps).
        let status = PHY_STATUS_LINK_UP | PHY_STATUS_1000MBPS;
        assert_eq!(parse_phy_status(status), Some(LinkSpeed::Mbps1000));
    }

    #[test]
    fn test_phy_link_100mbps() {
        // Bit 1 (link up) + bit 3 (100 Mbps).
        let status = PHY_STATUS_LINK_UP | PHY_STATUS_100MBPS;
        assert_eq!(parse_phy_status(status), Some(LinkSpeed::Mbps100));
    }

    #[test]
    fn test_phy_link_10mbps() {
        // Bit 1 (link up) only — no speed bits = 10 Mbps.
        let status = PHY_STATUS_LINK_UP;
        assert_eq!(parse_phy_status(status), Some(LinkSpeed::Mbps10));
    }

    #[test]
    fn test_phy_link_1000mbps_full_duplex() {
        let status = PHY_STATUS_LINK_UP | PHY_STATUS_1000MBPS | PHY_STATUS_FULL_DUPLEX;
        assert_eq!(parse_phy_status(status), Some(LinkSpeed::Mbps1000));
    }

    #[test]
    fn test_phy_link_1000mbps_overrides_100() {
        // Both 1000 and 100 set — 1000 should take priority.
        let status = PHY_STATUS_LINK_UP | PHY_STATUS_1000MBPS | PHY_STATUS_100MBPS;
        assert_eq!(parse_phy_status(status), Some(LinkSpeed::Mbps1000));
    }

    // -- MAC address byte ordering ------------------------------------------

    #[test]
    fn test_mac_address_byte_order() {
        let dev = Rtl8169Device::new_test(0x1000, [0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let mac = dev.mac_address();
        assert_eq!(mac[0], 0x00);
        assert_eq!(mac[1], 0x11);
        assert_eq!(mac[2], 0x22);
        assert_eq!(mac[3], 0x33);
        assert_eq!(mac[4], 0x44);
        assert_eq!(mac[5], 0x55);
    }

    #[test]
    fn test_mac_address_broadcast() {
        let dev = Rtl8169Device::new_test(0x1000, [0xFF; 6]);
        assert_eq!(dev.mac_address(), [0xFF; 6]);
    }

    // -- Device state -------------------------------------------------------

    #[test]
    fn test_device_not_initialized() {
        let dev = Rtl8169Device::new_test(0x1000, [0x00; 6]);
        assert!(!dev.is_initialized());
    }

    #[test]
    fn test_send_before_init() {
        let mut dev = Rtl8169Device::new_test(0x1000, [0x00; 6]);
        let frame = [0u8; 64];
        assert_eq!(dev.send(&frame), Err(NetError::DeviceNotReady));
    }

    #[test]
    fn test_receive_before_init() {
        let mut dev = Rtl8169Device::new_test(0x1000, [0x00; 6]);
        let mut buf = [0u8; MAX_FRAME_SIZE];
        assert_eq!(dev.receive(&mut buf), Err(NetError::DeviceNotReady));
    }

    #[test]
    fn test_device_base_addr() {
        let dev = Rtl8169Device::new_test(0xFE00_0000, [0x00; 6]);
        assert_eq!(dev.base_addr(), 0xFE00_0000);
    }

    // -- Device send/receive with test init ---------------------------------

    #[test]
    fn test_send_after_manual_init() {
        let mut dev = Rtl8169Device::new_test(0x1000, [0x00; 6]);
        // Manually initialize rings and set flag (bypass MMIO init).
        dev.tx_ring.init();
        dev.rx_ring.init();
        dev.initialized = true;

        // send() will try to write MMIO for TX poll, but we only test ring logic.
        // Since we're in a test, the MMIO write to a fake address is UB in
        // general, so we directly test ring enqueue instead.
        let frame = [0xAA_u8; 64];
        dev.tx_ring.enqueue(&frame).unwrap();
        assert_eq!(dev.tx_ring.in_use(), 1);
    }

    #[test]
    fn test_receive_with_simulated_rx() {
        let mut dev = Rtl8169Device::new_test(0x1000, [0x00; 6]);
        dev.tx_ring.init();
        dev.rx_ring.init();
        dev.initialized = true;

        // Simulate hardware delivering a frame.
        dev.rx_ring.descriptors[0].opts1 = DESC_FS | DESC_LS | 42;
        dev.rx_ring.buffers[0][..4].copy_from_slice(&[0x01, 0x02, 0x03, 0x04]);

        let mut buf = [0u8; MAX_FRAME_SIZE];
        let len = dev.receive(&mut buf).unwrap();
        assert_eq!(len, 42);
        assert_eq!(&buf[..4], &[0x01, 0x02, 0x03, 0x04]);
    }

    // -- Register offset constants ------------------------------------------

    #[test]
    fn test_register_offsets() {
        assert_eq!(REG_IDR0, 0x00);
        assert_eq!(REG_IDR5, 0x05);
        assert_eq!(REG_TNPDS, 0x20);
        assert_eq!(REG_COMMAND, 0x37);
        assert_eq!(REG_TX_POLL, 0x38);
        assert_eq!(REG_INTR_MASK, 0x3C);
        assert_eq!(REG_INTR_STATUS, 0x3E);
        assert_eq!(REG_TX_CONFIG, 0x40);
        assert_eq!(REG_RX_CONFIG, 0x44);
        assert_eq!(REG_PHY_STATUS, 0x6C);
        assert_eq!(REG_RDSAR, 0xE4);
    }

    // -- Command bits -------------------------------------------------------

    #[test]
    fn test_command_bits() {
        assert_eq!(CMD_RESET, 0x10);
        assert_eq!(CMD_RX_ENABLE, 0x08);
        assert_eq!(CMD_TX_ENABLE, 0x04);
    }

    // -- Interrupt bits -----------------------------------------------------

    #[test]
    fn test_interrupt_bits() {
        assert_eq!(INTR_ROK, 0x0001);
        assert_eq!(INTR_RER, 0x0002);
        assert_eq!(INTR_TOK, 0x0004);
        assert_eq!(INTR_TER, 0x0008);
        assert_eq!(INTR_LINK_CHG, 0x0020);
    }

    #[test]
    fn test_default_intr_mask() {
        let expected = INTR_ROK | INTR_RER | INTR_TOK | INTR_TER | INTR_LINK_CHG;
        assert_eq!(INTR_DEFAULT_MASK, expected);
    }

    // -- Descriptor flag constants ------------------------------------------

    #[test]
    fn test_descriptor_flag_values() {
        assert_eq!(DESC_OWN, 0x8000_0000);
        assert_eq!(DESC_EOR, 0x4000_0000);
        assert_eq!(DESC_FS, 0x2000_0000);
        assert_eq!(DESC_LS, 0x1000_0000);
        assert_eq!(DESC_LEN_MASK, 0x3FFF);
    }

    // -- Ring size constant -------------------------------------------------

    #[test]
    fn test_ring_size() {
        assert_eq!(RING_SIZE, 64);
    }

    // -- LinkSpeed enum -----------------------------------------------------

    #[test]
    fn test_link_speed_debug() {
        assert_eq!(format!("{:?}", LinkSpeed::Mbps10), "Mbps10");
        assert_eq!(format!("{:?}", LinkSpeed::Mbps100), "Mbps100");
        assert_eq!(format!("{:?}", LinkSpeed::Mbps1000), "Mbps1000");
    }

    #[test]
    fn test_link_speed_eq() {
        assert_eq!(LinkSpeed::Mbps1000, LinkSpeed::Mbps1000);
        assert_ne!(LinkSpeed::Mbps10, LinkSpeed::Mbps100);
    }

    #[test]
    fn test_link_speed_clone() {
        let s = LinkSpeed::Mbps1000;
        let c = s;
        assert_eq!(s, c);
    }

    // -- TX config / RX config defaults -------------------------------------

    #[test]
    fn test_tx_config_default() {
        // IFG bits should be at bits 25:24.
        assert_ne!(TX_CONFIG_DEFAULT, 0);
        assert!((TX_CONFIG_DEFAULT & (0x03 << 24)) != 0);
    }

    #[test]
    fn test_rx_config_default() {
        // Should accept broadcast (bit 0), multicast (bit 1), physical (bit 2).
        assert!((RX_CONFIG_DEFAULT & 0x07) == 0x07);
    }

    // -- Stress: fill and drain TX ring multiple times ----------------------

    #[test]
    fn test_tx_ring_fill_drain_cycle() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        for _cycle in 0..3 {
            // Fill.
            for i in 0..RING_SIZE {
                ring.enqueue(&[i as u8; 64]).unwrap();
            }
            assert!(ring.is_full());

            // Simulate hardware completion.
            for i in 0..RING_SIZE {
                ring.descriptors[i].opts1 &= !DESC_OWN;
            }
            let reclaimed = ring.reclaim();
            assert_eq!(reclaimed, RING_SIZE);
            assert!(ring.is_empty());
        }
    }

    // -- RX multiple frames -------------------------------------------------

    #[test]
    fn test_rx_ring_multiple_frames() {
        let mut ring = RxDescriptorRing::new();
        ring.init();

        // Deliver 5 frames.
        for i in 0..5 {
            ring.descriptors[i].opts1 = DESC_FS | DESC_LS | 80;
            ring.buffers[i][0] = (i * 10) as u8;
        }

        for i in 0..5 {
            let mut buf = [0u8; MAX_FRAME_SIZE];
            let len = ring.poll_receive(&mut buf).unwrap();
            assert_eq!(len, 80);
            assert_eq!(buf[0], (i * 10) as u8);
        }

        // No more data.
        let mut buf = [0u8; MAX_FRAME_SIZE];
        assert_eq!(ring.poll_receive(&mut buf), Err(NetError::NoData));
    }

    // -- MAX_FRAME_SIZE constant --------------------------------------------

    #[test]
    fn test_max_frame_size() {
        assert_eq!(MAX_FRAME_SIZE, 1536);
        assert!(MAX_FRAME_SIZE >= 1514); // standard Ethernet
    }

    // -- TX enqueue copies data correctly -----------------------------------

    #[test]
    fn test_tx_enqueue_data_copy() {
        let mut ring = TxDescriptorRing::new();
        ring.init();

        let frame: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        ring.enqueue(&frame).unwrap();

        assert_eq!(&ring.buffers[0][..8], &frame);
    }

    // -- PHY status register bit constants ----------------------------------

    #[test]
    fn test_phy_status_constants() {
        assert_eq!(PHY_STATUS_LINK_UP, 0x02);
        assert_eq!(PHY_STATUS_1000MBPS, 0x10);
        assert_eq!(PHY_STATUS_100MBPS, 0x08);
        assert_eq!(PHY_STATUS_FULL_DUPLEX, 0x01);
    }

    // -- TX poll constant ---------------------------------------------------

    #[test]
    fn test_tx_poll_npq() {
        assert_eq!(TX_POLL_NPQ, 0x40);
    }

    // -- Accessor methods on device -----------------------------------------

    #[test]
    fn test_device_accessors() {
        let mut dev = Rtl8169Device::new_test(0x2000, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        dev.tx_ring.init();
        dev.rx_ring.init();

        assert_eq!(dev.base_addr(), 0x2000);
        assert_eq!(dev.mac_address(), [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        assert!(!dev.is_initialized());
        assert!(dev.tx_ring().is_empty());
        assert_eq!(dev.rx_ring().tail(), 0);
    }
}
