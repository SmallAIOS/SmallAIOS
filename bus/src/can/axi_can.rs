// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AXI CAN controller driver for FPGA soft-IP.
//!
//! Implements the [`CanController`] trait for the Xilinx/AMD AXI CAN
//! soft-IP core (PG096), used on Zynq UltraScale+ and similar SoC FPGAs.
//! The controller is accessed through memory-mapped AXI4-Lite registers.
//!
//! # Hardware interface
//!
//! The AXI CAN IP is mapped into the CPU's address space as a contiguous
//! register block. This driver uses an [`MmioAccess`] trait to perform
//! 32-bit register reads and writes.
//!
//! # References
//!
//! - Xilinx PG096: AXI CAN Controller LogiCORE IP Product Guide
//! - ISO 11898-1 CAN data link layer

use crate::BusError;
use crate::can::controller::{CanBitTiming, CanController, CanMode, CanStats};
use crate::can::frame::{CanFrame, FrameType};
use crate::can::state::BusState;

// ---------------------------------------------------------------------------
// MMIO abstraction
// ---------------------------------------------------------------------------

/// Memory-mapped I/O access trait for AXI register reads/writes.
///
/// Platform HALs implement this trait to provide access to a specific
/// MMIO region. The base address is managed by the implementation.
pub trait MmioAccess {
    /// Read a 32-bit register at the given byte offset from the base address.
    fn read32(&self, offset: usize) -> u32;

    /// Write a 32-bit value to the register at the given byte offset.
    fn write32(&mut self, offset: usize, value: u32);
}

// ---------------------------------------------------------------------------
// AXI CAN register offsets (from PG096)
// ---------------------------------------------------------------------------

mod reg {
    /// Software Reset Register.
    pub const SRR: usize = 0x00;
    /// Mode Select Register.
    pub const MSR: usize = 0x04;
    /// Bus Timing Register.
    pub const BRPR: usize = 0x08;
    /// Bit Timing Register.
    pub const BTR: usize = 0x0C;
    /// Error Counter Register.
    pub const ECR: usize = 0x10;
    /// Error Status Register.
    pub const ESR: usize = 0x14;
    /// Status Register.
    pub const SR: usize = 0x18;
    /// TX FIFO ID register.
    pub const TXFIFO_ID: usize = 0x30;
    /// TX FIFO DLC register.
    pub const TXFIFO_DLC: usize = 0x34;
    /// TX FIFO Data Word 1 (bytes 0-3).
    pub const TXFIFO_DW1: usize = 0x38;
    /// TX FIFO Data Word 2 (bytes 4-7).
    pub const TXFIFO_DW2: usize = 0x3C;
    /// RX FIFO ID register.
    pub const RXFIFO_ID: usize = 0x50;
    /// RX FIFO DLC register.
    pub const RXFIFO_DLC: usize = 0x54;
    /// RX FIFO Data Word 1.
    pub const RXFIFO_DW1: usize = 0x58;
    /// RX FIFO Data Word 2.
    pub const RXFIFO_DW2: usize = 0x5C;
}

/// SRR register bits.
mod srr {
    /// CAN Enable bit.
    pub const CEN: u32 = 1 << 1;
    /// Software Reset bit.
    pub const SRST: u32 = 1 << 0;
}

/// MSR register bits.
mod msr {
    /// Sleep mode request.
    pub const SLEEP: u32 = 1 << 0;
    /// Loopback mode.
    pub const LBACK: u32 = 1 << 1;
}

/// SR register bits.
mod sr {
    /// Configuration mode indicator.
    pub const CONFIG: u32 = 1 << 0;
    /// TX FIFO full.
    pub const TXFLL: u32 = 1 << 4;
    /// RX FIFO valid (at least one frame available).
    pub const RXFVLD: u32 = 1 << 7;
}

/// ESR register bits for error state.
mod esr {
    /// Bus-Off state.
    pub const BOFF: u32 = 1 << 2;
    /// Error-Passive state.
    pub const EPSV: u32 = 1 << 1;
}

// ---------------------------------------------------------------------------
// AXI CAN controller
// ---------------------------------------------------------------------------

/// AXI CAN soft-IP controller driver.
///
/// Wraps an [`MmioAccess`] to provide [`CanController`] access to the
/// Xilinx AXI CAN IP core.
pub struct AxiCan<M: MmioAccess> {
    mmio: M,
    mode: CanMode,
    stats: CanStats,
    initialized: bool,
}

impl<M: MmioAccess> AxiCan<M> {
    /// Create a new AXI CAN driver wrapping the given MMIO accessor.
    pub fn new(mmio: M) -> Self {
        Self {
            mmio,
            mode: CanMode::Configuration,
            stats: CanStats::default(),
            initialized: false,
        }
    }

    /// Check if the controller is in configuration mode.
    fn is_config_mode(&self) -> bool {
        self.mmio.read32(reg::SR) & sr::CONFIG != 0
    }

    /// Enter configuration mode by clearing CEN in SRR.
    fn enter_config_mode(&mut self) -> Result<(), BusError> {
        let srr = self.mmio.read32(reg::SRR);
        self.mmio.write32(reg::SRR, srr & !srr::CEN);
        if !self.is_config_mode() {
            return Err(BusError::Timeout);
        }
        Ok(())
    }

    /// Enter operational mode by setting CEN in SRR.
    fn enter_operational_mode(&mut self) -> Result<(), BusError> {
        let srr = self.mmio.read32(reg::SRR);
        self.mmio.write32(reg::SRR, srr | srr::CEN);
        Ok(())
    }

    /// Read a CAN frame from the RX FIFO.
    fn read_rx_fifo(&mut self) -> Result<Option<CanFrame>, BusError> {
        let status = self.mmio.read32(reg::SR);
        if status & sr::RXFVLD == 0 {
            return Ok(None);
        }

        let id_reg = self.mmio.read32(reg::RXFIFO_ID);
        let dlc_reg = self.mmio.read32(reg::RXFIFO_DLC);
        let dw1 = self.mmio.read32(reg::RXFIFO_DW1);
        let dw2 = self.mmio.read32(reg::RXFIFO_DW2);

        let is_extended = id_reg & (1 << 19) != 0;
        let dlc = (dlc_reg & 0x0F) as usize;
        let dlc = core::cmp::min(dlc, 8);

        let (id, frame_type) = if is_extended {
            let id = id_reg & 0x1FFFFFFF;
            (id, FrameType::Extended)
        } else {
            let id = (id_reg >> 21) & 0x7FF;
            (id, FrameType::Standard)
        };

        // Reassemble data bytes from the two 32-bit words.
        let mut data = [0u8; 8];
        data[0] = (dw1 >> 24) as u8;
        data[1] = (dw1 >> 16) as u8;
        data[2] = (dw1 >> 8) as u8;
        data[3] = dw1 as u8;
        data[4] = (dw2 >> 24) as u8;
        data[5] = (dw2 >> 16) as u8;
        data[6] = (dw2 >> 8) as u8;
        data[7] = dw2 as u8;

        let frame = match frame_type {
            FrameType::Standard => CanFrame::new_standard(id, &data[..dlc])?,
            FrameType::Extended => CanFrame::new_extended(id, &data[..dlc])?,
        };

        self.stats.rx_frames += 1;
        Ok(Some(frame))
    }

    /// Write a CAN frame to the TX FIFO.
    fn write_tx_fifo(&mut self, frame: &CanFrame) -> Result<(), BusError> {
        let status = self.mmio.read32(reg::SR);
        if status & sr::TXFLL != 0 {
            return Err(BusError::BufferTooSmall);
        }

        let id_reg = match frame.frame_type {
            FrameType::Standard => (frame.id & 0x7FF) << 21,
            FrameType::Extended => (frame.id & 0x1FFFFFFF) | (1 << 19),
        };

        let dlc = core::cmp::min(frame.data.len(), 8) as u32;

        let mut d = [0u8; 8];
        let len = frame.data.len().min(8);
        d[..len].copy_from_slice(&frame.data[..len]);

        let dw1 = ((d[0] as u32) << 24) | ((d[1] as u32) << 16) | ((d[2] as u32) << 8) | d[3] as u32;
        let dw2 = ((d[4] as u32) << 24) | ((d[5] as u32) << 16) | ((d[6] as u32) << 8) | d[7] as u32;

        self.mmio.write32(reg::TXFIFO_ID, id_reg);
        self.mmio.write32(reg::TXFIFO_DLC, dlc);
        self.mmio.write32(reg::TXFIFO_DW1, dw1);
        self.mmio.write32(reg::TXFIFO_DW2, dw2);

        self.stats.tx_frames += 1;
        Ok(())
    }
}

impl<M: MmioAccess> CanController for AxiCan<M> {
    fn init(&mut self) -> Result<(), BusError> {
        // Software reset.
        self.mmio.write32(reg::SRR, srr::SRST);
        // Wait for reset to complete (SRST auto-clears).
        // Enter configuration mode (CEN=0).
        self.mmio.write32(reg::SRR, 0);
        self.mode = CanMode::Configuration;
        self.initialized = true;
        Ok(())
    }

    fn set_mode(&mut self, mode: CanMode) -> Result<(), BusError> {
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        match mode {
            CanMode::Configuration => {
                self.enter_config_mode()?;
            }
            CanMode::Normal => {
                self.mmio.write32(reg::MSR, 0);
                self.enter_operational_mode()?;
            }
            CanMode::Loopback => {
                self.mmio.write32(reg::MSR, msr::LBACK);
                self.enter_operational_mode()?;
            }
            CanMode::Sleep => {
                self.mmio.write32(reg::MSR, msr::SLEEP);
                self.enter_operational_mode()?;
            }
            CanMode::ListenOnly => {
                return Err(BusError::NotSupported);
            }
        }
        self.mode = mode;
        Ok(())
    }

    fn mode(&self) -> CanMode {
        self.mode
    }

    fn set_bit_timing(&mut self, timing: &CanBitTiming) -> Result<(), BusError> {
        if self.mode != CanMode::Configuration {
            return Err(BusError::NotSupported);
        }
        // BRPR: prescaler - 1
        self.mmio.write32(reg::BRPR, timing.prescaler.saturating_sub(1) as u32);
        // BTR: SJW(2) + TSEG2(3) + TSEG1(4)
        let btr = (((timing.sjw.saturating_sub(1) as u32) & 0x03) << 7)
            | (((timing.tseg2.saturating_sub(1) as u32) & 0x07) << 4)
            | ((timing.tseg1.saturating_sub(1) as u32) & 0x0F);
        self.mmio.write32(reg::BTR, btr);
        Ok(())
    }

    fn transmit(&mut self, frame: &CanFrame) -> Result<(), BusError> {
        if self.mode != CanMode::Normal && self.mode != CanMode::Loopback {
            return Err(BusError::NotSupported);
        }
        self.write_tx_fifo(frame)
    }

    fn receive(&mut self) -> Result<Option<CanFrame>, BusError> {
        if self.mode == CanMode::Configuration || self.mode == CanMode::Sleep {
            return Err(BusError::NotSupported);
        }
        self.read_rx_fifo()
    }

    fn bus_state(&self) -> BusState {
        let esr = self.mmio.read32(reg::ESR);
        if esr & esr::BOFF != 0 {
            BusState::BusOff
        } else if esr & esr::EPSV != 0 {
            BusState::ErrorPassive
        } else {
            BusState::ErrorActive
        }
    }

    fn stats(&self) -> CanStats {
        let ecr = self.mmio.read32(reg::ECR);
        CanStats {
            tx_frames: self.stats.tx_frames,
            rx_frames: self.stats.rx_frames,
            tx_errors: self.stats.tx_errors,
            rx_errors: self.stats.rx_errors,
            bus_off_count: self.stats.bus_off_count,
            error_passive_count: self.stats.error_passive_count,
            tec: (ecr & 0xFF) as u8,
            rec: ((ecr >> 8) & 0xFF) as u8,
        }
    }

    fn reset(&mut self) -> Result<(), BusError> {
        self.mmio.write32(reg::SRR, srr::SRST);
        self.mmio.write32(reg::SRR, 0);
        self.mode = CanMode::Configuration;
        self.stats = CanStats::default();
        self.initialized = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock MMIO accessor backed by a register array.
    struct MockMmio {
        regs: [u32; 64],
    }

    impl MockMmio {
        fn new() -> Self {
            Self { regs: [0u32; 64] }
        }
    }

    impl MmioAccess for MockMmio {
        fn read32(&self, offset: usize) -> u32 {
            let idx = offset / 4;
            if idx < self.regs.len() {
                self.regs[idx]
            } else {
                0
            }
        }

        fn write32(&mut self, offset: usize, value: u32) {
            let idx = offset / 4;
            if idx < self.regs.len() {
                self.regs[idx] = value;
            }
        }
    }

    #[test]
    fn test_axi_can_new() {
        let mmio = MockMmio::new();
        let drv = AxiCan::new(mmio);
        assert_eq!(drv.mode(), CanMode::Configuration);
        assert!(!drv.initialized);
    }

    #[test]
    fn test_axi_can_init() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        assert!(drv.initialized);
        assert_eq!(drv.mode(), CanMode::Configuration);
    }

    #[test]
    fn test_axi_can_set_bit_timing() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        let timing = CanBitTiming::standard_500kbps();
        drv.set_bit_timing(&timing).unwrap();
    }

    #[test]
    fn test_axi_can_set_bit_timing_wrong_mode() {
        let mut mmio = MockMmio::new();
        // Simulate Normal mode: set SR bit 3
        mmio.regs[reg::SR / 4] = 1 << 3;

        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        drv.mode = CanMode::Normal; // Force mode
        let timing = CanBitTiming::standard_500kbps();
        assert_eq!(drv.set_bit_timing(&timing).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_axi_can_transmit_wrong_mode() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        let frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        assert_eq!(drv.transmit(&frame).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_axi_can_receive_wrong_mode() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        assert_eq!(drv.receive().err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_axi_can_bus_state_active() {
        let mmio = MockMmio::new();
        let drv = AxiCan::new(mmio);
        assert_eq!(drv.bus_state(), BusState::ErrorActive);
    }

    #[test]
    fn test_axi_can_bus_state_bus_off() {
        let mut mmio = MockMmio::new();
        mmio.regs[reg::ESR / 4] = esr::BOFF;
        let drv = AxiCan::new(mmio);
        assert_eq!(drv.bus_state(), BusState::BusOff);
    }

    #[test]
    fn test_axi_can_bus_state_error_passive() {
        let mut mmio = MockMmio::new();
        mmio.regs[reg::ESR / 4] = esr::EPSV;
        let drv = AxiCan::new(mmio);
        assert_eq!(drv.bus_state(), BusState::ErrorPassive);
    }

    #[test]
    fn test_axi_can_reset() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        drv.reset().unwrap();
        assert!(!drv.initialized);
        assert_eq!(drv.mode(), CanMode::Configuration);
    }

    #[test]
    fn test_axi_can_receive_empty() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        drv.mode = CanMode::Normal; // Force mode
        assert!(drv.receive().unwrap().is_none());
    }

    #[test]
    fn test_axi_can_stats_with_ecr() {
        let mut mmio = MockMmio::new();
        // ECR: TEC=10 (low byte), REC=5 (next byte)
        mmio.regs[reg::ECR / 4] = 0x050A;
        let drv = AxiCan::new(mmio);
        let stats = drv.stats();
        assert_eq!(stats.tec, 10);
        assert_eq!(stats.rec, 5);
    }

    #[test]
    fn test_axi_can_set_mode_unsupported() {
        let mmio = MockMmio::new();
        let mut drv = AxiCan::new(mmio);
        drv.init().unwrap();
        assert_eq!(drv.set_mode(CanMode::ListenOnly).err(), Some(BusError::NotSupported));
    }
}
