// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! FPGA CSI-2 receiver IP driver.
//!
//! Implements the [`CsiReceiver`] trait for a generic FPGA-based CSI-2
//! receiver IP block accessed through the [`FpgaFabric`] trait. This
//! covers custom RTL or vendor-provided CSI-2 receive cores that present
//! a memory-mapped register interface over AXI.
//!
//! The register map assumes a typical FPGA CSI-2 RX core with:
//! - Control/status registers
//! - DMA descriptor ring
//! - Frame buffer management
//! - Interrupt handling

#![allow(dead_code)]

use smallaios_kernel::hal::{CsiConfig, CsiFrame, CsiIrqSource, CsiReceiver, FpgaFabric, HalError};

// ---------------------------------------------------------------------------
// Register offsets for FPGA CSI-2 RX IP
// ---------------------------------------------------------------------------

/// Core identification.
const CORE_ID: u32 = 0x000;
/// Core version.
const CORE_VERSION: u32 = 0x004;
/// Global control register.
const CTRL_REG: u32 = 0x008;
/// Global status register.
const STAT_REG: u32 = 0x00C;

/// CSI protocol configuration.
const CSI_LANES: u32 = 0x010;
const CSI_DATA_TYPE: u32 = 0x014;
const CSI_ACTIVE_WIDTH: u32 = 0x018;
const CSI_ACTIVE_HEIGHT: u32 = 0x01C;

/// DMA configuration.
const DMA_CTRL: u32 = 0x020;
const DMA_STATUS: u32 = 0x024;
const DMA_LINE_STRIDE: u32 = 0x028;
const DMA_FRAME_SIZE: u32 = 0x02C;

/// DMA buffer addresses (up to 4 buffers).
const DMA_BUF0_ADDR: u32 = 0x040;
const DMA_BUF1_ADDR: u32 = 0x044;
const DMA_BUF2_ADDR: u32 = 0x048;
const DMA_BUF3_ADDR: u32 = 0x04C;

/// Interrupt registers.
const IRQ_STATUS: u32 = 0x060;
const IRQ_MASK: u32 = 0x064;
const IRQ_CLEAR: u32 = 0x068;

/// Frame completion register (read: last completed buffer index + status).
const FRAME_STATUS: u32 = 0x070;
const FRAME_COUNT: u32 = 0x074;
const FRAME_TIMESTAMP: u32 = 0x078;

// Control register bits.
const CTRL_ENABLE: u32 = 1 << 0;
const CTRL_CAPTURE: u32 = 1 << 1;
const CTRL_CONTINUOUS: u32 = 1 << 2;
const CTRL_SOFT_RESET: u32 = 1 << 31;

// Status register bits.
const STAT_ENABLED: u32 = 1 << 0;
const STAT_CAPTURING: u32 = 1 << 1;
const STAT_SYNCED: u32 = 1 << 2;

// IRQ bits.
const IRQ_FRAME_DONE: u32 = 1 << 0;
const IRQ_OVERFLOW: u32 = 1 << 1;
const IRQ_CRC_ERR: u32 = 1 << 2;
const IRQ_ECC_ERR: u32 = 1 << 3;

/// Maximum frame buffers.
const MAX_BUFFERS: u8 = 4;

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// FPGA CSI-2 receiver IP driver, generic over FPGA fabric access.
pub struct FpgaCsi<F: FpgaFabric> {
    fabric: F,
    base_addr: u64,
    configured: bool,
    capturing: bool,
    config: Option<CsiConfig>,
    buffer_addrs: [u64; MAX_BUFFERS as usize],
    frame_size: usize,
    sequence: u32,
    current_buffer: u8,
}

impl<F: FpgaFabric> FpgaCsi<F> {
    /// Create a new FPGA CSI-2 receiver driver.
    pub fn new(fabric: F, base_addr: u64) -> Self {
        Self {
            fabric,
            base_addr,
            configured: false,
            capturing: false,
            config: None,
            buffer_addrs: [0; MAX_BUFFERS as usize],
            frame_size: 0,
            sequence: 0,
            current_buffer: 0,
        }
    }

    /// Return the base address.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Return whether the receiver is configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Return whether capture is active.
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }

    fn read_reg(&self, offset: u32) -> Result<u32, HalError> {
        self.fabric.read_reg(self.base_addr, offset)
    }

    fn write_reg(&mut self, offset: u32, value: u32) -> Result<(), HalError> {
        self.fabric.write_reg(self.base_addr, offset, value)
    }

    /// Encode the MIPI CSI-2 data type code for a pixel format.
    fn data_type_code(config: &CsiConfig) -> u32 {
        match config.pixel_format {
            smallaios_kernel::hal::CsiPixelFormat::Raw8 => 0x2A,
            smallaios_kernel::hal::CsiPixelFormat::Raw10 => 0x2B,
            smallaios_kernel::hal::CsiPixelFormat::Raw12 => 0x2C,
            smallaios_kernel::hal::CsiPixelFormat::Yuv422 => 0x1E,
            smallaios_kernel::hal::CsiPixelFormat::Rgb888 => 0x24,
        }
    }

    /// Compute the DMA line stride.
    fn compute_line_stride(config: &CsiConfig) -> u32 {
        let bpp = match config.pixel_format {
            smallaios_kernel::hal::CsiPixelFormat::Raw8 => 8u32,
            smallaios_kernel::hal::CsiPixelFormat::Raw10 => 10,
            smallaios_kernel::hal::CsiPixelFormat::Raw12 => 12,
            smallaios_kernel::hal::CsiPixelFormat::Yuv422 => 16,
            smallaios_kernel::hal::CsiPixelFormat::Rgb888 => 24,
        };
        // Align to AXI burst boundary (typically 16 bytes).
        let raw_stride = (config.width as u32 * bpp).div_ceil(8);
        (raw_stride + 15) & !15
    }

    /// Get the DMA buffer address register offset for a given index.
    fn dma_buf_reg(index: u8) -> Result<u32, HalError> {
        match index {
            0 => Ok(DMA_BUF0_ADDR),
            1 => Ok(DMA_BUF1_ADDR),
            2 => Ok(DMA_BUF2_ADDR),
            3 => Ok(DMA_BUF3_ADDR),
            _ => Err(HalError::OutOfRange),
        }
    }
}

impl<F: FpgaFabric> CsiReceiver for FpgaCsi<F> {
    fn init(&mut self, config: CsiConfig) -> Result<(), HalError> {
        crate::camera::validate_csi_config(&config)?;

        // Soft reset the core.
        self.write_reg(CTRL_REG, CTRL_SOFT_RESET)?;
        self.write_reg(CTRL_REG, 0)?;

        // Configure CSI parameters.
        self.write_reg(CSI_LANES, config.lanes as u32)?;
        self.write_reg(CSI_DATA_TYPE, Self::data_type_code(&config))?;
        self.write_reg(CSI_ACTIVE_WIDTH, config.width as u32)?;
        self.write_reg(CSI_ACTIVE_HEIGHT, config.height as u32)?;

        // Configure DMA.
        let stride = Self::compute_line_stride(&config);
        self.write_reg(DMA_LINE_STRIDE, stride)?;

        self.frame_size = crate::camera::compute_frame_size(&config);
        self.write_reg(DMA_FRAME_SIZE, self.frame_size as u32)?;

        // Mask all interrupts initially.
        self.write_reg(IRQ_MASK, 0)?;

        self.config = Some(config);
        self.configured = true;
        self.sequence = 0;
        self.current_buffer = 0;
        Ok(())
    }

    fn start_capture(&mut self) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        // Enable interrupts.
        self.write_reg(
            IRQ_MASK,
            IRQ_FRAME_DONE | IRQ_OVERFLOW | IRQ_CRC_ERR | IRQ_ECC_ERR,
        )?;
        // Start continuous capture.
        self.write_reg(CTRL_REG, CTRL_ENABLE | CTRL_CAPTURE | CTRL_CONTINUOUS)?;
        self.capturing = true;
        Ok(())
    }

    fn stop_capture(&mut self) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        self.write_reg(CTRL_REG, 0)?;
        self.write_reg(IRQ_MASK, 0)?;
        self.capturing = false;
        Ok(())
    }

    fn dequeue_frame(&mut self) -> Result<CsiFrame, HalError> {
        if !self.capturing {
            return Err(HalError::InitFailed);
        }
        let status = self.read_reg(FRAME_STATUS)?;
        let buf_idx = (status & 0x03) as u8;
        let seq = self.read_reg(FRAME_COUNT)?;
        let ts = self.read_reg(FRAME_TIMESTAMP)? as u64;

        Ok(CsiFrame {
            buffer_index: buf_idx,
            timestamp_us: ts,
            sequence: seq,
            bytes_used: self.frame_size as u32,
        })
    }

    fn enqueue_frame(&mut self, buffer_index: u8) -> Result<(), HalError> {
        let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
        crate::camera::validate_buffer_index(buffer_index, config.num_buffers)?;

        let reg = Self::dma_buf_reg(buffer_index)?;
        let addr = self.buffer_addrs[buffer_index as usize];
        self.write_reg(reg, addr as u32)
    }

    fn buffer_address(&self, buffer_index: u8) -> Result<u64, HalError> {
        let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
        crate::camera::validate_buffer_index(buffer_index, config.num_buffers)?;
        Ok(self.buffer_addrs[buffer_index as usize])
    }

    fn buffer_size(&self) -> usize {
        self.frame_size
    }

    fn irq_handler(&mut self) -> Result<CsiIrqSource, HalError> {
        let status = self.read_reg(IRQ_STATUS)?;
        // Clear all pending interrupts.
        self.write_reg(IRQ_CLEAR, status)?;

        if status & IRQ_OVERFLOW != 0 {
            return Ok(CsiIrqSource::Overflow);
        }
        if status & IRQ_CRC_ERR != 0 {
            return Ok(CsiIrqSource::CrcError);
        }
        if status & IRQ_ECC_ERR != 0 {
            return Ok(CsiIrqSource::EccError);
        }
        if status & IRQ_FRAME_DONE != 0 {
            let buf_idx = self.current_buffer;
            self.sequence += 1;
            let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
            self.current_buffer = (self.current_buffer + 1) % config.num_buffers;
            return Ok(CsiIrqSource::FrameComplete(buf_idx));
        }
        Err(HalError::InterruptError)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(CTRL_REG, CTRL_SOFT_RESET)?;
        self.write_reg(CTRL_REG, 0)?;
        self.write_reg(IRQ_MASK, 0)?;
        self.configured = false;
        self.capturing = false;
        self.config = None;
        self.frame_size = 0;
        self.sequence = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{CsiPixelFormat, DmaDescriptor, DmaToken};

    struct MockFpgaFabric {
        regs: [u32; 64],
    }

    impl MockFpgaFabric {
        fn new() -> Self {
            Self { regs: [0u32; 64] }
        }
    }

    impl FpgaFabric for MockFpgaFabric {
        fn read_reg(&self, _base: u64, offset: u32) -> Result<u32, HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                Ok(self.regs[idx])
            } else {
                Err(HalError::OutOfRange)
            }
        }

        fn write_reg(&mut self, _base: u64, offset: u32, value: u32) -> Result<(), HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                self.regs[idx] = value;
                Ok(())
            } else {
                Err(HalError::OutOfRange)
            }
        }

        fn dma_start(&mut self, _d: DmaDescriptor) -> Result<DmaToken, HalError> {
            Err(HalError::NotSupported)
        }

        fn dma_poll(&self, t: DmaToken) -> Result<DmaToken, HalError> {
            Ok(t)
        }

        fn dma_irq_ack(&mut self) -> Result<u8, HalError> {
            Ok(0)
        }
    }

    fn make_config() -> CsiConfig {
        CsiConfig {
            lanes: 2,
            lane_speed_mbps: 800,
            width: 1920,
            height: 1080,
            pixel_format: CsiPixelFormat::Raw10,
            fps: 30,
            num_buffers: 3,
        }
    }

    #[test]
    fn test_fpga_csi_new() {
        let drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        assert_eq!(drv.base_addr(), 0x4300_0000);
        assert!(!drv.is_configured());
        assert!(!drv.is_capturing());
    }

    #[test]
    fn test_fpga_csi_init() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        assert!(drv.init(make_config()).is_ok());
        assert!(drv.is_configured());
    }

    #[test]
    fn test_fpga_csi_start_capture() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.start_capture().is_ok());
        assert!(drv.is_capturing());
    }

    #[test]
    fn test_fpga_csi_stop_capture() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        drv.init(make_config()).unwrap();
        drv.start_capture().unwrap();
        assert!(drv.stop_capture().is_ok());
        assert!(!drv.is_capturing());
    }

    #[test]
    fn test_fpga_csi_buffer_size() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        drv.init(make_config()).unwrap();
        // 1920*1080*10/8 = 2,592,000 (rounded up)
        assert!(drv.buffer_size() > 0);
        assert_eq!(
            drv.buffer_size(),
            crate::camera::compute_frame_size(&make_config())
        );
    }

    #[test]
    fn test_fpga_csi_reset() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.reset().is_ok());
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_fpga_csi_invalid_config() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        let mut cfg = make_config();
        cfg.lanes = 3; // Invalid.
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_fpga_csi_start_capture_not_configured() {
        let mut drv = FpgaCsi::new(MockFpgaFabric::new(), 0x4300_0000);
        assert_eq!(drv.start_capture(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_data_type_code() {
        let cfg = make_config();
        assert_eq!(FpgaCsi::<MockFpgaFabric>::data_type_code(&cfg), 0x2B);
    }

    #[test]
    fn test_compute_line_stride() {
        let cfg = make_config();
        let stride = FpgaCsi::<MockFpgaFabric>::compute_line_stride(&cfg);
        // 1920*10/8 = 2400, aligned to 16 = 2400 (already aligned).
        assert_eq!(stride, 2400);
    }

    #[test]
    fn test_dma_buf_reg() {
        assert_eq!(FpgaCsi::<MockFpgaFabric>::dma_buf_reg(0), Ok(DMA_BUF0_ADDR));
        assert_eq!(FpgaCsi::<MockFpgaFabric>::dma_buf_reg(1), Ok(DMA_BUF1_ADDR));
        assert_eq!(
            FpgaCsi::<MockFpgaFabric>::dma_buf_reg(4),
            Err(HalError::OutOfRange)
        );
    }
}
