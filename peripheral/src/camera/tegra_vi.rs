// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA Tegra Video Input (VI) / CSI-2 receiver driver.
//!
//! The Tegra VI unit receives MIPI CSI-2 data from camera sensors and
//! writes frames into memory via DMA. This driver targets the Tegra X1/X2
//! generation VI hardware block.
//!
//! This is a stub MMIO driver for host testing — real register access
//! returns `HalError::MmioError`.
//!
//! # References
//!
//! - NVIDIA Tegra X1 TRM, Chapter 27: VI
//! - NVIDIA Tegra X2 TRM, Chapter 25: VI5

#![allow(dead_code)]

use smallaios_kernel::hal::{CsiConfig, CsiFrame, CsiIrqSource, CsiReceiver, HalError};

// ---------------------------------------------------------------------------
// Tegra VI register offsets (from Tegra X1/X2 TRM)
// ---------------------------------------------------------------------------

/// CSI channel base offset (per-channel stride: 0x10000).
const CSI_CHAN_STRIDE: u32 = 0x10000;

/// CSI pixel parser registers.
const CSI_PP_STATUS: u32 = 0x0000;
const CSI_PP_CTRL: u32 = 0x0004;
const CSI_PP_ACTIVE_SIZE: u32 = 0x0008;
const CSI_PP_FORMAT: u32 = 0x000C;

/// CSI PHY control.
const CSI_PHY_CTRL: u32 = 0x0100;
const CSI_PHY_STATUS: u32 = 0x0104;
const CSI_CIL_CTRL: u32 = 0x0108;
const CSI_CIL_STATUS: u32 = 0x010C;
const CSI_CIL_PAD_CONFIG: u32 = 0x0110;

/// VI channel registers.
const VI_CHAN_CTRL: u32 = 0x1000;
const VI_CHAN_STATUS: u32 = 0x1004;
const VI_CHAN_SURFACE_ADDR: u32 = 0x1008;
const VI_CHAN_SURFACE_STRIDE: u32 = 0x100C;
const VI_CHAN_SURFACE_SIZE: u32 = 0x1010;
const VI_CHAN_SYNC_CTRL: u32 = 0x1014;

/// VI DMA registers.
const VI_DMA_CTRL: u32 = 0x2000;
const VI_DMA_STATUS: u32 = 0x2004;
const VI_DMA_ADDR_0: u32 = 0x2008;
const VI_DMA_ADDR_1: u32 = 0x200C;
const VI_DMA_ADDR_2: u32 = 0x2010;
const VI_DMA_ADDR_3: u32 = 0x2014;
const VI_DMA_LINE_STRIDE: u32 = 0x2018;
const VI_DMA_FRAME_SIZE: u32 = 0x201C;

/// VI interrupt registers.
const VI_IRQ_STATUS: u32 = 0x3000;
const VI_IRQ_MASK: u32 = 0x3004;
const VI_IRQ_CLEAR: u32 = 0x3008;

/// VI control bits.
const VI_CTRL_ENABLE: u32 = 1 << 0;
const VI_CTRL_CAPTURE: u32 = 1 << 1;
const VI_CTRL_CONTINUOUS: u32 = 1 << 2;

/// VI IRQ bits.
const VI_IRQ_FRAME_DONE: u32 = 1 << 0;
const VI_IRQ_OVERFLOW: u32 = 1 << 1;
const VI_IRQ_CRC_ERR: u32 = 1 << 2;
const VI_IRQ_ECC_ERR: u32 = 1 << 3;

// ---------------------------------------------------------------------------
// Driver state
// ---------------------------------------------------------------------------

/// Maximum frame buffers supported.
const MAX_BUFFERS: u8 = 4;

/// Tegra VI CSI-2 receiver driver.
pub struct TegraVi {
    /// MMIO base address of the VI block.
    base_addr: u64,
    /// IRQ number.
    irq: u32,
    /// Whether the receiver has been initialized.
    configured: bool,
    /// Whether capture is active.
    capturing: bool,
    /// Current configuration.
    config: Option<CsiConfig>,
    /// Frame buffer base addresses (physical).
    buffer_addrs: [u64; MAX_BUFFERS as usize],
    /// Computed frame buffer size in bytes.
    frame_size: usize,
    /// Frame sequence counter.
    sequence: u32,
    /// Current write buffer index.
    current_buffer: u8,
}

impl TegraVi {
    /// Create a new Tegra VI driver.
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            capturing: false,
            config: None,
            buffer_addrs: [0; MAX_BUFFERS as usize],
            frame_size: 0,
            sequence: 0,
            current_buffer: 0,
        }
    }

    /// Return the MMIO base address.
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

    /// Read a register (stub — returns MmioError on host).
    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }

    /// Write a register (stub — returns MmioError on host).
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    /// Build the CSI pixel parser format register value.
    fn build_pp_format(config: &CsiConfig) -> u32 {
        let fmt_code = match config.pixel_format {
            smallaios_kernel::hal::CsiPixelFormat::Raw8 => 0x2A,
            smallaios_kernel::hal::CsiPixelFormat::Raw10 => 0x2B,
            smallaios_kernel::hal::CsiPixelFormat::Raw12 => 0x2C,
            smallaios_kernel::hal::CsiPixelFormat::Yuv422 => 0x1E,
            smallaios_kernel::hal::CsiPixelFormat::Rgb888 => 0x24,
        };
        (config.lanes as u32) << 16 | fmt_code
    }

    /// Build the active size register value.
    fn build_active_size(config: &CsiConfig) -> u32 {
        (config.height as u32) << 16 | config.width as u32
    }

    /// Compute the DMA line stride in bytes.
    fn compute_line_stride(config: &CsiConfig) -> u32 {
        let bpp = match config.pixel_format {
            smallaios_kernel::hal::CsiPixelFormat::Raw8 => 8u32,
            smallaios_kernel::hal::CsiPixelFormat::Raw10 => 10,
            smallaios_kernel::hal::CsiPixelFormat::Raw12 => 12,
            smallaios_kernel::hal::CsiPixelFormat::Yuv422 => 16,
            smallaios_kernel::hal::CsiPixelFormat::Rgb888 => 24,
        };
        // Align stride to 64 bytes for DMA efficiency.
        let raw_stride = (config.width as u32 * bpp).div_ceil(8);
        (raw_stride + 63) & !63
    }
}

impl CsiReceiver for TegraVi {
    fn init(&mut self, config: CsiConfig) -> Result<(), HalError> {
        crate::camera::validate_csi_config(&config)?;

        // Disable VI.
        self.write_reg(VI_CHAN_CTRL, 0)?;

        // Configure CSI PHY for the requested lane count.
        let phy_ctrl = match config.lanes {
            1 => 0x01,
            2 => 0x03,
            4 => 0x0F,
            _ => return Err(HalError::InvalidConfig),
        };
        self.write_reg(CSI_PHY_CTRL, phy_ctrl)?;

        // Configure pixel parser.
        self.write_reg(CSI_PP_FORMAT, Self::build_pp_format(&config))?;
        self.write_reg(CSI_PP_ACTIVE_SIZE, Self::build_active_size(&config))?;

        // Configure DMA stride.
        let stride = Self::compute_line_stride(&config);
        self.write_reg(VI_DMA_LINE_STRIDE, stride)?;

        // Compute frame size.
        self.frame_size = crate::camera::compute_frame_size(&config);

        self.config = Some(config);
        self.configured = true;
        self.sequence = 0;
        Ok(())
    }

    fn start_capture(&mut self) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        // Enable VI in continuous capture mode.
        self.write_reg(
            VI_CHAN_CTRL,
            VI_CTRL_ENABLE | VI_CTRL_CAPTURE | VI_CTRL_CONTINUOUS,
        )?;
        // Unmask frame-done and error interrupts.
        self.write_reg(
            VI_IRQ_MASK,
            VI_IRQ_FRAME_DONE | VI_IRQ_OVERFLOW | VI_IRQ_CRC_ERR,
        )?;
        self.capturing = true;
        Ok(())
    }

    fn stop_capture(&mut self) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        self.write_reg(VI_CHAN_CTRL, 0)?;
        self.write_reg(VI_IRQ_MASK, 0)?;
        self.capturing = false;
        Ok(())
    }

    fn dequeue_frame(&mut self) -> Result<CsiFrame, HalError> {
        if !self.capturing {
            return Err(HalError::InitFailed);
        }
        // In a real driver, this would check the DMA completion ring.
        let status = self.read_reg(VI_DMA_STATUS)?;
        let _ = status; // Use status to determine buffer index.
        Err(HalError::MmioError)
    }

    fn enqueue_frame(&mut self, buffer_index: u8) -> Result<(), HalError> {
        let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
        crate::camera::validate_buffer_index(buffer_index, config.num_buffers)?;

        // Program the DMA address register for this buffer slot.
        let dma_addr_offset = match buffer_index {
            0 => VI_DMA_ADDR_0,
            1 => VI_DMA_ADDR_1,
            2 => VI_DMA_ADDR_2,
            3 => VI_DMA_ADDR_3,
            _ => return Err(HalError::OutOfRange),
        };
        self.write_reg(
            dma_addr_offset,
            self.buffer_addrs[buffer_index as usize] as u32,
        )
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
        let status = self.read_reg(VI_IRQ_STATUS)?;
        // Clear the interrupt.
        self.write_reg(VI_IRQ_CLEAR, status)?;

        if status & VI_IRQ_OVERFLOW != 0 {
            return Ok(CsiIrqSource::Overflow);
        }
        if status & VI_IRQ_CRC_ERR != 0 {
            return Ok(CsiIrqSource::CrcError);
        }
        if status & VI_IRQ_ECC_ERR != 0 {
            return Ok(CsiIrqSource::EccError);
        }
        if status & VI_IRQ_FRAME_DONE != 0 {
            let buf_idx = self.current_buffer;
            self.sequence += 1;
            let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
            self.current_buffer = (self.current_buffer + 1) % config.num_buffers;
            return Ok(CsiIrqSource::FrameComplete(buf_idx));
        }
        Err(HalError::InterruptError)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(VI_CHAN_CTRL, 0)?;
        self.write_reg(VI_IRQ_MASK, 0)?;
        self.write_reg(CSI_PHY_CTRL, 0)?;
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
    use smallaios_kernel::hal::CsiPixelFormat;

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
    fn test_tegra_vi_new() {
        let drv = TegraVi::new(0x5420_0000, 50);
        assert_eq!(drv.base_addr(), 0x5420_0000);
        assert!(!drv.is_configured());
        assert!(!drv.is_capturing());
    }

    #[test]
    fn test_build_pp_format() {
        let config = make_config();
        let fmt = TegraVi::build_pp_format(&config);
        // Lanes=2 in bits 17:16, RAW10=0x2B in bits 7:0.
        assert_eq!(fmt & 0xFF, 0x2B);
        assert_eq!((fmt >> 16) & 0xFF, 2);
    }

    #[test]
    fn test_build_active_size() {
        let config = make_config();
        let size = TegraVi::build_active_size(&config);
        assert_eq!(size & 0xFFFF, 1920);
        assert_eq!((size >> 16) & 0xFFFF, 1080);
    }

    #[test]
    fn test_compute_line_stride_raw10() {
        let config = make_config();
        let stride = TegraVi::compute_line_stride(&config);
        // 1920 * 10 / 8 = 2400, aligned to 64 = 2432.
        assert_eq!(stride, 2432);
    }

    #[test]
    fn test_compute_line_stride_rgb888() {
        let mut config = make_config();
        config.pixel_format = CsiPixelFormat::Rgb888;
        let stride = TegraVi::compute_line_stride(&config);
        // 1920 * 3 = 5760, aligned to 64 = 5760 (already aligned).
        assert_eq!(stride, 5760);
    }

    #[test]
    fn test_init_returns_mmio_error() {
        let mut drv = TegraVi::new(0x5420_0000, 50);
        // Init will fail because write_reg returns MmioError.
        assert_eq!(drv.init(make_config()), Err(HalError::MmioError));
    }

    #[test]
    fn test_start_capture_not_configured() {
        let mut drv = TegraVi::new(0x5420_0000, 50);
        assert_eq!(drv.start_capture(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_buffer_size_after_manual_set() {
        let mut drv = TegraVi::new(0x5420_0000, 50);
        // Manually set frame_size to test buffer_size() accessor.
        drv.frame_size = 2_592_000;
        assert_eq!(drv.buffer_size(), 2_592_000);
    }

    #[test]
    fn test_buffer_address_not_configured() {
        let drv = TegraVi::new(0x5420_0000, 50);
        assert_eq!(drv.buffer_address(0), Err(HalError::InitFailed));
    }
}
