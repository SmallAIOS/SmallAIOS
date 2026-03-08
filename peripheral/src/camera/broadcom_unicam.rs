// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Broadcom Unicam CSI-2 receiver driver.
//!
//! The Unicam block is Broadcom's MIPI CSI-2 receiver IP found in BCM2835/2837/2711
//! SoCs (Raspberry Pi 1-4). It supports up to 2 CSI-2 data lanes with
//! DMA write-back of received frame data.
//!
//! This is a stub MMIO driver for host testing — real register access
//! returns `HalError::MmioError`.
//!
//! # References
//!
//! - BCM2835 ARM Peripherals Guide (Section 4: Unicam)
//! - Linux kernel: drivers/media/platform/bcm2835/bcm2835-unicam.c

#![allow(dead_code)]

use smallaios_kernel::hal::{CsiConfig, CsiFrame, CsiIrqSource, CsiReceiver, HalError};

// ---------------------------------------------------------------------------
// Unicam register offsets
// ---------------------------------------------------------------------------

/// Image data registers.
const UNICAM_CTRL: u32 = 0x000;
const UNICAM_STA: u32 = 0x004;
const UNICAM_ANA: u32 = 0x008;
const UNICAM_PRI: u32 = 0x00C;
const UNICAM_CLK: u32 = 0x010;
const UNICAM_CLT: u32 = 0x014;
const UNICAM_DAT0: u32 = 0x018;
const UNICAM_DAT1: u32 = 0x01C;

/// DMA control.
const UNICAM_DCS: u32 = 0x200;
const UNICAM_DBSA0: u32 = 0x204;
const UNICAM_DBEA0: u32 = 0x208;
const UNICAM_DBSA1: u32 = 0x20C;
const UNICAM_DBEA1: u32 = 0x210;
const UNICAM_DBCTL: u32 = 0x214;
const UNICAM_IBSA0: u32 = 0x218;
const UNICAM_IBEA0: u32 = 0x21C;
const UNICAM_IBSA1: u32 = 0x220;
const UNICAM_IBEA1: u32 = 0x224;
const UNICAM_IBLS: u32 = 0x228;

/// Interrupt registers.
const UNICAM_ICTL: u32 = 0x100;
const UNICAM_ISTA: u32 = 0x104;
const UNICAM_IDI0: u32 = 0x108;
const UNICAM_IPIPE: u32 = 0x10C;

/// Image capture.
const UNICAM_ICC: u32 = 0x300;
const UNICAM_ICS: u32 = 0x304;
const UNICAM_IDC: u32 = 0x308;
const UNICAM_IDPO: u32 = 0x30C;
const UNICAM_IDCA: u32 = 0x310;
const UNICAM_IDCD: u32 = 0x314;

// Control register bits.
const CTRL_MEN: u32 = 1 << 0; // Module enable.
const CTRL_SOE: u32 = 1 << 1; // Start of embedded data enable.
const CTRL_CPE: u32 = 1 << 2; // Clock phase.
const CTRL_OFEN: u32 = 1 << 3; // Overflow enable.

// Status register bits.
const STA_SYN: u32 = 1 << 0; // Synchronization.
const STA_CS: u32 = 1 << 1; // CRC status.
const STA_DL: u32 = 1 << 2; // Data loss.
const STA_SBE: u32 = 1 << 4; // Start of embedded data.
const STA_PBE: u32 = 1 << 5; // Picture boundary error.

// Interrupt bits.
const ICTL_FSI: u32 = 1 << 0; // Frame start interrupt.
const ICTL_FEI: u32 = 1 << 1; // Frame end interrupt.
const ICTL_LCI: u32 = 1 << 4; // Line capture interrupt.
const ICTL_OFO: u32 = 1 << 5; // FIFO overflow interrupt.
const ICTL_CRCE: u32 = 1 << 6; // CRC error interrupt.

/// Maximum frame buffers.
const MAX_BUFFERS: u8 = 4;

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Broadcom Unicam CSI-2 receiver driver.
pub struct BroadcomUnicam {
    /// MMIO base address.
    base_addr: u64,
    /// IRQ number.
    irq: u32,
    /// Whether initialized.
    configured: bool,
    /// Whether capture is active.
    capturing: bool,
    /// Current configuration.
    config: Option<CsiConfig>,
    /// Frame buffer base addresses.
    buffer_addrs: [u64; MAX_BUFFERS as usize],
    /// Computed frame buffer size.
    frame_size: usize,
    /// Frame sequence counter.
    sequence: u32,
    /// Current DMA buffer index.
    current_buffer: u8,
}

impl BroadcomUnicam {
    /// Create a new Unicam driver.
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

    /// Read a register (stub).
    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }

    /// Write a register (stub).
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    /// Build the UNICAM_IDI0 register value for the pixel format.
    ///
    /// This register selects the MIPI CSI-2 data type to receive.
    fn build_data_id(config: &CsiConfig) -> u32 {
        match config.pixel_format {
            smallaios_kernel::hal::CsiPixelFormat::Raw8 => 0x2A,
            smallaios_kernel::hal::CsiPixelFormat::Raw10 => 0x2B,
            smallaios_kernel::hal::CsiPixelFormat::Raw12 => 0x2C,
            smallaios_kernel::hal::CsiPixelFormat::Yuv422 => 0x1E,
            smallaios_kernel::hal::CsiPixelFormat::Rgb888 => 0x24,
        }
    }

    /// Compute the image line stride in bytes.
    fn compute_line_stride(config: &CsiConfig) -> u32 {
        let bpp = match config.pixel_format {
            smallaios_kernel::hal::CsiPixelFormat::Raw8 => 8u32,
            smallaios_kernel::hal::CsiPixelFormat::Raw10 => 10,
            smallaios_kernel::hal::CsiPixelFormat::Raw12 => 12,
            smallaios_kernel::hal::CsiPixelFormat::Yuv422 => 16,
            smallaios_kernel::hal::CsiPixelFormat::Rgb888 => 24,
        };
        // Unicam requires 32-byte aligned line stride.
        let raw_stride = (config.width as u32 * bpp).div_ceil(8);
        (raw_stride + 31) & !31
    }
}

impl CsiReceiver for BroadcomUnicam {
    fn init(&mut self, config: CsiConfig) -> Result<(), HalError> {
        crate::camera::validate_csi_config(&config)?;

        // Unicam supports max 2 lanes.
        if config.lanes > 2 {
            return Err(HalError::InvalidConfig);
        }

        // Disable module.
        self.write_reg(UNICAM_CTRL, 0)?;

        // Configure data lane settings.
        let dat0_val = config.lanes as u32;
        self.write_reg(UNICAM_DAT0, dat0_val)?;

        // Configure data ID filter.
        self.write_reg(UNICAM_IDI0, Self::build_data_id(&config))?;

        // Configure line stride for DMA.
        let stride = Self::compute_line_stride(&config);
        self.write_reg(UNICAM_IBLS, stride)?;

        // Compute and store frame size.
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

        // Enable frame-end and overflow interrupts.
        self.write_reg(UNICAM_ICTL, ICTL_FEI | ICTL_OFO | ICTL_CRCE)?;

        // Enable module and start capture.
        self.write_reg(UNICAM_CTRL, CTRL_MEN | CTRL_OFEN)?;

        self.capturing = true;
        Ok(())
    }

    fn stop_capture(&mut self) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        self.write_reg(UNICAM_CTRL, 0)?;
        self.write_reg(UNICAM_ICTL, 0)?;
        self.capturing = false;
        Ok(())
    }

    fn dequeue_frame(&mut self) -> Result<CsiFrame, HalError> {
        if !self.capturing {
            return Err(HalError::InitFailed);
        }
        // Read DMA status to check completion.
        let _status = self.read_reg(UNICAM_DCS)?;
        Err(HalError::MmioError)
    }

    fn enqueue_frame(&mut self, buffer_index: u8) -> Result<(), HalError> {
        let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
        crate::camera::validate_buffer_index(buffer_index, config.num_buffers)?;

        // Program image buffer start/end addresses.
        let addr = self.buffer_addrs[buffer_index as usize];
        let end_addr = addr + self.frame_size as u64;
        let (sa_reg, ea_reg) = if buffer_index.is_multiple_of(2) {
            (UNICAM_IBSA0, UNICAM_IBEA0)
        } else {
            (UNICAM_IBSA1, UNICAM_IBEA1)
        };
        self.write_reg(sa_reg, addr as u32)?;
        self.write_reg(ea_reg, end_addr as u32)
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
        let status = self.read_reg(UNICAM_ISTA)?;

        if status & ICTL_OFO != 0 {
            return Ok(CsiIrqSource::Overflow);
        }
        if status & ICTL_CRCE != 0 {
            return Ok(CsiIrqSource::CrcError);
        }
        if status & ICTL_FEI != 0 {
            let buf_idx = self.current_buffer;
            self.sequence += 1;
            let config = self.config.as_ref().ok_or(HalError::InitFailed)?;
            self.current_buffer = (self.current_buffer + 1) % config.num_buffers;
            return Ok(CsiIrqSource::FrameComplete(buf_idx));
        }
        Err(HalError::InterruptError)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(UNICAM_CTRL, 0)?;
        self.write_reg(UNICAM_ICTL, 0)?;
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
            num_buffers: 2,
        }
    }

    #[test]
    fn test_unicam_new() {
        let drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.base_addr(), 0x7E80_0000);
        assert!(!drv.is_configured());
        assert!(!drv.is_capturing());
    }

    #[test]
    fn test_build_data_id() {
        let config = make_config();
        assert_eq!(BroadcomUnicam::build_data_id(&config), 0x2B); // RAW10

        let mut cfg2 = config;
        cfg2.pixel_format = CsiPixelFormat::Yuv422;
        assert_eq!(BroadcomUnicam::build_data_id(&cfg2), 0x1E);
    }

    #[test]
    fn test_compute_line_stride_raw10() {
        let config = make_config();
        let stride = BroadcomUnicam::compute_line_stride(&config);
        // 1920*10/8 = 2400, aligned to 32 = 2400 (already aligned).
        assert_eq!(stride, 2400);
    }

    #[test]
    fn test_compute_line_stride_rgb888() {
        let mut config = make_config();
        config.pixel_format = CsiPixelFormat::Rgb888;
        let stride = BroadcomUnicam::compute_line_stride(&config);
        // 1920*3 = 5760, aligned to 32 = 5760.
        assert_eq!(stride, 5760);
    }

    #[test]
    fn test_init_rejects_4_lanes() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut config = make_config();
        config.lanes = 4; // Unicam only supports up to 2 lanes.
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_returns_mmio_error_on_valid_config() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        // Valid config but MMIO is stubbed.
        assert_eq!(drv.init(make_config()), Err(HalError::MmioError));
    }

    #[test]
    fn test_start_capture_not_configured() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.start_capture(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_buffer_size_default() {
        let drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.buffer_size(), 0);
    }

    // ----- Constructor and field access tests -----

    #[test]
    fn test_unicam_new_different_params() {
        let drv = BroadcomUnicam::new(0x7E80_1000, 100);
        assert_eq!(drv.base_addr(), 0x7E80_1000);
        assert_eq!(drv.irq, 100);
        assert_eq!(drv.sequence, 0);
        assert_eq!(drv.current_buffer, 0);
        assert_eq!(drv.frame_size, 0);
        assert_eq!(drv.buffer_addrs, [0u64; 4]);
        assert!(drv.config.is_none());
    }

    #[test]
    fn test_unicam_new_zero_base() {
        let drv = BroadcomUnicam::new(0, 0);
        assert_eq!(drv.base_addr(), 0);
        assert!(!drv.is_configured());
        assert!(!drv.is_capturing());
    }

    // ----- Lane configuration validation tests -----

    #[test]
    fn test_init_rejects_0_lanes() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.lanes = 0;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_rejects_3_lanes() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.lanes = 3;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_1_lane_returns_mmio_error() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.lanes = 1;
        // Valid lane count, but MMIO is stubbed.
        assert_eq!(drv.init(cfg), Err(HalError::MmioError));
    }

    #[test]
    fn test_init_2_lanes_returns_mmio_error() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let cfg = make_config(); // 2 lanes
        assert_eq!(drv.init(cfg), Err(HalError::MmioError));
    }

    #[test]
    fn test_init_invalid_width() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.width = 0;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_height() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.height = 5000;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_fps_zero() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.fps = 0;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_fps_too_high() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.fps = 121;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_num_buffers_zero() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.num_buffers = 0;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_num_buffers_too_many() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let mut cfg = make_config();
        cfg.num_buffers = 5;
        assert_eq!(drv.init(cfg), Err(HalError::InvalidConfig));
    }

    // ----- Data ID / pixel format tests -----

    #[test]
    fn test_build_data_id_raw8() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw8;
        assert_eq!(BroadcomUnicam::build_data_id(&cfg), 0x2A);
    }

    #[test]
    fn test_build_data_id_raw12() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw12;
        assert_eq!(BroadcomUnicam::build_data_id(&cfg), 0x2C);
    }

    #[test]
    fn test_build_data_id_rgb888() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Rgb888;
        assert_eq!(BroadcomUnicam::build_data_id(&cfg), 0x24);
    }

    // ----- Line stride computation tests -----

    #[test]
    fn test_compute_line_stride_raw8() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw8;
        cfg.width = 1920;
        let stride = BroadcomUnicam::compute_line_stride(&cfg);
        // 1920 * 8 / 8 = 1920, aligned to 32 = 1920.
        assert_eq!(stride, 1920);
        assert_eq!(stride % 32, 0);
    }

    #[test]
    fn test_compute_line_stride_raw12() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw12;
        cfg.width = 1920;
        let stride = BroadcomUnicam::compute_line_stride(&cfg);
        // 1920 * 12 / 8 = 2880, aligned to 32 = 2880.
        assert_eq!(stride, 2880);
        assert_eq!(stride % 32, 0);
    }

    #[test]
    fn test_compute_line_stride_yuv422() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Yuv422;
        cfg.width = 1920;
        let stride = BroadcomUnicam::compute_line_stride(&cfg);
        // 1920 * 16 / 8 = 3840, aligned to 32 = 3840.
        assert_eq!(stride, 3840);
        assert_eq!(stride % 32, 0);
    }

    #[test]
    fn test_compute_line_stride_alignment_needed() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw8;
        cfg.width = 100;
        let stride = BroadcomUnicam::compute_line_stride(&cfg);
        // 100 * 8 / 8 = 100, next multiple of 32 = 128.
        assert_eq!(stride, 128);
        assert_eq!(stride % 32, 0);
    }

    #[test]
    fn test_compute_line_stride_small_width() {
        let mut cfg = make_config();
        cfg.pixel_format = CsiPixelFormat::Raw8;
        cfg.width = 16; // Minimum valid width.
        let stride = BroadcomUnicam::compute_line_stride(&cfg);
        // 16 * 8 / 8 = 16, next multiple of 32 = 32.
        assert_eq!(stride, 32);
        assert_eq!(stride % 32, 0);
    }

    // ----- Frame capture control / state machine tests -----

    #[test]
    fn test_stop_capture_not_configured() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.stop_capture(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_dequeue_frame_not_capturing() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.dequeue_frame(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_enqueue_frame_not_configured() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.enqueue_frame(0), Err(HalError::InitFailed));
    }

    #[test]
    fn test_start_capture_with_manual_configured_flag() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.configured = true;
        // write_reg stubs return MmioError.
        assert_eq!(drv.start_capture(), Err(HalError::MmioError));
    }

    #[test]
    fn test_stop_capture_with_manual_configured_flag() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.configured = true;
        assert_eq!(drv.stop_capture(), Err(HalError::MmioError));
    }

    #[test]
    fn test_capturing_flag_not_set_without_config() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let _ = drv.start_capture();
        assert!(!drv.is_capturing());
    }

    #[test]
    fn test_configured_flag_not_set_on_mmio_failure() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let _ = drv.init(make_config());
        assert!(!drv.is_configured());
        assert!(drv.config.is_none());
    }

    // ----- Interrupt handling tests -----

    #[test]
    fn test_irq_handler_returns_mmio_error() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.irq_handler(), Err(HalError::MmioError));
    }

    #[test]
    fn test_reset_returns_mmio_error() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.reset(), Err(HalError::MmioError));
    }

    #[test]
    fn test_read_reg_always_returns_mmio_error() {
        let drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.read_reg(0), Err(HalError::MmioError));
        assert_eq!(drv.read_reg(UNICAM_ISTA), Err(HalError::MmioError));
    }

    #[test]
    fn test_write_reg_always_returns_mmio_error() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.write_reg(0, 0), Err(HalError::MmioError));
        assert_eq!(drv.write_reg(UNICAM_CTRL, 0xFF), Err(HalError::MmioError));
    }

    // ----- Buffer management state tests -----

    #[test]
    fn test_buffer_address_not_configured() {
        let drv = BroadcomUnicam::new(0x7E80_0000, 44);
        assert_eq!(drv.buffer_address(0), Err(HalError::InitFailed));
    }

    #[test]
    fn test_buffer_address_with_config() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.config = Some(make_config());
        drv.buffer_addrs[0] = 0xA000_0000;
        drv.buffer_addrs[1] = 0xA010_0000;
        assert_eq!(drv.buffer_address(0), Ok(0xA000_0000));
        assert_eq!(drv.buffer_address(1), Ok(0xA010_0000));
    }

    #[test]
    fn test_buffer_address_out_of_range() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.config = Some(make_config()); // num_buffers = 2
        assert_eq!(drv.buffer_address(2), Err(HalError::OutOfRange));
        assert_eq!(drv.buffer_address(3), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_enqueue_frame_with_config_returns_mmio() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.config = Some(make_config());
        drv.buffer_addrs[0] = 0xB000_0000;
        assert_eq!(drv.enqueue_frame(0), Err(HalError::MmioError));
    }

    #[test]
    fn test_enqueue_frame_buffer_out_of_range() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.config = Some(make_config()); // num_buffers = 2
        assert_eq!(drv.enqueue_frame(2), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_frame_size_stays_zero_on_failed_init() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let _ = drv.init(make_config());
        assert_eq!(drv.buffer_size(), 0);
    }

    #[test]
    fn test_manual_frame_size_tracking() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.frame_size = crate::camera::compute_frame_size(&make_config());
        assert_eq!(drv.buffer_size(), (1920 * 1080 * 10 + 7) / 8);
    }

    #[test]
    fn test_current_buffer_wraps_around() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        drv.config = Some(make_config()); // num_buffers = 2
        drv.current_buffer = 1;
        let next = (drv.current_buffer + 1) % drv.config.as_ref().unwrap().num_buffers;
        assert_eq!(next, 0);
    }

    #[test]
    fn test_sequence_counter_stays_zero_on_failed_init() {
        let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
        let _ = drv.init(make_config());
        assert_eq!(drv.sequence, 0);
    }

    #[test]
    fn test_buffer_size_for_all_pixel_formats() {
        let formats_and_sizes: &[(CsiPixelFormat, usize)] = &[
            (CsiPixelFormat::Raw8, 1920 * 1080),
            (CsiPixelFormat::Raw10, (1920 * 1080 * 10 + 7) / 8),
            (CsiPixelFormat::Raw12, (1920 * 1080 * 12 + 7) / 8),
            (CsiPixelFormat::Yuv422, 1920 * 1080 * 2),
            (CsiPixelFormat::Rgb888, 1920 * 1080 * 3),
        ];
        for &(fmt, expected) in formats_and_sizes {
            let mut cfg = make_config();
            cfg.pixel_format = fmt;
            let mut drv = BroadcomUnicam::new(0x7E80_0000, 44);
            drv.frame_size = crate::camera::compute_frame_size(&cfg);
            assert_eq!(drv.buffer_size(), expected);
        }
    }
}
