// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra X1 EDID reader and video mode detection via DPAUX1/DDC.
//!
//! Reads the 128-byte base EDID block from an HDMI monitor via the
//! DPAUX1 controller in I2C/DDC mode, parses the preferred timing
//! descriptor, and provides safe fallback to standard modes.
//!
//! Reference: VESA EDID Standard 1.4, CEA-861, Tegra X1 TRM chapter 35.

#![allow(dead_code)]

use crate::platform;

// ─── Base addresses ────────────────────────────────────────────────────────

const DPAUX1_BASE: usize = platform::DPAUX1_BASE;
const CAR_BASE: usize = platform::CAR_BASE;

// ─── DPAUX1 register offsets ───────────────────────────────────────────────

/// DPAUX hybrid pad control register (select I2C/DDC vs DP AUX mode).
pub const DPAUX_HYBRID_PADCTL: usize = 0x124;
/// DPAUX AUX data register (read/write transaction data).
pub const DPAUX_DP_AUXDATA: usize = 0x014;
/// DPAUX AUX address register (I2C slave address / AUX address).
pub const DPAUX_DP_AUXADDR: usize = 0x018;
/// DPAUX AUX control register (transaction type, size, start).
pub const DPAUX_DP_AUXCTL: usize = 0x010;
/// DPAUX AUX status register (transaction complete, error flags).
pub const DPAUX_DP_AUXSTAT: usize = 0x01C;

// ─── DPAUX control bits ───────────────────────────────────────────────────

/// AUXCTL: start transaction.
const DPAUX_CTL_START: u32 = 1 << 0;
/// AUXCTL: I2C read transaction type.
const DPAUX_CTL_I2C_READ: u32 = 0x1 << 8;
/// AUXCTL: I2C write transaction type (for address setup).
const DPAUX_CTL_I2C_WRITE: u32 = 0x0 << 8;
/// AUXSTAT: transaction complete bit.
const DPAUX_STAT_DONE: u32 = 1 << 0;
/// AUXSTAT: transaction error (NAK or timeout).
const DPAUX_STAT_ERROR: u32 = 1 << 1;
/// Hybrid pad: I2C/DDC mode.
const DPAUX_PADCTL_I2C_MODE: u32 = 0x1;

// ─── CAR offsets for DPAUX1 ───────────────────────────────────────────────

/// CLK_ENB_W_SET — set-only clock enable register (bank W, for DPAUX).
const CAR_CLK_ENB_W_SET: usize = 0x448;
/// RST_DEV_W_CLR — clear-only reset deassert register (bank W).
const CAR_RST_DEV_W_CLR: usize = 0x43C;
/// DPAUX1 clock bit position in bank W.
const CAR_CLK_DPAUX1: u32 = 1 << 25;

// ─── DDC constants ─────────────────────────────────────────────────────────

/// DDC slave address for EDID (I2C address 0x50).
const DDC_EDID_ADDR: u8 = 0x50;
/// EDID block size in bytes.
const EDID_BLOCK_SIZE: usize = 128;
/// EDID magic header: first 8 bytes of a valid EDID block.
const EDID_MAGIC: [u8; 8] = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00];
/// Maximum bytes per AUX transaction.
const AUX_MAX_BYTES: usize = 16;
/// Maximum poll iterations for AUX transaction completion.
const AUX_TIMEOUT: u32 = 50_000;

// ─── Error type ────────────────────────────────────────────────────────────

/// Errors that can occur during EDID reading and parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdidError {
    /// DDC I2C transaction timed out (no monitor or unresponsive DDC bus).
    Timeout,
    /// EDID magic header `[0x00, 0xFF, ..., 0x00]` validation failed.
    InvalidMagic,
    /// Failed to parse timing descriptor from EDID data.
    ParseError,
    /// DDC slave did not acknowledge (NAK).
    DdcNak,
    /// EDID checksum validation failed (sum of 128 bytes mod 256 != 0).
    ChecksumError,
}

// ─── VideoMode ─────────────────────────────────────────────────────────────

/// Video display mode parameters (resolution + timing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoMode {
    /// Active horizontal pixels.
    pub width: u16,
    /// Active vertical lines.
    pub height: u16,
    /// Pixel clock in kHz.
    pub pixel_clock_khz: u32,
    /// Horizontal front porch (pixels).
    pub h_front_porch: u16,
    /// Horizontal sync width (pixels).
    pub h_sync_width: u16,
    /// Horizontal back porch (pixels).
    pub h_back_porch: u16,
    /// Vertical front porch (lines).
    pub v_front_porch: u16,
    /// Vertical sync width (lines).
    pub v_sync_width: u16,
    /// Vertical back porch (lines).
    pub v_back_porch: u16,
}

impl VideoMode {
    /// Standard CEA-861 1920x1080 at 60 Hz timing.
    pub const fn default_1080p() -> Self {
        Self {
            width: 1920,
            height: 1080,
            pixel_clock_khz: 148_500,
            h_front_porch: 88,
            h_sync_width: 44,
            h_back_porch: 148,
            v_front_porch: 4,
            v_sync_width: 5,
            v_back_porch: 36,
        }
    }

    /// Standard CEA-861 1280x720 at 60 Hz timing.
    pub const fn default_720p() -> Self {
        Self {
            width: 1280,
            height: 720,
            pixel_clock_khz: 74_250,
            h_front_porch: 110,
            h_sync_width: 40,
            h_back_porch: 220,
            v_front_porch: 5,
            v_sync_width: 5,
            v_back_porch: 20,
        }
    }
}

// ─── MMIO helpers ──────────────────────────────────────────────────────────

/// Read a 32-bit DPAUX1 register.
///
/// # Safety
/// DPAUX1 clocks must be enabled.
#[inline(always)]
unsafe fn dpaux_read32(offset: usize) -> u32 {
    let ptr = (DPAUX1_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit DPAUX1 register.
///
/// # Safety
/// DPAUX1 clocks must be enabled.
#[inline(always)]
unsafe fn dpaux_write32(offset: usize, value: u32) {
    let ptr = (DPAUX1_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

/// Write a 32-bit CAR register.
#[inline(always)]
unsafe fn car_write(offset: usize, value: u32) {
    let ptr = (CAR_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

// ─── DPAUX initialization ──────────────────────────────────────────────────

/// Initialize DPAUX1 for DDC/I2C mode.
///
/// Enables DPAUX1 clock, deasserts reset, and configures the hybrid pad
/// for I2C (DDC) operation.
///
/// # Safety
/// Must be called before any DDC transaction.
pub unsafe fn dpaux_init() {
    // Enable DPAUX1 clock
    car_write(CAR_CLK_ENB_W_SET, CAR_CLK_DPAUX1);
    // Deassert DPAUX1 reset
    car_write(CAR_RST_DEV_W_CLR, CAR_CLK_DPAUX1);
    // Small delay for clock stabilization
    for _ in 0..1000 {
        core::hint::spin_loop();
    }
    // Configure hybrid pad for I2C/DDC mode
    dpaux_write32(DPAUX_HYBRID_PADCTL, DPAUX_PADCTL_I2C_MODE);
}

// ─── DDC I2C transactions ──────────────────────────────────────────────────

/// Perform an I2C read via the DPAUX1 AUX controller.
///
/// Reads up to `buf.len()` bytes from the specified I2C slave address
/// starting at the given register offset.
///
/// # Safety
/// DPAUX1 must be initialized via `dpaux_init()`.
pub unsafe fn dpaux_i2c_read(slave_addr: u8, offset: u8, buf: &mut [u8]) -> Result<(), EdidError> {
    // Step 1: Write the offset (I2C write with address byte only)
    dpaux_write32(DPAUX_DP_AUXADDR, (slave_addr as u32) << 1);
    dpaux_write32(DPAUX_DP_AUXDATA, offset as u32);
    dpaux_write32(
        DPAUX_DP_AUXCTL,
        DPAUX_CTL_I2C_WRITE | DPAUX_CTL_START | 1, // 1 byte (offset)
    );

    // Poll for write completion
    for _ in 0..AUX_TIMEOUT {
        let stat = dpaux_read32(DPAUX_DP_AUXSTAT);
        if stat & DPAUX_STAT_ERROR != 0 {
            return Err(EdidError::DdcNak);
        }
        if stat & DPAUX_STAT_DONE != 0 {
            break;
        }
        core::hint::spin_loop();
    }

    // Step 2: Read data in chunks of AUX_MAX_BYTES
    let mut pos = 0;
    while pos < buf.len() {
        let chunk_len = core::cmp::min(AUX_MAX_BYTES, buf.len() - pos);

        dpaux_write32(DPAUX_DP_AUXADDR, ((slave_addr as u32) << 1) | 1);
        dpaux_write32(
            DPAUX_DP_AUXCTL,
            DPAUX_CTL_I2C_READ | DPAUX_CTL_START | (chunk_len as u32),
        );

        // Poll for read completion
        let mut done = false;
        for _ in 0..AUX_TIMEOUT {
            let stat = dpaux_read32(DPAUX_DP_AUXSTAT);
            if stat & DPAUX_STAT_ERROR != 0 {
                return Err(EdidError::DdcNak);
            }
            if stat & DPAUX_STAT_DONE != 0 {
                done = true;
                break;
            }
            core::hint::spin_loop();
        }

        if !done {
            return Err(EdidError::Timeout);
        }

        // Read data from AUX data register (packed 4 bytes at a time)
        for i in 0..chunk_len {
            let word = dpaux_read32(DPAUX_DP_AUXDATA + (i / 4) * 4);
            buf[pos + i] = ((word >> ((i % 4) * 8)) & 0xFF) as u8;
        }

        pos += chunk_len;
    }

    Ok(())
}

// ─── EDID read and validation ──────────────────────────────────────────────

/// Validate the EDID magic header (bytes 0-7).
pub fn validate_edid_magic(edid: &[u8; EDID_BLOCK_SIZE]) -> bool {
    edid[0..8] == EDID_MAGIC
}

/// Validate the EDID checksum (sum of all 128 bytes mod 256 must be 0).
pub fn validate_edid_checksum(edid: &[u8; EDID_BLOCK_SIZE]) -> bool {
    let sum: u8 = edid.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    sum == 0
}

/// Read the 128-byte base EDID block from the DDC bus.
///
/// Validates both the EDID magic header and checksum.
///
/// # Safety
/// DPAUX1 must be initialized.
pub unsafe fn read_edid() -> Result<[u8; EDID_BLOCK_SIZE], EdidError> {
    let mut edid = [0u8; EDID_BLOCK_SIZE];
    dpaux_i2c_read(DDC_EDID_ADDR, 0, &mut edid)?;

    if !validate_edid_magic(&edid) {
        return Err(EdidError::InvalidMagic);
    }

    if !validate_edid_checksum(&edid) {
        return Err(EdidError::ChecksumError);
    }

    Ok(edid)
}

// ─── EDID timing parsing ──────────────────────────────────────────────────

/// Parse the preferred detailed timing descriptor from EDID bytes 54-71.
///
/// EDID detailed timing descriptor format (18 bytes starting at offset 54):
///   [0:1] pixel clock in 10 kHz units (little-endian)
///   [2]   h_active low 8 bits
///   [3]   h_blanking low 8 bits
///   [4]   h_active high 4 bits | h_blanking high 4 bits
///   [5]   v_active low 8 bits
///   [6]   v_blanking low 8 bits
///   [7]   v_active high 4 bits | v_blanking high 4 bits
///   [8]   h_front_porch low 8 bits
///   [9]   h_sync_width low 8 bits
///   [10]  v_front_porch low 4 bits | v_sync_width low 4 bits
///   [11]  h_front_porch high 2 | h_sync high 2 | v_front_porch high 2 | v_sync high 2
pub fn parse_preferred_timing(edid: &[u8; EDID_BLOCK_SIZE]) -> Result<VideoMode, EdidError> {
    let dtd = &edid[54..72]; // First detailed timing descriptor

    // Pixel clock in 10 kHz units
    let pixel_clock_10khz = (dtd[0] as u32) | ((dtd[1] as u32) << 8);
    if pixel_clock_10khz == 0 {
        return Err(EdidError::ParseError);
    }
    let pixel_clock_khz = pixel_clock_10khz * 10;

    // Horizontal active and blanking
    let h_active = (dtd[2] as u16) | (((dtd[4] >> 4) as u16) << 8);
    let h_blanking = (dtd[3] as u16) | (((dtd[4] & 0x0F) as u16) << 8);

    // Vertical active and blanking
    let v_active = (dtd[5] as u16) | (((dtd[7] >> 4) as u16) << 8);
    let _v_blanking = (dtd[6] as u16) | (((dtd[7] & 0x0F) as u16) << 8);

    // Horizontal sync details
    let h_front_porch = (dtd[8] as u16) | (((dtd[11] >> 6) as u16) << 8);
    let h_sync_width = (dtd[9] as u16) | ((((dtd[11] >> 4) & 0x03) as u16) << 8);

    // Vertical sync details
    let v_front_porch = ((dtd[10] >> 4) as u16) | ((((dtd[11] >> 2) & 0x03) as u16) << 4);
    let v_sync_width = ((dtd[10] & 0x0F) as u16) | (((dtd[11] & 0x03) as u16) << 4);

    // Back porch = blanking - front_porch - sync_width
    let h_back_porch = h_blanking.saturating_sub(h_front_porch + h_sync_width);
    let v_back_porch = _v_blanking.saturating_sub(v_front_porch + v_sync_width);

    if h_active == 0 || v_active == 0 {
        return Err(EdidError::ParseError);
    }

    Ok(VideoMode {
        width: h_active,
        height: v_active,
        pixel_clock_khz,
        h_front_porch,
        h_sync_width,
        h_back_porch,
        v_front_porch,
        v_sync_width,
        v_back_porch,
    })
}

// ─── Mode detection with fallback ──────────────────────────────────────────

/// Detect the preferred video mode from the connected monitor.
///
/// Attempts to read EDID via DDC and parse the preferred timing.
/// Falls back to `VideoMode::default_1080p()` on any failure, logging
/// a warning via UART.
///
/// # Safety
/// DPAUX1 must be initialized.
#[cfg(not(test))]
pub unsafe fn detect_mode() -> VideoMode {
    match read_edid() {
        Ok(edid) => match parse_preferred_timing(&edid) {
            Ok(mode) => mode,
            Err(_) => {
                crate::uart::puts(
                    "[edid] WARN: failed to parse EDID timing, using 1080p fallback\n",
                );
                VideoMode::default_1080p()
            }
        },
        Err(_) => {
            crate::uart::puts("[edid] WARN: EDID read failed, using 1080p fallback\n");
            VideoMode::default_1080p()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── DPAUX register offsets ────────────────────────────────────────

    #[test]
    fn test_dpaux_register_offsets() {
        assert_eq!(DPAUX_HYBRID_PADCTL, 0x124);
        assert_eq!(DPAUX_DP_AUXDATA, 0x014);
        assert_eq!(DPAUX_DP_AUXADDR, 0x018);
        assert_eq!(DPAUX_DP_AUXCTL, 0x010);
        assert_eq!(DPAUX_DP_AUXSTAT, 0x01C);
    }

    #[test]
    fn test_dpaux_base_address() {
        assert_eq!(DPAUX1_BASE, 0x545C_0000);
    }

    // ─── VideoMode defaults ────────────────────────────────────────────

    #[test]
    fn test_videomode_default_1080p() {
        let m = VideoMode::default_1080p();
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert_eq!(m.pixel_clock_khz, 148_500);
        assert_eq!(m.h_front_porch, 88);
        assert_eq!(m.h_sync_width, 44);
        assert_eq!(m.h_back_porch, 148);
        assert_eq!(m.v_front_porch, 4);
        assert_eq!(m.v_sync_width, 5);
        assert_eq!(m.v_back_porch, 36);
    }

    #[test]
    fn test_videomode_default_720p() {
        let m = VideoMode::default_720p();
        assert_eq!(m.width, 1280);
        assert_eq!(m.height, 720);
        assert_eq!(m.pixel_clock_khz, 74_250);
        assert_eq!(m.h_front_porch, 110);
        assert_eq!(m.h_sync_width, 40);
        assert_eq!(m.h_back_porch, 220);
        assert_eq!(m.v_front_porch, 5);
        assert_eq!(m.v_sync_width, 5);
        assert_eq!(m.v_back_porch, 20);
    }

    #[test]
    fn test_1080p_total_h_blanking() {
        let m = VideoMode::default_1080p();
        // Total horizontal blanking = fp + sync + bp = 88 + 44 + 148 = 280
        let total_h = m.h_front_porch + m.h_sync_width + m.h_back_porch;
        assert_eq!(total_h, 280);
        // Total horizontal = active + blanking = 1920 + 280 = 2200
        assert_eq!(m.width as u32 + total_h as u32, 2200);
    }

    #[test]
    fn test_1080p_total_v_blanking() {
        let m = VideoMode::default_1080p();
        // Total vertical blanking = 4 + 5 + 36 = 45
        let total_v = m.v_front_porch + m.v_sync_width + m.v_back_porch;
        assert_eq!(total_v, 45);
        // Total vertical = 1080 + 45 = 1125
        assert_eq!(m.height as u32 + total_v as u32, 1125);
    }

    #[test]
    fn test_720p_total_h_blanking() {
        let m = VideoMode::default_720p();
        let total_h = m.h_front_porch + m.h_sync_width + m.h_back_porch;
        assert_eq!(total_h, 370);
        assert_eq!(m.width as u32 + total_h as u32, 1650);
    }

    // ─── EDID magic validation ─────────────────────────────────────────

    #[test]
    fn test_valid_edid_magic() {
        let mut edid = [0u8; 128];
        // Set valid magic header
        edid[0] = 0x00;
        edid[1] = 0xFF;
        edid[2] = 0xFF;
        edid[3] = 0xFF;
        edid[4] = 0xFF;
        edid[5] = 0xFF;
        edid[6] = 0xFF;
        edid[7] = 0x00;
        assert!(validate_edid_magic(&edid));
    }

    #[test]
    fn test_invalid_edid_magic_all_zeros() {
        let edid = [0u8; 128];
        assert!(!validate_edid_magic(&edid));
    }

    #[test]
    fn test_invalid_edid_magic_all_ff() {
        let edid = [0xFFu8; 128];
        assert!(!validate_edid_magic(&edid));
    }

    #[test]
    fn test_invalid_edid_magic_wrong_byte() {
        let mut edid = [0u8; 128];
        edid[0] = 0x00;
        edid[1] = 0xFF;
        edid[2] = 0xFF;
        edid[3] = 0xFF;
        edid[4] = 0xFF;
        edid[5] = 0xFF;
        edid[6] = 0x00; // Should be 0xFF
        edid[7] = 0x00;
        assert!(!validate_edid_magic(&edid));
    }

    // ─── EDID checksum validation ──────────────────────────────────────

    #[test]
    fn test_edid_checksum_valid() {
        let mut edid = [0u8; 128];
        // Set valid magic
        edid[0] = 0x00;
        for i in 1..7 {
            edid[i] = 0xFF;
        }
        edid[7] = 0x00;
        // Calculate checksum: make sum of all bytes = 0 mod 256
        let sum: u8 = edid[..127].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        edid[127] = 0u8.wrapping_sub(sum);
        assert!(validate_edid_checksum(&edid));
    }

    #[test]
    fn test_edid_checksum_invalid() {
        let mut edid = [0u8; 128];
        edid[0] = 0x01; // Non-zero byte
        edid[127] = 0x00; // Wrong checksum
        assert!(!validate_edid_checksum(&edid));
    }

    // ─── EDID 1080p timing parsing ─────────────────────────────────────

    /// Construct a valid EDID block with 1080p60 preferred timing.
    fn make_edid_1080p() -> [u8; 128] {
        let mut edid = [0u8; 128];

        // Magic header
        edid[0] = 0x00;
        for i in 1..7 {
            edid[i] = 0xFF;
        }
        edid[7] = 0x00;

        // Detailed Timing Descriptor at bytes 54-71 for 1080p60:
        // Pixel clock: 148500 kHz / 10 = 14850 (0x3A02)
        edid[54] = 0x02; // pixel clock low byte (14850 & 0xFF = 0x02)
        edid[55] = 0x3A; // pixel clock high byte (14850 >> 8 = 0x3A)

        // H active = 1920 = 0x780
        // H blanking = 280 = 0x118
        edid[56] = 0x80; // h_active low 8 bits (0x780 & 0xFF = 0x80)
        edid[57] = 0x18; // h_blanking low 8 bits (0x118 & 0xFF = 0x18)
        edid[58] = 0x71; // h_active high 4 = 0x7, h_blanking high 4 = 0x1

        // V active = 1080 = 0x438
        // V blanking = 45 = 0x02D
        edid[59] = 0x38; // v_active low 8 bits (0x438 & 0xFF = 0x38)
        edid[60] = 0x2D; // v_blanking low 8 bits (0x02D & 0xFF = 0x2D)
        edid[61] = 0x40; // v_active high 4 = 0x4, v_blanking high 4 = 0x0

        // H front porch = 88 = 0x058
        // H sync width = 44 = 0x02C
        edid[62] = 0x58; // h_front_porch low 8 bits
        edid[63] = 0x2C; // h_sync_width low 8 bits

        // V front porch = 4, V sync width = 5
        edid[64] = 0x45; // v_front_porch low 4 (4) | v_sync_width low 4 (5)

        // High bits: h_fp[1:0]=0, h_sync[1:0]=0, v_fp[1:0]=0, v_sync[1:0]=0
        edid[65] = 0x00;

        // Checksum
        let sum: u8 = edid[..127].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        edid[127] = 0u8.wrapping_sub(sum);

        edid
    }

    #[test]
    fn test_parse_1080p_edid_timing() {
        let edid = make_edid_1080p();
        let mode = parse_preferred_timing(&edid).unwrap();

        assert_eq!(mode.width, 1920);
        assert_eq!(mode.height, 1080);
        assert_eq!(mode.pixel_clock_khz, 148_500);
        assert_eq!(mode.h_front_porch, 88);
        assert_eq!(mode.h_sync_width, 44);
        assert_eq!(mode.h_back_porch, 148);
        assert_eq!(mode.v_front_porch, 4);
        assert_eq!(mode.v_sync_width, 5);
        assert_eq!(mode.v_back_porch, 36);
    }

    #[test]
    fn test_parse_1080p_edid_matches_default() {
        let edid = make_edid_1080p();
        let parsed = parse_preferred_timing(&edid).unwrap();
        let default = VideoMode::default_1080p();
        assert_eq!(parsed, default);
    }

    /// Construct a valid EDID block with 720p60 preferred timing.
    fn make_edid_720p() -> [u8; 128] {
        let mut edid = [0u8; 128];

        // Magic header
        edid[0] = 0x00;
        for i in 1..7 {
            edid[i] = 0xFF;
        }
        edid[7] = 0x00;

        // Pixel clock: 74250 kHz / 10 = 7425 (0x1D01)
        edid[54] = 0x01; // low byte
        edid[55] = 0x1D; // high byte

        // H active = 1280 = 0x500
        // H blanking = 370 = 0x172
        edid[56] = 0x00; // h_active low 8 bits
        edid[57] = 0x72; // h_blanking low 8 bits
        edid[58] = 0x51; // h_active high 4 = 0x5, h_blanking high 4 = 0x1

        // V active = 720 = 0x2D0
        // V blanking = 30 = 0x01E
        edid[59] = 0xD0; // v_active low 8 bits
        edid[60] = 0x1E; // v_blanking low 8 bits
        edid[61] = 0x20; // v_active high 4 = 0x2, v_blanking high 4 = 0x0

        // H front porch = 110 = 0x06E
        // H sync width = 40 = 0x028
        edid[62] = 0x6E; // h_front_porch low 8 bits
        edid[63] = 0x28; // h_sync_width low 8 bits

        // V front porch = 5, V sync width = 5
        edid[64] = 0x55; // v_front_porch low 4 (5) | v_sync_width low 4 (5)

        // High bits all zero
        edid[65] = 0x00;

        // Checksum
        let sum: u8 = edid[..127].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        edid[127] = 0u8.wrapping_sub(sum);

        edid
    }

    #[test]
    fn test_parse_720p_edid_timing() {
        let edid = make_edid_720p();
        let mode = parse_preferred_timing(&edid).unwrap();

        assert_eq!(mode.width, 1280);
        assert_eq!(mode.height, 720);
        assert_eq!(mode.pixel_clock_khz, 74_250);
        assert_eq!(mode.h_front_porch, 110);
        assert_eq!(mode.h_sync_width, 40);
        assert_eq!(mode.h_back_porch, 220);
        assert_eq!(mode.v_front_porch, 5);
        assert_eq!(mode.v_sync_width, 5);
        assert_eq!(mode.v_back_porch, 20);
    }

    #[test]
    fn test_parse_720p_edid_matches_default() {
        let edid = make_edid_720p();
        let parsed = parse_preferred_timing(&edid).unwrap();
        let default = VideoMode::default_720p();
        assert_eq!(parsed, default);
    }

    // ─── EDID parse error cases ────────────────────────────────────────

    #[test]
    fn test_parse_zero_pixel_clock_returns_error() {
        let mut edid = [0u8; 128];
        // Valid magic
        edid[0] = 0x00;
        for i in 1..7 {
            edid[i] = 0xFF;
        }
        edid[7] = 0x00;
        // Pixel clock = 0 at bytes 54-55
        edid[54] = 0x00;
        edid[55] = 0x00;
        let result = parse_preferred_timing(&edid);
        assert_eq!(result, Err(EdidError::ParseError));
    }

    // ─── EdidError enum ────────────────────────────────────────────────

    #[test]
    fn test_edid_error_variants() {
        assert_eq!(EdidError::Timeout, EdidError::Timeout);
        assert_eq!(EdidError::InvalidMagic, EdidError::InvalidMagic);
        assert_eq!(EdidError::ParseError, EdidError::ParseError);
        assert_eq!(EdidError::DdcNak, EdidError::DdcNak);
        assert_eq!(EdidError::ChecksumError, EdidError::ChecksumError);
        // All variants are distinct
        assert_ne!(EdidError::Timeout, EdidError::InvalidMagic);
        assert_ne!(EdidError::InvalidMagic, EdidError::ParseError);
        assert_ne!(EdidError::ParseError, EdidError::DdcNak);
        assert_ne!(EdidError::DdcNak, EdidError::ChecksumError);
    }

    // ─── EDID checksum with test vectors ───────────────────────────────

    #[test]
    fn test_1080p_edid_checksum() {
        let edid = make_edid_1080p();
        assert!(validate_edid_checksum(&edid));
        assert!(validate_edid_magic(&edid));
    }

    #[test]
    fn test_720p_edid_checksum() {
        let edid = make_edid_720p();
        assert!(validate_edid_checksum(&edid));
        assert!(validate_edid_magic(&edid));
    }

    // ─── DDC constants ─────────────────────────────────────────────────

    #[test]
    fn test_ddc_constants() {
        assert_eq!(DDC_EDID_ADDR, 0x50);
        assert_eq!(EDID_BLOCK_SIZE, 128);
    }

    #[test]
    fn test_edid_magic_bytes() {
        assert_eq!(EDID_MAGIC, [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]);
    }
}
