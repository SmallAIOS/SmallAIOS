// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra X1 Display Controller (DC0) driver.
//!
//! Programs DC0 registers to scan a linear framebuffer via DMA to the
//! display output (SOR0/HDMI). Uses Window A in RGBA8888 format.
//!
//! Reference: Tegra X1 TRM, chapter 32 (Display Controller).
//!
//! All register offsets are relative to `DC0_BASE` (`0x5420_0000`).
//! All MMIO operations are volatile and marked `unsafe`.

#![allow(dead_code)]

use crate::platform;

// Re-export VideoMode so tegra_dc can accept it as a parameter.
// VideoMode is defined in tegra_edid to keep the EDID/mode logic together.
#[cfg(not(test))]
use crate::tegra_edid::VideoMode;

// ─── DC0 base address ──────────────────────────────────────────────────────

const DC0_BASE: usize = platform::DC0_BASE;
const CAR_BASE: usize = platform::CAR_BASE;

// ─── DC0 register offsets (TRM chapter 32) ─────────────────────────────────

/// Display controller control register (enable/disable scanning).
pub const DC_DISP_CTRL: usize = 0x042 * 4;
/// Horizontal timing register: front porch + sync + back porch.
pub const DC_H_TIMING: usize = 0x044 * 4;
/// Vertical timing register: front porch + sync + back porch.
pub const DC_V_TIMING: usize = 0x045 * 4;
/// Display size register: active width | (active height << 16).
pub const DC_DI_SIZE: usize = 0x043 * 4;
/// Window A start address (framebuffer physical address).
pub const DC_WIN_A_START_ADDR: usize = 0x0800 * 4;
/// Window A size: width | (height << 16).
pub const DC_WIN_A_SIZE: usize = 0x0801 * 4;
/// Window A line stride in bytes (must be 64-byte aligned).
pub const DC_WIN_A_LINE_STRIDE: usize = 0x0802 * 4;
/// Window A byte swap / pixel format control.
pub const DC_WIN_A_BYTE_SWAP: usize = 0x0803 * 4;
/// Window A buffer control (format bits, enable).
pub const DC_WIN_A_BUFFER_CTRL: usize = 0x0804 * 4;

// ─── DISP_CTRL bits ────────────────────────────────────────────────────────

/// Enable bit in DC_DISP_CTRL: starts framebuffer DMA scanning.
const DISP_CTRL_ENABLE: u32 = 1 << 0;

// ─── CAR offsets for DISP0 ─────────────────────────────────────────────────

/// CLK_ENB_L_SET — set-only clock enable register (bank L).
const CAR_CLK_ENB_L_SET: usize = 0x320;
/// RST_DEV_L_CLR — clear-only reset deassert register (bank L).
const CAR_RST_DEV_L_CLR: usize = 0x304;
/// DISP0 clock bit position in bank L.
const CAR_CLK_DISP0: u32 = 1 << 27;

// ─── Pixel format ──────────────────────────────────────────────────────────

/// Supported pixel formats for Window A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PixelFormat {
    /// 24-bit RGB (3 bytes per pixel).
    RGB888 = 0x0,
    /// 16-bit RGB (2 bytes per pixel).
    RGB565 = 0x1,
    /// 32-bit RGBA (4 bytes per pixel) — recommended format.
    RGBA8888 = 0x3,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::RGB888 => 3,
            PixelFormat::RGB565 => 2,
            PixelFormat::RGBA8888 => 4,
        }
    }
}

// ─── MMIO helpers ──────────────────────────────────────────────────────────

/// Read a 32-bit DC0 register.
///
/// # Safety
/// DC0 clocks must be enabled. `offset` must be a valid DC register offset.
#[inline(always)]
unsafe fn dc_read32(offset: usize) -> u32 {
    let ptr = (DC0_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit DC0 register.
///
/// # Safety
/// DC0 clocks must be enabled. `offset` must be a valid DC register offset.
#[inline(always)]
unsafe fn dc_write32(offset: usize, value: u32) {
    let ptr = (DC0_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

/// Read a 32-bit CAR register.
///
/// # Safety
/// CAR is always accessible.
#[inline(always)]
unsafe fn car_read(offset: usize) -> u32 {
    let ptr = (CAR_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit CAR register.
///
/// # Safety
/// Incorrect CAR writes can hang the SoC.
#[inline(always)]
unsafe fn car_write(offset: usize, value: u32) {
    let ptr = (CAR_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

// ─── Stride alignment ──────────────────────────────────────────────────────

/// Align a stride value up to the next 64-byte boundary.
///
/// DC0 requires the framebuffer line stride to be 64-byte aligned.
pub const fn align_stride_64(stride: usize) -> usize {
    (stride + 63) & !63
}

// ─── Clock and reset ───────────────────────────────────────────────────────

/// Enable DISP0 clock via CAR and deassert its reset.
///
/// Uses set-only / clear-only registers so no read-modify-write is needed.
///
/// # Safety
/// Must be called before any DC0 register access.
pub unsafe fn dc_enable_clock() {
    // Enable DISP0 clock
    car_write(CAR_CLK_ENB_L_SET, CAR_CLK_DISP0);
    // Deassert DISP0 reset
    car_write(CAR_RST_DEV_L_CLR, CAR_CLK_DISP0);
    // Small delay for clock stabilization
    for _ in 0..1000 {
        core::hint::spin_loop();
    }
}

// ─── Framebuffer configuration ─────────────────────────────────────────────

/// Configure DC0 Window A with the framebuffer parameters.
///
/// Programs the start address, dimensions, stride, and pixel format.
/// The stride is automatically rounded up to 64-byte alignment.
///
/// # Safety
/// DC0 clocks must be enabled via `dc_enable_clock()`.
pub unsafe fn dc_set_framebuffer(addr: usize, width: u32, height: u32, format: PixelFormat) {
    let bpp = format.bytes_per_pixel() as u32;
    let stride = align_stride_64((width * bpp) as usize) as u32;

    // Framebuffer physical address
    dc_write32(DC_WIN_A_START_ADDR, addr as u32);
    // Window dimensions: width | (height << 16)
    dc_write32(DC_WIN_A_SIZE, width | (height << 16));
    // Line stride (64-byte aligned)
    dc_write32(DC_WIN_A_LINE_STRIDE, stride);
    // Buffer control: pixel format code
    dc_write32(DC_WIN_A_BUFFER_CTRL, format as u32);
}

// ─── Timing configuration ──────────────────────────────────────────────────

/// Configure DC0 display timing from a VideoMode.
///
/// Programs horizontal timing (front porch, sync, back porch),
/// vertical timing, and active display size.
///
/// # Safety
/// DC0 clocks must be enabled via `dc_enable_clock()`.
#[cfg(not(test))]
pub unsafe fn dc_set_timing(mode: &VideoMode) {
    // Active display size
    dc_write32(DC_DI_SIZE, mode.width as u32 | ((mode.height as u32) << 16));

    // Horizontal timing: front_porch | (sync << 8) | (back_porch << 16)
    let h_timing = (mode.h_front_porch as u32)
        | ((mode.h_sync_width as u32) << 8)
        | ((mode.h_back_porch as u32) << 16);
    dc_write32(DC_H_TIMING, h_timing);

    // Vertical timing: front_porch | (sync << 8) | (back_porch << 16)
    let v_timing = (mode.v_front_porch as u32)
        | ((mode.v_sync_width as u32) << 8)
        | ((mode.v_back_porch as u32) << 16);
    dc_write32(DC_V_TIMING, v_timing);
}

// ─── Enable / disable ──────────────────────────────────────────────────────

/// Enable DC0 display output (start DMA scanning).
///
/// # Safety
/// Framebuffer and timing must be configured before calling this.
pub unsafe fn dc_enable() {
    let ctrl = dc_read32(DC_DISP_CTRL);
    dc_write32(DC_DISP_CTRL, ctrl | DISP_CTRL_ENABLE);
}

/// Disable DC0 display output (stop DMA scanning).
///
/// # Safety
/// Safe to call at any time after clocks are enabled.
pub unsafe fn dc_disable() {
    let ctrl = dc_read32(DC_DISP_CTRL);
    dc_write32(DC_DISP_CTRL, ctrl & !DISP_CTRL_ENABLE);
}

// ─── Full init sequence ────────────────────────────────────────────────────

/// Initialize DC0: enable clock, set framebuffer, set timing, enable output.
///
/// This is the top-level entry point for display controller setup.
///
/// # Safety
/// Must be called exactly once during boot.
#[cfg(not(test))]
pub unsafe fn dc_init(mode: &VideoMode, fb_addr: usize) {
    dc_enable_clock();
    dc_set_framebuffer(
        fb_addr,
        mode.width as u32,
        mode.height as u32,
        PixelFormat::RGBA8888,
    );
    dc_set_timing(mode);
    dc_enable();
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Register offsets ──────────────────────────────────────────────

    #[test]
    fn test_dc_register_offsets() {
        assert_eq!(DC_DISP_CTRL, 0x042 * 4);
        assert_eq!(DC_H_TIMING, 0x044 * 4);
        assert_eq!(DC_V_TIMING, 0x045 * 4);
        assert_eq!(DC_DI_SIZE, 0x043 * 4);
        assert_eq!(DC_WIN_A_START_ADDR, 0x0800 * 4);
        assert_eq!(DC_WIN_A_SIZE, 0x0801 * 4);
        assert_eq!(DC_WIN_A_LINE_STRIDE, 0x0802 * 4);
        assert_eq!(DC_WIN_A_BYTE_SWAP, 0x0803 * 4);
        assert_eq!(DC_WIN_A_BUFFER_CTRL, 0x0804 * 4);
    }

    #[test]
    fn test_disp_ctrl_enable_bit() {
        assert_eq!(DISP_CTRL_ENABLE, 1 << 0);
    }

    // ─── CAR constants ─────────────────────────────────────────────────

    #[test]
    fn test_car_disp0_clock_bit() {
        assert_eq!(CAR_CLK_DISP0, 1 << 27);
    }

    #[test]
    fn test_car_disp0_register_offsets() {
        assert_eq!(CAR_CLK_ENB_L_SET, 0x320);
        assert_eq!(CAR_RST_DEV_L_CLR, 0x304);
    }

    // ─── Pixel format ──────────────────────────────────────────────────

    #[test]
    fn test_pixel_format_codes() {
        assert_eq!(PixelFormat::RGB888 as u32, 0x0);
        assert_eq!(PixelFormat::RGB565 as u32, 0x1);
        assert_eq!(PixelFormat::RGBA8888 as u32, 0x3);
    }

    #[test]
    fn test_pixel_format_bpp() {
        assert_eq!(PixelFormat::RGB888.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::RGB565.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::RGBA8888.bytes_per_pixel(), 4);
    }

    // ─── Stride alignment ──────────────────────────────────────────────

    #[test]
    fn test_stride_alignment_already_aligned() {
        // 1920 * 4 = 7680, which is 64-byte aligned (7680 / 64 = 120)
        assert_eq!(align_stride_64(7680), 7680);
    }

    #[test]
    fn test_stride_alignment_needs_rounding() {
        // 1920 * 3 = 5760 -> next 64-byte boundary = 5824
        assert_eq!(align_stride_64(5760), 5824);
    }

    #[test]
    fn test_stride_alignment_zero() {
        assert_eq!(align_stride_64(0), 0);
    }

    #[test]
    fn test_stride_alignment_one() {
        assert_eq!(align_stride_64(1), 64);
    }

    #[test]
    fn test_stride_alignment_63() {
        assert_eq!(align_stride_64(63), 64);
    }

    #[test]
    fn test_stride_alignment_64() {
        assert_eq!(align_stride_64(64), 64);
    }

    #[test]
    fn test_stride_alignment_65() {
        assert_eq!(align_stride_64(65), 128);
    }

    // ─── WIN_A_SIZE packing ────────────────────────────────────────────

    #[test]
    fn test_win_a_size_packing_1080p() {
        let width: u32 = 1920;
        let height: u32 = 1080;
        let packed = width | (height << 16);
        assert_eq!(packed & 0xFFFF, 1920);
        assert_eq!((packed >> 16) & 0xFFFF, 1080);
    }

    #[test]
    fn test_win_a_size_packing_720p() {
        let width: u32 = 1280;
        let height: u32 = 720;
        let packed = width | (height << 16);
        assert_eq!(packed & 0xFFFF, 1280);
        assert_eq!((packed >> 16) & 0xFFFF, 720);
    }

    // ─── 1080p60 timing values (CEA-861) ───────────────────────────────

    #[test]
    fn test_1080p60_h_timing_packing() {
        // CEA-861 1080p60: H front porch=88, sync=44, back porch=148
        let h_fp: u32 = 88;
        let h_sync: u32 = 44;
        let h_bp: u32 = 148;
        let h_timing = h_fp | (h_sync << 8) | (h_bp << 16);
        assert_eq!(h_timing & 0xFF, 88);
        assert_eq!((h_timing >> 8) & 0xFF, 44);
        assert_eq!((h_timing >> 16) & 0xFFFF, 148);
    }

    #[test]
    fn test_1080p60_v_timing_packing() {
        // CEA-861 1080p60: V front porch=4, sync=5, back porch=36
        let v_fp: u32 = 4;
        let v_sync: u32 = 5;
        let v_bp: u32 = 36;
        let v_timing = v_fp | (v_sync << 8) | (v_bp << 16);
        assert_eq!(v_timing & 0xFF, 4);
        assert_eq!((v_timing >> 8) & 0xFF, 5);
        assert_eq!((v_timing >> 16) & 0xFFFF, 36);
    }

    // ─── 1080p stride ──────────────────────────────────────────────────

    #[test]
    fn test_1080p_rgba8888_stride() {
        // 1920 * 4 = 7680 bytes, already 64-byte aligned
        let stride = align_stride_64(1920 * PixelFormat::RGBA8888.bytes_per_pixel());
        assert_eq!(stride, 7680);
    }

    #[test]
    fn test_1080p_rgb888_stride() {
        // 1920 * 3 = 5760 bytes -> 5824 (next 64-byte boundary)
        let stride = align_stride_64(1920 * PixelFormat::RGB888.bytes_per_pixel());
        assert_eq!(stride, 5824);
    }

    #[test]
    fn test_720p_rgba8888_stride() {
        // 1280 * 4 = 5120 bytes, already 64-byte aligned
        let stride = align_stride_64(1280 * PixelFormat::RGBA8888.bytes_per_pixel());
        assert_eq!(stride, 5120);
    }
}
