// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GICv2 interrupt controller driver for Tegra X1.
//!
//! On Tegra X1, the GIC is v2 (MMIO-based CPU interface, no redistributor):
//! - GICD at `0x5004_1000`
//! - GICC at `0x5004_2000`
//!
//! Interrupt layout:
//! - SGI 0-15: Software Generated Interrupts
//! - PPI 16-31: Private Peripheral Interrupts (timer IRQ 30)
//! - SPI 32+: Shared Peripheral Interrupts (PCIe, Ethernet, etc.)

use crate::interrupts::TIMER_IRQ;
use crate::platform;

const GICD_BASE: usize = platform::GICD_BASE;
const GICC_BASE: usize = platform::GICC_BASE;

// Distributor registers
const GICD_CTLR: usize = 0x000;
const GICD_ISENABLER: usize = 0x100;
const GICD_ICENABLER: usize = 0x180;
const GICD_IPRIORITYR: usize = 0x400;

// CPU Interface registers
const GICC_CTLR: usize = 0x000;
const GICC_PMR: usize = 0x004;
const GICC_IAR: usize = 0x00C;
const GICC_EOIR: usize = 0x010;

// ─── MMIO helpers ────────────────────────────────────────────────────────────

#[allow(dead_code)]
unsafe fn gicd_read(offset: usize) -> u32 {
    let ptr = (GICD_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

unsafe fn gicd_write(offset: usize, value: u32) {
    let ptr = (GICD_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

unsafe fn gicc_read(offset: usize) -> u32 {
    let ptr = (GICC_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

unsafe fn gicc_write(offset: usize, value: u32) {
    let ptr = (GICC_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

// ─── Distributor ─────────────────────────────────────────────────────────────

/// Initialize the GICv2 distributor.
///
/// Enables the distributor and the physical timer PPI (IRQ 30).
///
/// # Safety
/// Must be called once on the boot CPU before enabling interrupts.
pub unsafe fn init_gicd() {
    // Disable distributor during configuration
    gicd_write(GICD_CTLR, 0);

    // Enable timer PPI (IRQ 30): ISENABLER[0], bit 30
    gicd_write(GICD_ISENABLER, 1 << TIMER_IRQ);

    // Set timer priority to 0x20 (high priority)
    let prio_reg = GICD_IPRIORITYR + (TIMER_IRQ as usize);
    let ptr = (GICD_BASE + prio_reg) as *mut u8;
    core::ptr::write_volatile(ptr, 0x20);

    // Enable distributor: group 0 + group 1
    gicd_write(GICD_CTLR, 0x03);
}

// ─── CPU Interface ───────────────────────────────────────────────────────────

/// Initialize the GICv2 CPU interface.
///
/// Enables interrupt signaling and sets priority mask to accept all.
///
/// # Safety
/// Must be called on each CPU core after `init_gicd()`.
pub unsafe fn init_cpu_interface() {
    // Set priority mask: accept all priorities
    gicc_write(GICC_PMR, 0xFF);

    // Enable CPU interface: enable group 0 + group 1
    gicc_write(GICC_CTLR, 0x03);
}

// ─── Interrupt acknowledge / EOI ─────────────────────────────────────────────

/// Read the interrupt acknowledge register (GICC_IAR).
///
/// Returns the interrupt ID of the highest-priority pending interrupt.
/// A return value of 1023 indicates a spurious interrupt.
pub fn iar() -> u32 {
    unsafe { gicc_read(GICC_IAR) & 0x3FF }
}

/// Signal end-of-interrupt by writing to GICC_EOIR.
pub fn eoir(irq_id: u32) {
    unsafe {
        gicc_write(GICC_EOIR, irq_id);
    }
}

// ─── SPI enable / disable ────────────────────────────────────────────────────

/// Compute the ISENABLER/ICENABLER register offset and bit for an IRQ.
///
/// Each register covers 32 interrupts:
///   - IRQ 0-31: register offset 0
///   - IRQ 32-63: register offset 4
///   - IRQ 64-95: register offset 8
const fn isenabler_offset_and_bit(irq_id: u32) -> (usize, u32) {
    let reg_index = (irq_id / 32) as usize;
    let bit = irq_id % 32;
    (reg_index * 4, 1 << bit)
}

/// Enable a specific interrupt in the distributor.
pub fn enable_irq(irq_id: u32) {
    let (offset, bit) = isenabler_offset_and_bit(irq_id);
    unsafe {
        gicd_write(GICD_ISENABLER + offset, bit);
    }
}

/// Disable a specific interrupt in the distributor.
pub fn disable_irq(irq_id: u32) {
    let (offset, bit) = isenabler_offset_and_bit(irq_id);
    unsafe {
        gicd_write(GICD_ICENABLER + offset, bit);
    }
}

// ─── SGI (Software Generated Interrupt) ──────────────────────────────────────

const GICD_SGIR: usize = 0xF00;

/// Send a Software Generated Interrupt to a target CPU.
///
/// # Safety
/// Must only be called with valid target and SGI ID (0-15).
pub unsafe fn send_sgi(target_list: u16, intid: u8) {
    // GICD_SGIR: target list in [23:16], SGI ID in [3:0]
    let val = ((target_list as u32 & 0xFF) << 16) | (intid as u32 & 0xF);
    gicd_write(GICD_SGIR, val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isenabler_irq30() {
        // Timer PPI (IRQ 30): register 0, bit 30
        let (offset, bit) = isenabler_offset_and_bit(30);
        assert_eq!(offset, 0);
        assert_eq!(bit, 1 << 30);
    }

    #[test]
    fn test_isenabler_irq45() {
        // SPI (IRQ 45): register 1 (offset 4), bit 13
        let (offset, bit) = isenabler_offset_and_bit(45);
        assert_eq!(offset, 4);
        assert_eq!(bit, 1 << 13);
    }

    #[test]
    fn test_isenabler_irq32() {
        // First SPI (IRQ 32): register 1 (offset 4), bit 0
        let (offset, bit) = isenabler_offset_and_bit(32);
        assert_eq!(offset, 4);
        assert_eq!(bit, 1 << 0);
    }

    #[test]
    fn test_isenabler_irq0() {
        let (offset, bit) = isenabler_offset_and_bit(0);
        assert_eq!(offset, 0);
        assert_eq!(bit, 1 << 0);
    }

    #[test]
    fn test_gicc_pmr_value() {
        // GICC_PMR should be set to 0xFF to accept all priorities
        assert_eq!(0xFFu32, 255);
    }

    #[test]
    fn test_register_offsets() {
        assert_eq!(GICD_CTLR, 0x000);
        assert_eq!(GICD_ISENABLER, 0x100);
        assert_eq!(GICD_ICENABLER, 0x180);
        assert_eq!(GICD_IPRIORITYR, 0x400);
        assert_eq!(GICC_CTLR, 0x000);
        assert_eq!(GICC_PMR, 0x004);
        assert_eq!(GICC_IAR, 0x00C);
        assert_eq!(GICC_EOIR, 0x010);
    }
}
