// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GICv3 interrupt controller driver.
//!
//! Used by the GICv3 platforms — QEMU `virt` (`qemu-virt`) and the
//! NVIDIA Jetson Orin family / Tegra234 (`tegra234`). The GICv2 driver
//! for Tegra X1 lives in the sibling [`crate::gicv2`] module.
//!
//! GICv3 differs from GICv2 in two ways this driver cares about:
//! - **No memory-mapped CPU interface.** The GICC MMIO frame is replaced
//!   by the `ICC_*_EL1` system registers (SRE/PMR/IGRPEN1/IAR1/EOIR1/
//!   SGI1R). They are addressed here by their raw `S3_0_C12_*`
//!   encodings (named in comments) so this compiles on the pinned
//!   nightly without relying on assembler register aliases.
//! - **Per-CPU redistributor.** SGI/PPI configuration (enable, priority)
//!   moves out of the distributor into the GICR SGI frame; the
//!   redistributor must be woken (clear `GICR_WAKER.ProcessorSleep`,
//!   wait for `ChildrenAsleep == 0`) before its banked interrupts work.
//!
//! Affinity-routed SPIs (INTID >= 32) are still configured in the
//! distributor. LPI/ITS are out of scope for Phase 2 (no MSI-capable
//! peripherals are exercised on the Orin bring-up path).
//!
//! Interrupt layout (shared with GICv2):
//! - SGI 0-15:  Software Generated Interrupts (IPI)
//! - PPI 16-31: Private Peripheral Interrupts (timer PPI = IRQ 30)
//! - SPI 32+:   Shared Peripheral Interrupts (I/O devices)
//!
//! The MMIO + system-register sequences here are the proven ones
//! previously inlined in `interrupts.rs` under `qemu-virt`; this module
//! extracts them (OpenSpec `unikernel-orin-bringup-v1` task 2.13) so the
//! `tegra234` platform can share the exact same code path. Wiring the
//! Generic Timer PPI through an IRQ exception vector into the scheduler
//! tick is a separate task (2.15).

use core::arch::asm;

use crate::interrupts::TIMER_IRQ;
use crate::platform;

const GICD_BASE: usize = platform::GICD_BASE;
const GICR_BASE: usize = platform::GICR_BASE;

// ─── Distributor (GICD) registers ────────────────────────────────────────────

const GICD_CTLR: usize = 0x000;
const GICD_ISENABLER: usize = 0x100;
const GICD_ICENABLER: usize = 0x180;
// SPI priority (`GICD_IPRIORITYR`) is intentionally not configured: no
// SPIs are exercised in Phase 2 (LPI/ITS/MSI out of scope), and the
// timer PPI's priority is set in the redistributor (see `init_gicr`).

// `GICD_CTLR` bits used here: EnableGrp0 (0), EnableGrp1NS (1),
// ARE_NS (4 — Affinity Routing Enable, required for GICv3 SPI routing).
const GICD_CTLR_ENABLE_ARE: u32 = (1 << 0) | (1 << 1) | (1 << 4);

// ─── Redistributor (GICR) registers ──────────────────────────────────────────
//
// Offsets are from the start of this CPU's redistributor frame. The SGI
// frame (ISENABLER0 / IPRIORITYR / ICENABLER0) sits at +0x10000; WAKER
// lives in the RD_base frame at +0x0014.

const GICR_WAKER: usize = 0x0014;
const GICR_ISENABLER0: usize = 0x10100;
const GICR_ICENABLER0: usize = 0x10180;
const GICR_IPRIORITYR0: usize = 0x10400;

// `GICR_WAKER` bits: ProcessorSleep (1), ChildrenAsleep (2).
const GICR_WAKER_PROCESSOR_SLEEP: u32 = 1 << 1;
const GICR_WAKER_CHILDREN_ASLEEP: u32 = 1 << 2;

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

unsafe fn gicr_read(offset: usize) -> u32 {
    let ptr = (GICR_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

unsafe fn gicr_write(offset: usize, value: u32) {
    let ptr = (GICR_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

// ─── Distributor ─────────────────────────────────────────────────────────────

/// Initialize the GICv3 distributor.
///
/// Disables the distributor, then re-enables it with affinity routing
/// (`ARE_NS`) plus Group 0 / Group 1 NS. SPI enable/priority is done
/// per-IRQ via [`enable_irq`]; SGI/PPI live in the redistributor (see
/// [`init_gicr`]).
///
/// # Safety
/// Must be called once on the boot CPU, with GICD mapped, before
/// enabling interrupts.
pub unsafe fn init_gicd() {
    gicd_write(GICD_CTLR, 0);
    gicd_write(GICD_CTLR, GICD_CTLR_ENABLE_ARE);
}

// ─── Redistributor ───────────────────────────────────────────────────────────

/// Initialize this CPU's GICv3 redistributor.
///
/// Wakes the redistributor (clears `ProcessorSleep`, waits for
/// `ChildrenAsleep` to clear), then enables the physical timer PPI
/// ([`TIMER_IRQ`]) and gives it a high priority (`0x20`).
///
/// # Safety
/// Must be called on each CPU after [`init_gicd`], with GICR mapped.
pub unsafe fn init_gicr() {
    let waker = gicr_read(GICR_WAKER);
    gicr_write(GICR_WAKER, waker & !GICR_WAKER_PROCESSOR_SLEEP);
    while gicr_read(GICR_WAKER) & GICR_WAKER_CHILDREN_ASLEEP != 0 {
        core::hint::spin_loop();
    }

    // Timer PPI is INTID < 32 → banked in the redistributor SGI frame.
    gicr_write(GICR_ISENABLER0, 1 << TIMER_IRQ);
    let prio_offset = GICR_IPRIORITYR0 + (TIMER_IRQ as usize);
    let ptr = (GICR_BASE + prio_offset) as *mut u8;
    core::ptr::write_volatile(ptr, 0x20);
}

// ─── CPU interface (ICC system registers) ────────────────────────────────────
//
// System-register encodings (AArch64, GICv3):
//   ICC_SRE_EL1    = S3_0_C12_C12_5   (System Register Enable)
//   ICC_PMR_EL1    = S3_0_C4_C6_0     (Priority Mask)
//   ICC_IGRPEN1_EL1= S3_0_C12_C12_7   (Group 1 enable)
//   ICC_BPR1_EL1   = S3_0_C12_C12_3   (Binary Point, Group 1)
//   ICC_IAR1_EL1   = S3_0_C12_C12_0   (Interrupt Acknowledge, Group 1)
//   ICC_EOIR1_EL1  = S3_0_C12_C12_1   (End Of Interrupt, Group 1)
//   ICC_SGI1R_EL1  = S3_0_C12_C11_5   (SGI Group 1 generate)

/// Initialize the GICv3 CPU interface for this core.
///
/// Enables the system-register interface (`ICC_SRE_EL1.SRE`), accepts
/// all priorities (`ICC_PMR_EL1 = 0xFF`), enables Group 1 interrupts
/// (`ICC_IGRPEN1_EL1 = 1`), and sets the Group 1 binary point to 0
/// (no sub-priority grouping).
///
/// # Safety
/// Must be called at EL1 with the `ICC_*_EL1` system registers
/// accessible (firmware must have routed them to EL1).
pub unsafe fn init_icc() {
    let icc_sre: u64;
    asm!("mrs {}, S3_0_C12_C12_5", out(reg) icc_sre, options(nomem, nostack));
    asm!("msr S3_0_C12_C12_5, {}", in(reg) icc_sre | 1, options(nomem, nostack));
    asm!("msr S3_0_C4_C6_0, {}", in(reg) 0xFFu64, options(nomem, nostack));
    asm!("msr S3_0_C12_C12_7, {}", in(reg) 1u64, options(nomem, nostack));
    asm!("msr S3_0_C12_C12_3, {}", in(reg) 0u64, options(nomem, nostack));
}

/// Acknowledge the highest-priority pending Group 1 interrupt.
///
/// Reads `ICC_IAR1_EL1`; the low 24 bits are the INTID. A value of
/// 1023 (`0x3FF`) indicates a spurious interrupt.
pub fn icc_iar() -> u32 {
    let irq: u64;
    unsafe {
        asm!("mrs {}, S3_0_C12_C12_0", out(reg) irq, options(nomem, nostack));
    }
    irq as u32
}

/// Signal end-of-interrupt for `irq` by writing `ICC_EOIR1_EL1`.
pub fn icc_eoir(irq: u32) {
    unsafe {
        asm!("msr S3_0_C12_C12_1, {}", in(reg) irq as u64, options(nomem, nostack));
    }
}

// ─── SPI enable / disable ────────────────────────────────────────────────────

/// Compute the `(byte offset, bit mask)` into the 32-bit-per-register
/// `ISENABLER`/`ICENABLER` banks for `irq_id`.
///
/// Each register covers 32 interrupts: IRQ 0-31 → offset 0, IRQ 32-63 →
/// offset 4, etc. Identical layout to GICv2 (see `crate::gicv2`).
const fn isenabler_offset_and_bit(irq_id: u32) -> (usize, u32) {
    let reg_index = (irq_id / 32) as usize;
    let bit = irq_id % 32;
    (reg_index * 4, 1 << bit)
}

/// Enable interrupt `irq_id`.
///
/// SGI/PPI (INTID < 32) are banked per-CPU in the redistributor SGI
/// frame; SPI (INTID >= 32) live in the distributor.
pub fn enable_irq(irq_id: u32) {
    if irq_id < 32 {
        unsafe {
            gicr_write(GICR_ISENABLER0, 1 << irq_id);
        }
    } else {
        let (offset, bit) = isenabler_offset_and_bit(irq_id);
        unsafe {
            gicd_write(GICD_ISENABLER + offset, bit);
        }
    }
}

/// Disable interrupt `irq_id`. Mirrors [`enable_irq`]'s SGI/PPI vs SPI
/// redistributor/distributor split.
pub fn disable_irq(irq_id: u32) {
    if irq_id < 32 {
        unsafe {
            gicr_write(GICR_ICENABLER0, 1 << irq_id);
        }
    } else {
        let (offset, bit) = isenabler_offset_and_bit(irq_id);
        unsafe {
            gicd_write(GICD_ICENABLER + offset, bit);
        }
    }
}

// ─── SGI (Software Generated Interrupt) ──────────────────────────────────────

/// Send a Software Generated Interrupt via `ICC_SGI1R_EL1`.
///
/// `target_list` is the GICv3 Aff0 target bitmap; `intid` is the SGI ID
/// (0-15). This routes to the current affinity (Aff3..1 = 0), which is
/// sufficient for the single-cluster Phase 2 bring-up.
///
/// # Safety
/// Must be called with the `ICC_*_EL1` system registers accessible.
pub unsafe fn send_sgi(target_list: u16, intid: u8) {
    let val: u64 = (target_list as u64) | ((intid as u64) << 24);
    asm!("msr S3_0_C12_C11_5, {}", in(reg) val, options(nomem, nostack));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_ppi_is_banked_below_32() {
        // PPI 30 (the EL1 physical timer) must be a redistributor-banked
        // INTID so `enable_irq` routes it to GICR, not GICD.
        assert!(TIMER_IRQ < 32);
    }

    #[test]
    fn isenabler_irq30() {
        let (offset, bit) = isenabler_offset_and_bit(30);
        assert_eq!(offset, 0);
        assert_eq!(bit, 1 << 30);
    }

    #[test]
    fn isenabler_first_spi_irq32() {
        let (offset, bit) = isenabler_offset_and_bit(32);
        assert_eq!(offset, 4);
        assert_eq!(bit, 1 << 0);
    }

    #[test]
    fn isenabler_irq45() {
        let (offset, bit) = isenabler_offset_and_bit(45);
        assert_eq!(offset, 4);
        assert_eq!(bit, 1 << 13);
    }

    #[test]
    fn gicd_ctlr_enables_are_and_both_groups() {
        // EnableGrp0 | EnableGrp1NS | ARE_NS
        assert_eq!(GICD_CTLR_ENABLE_ARE, 0b1_0011);
    }

    #[test]
    fn waker_bit_positions() {
        assert_eq!(GICR_WAKER_PROCESSOR_SLEEP, 0b10);
        assert_eq!(GICR_WAKER_CHILDREN_ASLEEP, 0b100);
    }

    #[test]
    fn register_offsets() {
        assert_eq!(GICD_CTLR, 0x000);
        assert_eq!(GICD_ISENABLER, 0x100);
        assert_eq!(GICD_ICENABLER, 0x180);
        assert_eq!(GICR_WAKER, 0x0014);
        assert_eq!(GICR_ISENABLER0, 0x10100);
        assert_eq!(GICR_ICENABLER0, 0x10180);
        assert_eq!(GICR_IPRIORITYR0, 0x10400);
    }
}
