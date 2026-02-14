// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 interrupt handling: GIC, Generic Timer, exception vectors.
//!
//! GICv3 functions are available under `qemu-virt`.
//! GICv2 functions live in the separate `gicv2` module under `tegra-x1`.
//!
//! Timer, VBAR, and PSCI are platform-independent (same on both).
//!
//! Interrupt layout:
//! - SGI 0-15: Software Generated Interrupts (IPI)
//! - PPI 16-31: Private Peripheral Interrupts (timer, PMU)
//! - SPI 32+: Shared Peripheral Interrupts (I/O devices)

use core::arch::asm;

#[cfg(feature = "qemu-virt")]
use crate::platform;

/// Interrupt IDs.
pub const TIMER_IRQ: u32 = 30; // EL1 physical timer PPI
pub const IPI_IRQ: u32 = 0; // SGI 0

// ─── GICv3 (QEMU virt only) ─────────────────────────────────────────────────

#[cfg(feature = "qemu-virt")]
mod gicv3 {
    use super::*;

    const GICD_BASE: usize = platform::GICD_BASE;
    const GICR_BASE: usize = platform::GICR_BASE;

    const GICD_CTLR: usize = 0x000;

    const GICR_ISENABLER0: usize = 0x10100;
    const GICR_IPRIORITYR0: usize = 0x10400;
    const GICR_WAKER: usize = 0x14;

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

    /// # Safety
    /// Must be called with GICD mapped and accessible.
    pub unsafe fn init_gicd() {
        gicd_write(GICD_CTLR, 0);
        // Enable with affinity routing: group 0, group 1, ARE_S
        gicd_write(GICD_CTLR, (1 << 0) | (1 << 1) | (1 << 4));
    }

    /// # Safety
    /// Must be called with GICR mapped and accessible.
    pub unsafe fn init_gicr() {
        let waker = gicr_read(GICR_WAKER);
        gicr_write(GICR_WAKER, waker & !(1 << 1));
        while gicr_read(GICR_WAKER) & (1 << 2) != 0 {
            core::hint::spin_loop();
        }
        gicr_write(GICR_ISENABLER0, 1 << TIMER_IRQ);
        let prio_offset = GICR_IPRIORITYR0 + (TIMER_IRQ as usize);
        let ptr = (GICR_BASE + prio_offset) as *mut u8;
        core::ptr::write_volatile(ptr, 0x20);
    }

    /// # Safety
    /// Must be called at EL1 with ICC system registers accessible.
    pub unsafe fn init_icc() {
        let icc_sre: u64;
        asm!("mrs {}, S3_0_C12_C12_5", out(reg) icc_sre, options(nomem, nostack));
        asm!("msr S3_0_C12_C12_5, {}", in(reg) icc_sre | 1, options(nomem, nostack));
        asm!("msr S3_0_C4_C6_0, {}", in(reg) 0xFFu64, options(nomem, nostack));
        asm!("msr S3_0_C12_C12_7, {}", in(reg) 1u64, options(nomem, nostack));
        asm!("msr S3_0_C12_C12_3, {}", in(reg) 0u64, options(nomem, nostack));
    }

    pub fn icc_iar() -> u32 {
        let irq: u64;
        unsafe {
            asm!("mrs {}, S3_0_C12_C12_0", out(reg) irq, options(nomem, nostack));
        }
        irq as u32
    }

    pub fn icc_eoir(irq: u32) {
        unsafe {
            asm!("msr S3_0_C12_C12_1, {}", in(reg) irq as u64, options(nomem, nostack));
        }
    }

    /// # Safety
    /// Must be called with ICC system registers accessible.
    pub unsafe fn send_sgi(target_list: u16, intid: u8) {
        let val: u64 = (target_list as u64) | ((intid as u64) << 24);
        asm!("msr S3_0_C12_C11_5, {}", in(reg) val, options(nomem, nostack));
    }
}

// Re-export GICv3 functions under qemu-virt
#[cfg(feature = "qemu-virt")]
pub use gicv3::{icc_eoir, icc_iar, init_gicd, init_gicr, init_icc, send_sgi};

// ─── ARM64 Generic Timer (platform-independent) ─────────────────────────────

/// Initialize the EL1 physical timer with a periodic-like countdown.
///
/// # Safety
/// Must be called after GIC is initialized.
pub unsafe fn init_timer(ticks: u64) {
    asm!("msr cntp_tval_el0, {}", in(reg) ticks, options(nomem, nostack));
    asm!("msr cntp_ctl_el0, {}", in(reg) 1u64, options(nomem, nostack));
}

/// Reload the timer for the next tick.
pub fn timer_reload(ticks: u64) {
    unsafe {
        asm!("msr cntp_tval_el0, {}", in(reg) ticks, options(nomem, nostack));
    }
}

/// Disable the timer interrupt.
pub fn timer_disable() {
    unsafe {
        asm!("msr cntp_ctl_el0, {}", in(reg) 0b10u64, options(nomem, nostack));
    }
}

/// Read the timer control register status.
pub fn timer_status() -> u64 {
    let val: u64;
    unsafe {
        asm!("mrs {}, cntp_ctl_el0", out(reg) val, options(nomem, nostack));
    }
    val
}

// ─── Exception Vector Table ─────────────────────────────────────────────────

/// Install the exception vector table.
///
/// # Safety
/// The vector table must be properly aligned (2 KiB) and contain
/// valid exception handlers.
pub unsafe fn set_vbar(vbar: u64) {
    asm!("msr vbar_el1, {}", in(reg) vbar, options(nomem, nostack));
    asm!("isb", options(nomem, nostack));
}

// ─── PSCI (Power State Coordination Interface) ──────────────────────────────

const PSCI_CPU_ON: u64 = 0xC400_0003;

/// Start an application processor using PSCI CPU_ON.
///
/// # Safety
/// `target_cpu` must be a valid MPIDR value, `entry_point` must be
/// a valid code address with an appropriate stack set up.
pub unsafe fn psci_cpu_on(target_cpu: u64, entry_point: u64, context_id: u64) -> i64 {
    let result: i64;
    asm!(
        "mov x0, {func}",
        "mov x1, {target}",
        "mov x2, {entry}",
        "mov x3, {ctx}",
        "smc #0",
        "mov {result}, x0",
        func = in(reg) PSCI_CPU_ON,
        target = in(reg) target_cpu,
        entry = in(reg) entry_point,
        ctx = in(reg) context_id,
        result = out(reg) result,
        out("x0") _,
        out("x1") _,
        out("x2") _,
        out("x3") _,
        options(nomem, nostack),
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interrupt_constants() {
        assert_eq!(TIMER_IRQ, 30);
        assert_eq!(IPI_IRQ, 0);
    }
}
