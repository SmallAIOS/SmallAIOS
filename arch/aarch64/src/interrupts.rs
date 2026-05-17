// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 interrupt handling: GIC, Generic Timer, exception vectors.
//!
//! GIC drivers live in separate modules: [`crate::gicv3`] for the
//! GICv3 platforms (`qemu-virt`, `tegra234`) and [`crate::gicv2`] for
//! `tegra-x1`. The GICv3 entry points are re-exported here so callers
//! using the `interrupts::` path keep working.
//!
//! Timer, VBAR, and PSCI are platform-independent (same on both).
//!
//! Interrupt layout:
//! - SGI 0-15: Software Generated Interrupts (IPI)
//! - PPI 16-31: Private Peripheral Interrupts (timer, PMU)
//! - SPI 32+: Shared Peripheral Interrupts (I/O devices)

use core::arch::asm;

/// Interrupt IDs.
pub const TIMER_IRQ: u32 = 30; // EL1 physical timer PPI
pub const IPI_IRQ: u32 = 0; // SGI 0

// ─── GICv3 re-export ─────────────────────────────────────────────────────────
//
// The GICv3 driver was extracted to `crate::gicv3` (OpenSpec
// `unikernel-orin-bringup-v1` task 2.13) so the `tegra234` platform can
// share the exact same code path as `qemu-virt`. Re-exported here, on
// the GICv3 platforms only, so existing `interrupts::`-path callers
// keep resolving (task 2.14: `gicv3` for `qemu-virt` + `tegra234`,
// `gicv2` stays for `tegra-x1`).
#[cfg(any(feature = "qemu-virt", feature = "tegra234"))]
pub use crate::gicv3::{icc_eoir, icc_iar, init_gicd, init_gicr, init_icc, send_sgi};

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
