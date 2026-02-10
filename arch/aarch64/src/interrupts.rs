// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 interrupt handling: GICv3, Generic Timer, exception vectors.
//!
//! Interrupt layout:
//! - SGI 0-15: Software Generated Interrupts (IPI)
//! - PPI 16-31: Private Peripheral Interrupts (timer, PMU)
//! - SPI 32+: Shared Peripheral Interrupts (I/O devices)
//!
//! Timer:
//! - CNTP_TVAL_EL0: Physical timer countdown value
//! - CNTP_CTL_EL0: Physical timer control (enable, mask, status)
//! - CNTPCT_EL0: Physical counter value (read-only monotonic)
//! - CNTFRQ_EL0: Counter frequency in Hz

use core::arch::asm;

// ─── GICv3 Distributor Registers ─────────────────────────────────────────────

/// GIC Distributor base address (QEMU virt).
const GICD_BASE: usize = 0x0800_0000;
/// GIC Redistributor base address (QEMU virt).
const GICR_BASE: usize = 0x080A_0000;

/// GICD register offsets.
const GICD_CTLR: usize = 0x000;
const _GICD_ISENABLER: usize = 0x100;
const _GICD_ICENABLER: usize = 0x180;
const _GICD_ISPENDR: usize = 0x200;
const _GICD_IPRIORITYR: usize = 0x400;
const _GICD_ITARGETSR: usize = 0x800;
const _GICD_ICFGR: usize = 0xC00;

/// GICR SGI/PPI register offsets (per-redistributor).
const GICR_ISENABLER0: usize = 0x10100;
const GICR_IPRIORITYR0: usize = 0x10400;
const GICR_WAKER: usize = 0x14;

/// Interrupt IDs.
pub const TIMER_IRQ: u32 = 30; // EL1 physical timer PPI
pub const IPI_IRQ: u32 = 0; // SGI 0

/// Read a 32-bit GIC Distributor register.
///
/// # Safety
/// GIC must be mapped.
unsafe fn gicd_read(offset: usize) -> u32 {
    let ptr = (GICD_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit GIC Distributor register.
///
/// # Safety
/// GIC must be mapped.
unsafe fn gicd_write(offset: usize, value: u32) {
    let ptr = (GICD_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

/// Read a 32-bit GIC Redistributor register.
///
/// # Safety
/// GIC must be mapped.
unsafe fn gicr_read(offset: usize) -> u32 {
    let ptr = (GICR_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit GIC Redistributor register.
///
/// # Safety
/// GIC must be mapped.
unsafe fn gicr_write(offset: usize, value: u32) {
    let ptr = (GICR_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

/// Initialize the GICv3 distributor.
///
/// # Safety
/// GIC registers must be identity-mapped.
pub unsafe fn init_gicd() {
    // Disable distributor
    gicd_write(GICD_CTLR, 0);

    // Enable distributor with affinity routing (ARE)
    // Bit 0: Enable group 0, Bit 1: Enable group 1, Bit 4: ARE_S
    gicd_write(GICD_CTLR, (1 << 0) | (1 << 1) | (1 << 4));
}

/// Initialize the GICv3 redistributor for the current core.
///
/// # Safety
/// GIC registers must be identity-mapped.
pub unsafe fn init_gicr() {
    // Wake up redistributor
    let waker = gicr_read(GICR_WAKER);
    gicr_write(GICR_WAKER, waker & !(1 << 1)); // Clear ProcessorSleep

    // Wait for ChildrenAsleep to clear
    while gicr_read(GICR_WAKER) & (1 << 2) != 0 {
        core::hint::spin_loop();
    }

    // Enable timer PPI (IRQ 30): set enable bit
    gicr_write(GICR_ISENABLER0, 1 << TIMER_IRQ);

    // Set timer priority to high (lower value = higher priority)
    // Priority register is byte-accessible, IRQ 30 is in IPRIORITYR7 (offset 30)
    let prio_offset = GICR_IPRIORITYR0 + (TIMER_IRQ as usize);
    let ptr = (GICR_BASE + prio_offset) as *mut u8;
    core::ptr::write_volatile(ptr, 0x20); // High priority
}

/// Initialize the GICv3 CPU interface (ICC system registers).
///
/// # Safety
/// Must be called on each core.
pub unsafe fn init_icc() {
    // Enable system register interface
    let icc_sre: u64;
    asm!("mrs {}, S3_0_C12_C12_5", out(reg) icc_sre, options(nomem, nostack));
    asm!("msr S3_0_C12_C12_5, {}", in(reg) icc_sre | 1, options(nomem, nostack));

    // Set priority mask (allow all priorities)
    asm!("msr S3_0_C4_C6_0, {}", in(reg) 0xFFu64, options(nomem, nostack)); // ICC_PMR_EL1

    // Enable group 1 interrupts
    asm!("msr S3_0_C12_C12_7, {}", in(reg) 1u64, options(nomem, nostack)); // ICC_IGRPEN1_EL1

    // Binary point = 0 (all bits used for group priority)
    asm!("msr S3_0_C12_C12_3, {}", in(reg) 0u64, options(nomem, nostack)); // ICC_BPR1_EL1
}

/// Acknowledge an interrupt (read IAR).
pub fn icc_iar() -> u32 {
    let irq: u64;
    unsafe {
        asm!("mrs {}, S3_0_C12_C12_0", out(reg) irq, options(nomem, nostack)); // ICC_IAR1_EL1
    }
    irq as u32
}

/// Signal End of Interrupt.
pub fn icc_eoir(irq: u32) {
    unsafe {
        asm!("msr S3_0_C12_C12_1, {}", in(reg) irq as u64, options(nomem, nostack));
        // ICC_EOIR1_EL1
    }
}

/// Send a Software Generated Interrupt (IPI) to a target core.
///
/// # Safety
/// Target must be a valid MPIDR affinity value.
pub unsafe fn send_sgi(target_list: u16, intid: u8) {
    // ICC_SGI1R_EL1: Aff3[55:48] | Aff2[39:32] | IRM[40] | Aff1[23:16] | TargetList[15:0] | INTID[27:24]
    let val: u64 = (target_list as u64) | ((intid as u64) << 24);
    asm!("msr S3_0_C12_C11_5, {}", in(reg) val, options(nomem, nostack)); // ICC_SGI1R_EL1
}

// ─── ARM64 Generic Timer ─────────────────────────────────────────────────────

/// Initialize the EL1 physical timer with a periodic-like countdown.
///
/// # Safety
/// Must be called after GIC is initialized.
pub unsafe fn init_timer(ticks: u64) {
    // Set countdown value
    asm!("msr cntp_tval_el0, {}", in(reg) ticks, options(nomem, nostack));
    // Enable timer, unmask interrupt (bit 0 = enable, bit 1 = mask)
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
        // Set mask bit (bit 1)
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

// ─── Exception Vector Table ──────────────────────────────────────────────────

/// Install the exception vector table.
///
/// # Safety
/// The vector table must be properly aligned (2 KiB) and contain
/// valid exception handlers.
pub unsafe fn set_vbar(vbar: u64) {
    asm!("msr vbar_el1, {}", in(reg) vbar, options(nomem, nostack));
    asm!("isb", options(nomem, nostack));
}

// ─── PSCI (Power State Coordination Interface) ───────────────────────────────

/// PSCI function IDs (SMC calling convention).
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
