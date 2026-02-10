// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 interrupt handling: IDT, Local APIC, APIC Timer, IPI.
//!
//! Interrupt layout:
//! - 0-31: CPU exceptions (division error, page fault, etc.)
//! - 32: APIC timer (scheduler tick)
//! - 33: IPI (inter-processor interrupt for work-stealing wakeup)
//! - 34-47: I/O APIC (keyboard, serial, etc.)
//! - 255: APIC spurious

use core::arch::asm;

// ─── Interrupt Vector Numbers ────────────────────────────────────────────────

/// APIC timer interrupt vector.
pub const TIMER_VECTOR: u8 = 32;
/// IPI interrupt vector.
pub const IPI_VECTOR: u8 = 33;
/// Spurious interrupt vector.
pub const SPURIOUS_VECTOR: u8 = 255;

// ─── Local APIC Registers (memory-mapped at 0xFEE00000) ─────────────────────

/// Default APIC base address.
const APIC_BASE: usize = 0xFEE0_0000;

/// APIC register offsets.
const APIC_ID: usize = 0x020;
const APIC_EOI: usize = 0x0B0;
const APIC_SVR: usize = 0x0F0;
const APIC_ICR_LOW: usize = 0x300;
const APIC_ICR_HIGH: usize = 0x310;
const APIC_TIMER_LVT: usize = 0x320;
const APIC_TIMER_INITIAL: usize = 0x380;
const APIC_TIMER_CURRENT: usize = 0x390;
const APIC_TIMER_DIVIDE: usize = 0x3E0;

/// Read a 32-bit APIC register.
///
/// # Safety
/// APIC must be mapped and accessible.
unsafe fn apic_read(offset: usize) -> u32 {
    let ptr = (APIC_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Write a 32-bit APIC register.
///
/// # Safety
/// APIC must be mapped and accessible.
unsafe fn apic_write(offset: usize, value: u32) {
    let ptr = (APIC_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, value);
}

/// Get the current CPU's APIC ID.
pub fn apic_id() -> u32 {
    unsafe { apic_read(APIC_ID) >> 24 }
}

/// Initialize the Local APIC.
///
/// # Safety
/// Must be called early in boot, after APIC base is identity-mapped.
pub unsafe fn init_apic() {
    // Enable APIC: set spurious vector, enable bit (bit 8)
    let svr = (SPURIOUS_VECTOR as u32) | (1 << 8);
    apic_write(APIC_SVR, svr);
}

/// Send End-Of-Interrupt to the APIC.
pub fn eoi() {
    unsafe {
        apic_write(APIC_EOI, 0);
    }
}

/// Configure the APIC timer in periodic mode.
///
/// # Safety
/// APIC must be initialized.
pub unsafe fn init_apic_timer(initial_count: u32) {
    // Set divider to 16
    apic_write(APIC_TIMER_DIVIDE, 0x03); // divide by 16

    // Set LVT timer entry: vector, periodic mode (bit 17)
    let lvt = (TIMER_VECTOR as u32) | (1 << 17); // periodic
    apic_write(APIC_TIMER_LVT, lvt);

    // Set initial count — timer fires when count reaches zero
    apic_write(APIC_TIMER_INITIAL, initial_count);
}

/// Get the current APIC timer count.
pub fn apic_timer_current() -> u32 {
    unsafe { apic_read(APIC_TIMER_CURRENT) }
}

/// Send an IPI (Inter-Processor Interrupt) to a specific core.
///
/// # Safety
/// Target APIC ID must be valid.
pub unsafe fn send_ipi(target_apic_id: u32, vector: u8) {
    // Set destination in ICR high
    apic_write(APIC_ICR_HIGH, target_apic_id << 24);
    // Send: fixed delivery, vector
    apic_write(APIC_ICR_LOW, vector as u32);
}

/// Send an IPI to all other cores.
///
/// # Safety
/// Must be called with valid interrupt state.
pub unsafe fn send_ipi_all_excluding_self(vector: u8) {
    apic_write(APIC_ICR_HIGH, 0);
    // All excluding self (shorthand = 0b11), fixed delivery
    apic_write(APIC_ICR_LOW, (vector as u32) | (0b11 << 18));
}

// ─── IDT (Interrupt Descriptor Table) ────────────────────────────────────────

/// IDT entry (16 bytes for long mode).
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct IdtEntry {
    offset_low: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    offset_mid: u16,
    offset_high: u32,
    _reserved: u32,
}

impl IdtEntry {
    const EMPTY: Self = Self {
        offset_low: 0,
        selector: 0,
        ist: 0,
        type_attr: 0,
        offset_mid: 0,
        offset_high: 0,
        _reserved: 0,
    };

    /// Create an IDT entry for an interrupt handler.
    pub fn new(handler: u64, selector: u16, ist: u8, gate_type: u8) -> Self {
        Self {
            offset_low: handler as u16,
            selector,
            ist,
            type_attr: gate_type | (1 << 7), // present bit
            offset_mid: (handler >> 16) as u16,
            offset_high: (handler >> 32) as u32,
            _reserved: 0,
        }
    }
}

/// IDT with 256 entries.
#[repr(C, align(16))]
pub struct Idt {
    pub entries: [IdtEntry; 256],
}

impl Idt {
    /// Create an empty IDT.
    pub const fn new() -> Self {
        Self {
            entries: [IdtEntry::EMPTY; 256],
        }
    }

    /// Set an entry for a given vector.
    pub fn set_handler(&mut self, vector: u8, handler: u64, ist: u8) {
        // 0x8E = 64-bit interrupt gate, DPL=0, present
        self.entries[vector as usize] = IdtEntry::new(handler, 0x08, ist, 0x8E);
    }

    /// Load this IDT into the CPU.
    ///
    /// # Safety
    /// The IDT must remain valid for the lifetime of this CPU.
    pub unsafe fn load(&'static self) {
        let ptr = IdtPointer {
            limit: (core::mem::size_of::<Self>() - 1) as u16,
            base: self as *const Self as u64,
        };
        asm!(
            "lidt [{}]",
            in(reg) &ptr,
            options(nostack),
        );
    }
}

/// IDTR pointer structure.
#[repr(C, packed)]
struct IdtPointer {
    limit: u16,
    base: u64,
}

// ─── SMP Boot (SIPI) ────────────────────────────────────────────────────────

/// Send INIT-SIPI-SIPI sequence to start an Application Processor.
///
/// The trampoline code must be placed at a page-aligned address below 1 MiB
/// (the SIPI vector is the page number of the entry point).
///
/// # Safety
/// Trampoline code must be set up at the target address.
pub unsafe fn start_ap(target_apic_id: u32, trampoline_page: u8) {
    // INIT IPI
    apic_write(APIC_ICR_HIGH, target_apic_id << 24);
    apic_write(APIC_ICR_LOW, 0x0000_C500); // INIT, level assert

    // Wait 10 ms (rough delay)
    for _ in 0..10_000 {
        core::hint::spin_loop();
    }

    // Deassert INIT
    apic_write(APIC_ICR_HIGH, target_apic_id << 24);
    apic_write(APIC_ICR_LOW, 0x0000_8500); // INIT, level deassert

    // SIPI (Startup IPI) — send twice per Intel spec
    for _ in 0..2 {
        apic_write(APIC_ICR_HIGH, target_apic_id << 24);
        apic_write(APIC_ICR_LOW, 0x0000_0600 | (trampoline_page as u32));

        // Wait 200 us
        for _ in 0..200 {
            core::hint::spin_loop();
        }
    }
}

/// Read the IA32_APIC_BASE MSR to get the APIC base address.
pub fn read_apic_base_msr() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!(
            "rdmsr",
            in("ecx") 0x1Bu32, // IA32_APIC_BASE
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}
