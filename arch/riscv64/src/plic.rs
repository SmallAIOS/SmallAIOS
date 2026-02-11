// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! PLIC (Platform-Level Interrupt Controller) driver (23.4).
//!
//! QEMU virt machine PLIC base: 0x0C00_0000
//!
//! Memory map:
//! - 0x0000_0000: Priority registers (4 bytes per source, 1024 sources)
//! - 0x0000_1000: Pending bits (128 bytes)
//! - 0x0000_2000: Enable bits (per context, 128 bytes per context)
//! - 0x0020_0000: Priority threshold + claim/complete (per context)
//!
//! Context = hart_id * 2 + 1 (S-mode context for each hart)

/// PLIC base address (QEMU virt).
const PLIC_BASE: usize = 0x0C00_0000;

/// Maximum interrupt sources.
pub const MAX_SOURCES: u32 = 1024;

/// Maximum priority value.
pub const MAX_PRIORITY: u32 = 7;

/// Register offsets.
const PRIORITY_BASE: usize = 0x0000_0000;
const PENDING_BASE: usize = 0x0000_1000;
const ENABLE_BASE: usize = 0x0000_2000;
const CONTEXT_BASE: usize = 0x0020_0000;

/// Bytes per context in the enable region.
const ENABLE_PER_CONTEXT: usize = 0x80;
/// Bytes per context in the threshold/claim region.
const CONTEXT_SIZE: usize = 0x1000;

/// Threshold register offset within a context.
const THRESHOLD_OFFSET: usize = 0x00;
/// Claim/complete register offset within a context.
const CLAIM_OFFSET: usize = 0x04;

/// Compute the S-mode context ID for a hart.
///
/// On QEMU virt, context = hart_id * 2 + 1 (context 0 is M-mode for hart 0).
pub fn s_mode_context(hart_id: usize) -> usize {
    hart_id * 2 + 1
}

/// Write a 32-bit value to a PLIC register.
///
/// # Safety
/// PLIC must be identity-mapped.
unsafe fn plic_write(offset: usize, val: u32) {
    let ptr = (PLIC_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, val);
}

/// Read a 32-bit value from a PLIC register.
///
/// # Safety
/// PLIC must be identity-mapped.
unsafe fn plic_read(offset: usize) -> u32 {
    let ptr = (PLIC_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Set the priority for an interrupt source (0 = disabled, 1-7 ascending).
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn set_priority(source: u32, priority: u32) {
    if source == 0 || source >= MAX_SOURCES || priority > MAX_PRIORITY {
        return;
    }
    let offset = PRIORITY_BASE + (source as usize) * 4;
    plic_write(offset, priority);
}

/// Enable an interrupt source for a specific S-mode context.
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn enable(context: usize, source: u32) {
    if source == 0 || source >= MAX_SOURCES {
        return;
    }
    let reg_offset = ENABLE_BASE + context * ENABLE_PER_CONTEXT + (source as usize / 32) * 4;
    let bit = 1u32 << (source % 32);
    let current = plic_read(reg_offset);
    plic_write(reg_offset, current | bit);
}

/// Disable an interrupt source for a specific S-mode context.
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn disable(context: usize, source: u32) {
    if source == 0 || source >= MAX_SOURCES {
        return;
    }
    let reg_offset = ENABLE_BASE + context * ENABLE_PER_CONTEXT + (source as usize / 32) * 4;
    let bit = 1u32 << (source % 32);
    let current = plic_read(reg_offset);
    plic_write(reg_offset, current & !bit);
}

/// Set the priority threshold for a context.
///
/// Interrupts with priority <= threshold are masked.
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn set_threshold(context: usize, threshold: u32) {
    let offset = CONTEXT_BASE + context * CONTEXT_SIZE + THRESHOLD_OFFSET;
    plic_write(offset, threshold);
}

/// Claim the highest-priority pending interrupt for a context.
///
/// Returns the interrupt source ID, or 0 if no interrupt is pending.
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn claim(context: usize) -> u32 {
    let offset = CONTEXT_BASE + context * CONTEXT_SIZE + CLAIM_OFFSET;
    plic_read(offset)
}

/// Complete interrupt handling for a claimed source.
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn complete(context: usize, source: u32) {
    let offset = CONTEXT_BASE + context * CONTEXT_SIZE + CLAIM_OFFSET;
    plic_write(offset, source);
}

/// Check if a specific interrupt source is pending.
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn is_pending(source: u32) -> bool {
    if source == 0 || source >= MAX_SOURCES {
        return false;
    }
    let offset = PENDING_BASE + (source as usize / 32) * 4;
    let val = plic_read(offset);
    val & (1 << (source % 32)) != 0
}

/// Initialize the PLIC for a hart's S-mode context.
///
/// Disables all sources and sets threshold to 0 (allow all priorities).
///
/// # Safety
/// PLIC must be identity-mapped.
pub unsafe fn init_for_hart(hart_id: usize) {
    let context = s_mode_context(hart_id);

    // Disable all sources for this context.
    for word in 0..(MAX_SOURCES as usize / 32) {
        let offset = ENABLE_BASE + context * ENABLE_PER_CONTEXT + word * 4;
        plic_write(offset, 0);
    }

    // Set threshold to 0 (allow all priorities).
    set_threshold(context, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_mode_context() {
        assert_eq!(s_mode_context(0), 1);
        assert_eq!(s_mode_context(1), 3);
        assert_eq!(s_mode_context(3), 7);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_SOURCES, 1024);
        assert_eq!(MAX_PRIORITY, 7);
        assert_eq!(PLIC_BASE, 0x0C00_0000);
    }
}
