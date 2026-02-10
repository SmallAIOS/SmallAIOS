// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 4-level page table management.
//!
//! Layout (48-bit canonical addresses):
//! - PML4 (L4): 512 entries, each covers 512 GiB
//! - PDPT (L3): 512 entries, each covers 1 GiB
//! - PD   (L2): 512 entries, each covers 2 MiB
//! - PT   (L1): 512 entries, each covers 4 KiB
//!
//! In unikernel mode, we use identity mapping (virt == phys) for all
//! kernel memory. The top half of the address space is used.

use smallaios_kernel::mem::{
    page::{page_table_index, PageFlags, PageLevel, PageTable, PageTableEntry},
    MemError, PhysAddr, VirtAddr, PAGE_SIZE_4K,
};

/// x86-64 PTE flag bits.
const PTE_PRESENT: u64 = 1 << 0;
const PTE_WRITABLE: u64 = 1 << 1;
const PTE_USER: u64 = 1 << 2;
const PTE_WRITE_THROUGH: u64 = 1 << 3;
const PTE_CACHE_DISABLE: u64 = 1 << 4;
const PTE_HUGE: u64 = 1 << 7;
const PTE_GLOBAL: u64 = 1 << 8;
const PTE_NO_EXECUTE: u64 = 1 << 63;

/// Mask for the physical address in a PTE (bits [12..52]).
const PTE_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// Convert platform-independent flags to x86-64 PTE bits.
fn flags_to_pte_bits(flags: PageFlags) -> u64 {
    let mut bits = 0u64;
    if flags.contains(PageFlags::PRESENT) {
        bits |= PTE_PRESENT;
    }
    if flags.contains(PageFlags::WRITABLE) {
        bits |= PTE_WRITABLE;
    }
    if flags.contains(PageFlags::USER) {
        bits |= PTE_USER;
    }
    if flags.contains(PageFlags::NO_EXECUTE) {
        bits |= PTE_NO_EXECUTE;
    }
    if flags.contains(PageFlags::WRITE_THROUGH) {
        bits |= PTE_WRITE_THROUGH;
    }
    if flags.contains(PageFlags::CACHE_DISABLE) {
        bits |= PTE_CACHE_DISABLE;
    }
    if flags.contains(PageFlags::HUGE_PAGE) {
        bits |= PTE_HUGE;
    }
    if flags.contains(PageFlags::GLOBAL) {
        bits |= PTE_GLOBAL;
    }
    bits
}

/// x86-64 page table manager.
///
/// Operates on identity-mapped page tables (the page table pages
/// themselves are at physical addresses that match their virtual
/// addresses in the identity map).
pub struct X86PageTable {
    /// Physical address of the PML4 (root) page table.
    pml4_addr: PhysAddr,
}

impl X86PageTable {
    /// Create a new page table manager wrapping an existing PML4.
    ///
    /// # Safety
    /// `pml4_addr` must point to a valid, zeroed PML4 page table.
    pub unsafe fn new(pml4_addr: PhysAddr) -> Self {
        Self { pml4_addr }
    }

    /// Get the physical address of the PML4 for loading into CR3.
    pub fn pml4_phys(&self) -> PhysAddr {
        self.pml4_addr
    }

    /// Load this page table into CR3 (activate it).
    ///
    /// # Safety
    /// The page table must have valid identity mappings for the
    /// currently executing code.
    pub unsafe fn activate(&self) {
        core::arch::asm!(
            "mov cr3, {}",
            in(reg) self.pml4_addr.as_usize(),
            options(nostack, preserves_flags),
        );
    }

    /// Invalidate a single TLB entry.
    pub fn invlpg(virt: VirtAddr) {
        unsafe {
            core::arch::asm!(
                "invlpg [{}]",
                in(reg) virt.as_usize(),
                options(nostack, preserves_flags),
            );
        }
    }

    /// Get a mutable reference to a page table at a physical address.
    /// Assumes identity mapping.
    unsafe fn table_at(&self, addr: PhysAddr) -> &mut PageTable {
        &mut *(addr.as_usize() as *mut PageTable)
    }

    /// Walk the page table hierarchy, creating intermediate tables as needed.
    /// `alloc_page` is called to allocate new page table pages.
    ///
    /// Returns a mutable reference to the L1 (leaf) page table and the index.
    unsafe fn walk_or_create(
        &mut self,
        virt: VirtAddr,
        alloc_page: &mut dyn FnMut() -> Result<PhysAddr, MemError>,
    ) -> Result<(&mut PageTable, usize), MemError> {
        let levels = [PageLevel::L4, PageLevel::L3, PageLevel::L2];
        let mut table_addr = self.pml4_addr;

        for &level in &levels {
            let table = self.table_at(table_addr);
            let idx = page_table_index(virt, level);
            let entry = &mut table.entries[idx];

            if !entry.is_present() {
                // Allocate a new page table page
                let new_page = alloc_page()?;
                // Zero the new page
                core::ptr::write_bytes(new_page.as_usize() as *mut u8, 0, PAGE_SIZE_4K);
                // Set the entry to point to the new table
                *entry = PageTableEntry(
                    (new_page.as_usize() as u64 & PTE_ADDR_MASK) | PTE_PRESENT | PTE_WRITABLE,
                );
            }

            table_addr = entry.address();
        }

        let leaf_table = self.table_at(table_addr);
        let leaf_idx = page_table_index(virt, PageLevel::L1);
        Ok((leaf_table, leaf_idx))
    }
}

/// Identity map a range of physical memory.
///
/// # Safety
/// Requires that `page_table` is valid and `alloc_page` returns valid pages.
pub unsafe fn identity_map_range(
    page_table: &mut X86PageTable,
    start: PhysAddr,
    size: usize,
    flags: PageFlags,
    alloc_page: &mut dyn FnMut() -> Result<PhysAddr, MemError>,
) -> Result<(), MemError> {
    let start_addr = start.align_down(PAGE_SIZE_4K).as_usize();
    let end_addr = (start.as_usize() + size + PAGE_SIZE_4K - 1) & !(PAGE_SIZE_4K - 1);

    let mut addr = start_addr;
    while addr < end_addr {
        let virt = VirtAddr::new(addr);
        let phys = PhysAddr::new(addr);

        let (table, idx) = page_table.walk_or_create(virt, alloc_page)?;
        if table.entries[idx].is_present() {
            return Err(MemError::AlreadyMapped);
        }

        let pte_bits = (phys.as_usize() as u64 & PTE_ADDR_MASK) | flags_to_pte_bits(flags);
        table.entries[idx] = PageTableEntry(pte_bits);

        addr += PAGE_SIZE_4K;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flags_to_pte() {
        let flags = PageFlags::PRESENT
            .union(PageFlags::WRITABLE)
            .union(PageFlags::NO_EXECUTE);
        let bits = flags_to_pte_bits(flags);
        assert_ne!(bits & PTE_PRESENT, 0);
        assert_ne!(bits & PTE_WRITABLE, 0);
        assert_ne!(bits & PTE_NO_EXECUTE, 0);
        assert_eq!(bits & PTE_USER, 0);
    }
}
