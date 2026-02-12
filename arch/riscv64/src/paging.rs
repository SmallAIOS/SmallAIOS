// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V SV48 4-level page table management (23.2, 23.3).
//!
//! SV48 layout (48-bit virtual address, 4 KiB pages):
//! - Level 3 (root/PGD): 512 entries, each covers 512 GiB
//! - Level 2 (PUD): 512 entries, each covers 1 GiB (can be giga-page leaf)
//! - Level 1 (PMD): 512 entries, each covers 2 MiB (can be mega-page leaf)
//! - Level 0 (PTE): 512 entries, each covers 4 KiB (page leaf)
//!
//! satp register: mode=9 (SV48) | ASID[59:44] | PPN[43:0]
//!
//! TLB flush via `sfence.vma` instruction.

use smallaios_kernel::mem::{
    page::{page_table_index, PageFlags, PageLevel, PageTable, PageTableEntry},
    MemError, PhysAddr, VirtAddr, PAGE_SIZE_4K,
};

// RISC-V PTE bits
const PTE_V: u64 = 1 << 0; // Valid
const PTE_R: u64 = 1 << 1; // Read
const PTE_W: u64 = 1 << 2; // Write
const PTE_X: u64 = 1 << 3; // Execute
const PTE_U: u64 = 1 << 4; // User-accessible
const PTE_G: u64 = 1 << 5; // Global
const PTE_A: u64 = 1 << 6; // Accessed
const PTE_D: u64 = 1 << 7; // Dirty

/// Mask for PPN in a page table entry (bits [53:10]).
const PPN_MASK: u64 = 0x003F_FFFF_FFFF_FC00;
/// Shift for PPN field.
const PPN_SHIFT: u32 = 10;

/// SV48 mode value for satp register.
pub const SATP_MODE_SV48: u64 = 9 << 60;

/// Convert platform-independent flags to RISC-V PTE bits.
fn flags_to_pte_bits(flags: PageFlags) -> u64 {
    let mut bits = PTE_V | PTE_A;

    if flags.contains(PageFlags::WRITABLE) {
        bits |= PTE_R | PTE_W | PTE_D;
    } else {
        bits |= PTE_R;
    }

    if !flags.contains(PageFlags::NO_EXECUTE) {
        bits |= PTE_X;
    }

    if flags.contains(PageFlags::USER) {
        bits |= PTE_U;
    }

    if flags.contains(PageFlags::GLOBAL) {
        bits |= PTE_G;
    }

    bits
}

/// Extract PPN from a PTE value.
fn pte_ppn(pte: u64) -> u64 {
    (pte & PPN_MASK) >> PPN_SHIFT
}

/// Construct a PTE from a physical page number and flag bits.
fn make_pte(ppn: u64, bits: u64) -> u64 {
    (ppn << PPN_SHIFT) | bits
}

/// RISC-V SV48 page table manager.
pub struct Riscv64PageTable {
    /// Physical address of the root (level 3) page table.
    root_addr: PhysAddr,
}

impl Riscv64PageTable {
    /// Create a new page table manager wrapping an existing root table.
    ///
    /// # Safety
    /// `root_addr` must point to a valid, zeroed page table page.
    pub unsafe fn new(root_addr: PhysAddr) -> Self {
        Self { root_addr }
    }

    /// Get the satp register value for this page table (mode=SV48, ASID=0).
    pub fn satp_value(&self) -> u64 {
        let ppn = (self.root_addr.as_usize() as u64) >> 12;
        SATP_MODE_SV48 | ppn
    }

    /// Activate this page table by writing to satp and flushing TLB.
    ///
    /// # Safety
    /// The page table must have valid mappings for the currently executing code.
    pub unsafe fn activate(&self) {
        let satp = self.satp_value();
        core::arch::asm!(
            "csrw satp, {satp}",
            "sfence.vma",
            satp = in(reg) satp,
            options(nostack),
        );
    }

    /// Flush TLB for a specific virtual address.
    pub fn sfence_vma(vaddr: VirtAddr) {
        unsafe {
            core::arch::asm!(
                "sfence.vma {addr}, zero",
                addr = in(reg) vaddr.as_usize(),
                options(nostack),
            );
        }
    }

    /// Flush all TLB entries.
    pub fn sfence_vma_all() {
        unsafe {
            core::arch::asm!("sfence.vma", options(nostack));
        }
    }

    /// Get a mutable reference to a page table at a physical address.
    /// Assumes identity mapping.
    unsafe fn table_at(&self, addr: PhysAddr) -> &mut PageTable {
        &mut *(addr.as_usize() as *mut PageTable)
    }

    /// Walk the page table hierarchy, creating intermediate tables as needed.
    ///
    /// SV48 walk order: Level 3 (root) -> Level 2 -> Level 1 -> Level 0 (leaf)
    /// Mapped to platform levels: L4 -> L3 -> L2 -> L1
    unsafe fn walk_or_create(
        &mut self,
        virt: VirtAddr,
        alloc_page: &mut dyn FnMut() -> Result<PhysAddr, MemError>,
    ) -> Result<(&mut PageTable, usize), MemError> {
        let levels = [PageLevel::L4, PageLevel::L3, PageLevel::L2];
        let mut table_addr = self.root_addr;

        for &level in &levels {
            let table = self.table_at(table_addr);
            let idx = page_table_index(virt, level);
            let entry = &mut table.entries[idx];

            if entry.0 & PTE_V == 0 {
                // Allocate a new page table page.
                let new_page = alloc_page()?;
                core::ptr::write_bytes(new_page.as_usize() as *mut u8, 0, PAGE_SIZE_4K);
                // Non-leaf PTE: V=1 but no R/W/X.
                let ppn = (new_page.as_usize() as u64) >> 12;
                *entry = PageTableEntry(make_pte(ppn, PTE_V));
            }

            // Extract next-level table address from PPN.
            let next_ppn = pte_ppn(entry.0);
            table_addr = PhysAddr::new((next_ppn << 12) as usize);
        }

        let leaf_table = self.table_at(table_addr);
        let leaf_idx = page_table_index(virt, PageLevel::L1);
        Ok((leaf_table, leaf_idx))
    }

    /// Map a 4 KiB virtual page to a physical frame.
    ///
    /// # Safety
    /// Requires valid page table state and `alloc_page` returning valid pages.
    pub unsafe fn map_page(
        &mut self,
        virt: VirtAddr,
        phys: PhysAddr,
        flags: PageFlags,
        alloc_page: &mut dyn FnMut() -> Result<PhysAddr, MemError>,
    ) -> Result<(), MemError> {
        let (table, idx) = self.walk_or_create(virt, alloc_page)?;
        if table.entries[idx].0 & PTE_V != 0 {
            return Err(MemError::AlreadyMapped);
        }

        let ppn = (phys.as_usize() as u64) >> 12;
        let bits = flags_to_pte_bits(flags);
        table.entries[idx] = PageTableEntry(make_pte(ppn, bits));

        Ok(())
    }

    /// Unmap a 4 KiB virtual page.
    pub unsafe fn unmap_page(&mut self, virt: VirtAddr) -> Result<PhysAddr, MemError> {
        let levels = [PageLevel::L4, PageLevel::L3, PageLevel::L2];
        let mut table_addr = self.root_addr;

        for &level in &levels {
            let table = self.table_at(table_addr);
            let idx = page_table_index(virt, level);
            let entry = table.entries[idx];

            if entry.0 & PTE_V == 0 {
                return Err(MemError::NotMapped);
            }
            let next_ppn = pte_ppn(entry.0);
            table_addr = PhysAddr::new((next_ppn << 12) as usize);
        }

        let leaf_table = self.table_at(table_addr);
        let leaf_idx = page_table_index(virt, PageLevel::L1);
        let entry = leaf_table.entries[leaf_idx];

        if entry.0 & PTE_V == 0 {
            return Err(MemError::NotMapped);
        }

        let phys_ppn = pte_ppn(entry.0);
        let phys = PhysAddr::new((phys_ppn << 12) as usize);
        leaf_table.entries[leaf_idx] = PageTableEntry(0);

        // Flush TLB for this address.
        Self::sfence_vma(virt);

        Ok(phys)
    }

    /// Change the permissions on an existing mapping without changing the
    /// physical address. Issues sfence.vma after modification.
    pub unsafe fn protect_page(
        &mut self,
        virt: VirtAddr,
        flags: PageFlags,
    ) -> Result<(), MemError> {
        let levels = [PageLevel::L4, PageLevel::L3, PageLevel::L2];
        let mut table_addr = self.root_addr;

        for &level in &levels {
            let table = self.table_at(table_addr);
            let idx = page_table_index(virt, level);
            let entry = table.entries[idx];
            if entry.0 & PTE_V == 0 {
                return Err(MemError::NotMapped);
            }
            let next_ppn = pte_ppn(entry.0);
            table_addr = PhysAddr::new((next_ppn << 12) as usize);
        }

        let leaf_table = self.table_at(table_addr);
        let leaf_idx = page_table_index(virt, PageLevel::L1);
        let entry = leaf_table.entries[leaf_idx];

        if entry.0 & PTE_V == 0 {
            return Err(MemError::NotMapped);
        }

        // Keep the PPN, update the flags.
        let ppn = pte_ppn(entry.0);
        let bits = flags_to_pte_bits(flags);
        leaf_table.entries[leaf_idx] = PageTableEntry(make_pte(ppn, bits));

        Self::sfence_vma(virt);
        Ok(())
    }
}

/// Identity map a range of physical memory.
///
/// # Safety
/// Requires valid page table and page allocator.
pub unsafe fn identity_map_range(
    page_table: &mut Riscv64PageTable,
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
        page_table.map_page(virt, phys, flags, alloc_page)?;
        addr += PAGE_SIZE_4K;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flags_to_pte_rw() {
        let flags = PageFlags::PRESENT.union(PageFlags::WRITABLE);
        let bits = flags_to_pte_bits(flags);
        assert_ne!(bits & PTE_V, 0);
        assert_ne!(bits & PTE_R, 0);
        assert_ne!(bits & PTE_W, 0);
        assert_ne!(bits & PTE_D, 0);
        assert_ne!(bits & PTE_X, 0); // execute allowed by default
    }

    #[test]
    fn test_flags_to_pte_ro_nx() {
        let flags = PageFlags::PRESENT.union(PageFlags::NO_EXECUTE);
        let bits = flags_to_pte_bits(flags);
        assert_ne!(bits & PTE_V, 0);
        assert_ne!(bits & PTE_R, 0);
        assert_eq!(bits & PTE_W, 0);
        assert_eq!(bits & PTE_X, 0);
    }

    #[test]
    fn test_flags_user_global() {
        let flags = PageFlags::PRESENT
            .union(PageFlags::USER)
            .union(PageFlags::GLOBAL);
        let bits = flags_to_pte_bits(flags);
        assert_ne!(bits & PTE_U, 0);
        assert_ne!(bits & PTE_G, 0);
    }

    #[test]
    fn test_pte_ppn_roundtrip() {
        let ppn = 0x1_2345;
        let pte = make_pte(ppn, PTE_V | PTE_R);
        assert_eq!(pte_ppn(pte), ppn);
        assert_ne!(pte & PTE_V, 0);
        assert_ne!(pte & PTE_R, 0);
    }

    #[test]
    fn test_satp_mode_sv48() {
        assert_eq!(SATP_MODE_SV48, 9 << 60);
    }
}
