// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 4-level page table management (4 KiB granule).
//!
//! Layout (48-bit VA, 4 KiB granule):
//! - L0: 512 entries, each covers 512 GiB
//! - L1: 512 entries, each covers 1 GiB (can be block descriptor)
//! - L2: 512 entries, each covers 2 MiB (can be block descriptor)
//! - L3: 512 entries, each covers 4 KiB (page descriptor)
//!
//! In unikernel mode, we use identity mapping for all kernel memory.
//! TTBR0_EL1 is used for the lower half (user/kernel in unikernel).

use smallaios_kernel::mem::{
    page::{page_table_index, PageFlags, PageLevel, PageTable, PageTableEntry},
    MemError, PhysAddr, VirtAddr, PAGE_SIZE_4K,
};

/// ARM64 descriptor types.
const DESC_VALID: u64 = 1 << 0;
const DESC_TABLE: u64 = 1 << 1; // For L0-L2: table descriptor
const DESC_PAGE: u64 = 1 << 1; // For L3: page descriptor

/// Lower attributes (bits [11:2]).
const ATTR_IDX_NORMAL: u64 = 0 << 2; // MAIR index 0: Normal WB
const ATTR_IDX_DEVICE: u64 = 1 << 2; // MAIR index 1: Device-nGnRnE
const _ATTR_NS: u64 = 1 << 5; // Non-secure (used in NS bit setup)
const ATTR_AP_RW_EL1: u64 = 0b00 << 6; // EL1 read/write
const ATTR_AP_RO_EL1: u64 = 0b10 << 6; // EL1 read-only
const ATTR_SH_INNER: u64 = 0b11 << 8; // Inner shareable
const ATTR_AF: u64 = 1 << 10; // Access flag (must be 1)

/// Upper attributes (bits [63:50]).
const ATTR_PXN: u64 = 1 << 53; // Privileged execute never
const ATTR_UXN: u64 = 1 << 54; // Unprivileged execute never
const ATTR_NG: u64 = 1 << 11; // Not global (nG)

/// Mask for output address in a table/page descriptor (bits [47:12]).
const OUTPUT_ADDR_MASK: u64 = 0x0000_FFFF_FFFF_F000;

/// Convert platform-independent flags to ARM64 descriptor attributes.
fn flags_to_desc_attrs(flags: PageFlags) -> u64 {
    let mut attrs = ATTR_AF | ATTR_SH_INNER;

    if flags.contains(PageFlags::CACHE_DISABLE) {
        attrs |= ATTR_IDX_DEVICE;
    } else {
        attrs |= ATTR_IDX_NORMAL;
    }

    if !flags.contains(PageFlags::WRITABLE) {
        attrs |= ATTR_AP_RO_EL1;
    } else {
        attrs |= ATTR_AP_RW_EL1;
    }

    if flags.contains(PageFlags::NO_EXECUTE) {
        attrs |= ATTR_PXN | ATTR_UXN;
    }

    if !flags.contains(PageFlags::GLOBAL) {
        attrs |= ATTR_NG;
    }

    attrs
}

/// ARM64 page table manager.
pub struct Aarch64PageTable {
    /// Physical address of the L0 (root) translation table.
    l0_addr: PhysAddr,
}

impl Aarch64PageTable {
    /// Create a new page table manager wrapping an existing L0 table.
    ///
    /// # Safety
    /// `l0_addr` must point to a valid, zeroed L0 translation table page.
    pub unsafe fn new(l0_addr: PhysAddr) -> Self {
        Self { l0_addr }
    }

    /// Get the physical address for loading into TTBR0_EL1.
    pub fn ttbr0_phys(&self) -> PhysAddr {
        self.l0_addr
    }

    /// Load this page table into TTBR0_EL1 and issue TLB invalidation.
    ///
    /// # Safety
    /// The page table must have valid identity mappings for the
    /// currently executing code.
    pub unsafe fn activate(&self) {
        core::arch::asm!(
            "msr ttbr0_el1, {addr}",
            "isb",
            "tlbi vmalle1",
            "dsb sy",
            "isb",
            addr = in(reg) self.l0_addr.as_usize(),
            options(nostack),
        );
    }

    /// Invalidate a single TLB entry by VA.
    pub fn tlbi_vae1(virt: VirtAddr) {
        unsafe {
            // TLBI VAE1, Xt — VA is bits [55:12] of the address
            let va_bits = (virt.as_usize() >> 12) as u64;
            core::arch::asm!(
                "tlbi vae1, {}",
                "dsb sy",
                "isb",
                in(reg) va_bits,
                options(nostack),
            );
        }
    }

    /// Get a mutable reference to a page table at a physical address.
    /// Assumes identity mapping.
    #[allow(clippy::mut_from_ref)]
    unsafe fn table_at(&self, addr: PhysAddr) -> &mut PageTable {
        &mut *(addr.as_usize() as *mut PageTable)
    }

    /// Walk the page table hierarchy, creating intermediate tables as needed.
    ///
    /// ARM64 levels for 4K granule: L0 → L1 → L2 → L3
    /// We map L0=L4, L1=L3, L2=L2, L3=L1 in the platform-independent naming.
    unsafe fn walk_or_create(
        &mut self,
        virt: VirtAddr,
        alloc_page: &mut dyn FnMut() -> Result<PhysAddr, MemError>,
    ) -> Result<(&mut PageTable, usize), MemError> {
        let levels = [PageLevel::L4, PageLevel::L3, PageLevel::L2];
        let mut table_addr = self.l0_addr;

        for &level in &levels {
            let table = self.table_at(table_addr);
            let idx = page_table_index(virt, level);
            let entry = &mut table.entries[idx];

            if entry.0 & DESC_VALID == 0 {
                // Allocate a new page table page
                let new_page = alloc_page()?;
                core::ptr::write_bytes(new_page.as_usize() as *mut u8, 0, PAGE_SIZE_4K);
                // Table descriptor: valid + table
                *entry = PageTableEntry(
                    (new_page.as_usize() as u64 & OUTPUT_ADDR_MASK) | DESC_VALID | DESC_TABLE,
                );
            }

            // Extract next-level table address
            table_addr = PhysAddr::new((entry.0 & OUTPUT_ADDR_MASK) as usize);
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
        if table.entries[idx].0 & DESC_VALID != 0 {
            return Err(MemError::AlreadyMapped);
        }

        let desc = (phys.as_usize() as u64 & OUTPUT_ADDR_MASK)
            | DESC_VALID
            | DESC_PAGE
            | flags_to_desc_attrs(flags);
        table.entries[idx] = PageTableEntry(desc);

        Ok(())
    }

    /// Unmap a 4 KiB virtual page.
    ///
    /// # Safety
    /// The virtual address must have been previously mapped.
    pub unsafe fn unmap_page(&mut self, virt: VirtAddr) -> Result<PhysAddr, MemError> {
        let levels = [PageLevel::L4, PageLevel::L3, PageLevel::L2];
        let mut table_addr = self.l0_addr;

        for &level in &levels {
            let table = self.table_at(table_addr);
            let idx = page_table_index(virt, level);
            let entry = table.entries[idx];

            if entry.0 & DESC_VALID == 0 {
                return Err(MemError::NotMapped);
            }
            table_addr = PhysAddr::new((entry.0 & OUTPUT_ADDR_MASK) as usize);
        }

        let leaf_table = self.table_at(table_addr);
        let leaf_idx = page_table_index(virt, PageLevel::L1);
        let entry = leaf_table.entries[leaf_idx];

        if entry.0 & DESC_VALID == 0 {
            return Err(MemError::NotMapped);
        }

        let phys = PhysAddr::new((entry.0 & OUTPUT_ADDR_MASK) as usize);
        leaf_table.entries[leaf_idx] = PageTableEntry(0);

        Self::tlbi_vae1(virt);

        Ok(phys)
    }
}

/// Identity map a range of physical memory.
///
/// # Safety
/// Requires valid page table and page allocator.
pub unsafe fn identity_map_range(
    page_table: &mut Aarch64PageTable,
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

/// Configure MAIR_EL1 with memory attribute indices.
///
/// # Safety
/// Must be called early in boot before enabling the MMU.
pub unsafe fn configure_mair() {
    // Index 0: Normal Write-Back, Non-transient, Read-Write Allocate
    // Index 1: Device-nGnRnE (strongly ordered)
    let mair: u64 = 0xFF_00; // idx0 = 0xFF (Normal WB), idx1 = 0x00 (Device)
    core::arch::asm!(
        "msr mair_el1, {}",
        in(reg) mair,
        options(nostack),
    );
}

/// Configure TCR_EL1 for 4 KiB granule, 48-bit address space.
///
/// # Safety
/// Must be called early in boot before enabling the MMU.
pub unsafe fn configure_tcr() {
    // T0SZ = 16 (48-bit VA for TTBR0)
    // TG0 = 0b00 (4 KiB granule)
    // SH0 = 0b11 (Inner Shareable)
    // ORGN0 = 0b01 (WB RA WA)
    // IRGN0 = 0b01 (WB RA WA)
    #[allow(clippy::identity_op)]
    let tcr: u64 = (16 << 0)    // T0SZ (bits 5:0)
        | (0b00 << 14)          // TG0: 4 KiB
        | (0b11 << 12)          // SH0: Inner Shareable
        | (0b01 << 10)          // ORGN0: WB RA WA
        | (0b01 << 8); // IRGN0: WB RA WA
    core::arch::asm!(
        "msr tcr_el1, {}",
        in(reg) tcr,
        options(nostack),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flags_to_desc() {
        let flags = PageFlags::PRESENT
            .union(PageFlags::WRITABLE)
            .union(PageFlags::NO_EXECUTE);
        let attrs = flags_to_desc_attrs(flags);
        assert_ne!(attrs & ATTR_AF, 0);
        assert_ne!(attrs & ATTR_PXN, 0);
        assert_eq!(attrs & ATTR_AP_RO_EL1, 0); // Should be RW
    }
}
