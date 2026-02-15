// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU MMU (GMMU) Page Table management for GM20B (Tegra X1 / T210).
//!
//! The GMMU maps GPU virtual addresses to physical memory. GM20B uses
//! shared system DRAM (unified memory) rather than discrete VRAM,
//! making identity mapping the natural configuration.

#![allow(dead_code)]

extern crate alloc;
use alloc::vec::Vec;
use crate::GpuError;

const GMMU_PAGE_SIZE_SMALL: usize = 4096;
const GMMU_PAGE_SIZE_BIG: usize = 65536;
const GMMU_PTE_VALID: u64 = 1 << 0;
const GMMU_PTE_READ: u64 = 1 << 1;
const GMMU_PTE_WRITE: u64 = 1 << 2;
const GMMU_PDB_BASE_REG: u32 = 0x10_0000; // Instance block PDB pointer
const GMMU_TLB_INVALIDATE: u32 = 0x10_0CBC;

/// MMIO register access trait for hardware interaction.
pub trait MmioAccess {
    fn read32(&self, addr: u64) -> u32;
    fn write32(&self, addr: u64, val: u32);
}

/// GMMU initialization state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GmmuState {
    Uninitialized,
    PageTablesCreated,
    Active,
}

/// Page size selection for GMMU mappings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PageSize {
    /// 4 KiB small pages
    Small,
    /// 64 KiB big pages
    Big,
}

impl PageSize {
    /// Return the page size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            PageSize::Small => GMMU_PAGE_SIZE_SMALL,
            PageSize::Big => GMMU_PAGE_SIZE_BIG,
        }
    }
}

/// Build a Page Directory Entry pointing to a page table.
pub fn build_pde(pt_addr: u64, aperture: u32) -> u64 {
    ((pt_addr >> 12) << 12) | (aperture as u64 & 0x3)
}

/// Build a Page Table Entry mapping a physical page.
pub fn build_pte(phys_addr: u64, readable: bool, writable: bool) -> u64 {
    let mut pte = (phys_addr >> 12) << 12;
    pte |= GMMU_PTE_VALID;
    if readable {
        pte |= GMMU_PTE_READ;
    }
    if writable {
        pte |= GMMU_PTE_WRITE;
    }
    pte
}

/// A tracked memory mapping region.
#[derive(Clone, Debug, PartialEq)]
struct MappedRegion {
    virt_start: u64,
    phys_start: u64,
    size: u64,
}

/// GPU MMU page table manager.
pub struct GmmuPageTable {
    state: GmmuState,
    pdb_addr: u64,
    page_size: PageSize,
    mapped_regions: Vec<MappedRegion>,
}

impl GmmuPageTable {
    /// Create a new GMMU page table with the given page size.
    pub fn new(page_size: PageSize) -> Self {
        Self {
            state: GmmuState::Uninitialized,
            pdb_addr: 0,
            page_size,
            mapped_regions: Vec::new(),
        }
    }

    /// Create an identity mapping (GPU VA == physical address).
    ///
    /// The mapping covers `[phys_start, phys_start + size)`.
    pub fn create_identity_map(
        &mut self,
        phys_start: u64,
        size: u64,
    ) -> Result<(), GpuError> {
        if size == 0 {
            return Err(GpuError::InvalidConfig);
        }

        // Align to page boundaries
        let page_size = self.page_size.size_bytes() as u64;
        let aligned_start = phys_start & !(page_size - 1);
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        self.mapped_regions.push(MappedRegion {
            virt_start: aligned_start,
            phys_start: aligned_start,
            size: aligned_size,
        });

        self.state = GmmuState::PageTablesCreated;
        Ok(())
    }

    /// Activate the page table by writing the PDB base to GPU registers.
    pub fn activate(
        &mut self,
        bar0_base: u64,
        mmio: &impl MmioAccess,
    ) -> Result<(), GpuError> {
        if self.state != GmmuState::PageTablesCreated {
            return Err(GpuError::InvalidState);
        }

        // Write page directory base to the instance block register
        let pdb_reg_addr = bar0_base + GMMU_PDB_BASE_REG as u64;
        mmio.write32(pdb_reg_addr, (self.pdb_addr >> 12) as u32);

        self.state = GmmuState::Active;
        Ok(())
    }

    /// Invalidate the GPU TLB to flush stale translations.
    pub fn invalidate_tlb(&self, bar0_base: u64, mmio: &impl MmioAccess) {
        let tlb_addr = bar0_base + GMMU_TLB_INVALIDATE as u64;
        mmio.write32(tlb_addr, 1); // Trigger TLB invalidate
    }

    /// Check whether the GMMU is active.
    pub fn is_active(&self) -> bool {
        self.state == GmmuState::Active
    }

    /// Total bytes mapped across all regions.
    pub fn mapped_size(&self) -> u64 {
        self.mapped_regions.iter().map(|r| r.size).sum()
    }

    /// Return the current state.
    pub fn state(&self) -> GmmuState {
        self.state
    }

    /// Return the configured page size.
    pub fn page_size(&self) -> PageSize {
        self.page_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;

    struct MockMmio {
        writes: RefCell<alloc::vec::Vec<(u64, u32)>>,
    }

    impl MockMmio {
        fn new() -> Self {
            Self {
                writes: RefCell::new(alloc::vec::Vec::new()),
            }
        }
    }

    impl MmioAccess for MockMmio {
        fn read32(&self, _addr: u64) -> u32 {
            0
        }

        fn write32(&self, addr: u64, val: u32) {
            self.writes.borrow_mut().push((addr, val));
        }
    }

    // --- PDE/PTE construction tests ---

    #[test]
    fn test_build_pde_alignment() {
        let pde = build_pde(0x0010_0000, 1);
        // Address bits [63:12] preserved, aperture in [1:0]
        assert_eq!(pde & 0x3, 1);
        assert_eq!(pde & !0xFFF, 0x0010_0000);
    }

    #[test]
    fn test_build_pde_aperture_mask() {
        let pde = build_pde(0x2000_0000, 0xFF);
        // Only lower 2 bits of aperture should be kept
        assert_eq!(pde & 0x3, 3);
    }

    #[test]
    fn test_build_pte_valid_bit() {
        let pte = build_pte(0x1000, false, false);
        assert_ne!(pte & GMMU_PTE_VALID, 0);
    }

    #[test]
    fn test_build_pte_read_flag() {
        let pte = build_pte(0x1000, true, false);
        assert_ne!(pte & GMMU_PTE_READ, 0);
        assert_eq!(pte & GMMU_PTE_WRITE, 0);
    }

    #[test]
    fn test_build_pte_write_flag() {
        let pte = build_pte(0x1000, false, true);
        assert_eq!(pte & GMMU_PTE_READ, 0);
        assert_ne!(pte & GMMU_PTE_WRITE, 0);
    }

    #[test]
    fn test_build_pte_read_write() {
        let pte = build_pte(0x8000_0000, true, true);
        assert_ne!(pte & GMMU_PTE_VALID, 0);
        assert_ne!(pte & GMMU_PTE_READ, 0);
        assert_ne!(pte & GMMU_PTE_WRITE, 0);
        // Physical address preserved in upper bits
        assert_eq!(pte & !0xFFF, 0x8000_0000);
    }

    #[test]
    fn test_build_pte_address_alignment() {
        // Unaligned address should be masked to page boundary
        let pte = build_pte(0x1234, true, true);
        assert_eq!(pte & !0xFFF, 0x1000);
    }

    // --- Page size tests ---

    #[test]
    fn test_page_size_small() {
        assert_eq!(PageSize::Small.size_bytes(), 4096);
    }

    #[test]
    fn test_page_size_big() {
        assert_eq!(PageSize::Big.size_bytes(), 65536);
    }

    // --- GmmuPageTable state machine tests ---

    #[test]
    fn test_gmmu_initial_state() {
        let pt = GmmuPageTable::new(PageSize::Small);
        assert_eq!(pt.state(), GmmuState::Uninitialized);
        assert!(!pt.is_active());
        assert_eq!(pt.mapped_size(), 0);
    }

    #[test]
    fn test_identity_map() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        pt.create_identity_map(0x8000_0000, 0x10_0000).unwrap();

        assert_eq!(pt.state(), GmmuState::PageTablesCreated);
        assert_eq!(pt.mapped_size(), 0x10_0000);
    }

    #[test]
    fn test_identity_map_zero_size() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        assert_eq!(
            pt.create_identity_map(0x1000, 0),
            Err(GpuError::InvalidConfig)
        );
    }

    #[test]
    fn test_identity_map_alignment_small() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        // Request 1 byte — should be rounded up to one 4K page
        pt.create_identity_map(0x1000, 1).unwrap();
        assert_eq!(pt.mapped_size(), 4096);
    }

    #[test]
    fn test_identity_map_alignment_big() {
        let mut pt = GmmuPageTable::new(PageSize::Big);
        // Request 1 byte — should be rounded up to one 64K page
        pt.create_identity_map(0x1_0000, 1).unwrap();
        assert_eq!(pt.mapped_size(), 65536);
    }

    #[test]
    fn test_multiple_mappings() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        pt.create_identity_map(0x8000_0000, 0x10_0000).unwrap();
        pt.create_identity_map(0x9000_0000, 0x20_0000).unwrap();

        assert_eq!(pt.mapped_size(), 0x10_0000 + 0x20_0000);
    }

    #[test]
    fn test_activate() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        pt.create_identity_map(0x8000_0000, 0x10_0000).unwrap();

        let mmio = MockMmio::new();
        let bar0 = 0x5000_0000u64;
        pt.activate(bar0, &mmio).unwrap();

        assert!(pt.is_active());
        assert_eq!(pt.state(), GmmuState::Active);
    }

    #[test]
    fn test_activate_without_mapping() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        let mmio = MockMmio::new();

        // Can't activate from Uninitialized
        assert_eq!(pt.activate(0x5000_0000, &mmio), Err(GpuError::InvalidState));
    }

    #[test]
    fn test_activate_writes_pdb_register() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        pt.create_identity_map(0x8000_0000, 0x1000).unwrap();

        let mmio = MockMmio::new();
        let bar0 = 0x5000_0000u64;
        pt.activate(bar0, &mmio).unwrap();

        let writes = mmio.writes.borrow();
        let pdb_addr = bar0 + GMMU_PDB_BASE_REG as u64;
        assert!(writes.iter().any(|(addr, _)| *addr == pdb_addr));
    }

    #[test]
    fn test_invalidate_tlb() {
        let mut pt = GmmuPageTable::new(PageSize::Small);
        pt.create_identity_map(0x8000_0000, 0x1000).unwrap();

        let mmio = MockMmio::new();
        let bar0 = 0x5000_0000u64;
        pt.activate(bar0, &mmio).unwrap();
        pt.invalidate_tlb(bar0, &mmio);

        let writes = mmio.writes.borrow();
        let tlb_addr = bar0 + GMMU_TLB_INVALIDATE as u64;
        assert!(writes.iter().any(|(addr, _)| *addr == tlb_addr));
    }

    #[test]
    fn test_page_size_selection() {
        let small = GmmuPageTable::new(PageSize::Small);
        assert_eq!(small.page_size(), PageSize::Small);

        let big = GmmuPageTable::new(PageSize::Big);
        assert_eq!(big.page_size(), PageSize::Big);
    }
}
