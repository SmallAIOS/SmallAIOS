// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Platform-independent page table interface and types.
//!
//! Both x86-64 and ARM64 use 4-level page tables with 4 KiB pages.
//! Architecture-specific implementations live in the arch crates and
//! implement the `PageTableOps` trait.

use super::{MemError, PhysAddr, VirtAddr};

/// Page table entry flags (platform-independent).
#[derive(Debug, Clone, Copy)]
pub struct PageFlags {
    bits: u64,
}

impl PageFlags {
    /// No flags set.
    pub const NONE: Self = Self { bits: 0 };
    /// Page is present/valid.
    pub const PRESENT: Self = Self { bits: 1 << 0 };
    /// Page is writable.
    pub const WRITABLE: Self = Self { bits: 1 << 1 };
    /// Page is accessible from user mode (not used in unikernel mode).
    pub const USER: Self = Self { bits: 1 << 2 };
    /// Page is not executable.
    pub const NO_EXECUTE: Self = Self { bits: 1 << 3 };
    /// Page is write-through cached.
    pub const WRITE_THROUGH: Self = Self { bits: 1 << 4 };
    /// Page caching is disabled (for MMIO regions).
    pub const CACHE_DISABLE: Self = Self { bits: 1 << 5 };
    /// Huge page (2 MiB or 1 GiB).
    pub const HUGE_PAGE: Self = Self { bits: 1 << 6 };
    /// Global page (not flushed on CR3 switch).
    pub const GLOBAL: Self = Self { bits: 1 << 7 };

    /// Create flags from a raw bitmask.
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    /// Get the raw bitmask.
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// Combine two sets of flags (bitwise OR).
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Check if specific flags are set.
    pub const fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }
}

/// Standard page flags for kernel code.
pub const KERNEL_CODE_FLAGS: PageFlags =
    PageFlags::from_bits(PageFlags::PRESENT.bits() | PageFlags::GLOBAL.bits());

/// Standard page flags for kernel data.
pub const KERNEL_DATA_FLAGS: PageFlags = PageFlags::from_bits(
    PageFlags::PRESENT.bits()
        | PageFlags::WRITABLE.bits()
        | PageFlags::NO_EXECUTE.bits()
        | PageFlags::GLOBAL.bits(),
);

/// Standard page flags for MMIO regions.
pub const MMIO_FLAGS: PageFlags = PageFlags::from_bits(
    PageFlags::PRESENT.bits()
        | PageFlags::WRITABLE.bits()
        | PageFlags::NO_EXECUTE.bits()
        | PageFlags::CACHE_DISABLE.bits(),
);

/// Standard page flags for tensor buffers (writable, NX, DMA-capable).
pub const TENSOR_FLAGS: PageFlags = PageFlags::from_bits(
    PageFlags::PRESENT.bits()
        | PageFlags::WRITABLE.bits()
        | PageFlags::NO_EXECUTE.bits()
        | PageFlags::GLOBAL.bits(),
);

/// Page table level (for 4-level paging).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLevel {
    /// Level 4: PML4 (x86-64) / L0 (ARM64) — 512 GiB per entry
    L4,
    /// Level 3: PDPT (x86-64) / L1 (ARM64) — 1 GiB per entry
    L3,
    /// Level 2: PD (x86-64) / L2 (ARM64) — 2 MiB per entry
    L2,
    /// Level 1: PT (x86-64) / L3 (ARM64) — 4 KiB per entry
    L1,
}

/// A raw 64-bit page table entry.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct PageTableEntry(pub u64);

impl PageTableEntry {
    pub const EMPTY: Self = Self(0);

    pub fn is_present(self) -> bool {
        self.0 & 1 != 0
    }

    pub fn is_huge(self) -> bool {
        self.0 & (1 << 7) != 0
    }

    pub fn address(self) -> PhysAddr {
        // Physical address is in bits [12..52] on x86-64
        // and bits [12..48] on ARM64 (at 4 KiB granule)
        PhysAddr::new((self.0 & 0x000F_FFFF_FFFF_F000) as usize)
    }
}

/// A page table: 512 entries, page-aligned.
#[repr(C, align(4096))]
pub struct PageTable {
    pub entries: [PageTableEntry; 512],
}

impl PageTable {
    /// Create an empty page table.
    pub const fn new() -> Self {
        Self {
            entries: [PageTableEntry::EMPTY; 512],
        }
    }

    /// Zero all entries.
    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = PageTableEntry::EMPTY;
        }
    }
}

/// Page table operations trait — implemented by arch-specific code.
pub trait PageTableOps {
    /// Map a 4 KiB virtual page to a physical frame.
    fn map_page(
        &mut self,
        virt: VirtAddr,
        phys: PhysAddr,
        flags: PageFlags,
    ) -> Result<(), MemError>;

    /// Unmap a 4 KiB virtual page.
    fn unmap_page(&mut self, virt: VirtAddr) -> Result<PhysAddr, MemError>;

    /// Translate a virtual address to its physical address.
    fn translate(&self, virt: VirtAddr) -> Result<PhysAddr, MemError>;

    /// Update the protection flags of a mapped page.
    fn protect(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MemError>;
}

/// Extract the page table index at a given level from a virtual address.
///
/// x86-64 4-level paging:
/// - L4 (PML4): bits [47:39]
/// - L3 (PDPT): bits [38:30]
/// - L2 (PD):   bits [29:21]
/// - L1 (PT):   bits [20:12]
pub const fn page_table_index(virt: VirtAddr, level: PageLevel) -> usize {
    let shift = match level {
        PageLevel::L4 => 39,
        PageLevel::L3 => 30,
        PageLevel::L2 => 21,
        PageLevel::L1 => 12,
    };
    (virt.as_usize() >> shift) & 0x1FF
}

/// Calculate the page offset (bits [11:0]) from a virtual address.
pub const fn page_offset(virt: VirtAddr) -> usize {
    virt.as_usize() & 0xFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_flags() {
        let flags = PageFlags::PRESENT.union(PageFlags::WRITABLE);
        assert!(flags.contains(PageFlags::PRESENT));
        assert!(flags.contains(PageFlags::WRITABLE));
        assert!(!flags.contains(PageFlags::USER));
    }

    #[test]
    fn test_page_table_index() {
        // Address 0xFFFF_8000_0010_0000 (kernel space, 1 MiB offset)
        let virt = VirtAddr::new(0x0000_0000_0010_0000);
        assert_eq!(page_table_index(virt, PageLevel::L4), 0);
        assert_eq!(page_table_index(virt, PageLevel::L3), 0);
        assert_eq!(page_table_index(virt, PageLevel::L2), 0);
        // 0x100000 >> 12 = 0x100 = 256
        assert_eq!(page_table_index(virt, PageLevel::L1), 256);
    }

    #[test]
    fn test_page_offset() {
        assert_eq!(page_offset(VirtAddr::new(0x1234)), 0x234);
        assert_eq!(page_offset(VirtAddr::new(0x1000)), 0x000);
    }

    #[test]
    fn test_page_table_entry() {
        let entry = PageTableEntry(0x0000_0000_0020_0003); // present, writable, addr 0x200000
        assert!(entry.is_present());
        assert_eq!(entry.address().as_usize(), 0x20_0000);
    }

    #[test]
    fn test_empty_page_table() {
        let pt = PageTable::new();
        for entry in pt.entries.iter() {
            assert!(!entry.is_present());
        }
    }
}
