// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory mapping operations for the POSIX compatibility layer.
//!
//! Provides `mmap`, `munmap`, and `mprotect` interfaces. Currently these are
//! validated stubs — argument checks are performed but no actual page table
//! modifications happen until the kernel memory manager is wired up.
//!
//! Only `MAP_ANONYMOUS | MAP_PRIVATE` mappings are supported (sufficient for
//! ONNX runtime tensor allocation). File-backed and shared mappings are
//! rejected with `ENOSYS`.

use crate::errno::Errno;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Page size in bytes (4 KiB).
pub const PAGE_SIZE: usize = 4096;

/// Sentinel value returned by libc `mmap` on failure.
pub const MAP_FAILED: usize = usize::MAX;

// ─── Protection Flags ────────────────────────────────────────────────────────

/// No access permitted.
pub const PROT_NONE: u32 = 0;

/// Pages may be read.
pub const PROT_READ: u32 = 1;

/// Pages may be written.
pub const PROT_WRITE: u32 = 2;

/// Pages may be executed.
pub const PROT_EXEC: u32 = 4;

/// Mask of all valid protection bits.
const PROT_MASK: u32 = PROT_READ | PROT_WRITE | PROT_EXEC;

// ─── Map Flags ───────────────────────────────────────────────────────────────

/// Share mapping with other processes (not supported in unikernel).
pub const MAP_SHARED: u32 = 1;

/// Create a private copy-on-write mapping.
pub const MAP_PRIVATE: u32 = 2;

/// Place mapping at exactly the specified address.
pub const MAP_FIXED: u32 = 0x10;

/// Mapping is not backed by any file (anonymous).
pub const MAP_ANONYMOUS: u32 = 0x20;

/// Use huge pages for the mapping.
pub const MAP_HUGETLB: u32 = 0x40000;

// ─── Mmap Region ─────────────────────────────────────────────────────────────

/// Maximum number of tracked memory-mapped regions per task.
pub const MAX_MMAP_REGIONS: usize = 128;

/// Describes a single memory-mapped region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmapRegion {
    /// Base virtual address of the mapping.
    pub base: usize,
    /// Length of the mapping in bytes (page-aligned).
    pub length: usize,
    /// Protection flags (`PROT_*`).
    pub prot: u32,
    /// Mapping flags (`MAP_*`).
    pub flags: u32,
}

impl MmapRegion {
    /// Create a new mmap region descriptor.
    pub const fn new(base: usize, length: usize, prot: u32, flags: u32) -> Self {
        Self {
            base,
            length,
            prot,
            flags,
        }
    }
}

// ─── Mmap Table ──────────────────────────────────────────────────────────────

/// Per-task table tracking active memory mappings.
pub struct MmapTable {
    /// Region entries. `None` = slot is free.
    regions: [Option<MmapRegion>; MAX_MMAP_REGIONS],
    /// Number of active mappings.
    count: usize,
}

impl MmapTable {
    /// Create an empty mmap table.
    pub fn new() -> Self {
        Self {
            regions: [None; MAX_MMAP_REGIONS],
            count: 0,
        }
    }

    /// Return the number of active mappings.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Find the first free slot in the table.
    fn find_free_slot(&self) -> Result<usize, Errno> {
        for (i, slot) in self.regions.iter().enumerate() {
            if slot.is_none() {
                return Ok(i);
            }
        }
        Err(Errno::ENOMEM)
    }

    /// Find the slot index containing a region with the given base address.
    fn find_by_base(&self, base: usize) -> Option<usize> {
        for (i, slot) in self.regions.iter().enumerate() {
            if let Some(region) = slot {
                if region.base == base {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Create a new memory mapping.
    ///
    /// # Arguments
    ///
    /// * `addr`   — Hint address (0 = kernel chooses). `MAP_FIXED` forces exact placement.
    /// * `length` — Size of the mapping (must be > 0).
    /// * `prot`   — Protection flags (`PROT_*`).
    /// * `flags`  — Mapping flags (`MAP_*`).
    /// * `fd`     — File descriptor (-1 for anonymous mappings).
    /// * `offset` — Offset into the file (must be page-aligned).
    ///
    /// # Current Status
    ///
    /// This is a validated stub. Argument validation is performed, but the
    /// actual mapping is not yet backed by the kernel page allocator.
    /// Returns `ENOSYS` after validation passes.
    pub fn mmap(
        &mut self,
        addr: usize,
        length: usize,
        prot: u32,
        flags: u32,
        fd: i32,
        offset: u64,
    ) -> Result<usize, Errno> {
        // Length must be non-zero
        if length == 0 {
            return Err(Errno::EINVAL);
        }

        // Address hint must be page-aligned (if non-zero)
        if addr != 0 && addr % PAGE_SIZE != 0 {
            return Err(Errno::EINVAL);
        }

        // File offset must be page-aligned
        if offset % PAGE_SIZE as u64 != 0 {
            return Err(Errno::EINVAL);
        }

        // Validate protection flags
        if prot & !PROT_MASK != 0 {
            return Err(Errno::EINVAL);
        }

        // Must specify exactly one of MAP_SHARED or MAP_PRIVATE
        let sharing = flags & (MAP_SHARED | MAP_PRIVATE);
        if sharing == 0 || sharing == (MAP_SHARED | MAP_PRIVATE) {
            return Err(Errno::EINVAL);
        }

        // Only MAP_ANONYMOUS | MAP_PRIVATE is supported for now
        let is_anon = flags & MAP_ANONYMOUS != 0;
        let is_private = flags & MAP_PRIVATE != 0;
        if !is_anon || !is_private {
            return Err(Errno::ENOSYS);
        }

        // Anonymous mappings must have fd == -1
        if is_anon && fd != -1 {
            return Err(Errno::EINVAL);
        }

        // Anonymous mappings must have offset == 0
        if is_anon && offset != 0 {
            return Err(Errno::EINVAL);
        }

        // Ensure we have capacity
        let _slot = self.find_free_slot()?;

        // Stub: validation passes but actual allocation is not yet implemented.
        // Return ENOSYS to indicate the kernel needs to provide backing pages.
        Err(Errno::ENOSYS)
    }

    /// Unmap a previously mapped region.
    ///
    /// # Arguments
    ///
    /// * `addr`   — Start address (must be page-aligned).
    /// * `length` — Length to unmap (must be > 0).
    ///
    /// # Current Status
    ///
    /// Validated stub — checks alignment and region existence.
    pub fn munmap(&mut self, addr: usize, length: usize) -> Result<(), Errno> {
        // Length must be non-zero
        if length == 0 {
            return Err(Errno::EINVAL);
        }

        // Address must be page-aligned
        if addr % PAGE_SIZE != 0 {
            return Err(Errno::EINVAL);
        }

        // Look up the region (stub: would also need to handle partial unmaps)
        match self.find_by_base(addr) {
            Some(_idx) => {
                // Stub: actual page table teardown not yet implemented
                Err(Errno::ENOSYS)
            }
            None => Err(Errno::EINVAL),
        }
    }

    /// Change protection on a memory region.
    ///
    /// # Arguments
    ///
    /// * `addr`   — Start address (must be page-aligned).
    /// * `length` — Length of region (must be > 0).
    /// * `prot`   — New protection flags (`PROT_*`).
    ///
    /// # Current Status
    ///
    /// Validated stub — checks alignment and protection flags.
    pub fn mprotect(&mut self, addr: usize, length: usize, prot: u32) -> Result<(), Errno> {
        // Length must be non-zero
        if length == 0 {
            return Err(Errno::EINVAL);
        }

        // Address must be page-aligned
        if addr % PAGE_SIZE != 0 {
            return Err(Errno::EINVAL);
        }

        // Validate protection flags
        if prot & !PROT_MASK != 0 {
            return Err(Errno::EINVAL);
        }

        // Stub: actual page table permission changes not yet implemented
        Err(Errno::ENOSYS)
    }

    /// Record a mapping in the table (for use by the kernel allocator once
    /// the stub implementations are replaced).
    #[allow(dead_code)]
    fn record_region(&mut self, region: MmapRegion) -> Result<usize, Errno> {
        let slot = self.find_free_slot()?;
        self.regions[slot] = Some(region);
        self.count += 1;
        Ok(slot)
    }

    /// Remove a mapping record from the table.
    #[allow(dead_code)]
    fn remove_region(&mut self, idx: usize) {
        if idx < MAX_MMAP_REGIONS && self.regions[idx].is_some() {
            self.regions[idx] = None;
            self.count -= 1;
        }
    }
}

// ─── Helper Functions ────────────────────────────────────────────────────────

/// Round a length up to the next page boundary.
pub const fn page_align_up(len: usize) -> usize {
    (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Check whether an address is page-aligned.
pub const fn is_page_aligned(addr: usize) -> bool {
    addr % PAGE_SIZE == 0
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_zero_length_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(0, 0, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_unaligned_addr_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(
            4097, // not page-aligned
            PAGE_SIZE,
            PROT_READ,
            MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            0,
        );
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_unaligned_offset_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(0, PAGE_SIZE, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 1);
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_invalid_prot_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(
            0,
            PAGE_SIZE,
            0xFF, // invalid protection bits
            MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            0,
        );
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_no_sharing_flag_returns_einval() {
        let mut table = MmapTable::new();
        // Neither MAP_SHARED nor MAP_PRIVATE
        let result = table.mmap(0, PAGE_SIZE, PROT_READ, MAP_ANONYMOUS, -1, 0);
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_both_sharing_flags_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(
            0,
            PAGE_SIZE,
            PROT_READ,
            MAP_ANONYMOUS | MAP_SHARED | MAP_PRIVATE,
            -1,
            0,
        );
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_shared_returns_enosys() {
        let mut table = MmapTable::new();
        // MAP_SHARED is not supported
        let result = table.mmap(0, PAGE_SIZE, PROT_READ, MAP_SHARED, 3, 0);
        assert_eq!(result, Err(Errno::ENOSYS));
    }

    #[test]
    fn mmap_valid_anonymous_returns_enosys_stub() {
        let mut table = MmapTable::new();
        let result = table.mmap(
            0,
            PAGE_SIZE,
            PROT_READ | PROT_WRITE,
            MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            0,
        );
        // Stub: returns ENOSYS even after validation passes
        assert_eq!(result, Err(Errno::ENOSYS));
    }

    #[test]
    fn mmap_anon_with_fd_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(
            0,
            PAGE_SIZE,
            PROT_READ,
            MAP_ANONYMOUS | MAP_PRIVATE,
            3, // fd must be -1 for anonymous
            0,
        );
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn mmap_anon_with_nonzero_offset_returns_einval() {
        let mut table = MmapTable::new();
        let result = table.mmap(
            0,
            PAGE_SIZE,
            PROT_READ,
            MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            PAGE_SIZE as u64, // offset must be 0 for anonymous
        );
        assert_eq!(result, Err(Errno::EINVAL));
    }

    #[test]
    fn munmap_zero_length_returns_einval() {
        let mut table = MmapTable::new();
        assert_eq!(table.munmap(PAGE_SIZE, 0), Err(Errno::EINVAL));
    }

    #[test]
    fn munmap_unaligned_returns_einval() {
        let mut table = MmapTable::new();
        assert_eq!(table.munmap(1, PAGE_SIZE), Err(Errno::EINVAL));
    }

    #[test]
    fn munmap_no_such_region_returns_einval() {
        let mut table = MmapTable::new();
        assert_eq!(table.munmap(PAGE_SIZE, PAGE_SIZE), Err(Errno::EINVAL));
    }

    #[test]
    fn mprotect_zero_length_returns_einval() {
        let mut table = MmapTable::new();
        assert_eq!(table.mprotect(PAGE_SIZE, 0, PROT_READ), Err(Errno::EINVAL));
    }

    #[test]
    fn mprotect_unaligned_returns_einval() {
        let mut table = MmapTable::new();
        assert_eq!(
            table.mprotect(1, PAGE_SIZE, PROT_READ),
            Err(Errno::EINVAL)
        );
    }

    #[test]
    fn mprotect_invalid_prot_returns_einval() {
        let mut table = MmapTable::new();
        assert_eq!(
            table.mprotect(PAGE_SIZE, PAGE_SIZE, 0xFF),
            Err(Errno::EINVAL)
        );
    }

    #[test]
    fn mprotect_valid_returns_enosys_stub() {
        let mut table = MmapTable::new();
        assert_eq!(
            table.mprotect(PAGE_SIZE, PAGE_SIZE, PROT_READ | PROT_WRITE),
            Err(Errno::ENOSYS)
        );
    }

    #[test]
    fn page_align_up_works() {
        assert_eq!(page_align_up(0), 0);
        assert_eq!(page_align_up(1), PAGE_SIZE);
        assert_eq!(page_align_up(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(page_align_up(PAGE_SIZE + 1), 2 * PAGE_SIZE);
        assert_eq!(page_align_up(4095), PAGE_SIZE);
    }

    #[test]
    fn is_page_aligned_works() {
        assert!(is_page_aligned(0));
        assert!(is_page_aligned(PAGE_SIZE));
        assert!(is_page_aligned(2 * PAGE_SIZE));
        assert!(!is_page_aligned(1));
        assert!(!is_page_aligned(PAGE_SIZE + 1));
    }

    #[test]
    fn mmap_table_starts_empty() {
        let table = MmapTable::new();
        assert_eq!(table.count(), 0);
    }

    #[test]
    fn mmap_region_constructor() {
        let region = MmapRegion::new(0x1000, 0x2000, PROT_READ | PROT_WRITE, MAP_PRIVATE);
        assert_eq!(region.base, 0x1000);
        assert_eq!(region.length, 0x2000);
        assert_eq!(region.prot, PROT_READ | PROT_WRITE);
        assert_eq!(region.flags, MAP_PRIVATE);
    }

    #[test]
    fn map_failed_sentinel() {
        assert_eq!(MAP_FAILED, usize::MAX);
    }
}
