// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU VRAM memory allocator.
//!
//! Manages two regions of video memory:
//! - **Static** — long-lived allocations for model weights, rarely freed.
//! - **Dynamic** — short-lived workspace buffers for intermediate tensors.
//!
//! All allocations are aligned to [`GPU_PAGE_SIZE`] (64 KiB).

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::GpuError;

/// GPU page size — 64 KiB, the native page granularity for NVIDIA GPUs.
pub const GPU_PAGE_SIZE: usize = 65536;

/// Maximum number of live allocations tracked by the allocator.
pub const MAX_ALLOCATIONS: usize = 4096;

/// Describes which VRAM region an allocation belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryRegion {
    /// Static region — model weights and other long-lived data.
    Static,
    /// Dynamic region — workspace buffers and temporaries.
    Dynamic,
}

/// A single VRAM allocation record.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuAllocation {
    /// Unique allocation identifier.
    pub id: u64,
    /// Byte offset within the VRAM region.
    pub offset: u64,
    /// Aligned size in bytes.
    pub size: u64,
    /// Which region this allocation lives in.
    pub region: MemoryRegion,
    /// Whether this allocation is currently in use.
    pub in_use: bool,
}

/// Bump-style VRAM allocator with static and dynamic regions.
#[derive(Clone, Debug, PartialEq)]
pub struct VramAllocator {
    /// Total VRAM managed by this allocator.
    total_size: u64,

    /// Base address of the static region.
    static_base: u64,
    /// Total capacity of the static region.
    static_size: u64,
    /// Bytes currently consumed in the static region.
    static_used: u64,

    /// Base address of the dynamic region.
    dynamic_base: u64,
    /// Total capacity of the dynamic region.
    dynamic_size: u64,
    /// Bytes currently consumed in the dynamic region.
    dynamic_used: u64,

    /// All tracked allocations (both live and freed).
    allocations: Vec<GpuAllocation>,

    /// Monotonically increasing allocation-id generator.
    next_id: u64,
}

/// Align `size` up to the nearest multiple of [`GPU_PAGE_SIZE`].
fn align_up(size: u64) -> u64 {
    let page = GPU_PAGE_SIZE as u64;
    (size + page - 1) & !(page - 1)
}

impl VramAllocator {
    /// Create a new VRAM allocator.
    ///
    /// `total_size` — total VRAM in bytes.
    /// `static_fraction` — fraction of VRAM reserved for the static region
    /// (0.0..=1.0; typically 0.7 for 70 % weights / 30 % workspace).
    pub fn new(total_size: u64, static_fraction: f64) -> Self {
        let raw_static = (total_size as f64 * static_fraction) as u64;
        let static_size = align_up(raw_static);
        // Clamp so static never exceeds total.
        let static_size = if static_size > total_size {
            total_size
        } else {
            static_size
        };
        let dynamic_size = total_size - static_size;

        Self {
            total_size,
            static_base: 0,
            static_size,
            static_used: 0,
            dynamic_base: static_size,
            dynamic_size,
            dynamic_used: 0,
            allocations: Vec::new(),
            next_id: 1,
        }
    }

    /// Allocate `size` bytes from the requested `region`.
    ///
    /// Returns the unique allocation id on success.
    pub fn alloc(&mut self, size: u64, region: MemoryRegion) -> Result<u64, GpuError> {
        if size == 0 {
            return Err(GpuError::MemoryError);
        }

        if self.allocation_count() >= MAX_ALLOCATIONS {
            return Err(GpuError::MemoryError);
        }

        let aligned = align_up(size);

        let (base, region_size, used) = match region {
            MemoryRegion::Static => (self.static_base, self.static_size, &mut self.static_used),
            MemoryRegion::Dynamic => (self.dynamic_base, self.dynamic_size, &mut self.dynamic_used),
        };

        if *used + aligned > region_size {
            return Err(GpuError::MemoryError);
        }

        let offset = base + *used;
        *used += aligned;

        let id = self.next_id;
        self.next_id += 1;

        self.allocations.push(GpuAllocation {
            id,
            offset,
            size: aligned,
            region,
            in_use: true,
        });

        Ok(id)
    }

    /// Free a previous allocation by id.
    ///
    /// The freed bytes are returned to the region's used counter so they can be
    /// reclaimed.  Double-free is an error.
    pub fn free(&mut self, id: u64) -> Result<(), GpuError> {
        let alloc = self
            .allocations
            .iter_mut()
            .find(|a| a.id == id)
            .ok_or(GpuError::NotFound)?;

        if !alloc.in_use {
            return Err(GpuError::InvalidState);
        }

        alloc.in_use = false;

        match alloc.region {
            MemoryRegion::Static => self.static_used -= alloc.size,
            MemoryRegion::Dynamic => self.dynamic_used -= alloc.size,
        }

        Ok(())
    }

    /// Bytes currently used in `region`.
    pub fn used_bytes(&self, region: MemoryRegion) -> u64 {
        match region {
            MemoryRegion::Static => self.static_used,
            MemoryRegion::Dynamic => self.dynamic_used,
        }
    }

    /// Bytes still available in `region`.
    pub fn free_bytes(&self, region: MemoryRegion) -> u64 {
        match region {
            MemoryRegion::Static => self.static_size - self.static_used,
            MemoryRegion::Dynamic => self.dynamic_size - self.dynamic_used,
        }
    }

    /// Total bytes used across both regions.
    pub fn total_used(&self) -> u64 {
        self.static_used + self.dynamic_used
    }

    /// Total bytes free across both regions.
    pub fn total_free(&self) -> u64 {
        self.total_size - self.total_used()
    }

    /// Number of currently active (in-use) allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.iter().filter(|a| a.in_use).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_GIB: u64 = 1024 * 1024 * 1024;
    const PAGE: u64 = GPU_PAGE_SIZE as u64;

    // 1. New allocator splits correctly.
    #[test]
    fn test_new_allocator_splits_correctly() {
        let alloc = VramAllocator::new(ONE_GIB, 0.7);
        // Static region should be ~70 % of 1 GiB, aligned to 64 KiB.
        let expected_static = align_up((ONE_GIB as f64 * 0.7) as u64);
        assert_eq!(alloc.static_size, expected_static);
        assert_eq!(alloc.dynamic_size, ONE_GIB - expected_static);
        assert_eq!(alloc.static_base, 0);
        assert_eq!(alloc.dynamic_base, expected_static);
        assert_eq!(alloc.total_size, ONE_GIB);
    }

    // 2. Alloc in static region.
    #[test]
    fn test_alloc_static() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.7);
        let id = alloc.alloc(1024, MemoryRegion::Static).unwrap();
        assert_eq!(id, 1);
        assert_eq!(alloc.used_bytes(MemoryRegion::Static), PAGE);
    }

    // 3. Alloc in dynamic region.
    #[test]
    fn test_alloc_dynamic() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.7);
        let id = alloc.alloc(1024, MemoryRegion::Dynamic).unwrap();
        assert_eq!(id, 1);
        assert_eq!(alloc.used_bytes(MemoryRegion::Dynamic), PAGE);
    }

    // 4. Free allocation.
    #[test]
    fn test_free_allocation() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.7);
        let id = alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        assert_eq!(alloc.used_bytes(MemoryRegion::Static), PAGE);
        alloc.free(id).unwrap();
        assert_eq!(alloc.used_bytes(MemoryRegion::Static), 0);
    }

    // 5. Double free returns error.
    #[test]
    fn test_double_free_error() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.7);
        let id = alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        alloc.free(id).unwrap();
        assert_eq!(alloc.free(id), Err(GpuError::InvalidState));
    }

    // 6. Out of memory in region.
    #[test]
    fn test_out_of_memory() {
        // Tiny allocator: 2 pages total, 50 % each region = 1 page each.
        let mut alloc = VramAllocator::new(PAGE * 2, 0.5);
        alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        assert_eq!(
            alloc.alloc(PAGE, MemoryRegion::Static),
            Err(GpuError::MemoryError)
        );
    }

    // 7. Alignment to GPU_PAGE_SIZE.
    #[test]
    fn test_alignment() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.7);
        // Request 1 byte — should still consume one full page.
        alloc.alloc(1, MemoryRegion::Static).unwrap();
        assert_eq!(alloc.used_bytes(MemoryRegion::Static), PAGE);
    }

    // 8. Used / free byte tracking.
    #[test]
    fn test_used_free_tracking() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.5);
        let half = align_up((ONE_GIB as f64 * 0.5) as u64);
        assert_eq!(alloc.free_bytes(MemoryRegion::Static), half);
        alloc.alloc(PAGE * 3, MemoryRegion::Static).unwrap();
        assert_eq!(alloc.used_bytes(MemoryRegion::Static), PAGE * 3);
        assert_eq!(alloc.free_bytes(MemoryRegion::Static), half - PAGE * 3);
    }

    // 9. Total used / free.
    #[test]
    fn test_total_used_free() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.5);
        alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        alloc.alloc(PAGE * 2, MemoryRegion::Dynamic).unwrap();
        assert_eq!(alloc.total_used(), PAGE * 3);
        assert_eq!(alloc.total_free(), ONE_GIB - PAGE * 3);
    }

    // 10. Allocation count.
    #[test]
    fn test_allocation_count() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.5);
        assert_eq!(alloc.allocation_count(), 0);
        let id1 = alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        let _id2 = alloc.alloc(PAGE, MemoryRegion::Dynamic).unwrap();
        assert_eq!(alloc.allocation_count(), 2);
        alloc.free(id1).unwrap();
        assert_eq!(alloc.allocation_count(), 1);
    }

    // 11. Max allocations limit.
    #[test]
    fn test_max_allocations() {
        // We need enough VRAM for MAX_ALLOCATIONS pages.
        let big = PAGE * (MAX_ALLOCATIONS as u64 + 10);
        let mut alloc = VramAllocator::new(big, 1.0);
        for _ in 0..MAX_ALLOCATIONS {
            alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        }
        assert_eq!(
            alloc.alloc(PAGE, MemoryRegion::Static),
            Err(GpuError::MemoryError)
        );
    }

    // 12. Zero-size allocation handling.
    #[test]
    fn test_zero_size_alloc() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.5);
        assert_eq!(
            alloc.alloc(0, MemoryRegion::Static),
            Err(GpuError::MemoryError)
        );
    }

    // 13. Large allocation.
    #[test]
    fn test_large_allocation() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.7);
        let static_cap = alloc.free_bytes(MemoryRegion::Static);
        // Allocate entire static region.
        let id = alloc.alloc(static_cap, MemoryRegion::Static).unwrap();
        assert_eq!(alloc.free_bytes(MemoryRegion::Static), 0);
        alloc.free(id).unwrap();
        assert_eq!(alloc.free_bytes(MemoryRegion::Static), static_cap);
    }

    // 14. Free non-existent allocation returns NotFound.
    #[test]
    fn test_free_nonexistent() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.5);
        assert_eq!(alloc.free(999), Err(GpuError::NotFound));
    }

    // 15. Multiple sequential allocations have increasing offsets.
    #[test]
    fn test_sequential_offsets() {
        let mut alloc = VramAllocator::new(ONE_GIB, 0.5);
        let id1 = alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        let id2 = alloc.alloc(PAGE, MemoryRegion::Static).unwrap();
        let a1 = alloc.allocations.iter().find(|a| a.id == id1).unwrap();
        let a2 = alloc.allocations.iter().find(|a| a.id == id2).unwrap();
        assert_eq!(a2.offset, a1.offset + PAGE);
    }

    // 16. 100 % static fraction leaves zero dynamic.
    #[test]
    fn test_full_static_fraction() {
        let alloc = VramAllocator::new(ONE_GIB, 1.0);
        assert_eq!(alloc.static_size, ONE_GIB);
        assert_eq!(alloc.dynamic_size, 0);
    }

    // 17. 0 % static fraction leaves zero static.
    #[test]
    fn test_zero_static_fraction() {
        let alloc = VramAllocator::new(ONE_GIB, 0.0);
        assert_eq!(alloc.static_size, 0);
        assert_eq!(alloc.dynamic_size, ONE_GIB);
    }
}
