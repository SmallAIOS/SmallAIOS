// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Buddy allocator — physical page-granularity memory allocation.
//!
//! Supports orders 0 through MAX_ORDER:
//! - Order 0: 4 KiB (1 page)
//! - Order 9: 2 MiB (huge page)
//! - Order 18: 1 GiB (gigantic page)
//! - Order 21: 8 GiB (maximum)
//!
//! The allocator uses a bitmap per order to track free blocks.
//! Split and merge operations maintain the buddy invariant:
//! a block can only merge with its buddy (XOR of the block's index).

use super::{MemError, PhysAddr, PAGE_SIZE_4K};

/// Maximum order (2^21 pages = 8 GiB).
pub const MAX_ORDER: usize = 21;

/// Number of buddy orders (0..=MAX_ORDER).
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// Maximum number of blocks we can track at order 0.
/// Default: 256K pages (1 GiB). Use `large-memory` feature for 16M pages (64 GiB).
/// Tests: 1024 pages (4 MiB) to fit on stack.
///
/// Note: Each order bitmap is MAX_PAGES/64 × 8 bytes. With 22 orders + alloc bitmap,
/// total static size ≈ MAX_PAGES × 23 / 8 bytes. At 256K pages ≈ 736 KB; at 16M ≈ 46 MB.
#[cfg(not(test))]
#[cfg(feature = "large-memory")]
const MAX_PAGES: usize = 16 * 1024 * 1024;
#[cfg(not(test))]
#[cfg(not(feature = "large-memory"))]
const MAX_PAGES: usize = 256 * 1024;
#[cfg(test)]
const MAX_PAGES: usize = 1024;

/// Bitmap word type.
type BitmapWord = u64;
const BITS_PER_WORD: usize = 64;

/// Number of bitmap words needed for a given number of bits.
const fn bitmap_words(bits: usize) -> usize {
    bits.div_ceil(BITS_PER_WORD)
}

/// Maximum bitmap words needed at any order.
/// Order 0 has the most blocks (MAX_PAGES), higher orders have fewer.
const MAX_BITMAP_WORDS: usize = bitmap_words(MAX_PAGES);

/// A free-list bitmap for one order level.
struct OrderBitmap {
    /// Each bit: 1 = free, 0 = allocated or not present.
    bits: [BitmapWord; MAX_BITMAP_WORDS],
    /// Number of valid blocks at this order.
    num_blocks: usize,
    /// Count of free blocks.
    free_count: usize,
}

impl OrderBitmap {
    const fn new() -> Self {
        Self {
            bits: [0; MAX_BITMAP_WORDS],
            num_blocks: 0,
            free_count: 0,
        }
    }

    /// Mark block `idx` as free.
    fn set_free(&mut self, idx: usize) {
        let word = idx / BITS_PER_WORD;
        let bit = idx % BITS_PER_WORD;
        if self.bits[word] & (1 << bit) == 0 {
            self.bits[word] |= 1 << bit;
            self.free_count += 1;
        }
    }

    /// Mark block `idx` as allocated.
    fn set_allocated(&mut self, idx: usize) {
        let word = idx / BITS_PER_WORD;
        let bit = idx % BITS_PER_WORD;
        if self.bits[word] & (1 << bit) != 0 {
            self.bits[word] &= !(1 << bit);
            self.free_count -= 1;
        }
    }

    /// Check if block `idx` is free.
    fn is_free(&self, idx: usize) -> bool {
        let word = idx / BITS_PER_WORD;
        let bit = idx % BITS_PER_WORD;
        self.bits[word] & (1 << bit) != 0
    }

    /// Find any free block. Returns its index or None.
    fn find_free(&self) -> Option<usize> {
        for (wi, &word) in self.bits.iter().enumerate() {
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                let idx = wi * BITS_PER_WORD + bit;
                if idx < self.num_blocks {
                    return Some(idx);
                }
            }
        }
        None
    }
}

/// Simple page-level bitmap for tracking which pages are allocated.
/// Used for double-free detection — one bit per page.
struct AllocBitmap {
    bits: [BitmapWord; MAX_BITMAP_WORDS],
}

impl AllocBitmap {
    const fn new() -> Self {
        Self {
            bits: [0; MAX_BITMAP_WORDS],
        }
    }

    /// Mark a range of pages as allocated.
    fn mark_allocated(&mut self, page_idx: usize, count: usize) {
        for i in page_idx..page_idx + count {
            let word = i / BITS_PER_WORD;
            let bit = i % BITS_PER_WORD;
            self.bits[word] |= 1 << bit;
        }
    }

    /// Mark a range of pages as free.
    fn mark_free(&mut self, page_idx: usize, count: usize) {
        for i in page_idx..page_idx + count {
            let word = i / BITS_PER_WORD;
            let bit = i % BITS_PER_WORD;
            self.bits[word] &= !(1 << bit);
        }
    }

    /// Check if a page is allocated.
    fn is_allocated(&self, page_idx: usize) -> bool {
        let word = page_idx / BITS_PER_WORD;
        let bit = page_idx % BITS_PER_WORD;
        self.bits[word] & (1 << bit) != 0
    }
}

/// Buddy allocator for physical page allocation.
pub struct BuddyAllocator {
    /// Base physical address of the managed region.
    base: PhysAddr,
    /// Total number of pages in the managed region.
    total_pages: usize,
    /// Bitmap for each order level.
    orders: [OrderBitmap; NUM_ORDERS],
    /// Tracks which pages have been handed out (for double-free detection).
    alloc_map: AllocBitmap,
    /// Total free pages (in order-0 equivalents).
    free_pages: usize,
}

impl Default for BuddyAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl BuddyAllocator {
    /// Create a new allocator. All memory starts as allocated;
    /// call `add_free_region` to register usable memory.
    pub const fn new() -> Self {
        const NEW_BITMAP: OrderBitmap = OrderBitmap::new();
        Self {
            base: PhysAddr(0),
            total_pages: 0,
            orders: [NEW_BITMAP; NUM_ORDERS],
            alloc_map: AllocBitmap::new(),
            free_pages: 0,
        }
    }

    /// Initialize the allocator for a physical memory range.
    ///
    /// `base` must be page-aligned. `size` is in bytes.
    pub fn init(&mut self, base: PhysAddr, size: usize) {
        assert!(base.is_aligned(PAGE_SIZE_4K), "base must be page-aligned");
        self.base = base;
        self.total_pages = size / PAGE_SIZE_4K;

        // Set num_blocks at each order
        for order in 0..NUM_ORDERS {
            let block_pages = 1 << order;
            self.orders[order].num_blocks = self.total_pages / block_pages;
        }
    }

    /// Add a free region to the allocator. The region is broken into
    /// the largest possible buddy blocks.
    ///
    /// `start` and `size` must be page-aligned.
    pub fn add_free_region(&mut self, start: PhysAddr, size: usize) {
        if size == 0 || !start.is_aligned(PAGE_SIZE_4K) {
            return;
        }

        let mut addr = start.as_usize();
        let end = addr + (size & !(PAGE_SIZE_4K - 1));

        while addr < end {
            // Find the largest order block that:
            // 1. Fits within remaining space
            // 2. Is naturally aligned
            let offset = addr - self.base.as_usize();
            let page_idx = offset / PAGE_SIZE_4K;
            let remaining_pages = (end - addr) / PAGE_SIZE_4K;

            let mut order = 0;
            while order < MAX_ORDER {
                let next_block_pages = 1usize << (order + 1);
                // Must be aligned to next order and have enough space
                if !page_idx.is_multiple_of(next_block_pages) || remaining_pages < next_block_pages
                {
                    break;
                }
                order += 1;
            }

            let block_pages = 1usize << order;
            let block_idx = page_idx / block_pages;
            self.orders[order].set_free(block_idx);
            self.free_pages += block_pages;

            addr += block_pages * PAGE_SIZE_4K;
        }
    }

    /// Allocate a block of `2^order` pages.
    ///
    /// Returns the physical address of the allocated block.
    pub fn allocate(&mut self, order: usize) -> Result<PhysAddr, MemError> {
        if order > MAX_ORDER {
            return Err(MemError::OrderTooLarge);
        }

        // Find the smallest order >= requested that has a free block
        let mut found_order = order;
        while found_order <= MAX_ORDER {
            if self.orders[found_order].find_free().is_some() {
                break;
            }
            found_order += 1;
        }

        if found_order > MAX_ORDER {
            return Err(MemError::OutOfMemory);
        }

        // Take the free block at found_order
        let block_idx = self.orders[found_order].find_free().unwrap();
        self.orders[found_order].set_allocated(block_idx);

        // Split down to the requested order
        let mut current_order = found_order;
        let mut current_idx = block_idx;
        while current_order > order {
            current_order -= 1;
            // Split: the current block becomes two blocks at the next lower order
            let left_idx = current_idx * 2;
            let right_idx = left_idx + 1;
            // Keep left, free right (the buddy)
            self.orders[current_order].set_free(right_idx);
            current_idx = left_idx;
        }

        let page_idx = current_idx * (1 << order);
        let addr = self.base.as_usize() + page_idx * PAGE_SIZE_4K;
        self.free_pages -= 1 << order;
        self.alloc_map.mark_allocated(page_idx, 1 << order);

        Ok(PhysAddr::new(addr))
    }

    /// Allocate a single 4 KiB page.
    pub fn allocate_page(&mut self) -> Result<PhysAddr, MemError> {
        self.allocate(0)
    }

    /// Free a block of `2^order` pages at the given physical address.
    pub fn free(&mut self, addr: PhysAddr, order: usize) -> Result<(), MemError> {
        if order > MAX_ORDER {
            return Err(MemError::OrderTooLarge);
        }
        if !addr.is_aligned(PAGE_SIZE_4K) {
            return Err(MemError::BadAlignment);
        }

        let offset = addr
            .as_usize()
            .checked_sub(self.base.as_usize())
            .ok_or(MemError::InvalidAddress)?;
        let page_idx = offset / PAGE_SIZE_4K;
        let block_pages = 1usize << order;

        if !page_idx.is_multiple_of(block_pages) {
            return Err(MemError::BadAlignment);
        }

        // Check for double free using the allocation bitmap
        if !self.alloc_map.is_allocated(page_idx) {
            return Err(MemError::DoubleFree);
        }
        self.alloc_map.mark_free(page_idx, block_pages);

        let mut current_order = order;
        let mut current_idx = page_idx / block_pages;

        self.free_pages += block_pages;

        // Merge with buddies as long as possible
        while current_order < MAX_ORDER {
            let buddy_idx = current_idx ^ 1;

            // Check buddy is valid and free
            if buddy_idx >= self.orders[current_order].num_blocks {
                break;
            }
            if !self.orders[current_order].is_free(buddy_idx) {
                break;
            }

            // Merge: remove buddy from this order, move up
            self.orders[current_order].set_allocated(buddy_idx);
            current_idx /= 2;
            current_order += 1;
        }

        // Mark the merged block as free
        self.orders[current_order].set_free(current_idx);

        Ok(())
    }

    /// Free a single 4 KiB page.
    pub fn free_page(&mut self, addr: PhysAddr) -> Result<(), MemError> {
        self.free(addr, 0)
    }

    /// Total managed pages.
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    /// Number of free pages.
    pub fn free_pages(&self) -> usize {
        self.free_pages
    }

    /// Number of free blocks at a given order.
    pub fn free_blocks_at_order(&self, order: usize) -> usize {
        if order > MAX_ORDER {
            return 0;
        }
        self.orders[order].free_count
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::boxed::Box;

    // BuddyAllocator can be large (up to ~44 MB with large-memory feature), must be heap-allocated in tests.
    fn make_allocator(pages: usize) -> Box<BuddyAllocator> {
        let mut alloc = Box::new(BuddyAllocator::new());
        let base = PhysAddr::new(0x10_0000); // 1 MiB
        let size = pages * PAGE_SIZE_4K;
        alloc.init(base, size);
        alloc.add_free_region(base, size);
        alloc
    }

    #[test]
    fn test_allocate_single_page() {
        let mut alloc = make_allocator(16);
        let addr = alloc.allocate_page().unwrap();
        assert!(addr.is_aligned(PAGE_SIZE_4K));
        assert_eq!(alloc.free_pages(), 15);
    }

    #[test]
    fn test_allocate_and_free() {
        let mut alloc = make_allocator(16);
        let addr = alloc.allocate_page().unwrap();
        alloc.free_page(addr).unwrap();
        assert_eq!(alloc.free_pages(), 16);
    }

    #[test]
    fn test_double_free_detected() {
        let mut alloc = make_allocator(16);
        let addr = alloc.allocate_page().unwrap();
        alloc.free_page(addr).unwrap();
        assert_eq!(alloc.free_page(addr), Err(MemError::DoubleFree));
    }

    #[test]
    fn test_exhaust_memory() {
        let mut alloc = make_allocator(4);
        let _a = alloc.allocate_page().unwrap();
        let _b = alloc.allocate_page().unwrap();
        let _c = alloc.allocate_page().unwrap();
        let _d = alloc.allocate_page().unwrap();
        assert_eq!(alloc.allocate_page(), Err(MemError::OutOfMemory));
    }

    #[test]
    fn test_buddy_merge() {
        let mut alloc = make_allocator(4);
        let a = alloc.allocate_page().unwrap();
        let b = alloc.allocate_page().unwrap();
        let c = alloc.allocate_page().unwrap();
        let d = alloc.allocate_page().unwrap();

        // Free all — should coalesce back to a single order-2 block
        alloc.free_page(a).unwrap();
        alloc.free_page(b).unwrap();
        alloc.free_page(c).unwrap();
        alloc.free_page(d).unwrap();
        assert_eq!(alloc.free_pages(), 4);

        // Should be able to allocate an order-2 block now
        let big = alloc.allocate(2).unwrap();
        assert!(big.is_aligned(PAGE_SIZE_4K * 4));
        assert_eq!(alloc.free_pages(), 0);
    }

    #[test]
    fn test_large_order_allocation() {
        // 256 pages = 1 MiB
        let mut alloc = make_allocator(256);
        // Order 8 = 256 pages
        let addr = alloc.allocate(8).unwrap();
        assert!(addr.is_aligned(PAGE_SIZE_4K * 256));
        assert_eq!(alloc.free_pages(), 0);
        alloc.free(addr, 8).unwrap();
        assert_eq!(alloc.free_pages(), 256);
    }

    #[test]
    fn test_split_and_recombine() {
        let mut alloc = make_allocator(8);
        // Allocate order 0 splits the order-3 block down
        let p1 = alloc.allocate_page().unwrap();
        assert_eq!(alloc.free_pages(), 7);
        // Free it and everything should coalesce
        alloc.free_page(p1).unwrap();
        assert_eq!(alloc.free_pages(), 8);
        // Order 3 (8 pages) should be available
        let big = alloc.allocate(3).unwrap();
        assert_eq!(alloc.free_pages(), 0);
        alloc.free(big, 3).unwrap();
    }

    #[test]
    fn test_order_too_large() {
        let mut alloc = make_allocator(16);
        assert_eq!(alloc.allocate(MAX_ORDER + 1), Err(MemError::OrderTooLarge));
        assert_eq!(
            alloc.free(PhysAddr::new(0x10_0000), MAX_ORDER + 1),
            Err(MemError::OrderTooLarge)
        );
    }

    #[test]
    fn test_alignment_error() {
        let mut alloc = make_allocator(16);
        assert_eq!(
            alloc.free(PhysAddr::new(0x10_0001), 0),
            Err(MemError::BadAlignment)
        );
    }

    #[test]
    fn test_invalid_address() {
        let mut alloc = make_allocator(16);
        // Address below base
        assert_eq!(
            alloc.free(PhysAddr::new(0x1000), 0),
            Err(MemError::InvalidAddress)
        );
    }
}
