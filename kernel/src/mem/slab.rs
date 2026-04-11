// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Slab allocator — sub-page kernel object allocation.
//!
//! Size classes: 16, 32, 64, 128, 256, 512, 1024, 2048 bytes.
//! Each slab is one 4 KiB page. Objects are allocated from a free-list
//! within the slab. When a slab is exhausted, a new page is requested
//! from the buddy allocator.
//!
//! The slab header is stored at the beginning of the page, with objects
//! occupying the remainder. Free objects form a linked list via their
//! first 8 bytes (pointer to next free object).

use super::{MemError, PhysAddr, PAGE_SIZE_4K};

/// Size classes supported by the slab allocator.
pub const SIZE_CLASSES: [usize; 8] = [16, 32, 64, 128, 256, 512, 1024, 2048];

/// Number of size classes.
const NUM_CLASSES: usize = SIZE_CLASSES.len();

/// Maximum number of slabs per size class.
const MAX_SLABS_PER_CLASS: usize = 256;

/// Find the size class index for a given allocation size.
/// Returns None if the size exceeds the largest class.
pub fn size_class_index(size: usize) -> Option<usize> {
    for (i, &class_size) in SIZE_CLASSES.iter().enumerate() {
        if size <= class_size {
            return Some(i);
        }
    }
    None
}

/// Header stored at the beginning of each slab page.
#[repr(C)]
struct SlabHeader {
    /// Pointer to the first free object in this slab.
    free_list: *mut u8,
    /// Number of allocated objects.
    allocated: u16,
    /// Total capacity (objects per slab).
    capacity: u16,
    /// Object size for this slab.
    obj_size: u16,
    /// Padding for alignment.
    _pad: u16,
}

/// Per-class slab list.
struct SlabClass {
    /// Physical addresses of slab pages.
    slabs: [PhysAddr; MAX_SLABS_PER_CLASS],
    /// Number of active slabs.
    slab_count: usize,
    /// Object size for this class.
    obj_size: usize,
}

impl SlabClass {
    const fn new(obj_size: usize) -> Self {
        Self {
            slabs: [PhysAddr(0); MAX_SLABS_PER_CLASS],
            slab_count: 0,
            obj_size,
        }
    }

    /// Initialize a new slab page.
    ///
    /// # Safety
    /// `page_addr` must point to a valid, writable 4 KiB page.
    unsafe fn init_slab(&mut self, page_addr: PhysAddr) -> bool {
        if self.slab_count >= MAX_SLABS_PER_CLASS {
            return false;
        }

        let page_ptr = page_addr.as_usize() as *mut u8;
        let header_size = core::mem::size_of::<SlabHeader>();
        // Align object start to obj_size (or 16 minimum)
        let align = if self.obj_size >= 16 {
            self.obj_size
        } else {
            16
        };
        let data_start = (header_size + align - 1) & !(align - 1);
        let usable = PAGE_SIZE_4K - data_start;
        let capacity = usable / self.obj_size;

        if capacity == 0 {
            return false;
        }

        // Build free list: each free object's first 8 bytes point to the next
        let mut prev: *mut u8 = core::ptr::null_mut();
        for i in (0..capacity).rev() {
            let obj_ptr = page_ptr.add(data_start + i * self.obj_size);
            // Write the next pointer (or null for the last)
            (obj_ptr as *mut *mut u8).write(prev);
            prev = obj_ptr;
        }

        // Write slab header
        let header = page_ptr as *mut SlabHeader;
        (*header).free_list = prev;
        (*header).allocated = 0;
        (*header).capacity = capacity as u16;
        (*header).obj_size = self.obj_size as u16;
        (*header)._pad = 0;

        self.slabs[self.slab_count] = page_addr;
        self.slab_count += 1;
        true
    }

    /// Allocate an object from this size class.
    ///
    /// # Safety
    /// Slab pages must be valid and writable.
    unsafe fn allocate(&mut self) -> Option<*mut u8> {
        // Search existing slabs for a free object
        for i in 0..self.slab_count {
            let page_ptr = self.slabs[i].as_usize() as *mut u8;
            let header = page_ptr as *mut SlabHeader;

            if !(*header).free_list.is_null() {
                let obj = (*header).free_list;
                (*header).free_list = (obj as *const *mut u8).read();
                (*header).allocated += 1;
                return Some(obj);
            }
        }
        None
    }

    /// Free an object back to its slab.
    ///
    /// # Safety
    /// `ptr` must point to a previously allocated object within a slab page.
    unsafe fn free(&mut self, ptr: *mut u8) -> Result<(), MemError> {
        // Find which slab page this object belongs to
        let obj_addr = ptr as usize;
        let page_base = obj_addr & !(PAGE_SIZE_4K - 1);

        for i in 0..self.slab_count {
            if self.slabs[i].as_usize() == page_base {
                let header = page_base as *mut SlabHeader;
                // Push onto free list
                (ptr as *mut *mut u8).write((*header).free_list);
                (*header).free_list = ptr;
                (*header).allocated -= 1;
                return Ok(());
            }
        }

        Err(MemError::InvalidAddress)
    }
}

/// Slab allocator with multiple size classes.
pub struct SlabAllocator {
    classes: [SlabClass; NUM_CLASSES],
}

impl Default for SlabAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SlabAllocator {
    /// Create a new slab allocator (all classes empty).
    pub const fn new() -> Self {
        Self {
            classes: [
                SlabClass::new(16),
                SlabClass::new(32),
                SlabClass::new(64),
                SlabClass::new(128),
                SlabClass::new(256),
                SlabClass::new(512),
                SlabClass::new(1024),
                SlabClass::new(2048),
            ],
        }
    }

    /// Add a new slab page for a given size class.
    ///
    /// # Safety
    /// `page_addr` must point to a valid 4 KiB page from the buddy allocator.
    pub unsafe fn add_slab_page(&mut self, class_idx: usize, page_addr: PhysAddr) -> bool {
        if class_idx >= NUM_CLASSES {
            return false;
        }
        self.classes[class_idx].init_slab(page_addr)
    }

    /// Allocate an object of at least `size` bytes.
    ///
    /// # Safety
    /// The caller must ensure that slab pages are valid memory.
    pub unsafe fn allocate(&mut self, size: usize) -> Result<*mut u8, MemError> {
        let class_idx = size_class_index(size).ok_or(MemError::SizeTooLarge)?;
        self.classes[class_idx]
            .allocate()
            .ok_or(MemError::SlabExhausted)
    }

    /// Free a previously allocated object.
    ///
    /// # Safety
    /// `ptr` must be a valid pointer from a previous `allocate` call.
    /// `size` must be the original requested size (to determine size class).
    pub unsafe fn free(&mut self, ptr: *mut u8, size: usize) -> Result<(), MemError> {
        let class_idx = size_class_index(size).ok_or(MemError::SizeTooLarge)?;
        self.classes[class_idx].free(ptr)
    }

    /// Get the actual allocation size for a requested size.
    pub fn actual_size(size: usize) -> Option<usize> {
        size_class_index(size).map(|i| SIZE_CLASSES[i])
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;

    /// Allocate a page-aligned buffer for slab tests.
    fn page_aligned_buffer() -> alloc::vec::Vec<u8> {
        // Allocate 2 pages so we can find a page-aligned region within
        let buf = vec![0u8; PAGE_SIZE_4K * 2];
        buf
    }

    /// Get a page-aligned address within a buffer.
    fn page_aligned_addr(buf: &mut [u8]) -> PhysAddr {
        let ptr = buf.as_mut_ptr() as usize;
        let aligned = (ptr + PAGE_SIZE_4K - 1) & !(PAGE_SIZE_4K - 1);
        PhysAddr::new(aligned)
    }

    #[test]
    fn test_size_class_index() {
        assert_eq!(size_class_index(1), Some(0)); // 16
        assert_eq!(size_class_index(16), Some(0)); // 16
        assert_eq!(size_class_index(17), Some(1)); // 32
        assert_eq!(size_class_index(256), Some(4)); // 256
        assert_eq!(size_class_index(2048), Some(7)); // 2048
        assert_eq!(size_class_index(2049), None); // too large
    }

    #[test]
    fn test_actual_size() {
        assert_eq!(SlabAllocator::actual_size(1), Some(16));
        assert_eq!(SlabAllocator::actual_size(100), Some(128));
        assert_eq!(SlabAllocator::actual_size(3000), None);
    }

    #[test]
    fn test_slab_alloc_free() {
        // Create a page-aligned buffer (slab free uses page_base masking)
        let mut buf = page_aligned_buffer();
        let page_addr = page_aligned_addr(&mut buf);

        let mut slab = SlabAllocator::new();
        unsafe {
            assert!(slab.add_slab_page(0, page_addr)); // 16-byte class

            // Allocate several objects
            let a = slab.allocate(16).unwrap();
            let b = slab.allocate(16).unwrap();
            assert_ne!(a, b);

            // Free and re-allocate
            slab.free(a, 16).unwrap();
            let c = slab.allocate(16).unwrap();
            assert_eq!(c, a); // Should reuse freed slot
        }
    }

    #[test]
    fn test_slab_exhaustion() {
        let mut buf = page_aligned_buffer();
        let page_addr = page_aligned_addr(&mut buf);

        let mut slab = SlabAllocator::new();
        unsafe {
            slab.add_slab_page(7, page_addr); // 2048-byte class: only 1 object per page

            // Should get at least 1 allocation
            let _a = slab.allocate(2048).unwrap();
            // Eventually should exhaust
            let result = slab.allocate(2048);
            assert_eq!(result, Err(MemError::SlabExhausted));
        }
    }
}

// ─── Kani Proofs ────────────────────────────────────────────────────────────

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove: size_class_index never panics for any input.
    #[kani::proof]
    fn slab_size_class_no_panic() {
        let size: usize = kani::any();
        kani::assume(size <= 8192);
        let _ = size_class_index(size);
    }

    /// Prove: size_class_index returns correct class for valid sizes.
    #[kani::proof]
    fn slab_size_class_correct() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 2048);
        let idx = size_class_index(size).unwrap();
        assert!(SIZE_CLASSES[idx] >= size);
        if idx > 0 {
            assert!(SIZE_CLASSES[idx - 1] < size);
        }
    }

    /// Prove: actual_size never panics for any input.
    #[kani::proof]
    fn slab_actual_size_no_panic() {
        let size: usize = kani::any();
        kani::assume(size <= 8192);
        let _ = SlabAllocator::actual_size(size);
    }
}
