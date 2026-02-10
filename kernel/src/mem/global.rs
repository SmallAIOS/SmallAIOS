// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Global allocator — implements Rust's `GlobalAlloc` trait.
//!
//! Routes allocations through the slab allocator (for objects <= 2048 bytes)
//! or the buddy allocator (for page-sized and larger allocations).
//!
//! This enables `alloc::vec::Vec`, `alloc::boxed::Box`, and other
//! standard library heap types throughout the kernel.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;

use super::buddy::BuddyAllocator;
use super::slab::{self, SlabAllocator};
use super::PhysAddr;
use super::PAGE_SIZE_4K;

/// The global kernel allocator.
///
/// Thread safety: In the unikernel model, we use cooperative scheduling
/// with no preemption during allocation. Interrupts should be disabled
/// or allocations should not occur in interrupt context.
pub struct KernelAllocator {
    inner: UnsafeCell<KernelAllocatorInner>,
}

struct KernelAllocatorInner {
    buddy: BuddyAllocator,
    slab: SlabAllocator,
    initialized: bool,
}

// SAFETY: SmallAIOS is a unikernel with cooperative scheduling.
// Allocations are not preempted. The caller must ensure no concurrent
// access from interrupt handlers.
unsafe impl Sync for KernelAllocator {}

impl KernelAllocator {
    /// Create a new uninitialized global allocator.
    pub const fn new() -> Self {
        Self {
            inner: UnsafeCell::new(KernelAllocatorInner {
                buddy: BuddyAllocator::new(),
                slab: SlabAllocator::new(),
                initialized: false,
            }),
        }
    }

    /// Initialize the allocator with a physical memory region.
    ///
    /// # Safety
    /// Must be called exactly once during early boot, before any allocations.
    /// `base` and `size` must describe a valid, usable memory region.
    pub unsafe fn init(&self, base: PhysAddr, size: usize) {
        let inner = &mut *self.inner.get();
        inner.buddy.init(base, size);
        inner.buddy.add_free_region(base, size);

        // Pre-allocate slab pages for commonly used size classes
        for class_idx in 0..slab::SIZE_CLASSES.len() {
            if let Ok(page) = inner.buddy.allocate_page() {
                inner.slab.add_slab_page(class_idx, page);
            }
        }

        inner.initialized = true;
    }

    /// Get the buddy allocator for direct page allocations.
    ///
    /// # Safety
    /// Caller must ensure exclusive access (no concurrent allocations).
    pub unsafe fn buddy(&self) -> &mut BuddyAllocator {
        &mut (*self.inner.get()).buddy
    }

    /// Get the slab allocator for sub-page allocations.
    ///
    /// # Safety
    /// Caller must ensure exclusive access.
    pub unsafe fn slab(&self) -> &mut SlabAllocator {
        &mut (*self.inner.get()).slab
    }

    /// Check if the allocator has been initialized.
    pub fn is_initialized(&self) -> bool {
        unsafe { (*self.inner.get()).initialized }
    }
}

unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let inner = &mut *self.inner.get();
        if !inner.initialized {
            return ptr::null_mut();
        }

        let size = layout.size();
        let align = layout.align();

        // For small allocations that fit in slab classes, use slab allocator.
        // Slab guarantees at least 16-byte alignment which covers most cases.
        if size <= 2048 && align <= size.max(16) {
            match inner.slab.allocate(size) {
                Ok(ptr) => return ptr,
                Err(_) => {
                    // Try to grow the slab by allocating a new page from buddy
                    if let Some(class_idx) = slab::size_class_index(size) {
                        if let Ok(page) = inner.buddy.allocate_page() {
                            if inner.slab.add_slab_page(class_idx, page) {
                                if let Ok(ptr) = inner.slab.allocate(size) {
                                    return ptr;
                                }
                            }
                        }
                    }
                }
            }
        }

        // For large allocations or high-alignment requests, use buddy allocator.
        // Round up to pages.
        let alloc_size = if align > PAGE_SIZE_4K {
            // Need extra pages for alignment
            size + align
        } else {
            size
        };

        let pages_needed = (alloc_size + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
        let order = pages_needed.next_power_of_two().trailing_zeros() as usize;

        match inner.buddy.allocate(order) {
            Ok(addr) => {
                let ptr = addr.as_usize() as *mut u8;
                if align > PAGE_SIZE_4K {
                    // Align within the allocation
                    let aligned = ((ptr as usize) + align - 1) & !(align - 1);
                    aligned as *mut u8
                } else {
                    ptr
                }
            }
            Err(_) => ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let inner = &mut *self.inner.get();
        if !inner.initialized || ptr.is_null() {
            return;
        }

        let size = layout.size();
        let align = layout.align();

        // Slab-sized allocations
        if size <= 2048 && align <= size.max(16) {
            let _ = inner.slab.free(ptr, size);
            return;
        }

        // Buddy allocations — determine the order from the size
        let alloc_size = if align > PAGE_SIZE_4K {
            size + align
        } else {
            size
        };
        let pages_needed = (alloc_size + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
        let order = pages_needed.next_power_of_two().trailing_zeros() as usize;

        // For over-aligned allocations, we need to recover the original buddy address.
        // The buddy address is page-aligned and at or before `ptr`.
        let buddy_addr = if align > PAGE_SIZE_4K {
            PhysAddr::new((ptr as usize) & !(PAGE_SIZE_4K - 1))
        } else {
            PhysAddr::new(ptr as usize)
        };

        let _ = inner.buddy.free(buddy_addr, order);
    }
}

/// The global allocator instance.
/// Only active in non-test builds — tests use the host system allocator.
#[cfg(not(test))]
#[global_allocator]
static ALLOCATOR: KernelAllocator = KernelAllocator::new();

#[cfg(not(test))]
/// Get a reference to the global allocator.
pub fn global_allocator() -> &'static KernelAllocator {
    &ALLOCATOR
}
