// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory management subsystem.
//!
//! Layers:
//! 1. Physical memory map (boot-time enumeration)
//! 2. Buddy allocator (page-granularity, 4K-8G)
//! 3. Slab allocator (sub-page kernel objects, 16B-2048B)
//! 4. Tensor memory pool (aligned, reference-counted, GPU-mappable)

pub mod buddy;
pub mod global;
pub mod page;
pub mod phys;
pub mod slab;
pub mod tensor;

/// Physical address (usize-width, identity-mapped in unikernel mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct PhysAddr(pub usize);

/// Virtual address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct VirtAddr(pub usize);

impl PhysAddr {
    pub const fn new(addr: usize) -> Self {
        Self(addr)
    }

    pub const fn as_usize(self) -> usize {
        self.0
    }

    pub const fn align_up(self, align: usize) -> Self {
        Self((self.0 + align - 1) & !(align - 1))
    }

    pub const fn align_down(self, align: usize) -> Self {
        Self(self.0 & !(align - 1))
    }

    pub const fn is_aligned(self, align: usize) -> bool {
        self.0 & (align - 1) == 0
    }

    pub const fn offset(self, offset: usize) -> Self {
        Self(self.0 + offset)
    }
}

impl VirtAddr {
    pub const fn new(addr: usize) -> Self {
        Self(addr)
    }

    pub const fn as_usize(self) -> usize {
        self.0
    }

    pub const fn align_up(self, align: usize) -> Self {
        Self((self.0 + align - 1) & !(align - 1))
    }

    pub const fn align_down(self, align: usize) -> Self {
        Self(self.0 & !(align - 1))
    }

    pub const fn is_aligned(self, align: usize) -> bool {
        self.0 & (align - 1) == 0
    }

    pub const fn as_ptr<T>(self) -> *const T {
        self.0 as *const T
    }

    pub const fn as_mut_ptr<T>(self) -> *mut T {
        self.0 as *mut T
    }
}

/// Standard page sizes.
pub const PAGE_SIZE_4K: usize = 4096;
pub const PAGE_SIZE_2M: usize = 2 * 1024 * 1024;
pub const PAGE_SIZE_1G: usize = 1024 * 1024 * 1024;

/// Errors from memory subsystem operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemError {
    /// No memory available for the requested allocation.
    OutOfMemory,
    /// The requested address or size is not properly aligned.
    BadAlignment,
    /// The physical address is outside known memory regions.
    InvalidAddress,
    /// Attempted to free memory that is not allocated.
    DoubleFree,
    /// The requested allocation size is too large.
    SizeTooLarge,
    /// The requested order exceeds the maximum buddy order.
    OrderTooLarge,
    /// No slabs available for the requested size class.
    SlabExhausted,
    /// Tensor pool is exhausted.
    TensorPoolExhausted,
    /// Page table mapping error.
    MappingError,
    /// Attempted to unmap a page that is not mapped.
    NotMapped,
    /// Page table entry already exists.
    AlreadyMapped,
}
