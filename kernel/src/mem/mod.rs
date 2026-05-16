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

/// Internal const helper: round `addr` up to the next multiple of `align`.
///
/// `align` must be a power of two. Used by both `PhysAddr::align_up` and
/// `VirtAddr::align_up` to share their alignment arithmetic without
/// forcing the address newtypes through a (non-`const`) trait.
#[inline]
const fn align_up_usize(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// Internal const helper: round `addr` down to the previous multiple of `align`.
#[inline]
const fn align_down_usize(addr: usize, align: usize) -> usize {
    addr & !(align - 1)
}

/// Internal const helper: check that `addr` is a multiple of `align`.
#[inline]
const fn is_aligned_usize(addr: usize, align: usize) -> bool {
    addr & (align - 1) == 0
}

/// Generates the shared `const fn` API (`new`, `as_usize`, `align_up`,
/// `align_down`, `is_aligned`) on a `#[repr(transparent)]` address newtype.
///
/// `PhysAddr` and `VirtAddr` are intentionally distinct types (mixing them
/// is a kernel-correctness bug), so we keep the strong typing but share
/// the body via a macro instead of duplicating five identical methods per
/// type. This preserves the existing public API byte-for-byte.
macro_rules! impl_addr_newtype {
    ($t:ty) => {
        impl $t {
            #[inline]
            pub const fn new(addr: usize) -> Self {
                Self(addr)
            }

            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0
            }

            #[inline]
            pub const fn align_up(self, align: usize) -> Self {
                Self(align_up_usize(self.0, align))
            }

            #[inline]
            pub const fn align_down(self, align: usize) -> Self {
                Self(align_down_usize(self.0, align))
            }

            #[inline]
            pub const fn is_aligned(self, align: usize) -> bool {
                is_aligned_usize(self.0, align)
            }
        }
    };
}

impl_addr_newtype!(PhysAddr);
impl_addr_newtype!(VirtAddr);

impl PhysAddr {
    pub const fn offset(self, offset: usize) -> Self {
        Self(self.0 + offset)
    }
}

impl VirtAddr {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts that an address newtype's shared `new` / `as_usize` /
    /// `align_up` / `align_down` / `is_aligned` API behaves correctly.
    ///
    /// Used by both the `PhysAddr` and `VirtAddr` test cases below to
    /// avoid duplicating the same five assertions per type.
    macro_rules! assert_addr_api {
        ($t:ty, $base:expr) => {{
            let base: usize = $base;
            // new + as_usize round-trip.
            assert_eq!(<$t>::new(base).as_usize(), base);
            // align_up rounds an off-by-one address up to the next page.
            assert_eq!(
                <$t>::new(base | 0x1).align_up(0x1000).as_usize(),
                base + 0x1000
            );
            // align_down strips the low bits.
            assert_eq!(<$t>::new(base | 0xFFF).align_down(0x1000).as_usize(), base);
            // is_aligned discriminates page-aligned vs unaligned.
            assert!(<$t>::new(base).is_aligned(0x1000));
            assert!(!<$t>::new(base | 0x1).is_aligned(0x1000));
        }};
    }

    #[test]
    fn phys_addr_basic_and_align() {
        assert_addr_api!(PhysAddr, 0x1000);
    }

    #[test]
    fn phys_addr_offset() {
        assert_eq!(PhysAddr::new(0x1000).offset(0x500).as_usize(), 0x1500);
    }

    #[test]
    fn virt_addr_basic_and_align() {
        assert_addr_api!(VirtAddr, 0x2000);
    }

    #[test]
    fn virt_addr_as_ptr() {
        let addr = VirtAddr::new(0x4000);
        assert_eq!(addr.as_ptr::<u8>() as usize, 0x4000);
        assert_eq!(addr.as_mut_ptr::<u8>() as usize, 0x4000);
    }

    #[test]
    fn phys_addr_ordering() {
        assert!(PhysAddr::new(0x1000) < PhysAddr::new(0x2000));
        assert!(PhysAddr::new(0x2000) > PhysAddr::new(0x1000));
        assert_eq!(PhysAddr::new(0x1000), PhysAddr::new(0x1000));
    }

    #[test]
    fn virt_addr_ordering() {
        assert!(VirtAddr::new(0x1000) < VirtAddr::new(0x2000));
        assert_eq!(VirtAddr::new(0x1000), VirtAddr::new(0x1000));
    }

    #[test]
    fn page_sizes() {
        assert_eq!(PAGE_SIZE_4K, 4096);
        assert_eq!(PAGE_SIZE_2M, 2 * 1024 * 1024);
        assert_eq!(PAGE_SIZE_1G, 1024 * 1024 * 1024);
    }
}
