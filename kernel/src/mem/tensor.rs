// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tensor memory pool — aligned, reference-counted buffer management for ONNX
//! inference workloads.
//!
//! Design:
//! - Pre-allocated arena of SIMD-aligned buffers (64-byte minimum alignment)
//! - Bump-pointer allocation within the arena
//! - Reference counting for zero-copy tensor sharing between operators
//! - DMA-capable buffers for GPU transfer eligibility
//! - Deterministic allocation (no fragmentation within a session)

use super::{MemError, PhysAddr};
use core::sync::atomic::{AtomicU32, Ordering};

/// Minimum alignment for tensor buffers (AVX-512 / cache line).
pub const TENSOR_ALIGNMENT: usize = 64;

/// Maximum number of tensor buffers tracked in the pool.
const MAX_TENSOR_BUFFERS: usize = 4096;

/// A handle to a tensor buffer in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct TensorHandle(u32);

impl TensorHandle {
    /// Get the raw index.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Create a handle from a raw index.
    pub const fn from_index(idx: u32) -> Self {
        Self(idx)
    }
}

/// Metadata for a single tensor buffer.
struct TensorEntry {
    /// Offset from pool base.
    offset: usize,
    /// Size in bytes.
    size: usize,
    /// Reference count (atomic for concurrent access).
    refcount: AtomicU32,
    /// Whether this entry is in use.
    active: bool,
}

impl TensorEntry {
    const fn empty() -> Self {
        Self {
            offset: 0,
            size: 0,
            refcount: AtomicU32::new(0),
            active: false,
        }
    }
}

/// Tensor memory pool with arena allocation.
pub struct TensorPool {
    /// Base address of the arena.
    base: PhysAddr,
    /// Total size of the arena in bytes.
    capacity: usize,
    /// Current bump pointer offset.
    bump_offset: usize,
    /// Buffer metadata.
    entries: [TensorEntry; MAX_TENSOR_BUFFERS],
    /// Number of active entries.
    active_count: usize,
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorPool {
    /// Create an uninitialized tensor pool.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const EMPTY: TensorEntry = TensorEntry::empty();
        Self {
            base: PhysAddr(0),
            capacity: 0,
            bump_offset: 0,
            entries: [EMPTY; MAX_TENSOR_BUFFERS],
            active_count: 0,
        }
    }

    /// Initialize the tensor pool with a contiguous arena.
    ///
    /// `base` should be page-aligned and the region should be DMA-capable.
    pub fn init(&mut self, base: PhysAddr, capacity: usize) {
        self.base = base;
        self.capacity = capacity;
        self.bump_offset = 0;
        self.active_count = 0;
    }

    /// Allocate a tensor buffer of `size` bytes with at least `TENSOR_ALIGNMENT`
    /// alignment. Returns a handle and the physical address.
    pub fn allocate(&mut self, size: usize) -> Result<(TensorHandle, PhysAddr), MemError> {
        if size == 0 {
            return Err(MemError::BadAlignment);
        }

        // Align the bump pointer
        let aligned_offset = (self.bump_offset + TENSOR_ALIGNMENT - 1) & !(TENSOR_ALIGNMENT - 1);
        let end = aligned_offset + size;

        if end > self.capacity {
            return Err(MemError::TensorPoolExhausted);
        }

        // Find a free entry slot
        let slot = self.find_free_slot().ok_or(MemError::TensorPoolExhausted)?;

        self.entries[slot] = TensorEntry {
            offset: aligned_offset,
            size,
            refcount: AtomicU32::new(1),
            active: true,
        };

        self.bump_offset = end;
        self.active_count += 1;

        let addr = self.base.offset(aligned_offset);
        Ok((TensorHandle(slot as u32), addr))
    }

    /// Increment the reference count of a tensor buffer.
    pub fn retain(&self, handle: TensorHandle) -> Result<(), MemError> {
        let entry = self.get_entry(handle)?;
        entry.refcount.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Decrement the reference count. The buffer is logically freed when it
    /// reaches zero (but arena space is not reclaimed until reset).
    pub fn release(&mut self, handle: TensorHandle) -> Result<(), MemError> {
        let idx = handle.index();
        if idx >= MAX_TENSOR_BUFFERS || !self.entries[idx].active {
            return Err(MemError::InvalidAddress);
        }

        let prev = self.entries[idx].refcount.fetch_sub(1, Ordering::Release);
        if prev == 1 {
            // Last reference — mark inactive
            self.entries[idx].active = false;
            self.active_count -= 1;
        }
        Ok(())
    }

    /// Get the physical address of a tensor buffer.
    pub fn address(&self, handle: TensorHandle) -> Result<PhysAddr, MemError> {
        let entry = self.get_entry(handle)?;
        Ok(self.base.offset(entry.offset))
    }

    /// Get the size of a tensor buffer.
    pub fn size(&self, handle: TensorHandle) -> Result<usize, MemError> {
        let entry = self.get_entry(handle)?;
        Ok(entry.size)
    }

    /// Get the current reference count.
    pub fn refcount(&self, handle: TensorHandle) -> Result<u32, MemError> {
        let entry = self.get_entry(handle)?;
        Ok(entry.refcount.load(Ordering::Relaxed))
    }

    /// Reset the pool — free all allocations and reset the bump pointer.
    /// This is called between inference sessions to reclaim all memory.
    pub fn reset(&mut self) {
        self.bump_offset = 0;
        self.active_count = 0;
        for entry in self.entries.iter_mut() {
            entry.active = false;
            *entry.refcount.get_mut() = 0;
        }
    }

    /// Remaining capacity in bytes.
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.bump_offset)
    }

    /// Number of active tensor buffers.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Peak usage (high-water mark of bump pointer).
    pub fn peak_usage(&self) -> usize {
        self.bump_offset
    }

    fn find_free_slot(&self) -> Option<usize> {
        (0..MAX_TENSOR_BUFFERS).find(|&i| !self.entries[i].active)
    }

    fn get_entry(&self, handle: TensorHandle) -> Result<&TensorEntry, MemError> {
        let idx = handle.index();
        if idx >= MAX_TENSOR_BUFFERS || !self.entries[idx].active {
            return Err(MemError::InvalidAddress);
        }
        Ok(&self.entries[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(size: usize) -> TensorPool {
        let mut pool = TensorPool::new();
        pool.init(PhysAddr::new(0x100_0000), size);
        pool
    }

    #[test]
    fn test_basic_allocation() {
        let mut pool = make_pool(1024 * 1024); // 1 MiB
        let (h, addr) = pool.allocate(1024).unwrap();
        assert!(addr.is_aligned(TENSOR_ALIGNMENT));
        assert_eq!(pool.size(h).unwrap(), 1024);
        assert_eq!(pool.refcount(h).unwrap(), 1);
    }

    #[test]
    fn test_alignment() {
        let mut pool = make_pool(1024 * 1024);
        // First allocation
        let (_, a1) = pool.allocate(100).unwrap();
        assert!(a1.is_aligned(TENSOR_ALIGNMENT));
        // Second allocation should also be aligned
        let (_, a2) = pool.allocate(200).unwrap();
        assert!(a2.is_aligned(TENSOR_ALIGNMENT));
        assert!(a2.as_usize() > a1.as_usize());
    }

    #[test]
    fn test_reference_counting() {
        let mut pool = make_pool(1024 * 1024);
        let (h, _) = pool.allocate(512).unwrap();
        assert_eq!(pool.refcount(h).unwrap(), 1);

        pool.retain(h).unwrap();
        assert_eq!(pool.refcount(h).unwrap(), 2);

        pool.release(h).unwrap();
        assert_eq!(pool.refcount(h).unwrap(), 1);

        pool.release(h).unwrap();
        // Handle is now invalid (released)
        assert_eq!(pool.refcount(h), Err(MemError::InvalidAddress));
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut pool = make_pool(256); // Small pool
        let _ = pool.allocate(128).unwrap();
        let _ = pool.allocate(64).unwrap();
        // No room left for another 128-byte allocation (with alignment)
        assert_eq!(pool.allocate(128), Err(MemError::TensorPoolExhausted));
    }

    #[test]
    fn test_reset() {
        let mut pool = make_pool(4096);
        let (h, _) = pool.allocate(1024).unwrap();
        pool.allocate(1024).unwrap();
        assert_eq!(pool.active_count(), 2);

        pool.reset();
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.remaining(), 4096);
        // Old handle should be invalid
        assert_eq!(pool.size(h), Err(MemError::InvalidAddress));
    }

    #[test]
    fn test_zero_size_allocation() {
        let mut pool = make_pool(4096);
        assert_eq!(pool.allocate(0), Err(MemError::BadAlignment));
    }
}
