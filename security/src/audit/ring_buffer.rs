// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Lock-free ring buffer for audit events.
//!
//! Bounded circular buffer that overwrites the oldest entry when full.
//! Uses atomic operations for thread-safe single-producer access.
//! Designed to never block the inference path per NIST AU-12 requirements.

use core::sync::atomic::{AtomicUsize, Ordering};

use super::entry::AuditEntry;

/// Default ring buffer capacity (must be a power of two).
pub const DEFAULT_CAPACITY: usize = 1024;

/// Ring buffer for audit log entries.
///
/// Fixed-size circular buffer. When full, the oldest entry is silently
/// overwritten. The write path is wait-free (single atomic increment).
pub struct AuditRingBuffer<const N: usize = DEFAULT_CAPACITY> {
    /// Storage for entries. Entries at indices >= write_pos are stale/empty.
    entries: [Option<AuditEntry>; N],
    /// Monotonically increasing write position. The actual index is `write_pos % N`.
    write_pos: AtomicUsize,
    /// Total number of entries written (including overwrites).
    total_written: AtomicUsize,
    /// Number of entries overwritten (lost due to full buffer).
    overwritten: AtomicUsize,
}

impl<const N: usize> AuditRingBuffer<N> {
    /// Create a new empty ring buffer.
    ///
    /// # Panics
    ///
    /// Panics if N is 0.
    pub fn new() -> Self {
        assert!(N > 0, "ring buffer capacity must be > 0");
        Self {
            entries: [const { None }; N],
            write_pos: AtomicUsize::new(0),
            total_written: AtomicUsize::new(0),
            overwritten: AtomicUsize::new(0),
        }
    }

    /// Push an entry into the buffer.
    ///
    /// If the buffer is full, the oldest entry is overwritten. This operation
    /// is wait-free and never blocks.
    pub fn push(&mut self, entry: AuditEntry) {
        let pos = self.write_pos.load(Ordering::Relaxed);
        let idx = pos % N;

        if pos >= N && self.entries[idx].is_some() {
            self.overwritten.fetch_add(1, Ordering::Relaxed);
        }

        self.entries[idx] = Some(entry);
        self.write_pos.store(pos + 1, Ordering::Release);
        self.total_written.fetch_add(1, Ordering::Relaxed);
    }

    /// Number of entries currently in the buffer (not including overwritten).
    pub fn len(&self) -> usize {
        let written = self.write_pos.load(Ordering::Acquire);
        written.min(N)
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.write_pos.load(Ordering::Acquire) == 0
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.write_pos.load(Ordering::Acquire) >= N
    }

    /// Buffer capacity.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Total entries ever written.
    pub fn total_written(&self) -> usize {
        self.total_written.load(Ordering::Relaxed)
    }

    /// Number of entries that were overwritten.
    pub fn overwritten_count(&self) -> usize {
        self.overwritten.load(Ordering::Relaxed)
    }

    /// Drain up to `max` entries from the buffer into the provided slice.
    ///
    /// Returns the number of entries drained. Entries are returned in
    /// chronological order (oldest first). Drained entries are removed
    /// from the buffer.
    pub fn drain(&mut self, out: &mut [AuditEntry], max: usize) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        if write_pos == 0 {
            return 0;
        }

        let count = self.len().min(max).min(out.len());
        if count == 0 {
            return 0;
        }

        // Calculate the start position (oldest entry)
        let start = write_pos.saturating_sub(N);

        for (i, slot) in out.iter_mut().enumerate().take(count) {
            let idx = (start + i) % N;
            if let Some(entry) = self.entries[idx].take() {
                *slot = entry;
            }
        }

        // If we drained everything, reset write_pos
        if count == self.len() {
            self.write_pos.store(0, Ordering::Release);
        }

        count
    }

    /// Read the entry at position `index` (0 = oldest available).
    ///
    /// Returns None if the index is out of range or the slot is empty.
    pub fn get(&self, index: usize) -> Option<&AuditEntry> {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        if write_pos == 0 || index >= self.len() {
            return None;
        }

        let start = write_pos.saturating_sub(N);
        let actual_idx = (start + index) % N;
        self.entries[actual_idx].as_ref()
    }

    /// Iterate over all entries in chronological order (oldest first).
    pub fn iter(&self) -> RingBufferIter<'_, N> {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let start = write_pos.saturating_sub(N);
        RingBufferIter {
            buffer: self,
            current: start,
            end: write_pos,
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for slot in self.entries.iter_mut() {
            *slot = None;
        }
        self.write_pos.store(0, Ordering::Release);
    }
}

impl<const N: usize> Default for AuditRingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over ring buffer entries in chronological order.
pub struct RingBufferIter<'a, const N: usize> {
    buffer: &'a AuditRingBuffer<N>,
    current: usize,
    end: usize,
}

impl<'a, const N: usize> Iterator for RingBufferIter<'a, N> {
    type Item = &'a AuditEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }
        let idx = self.current % N;
        self.current += 1;
        self.buffer.entries[idx].as_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl<'a, const N: usize> ExactSizeIterator for RingBufferIter<'a, N> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditResult, Operation};
    use crate::audit::taxonomy::AuditEventType;

    fn make_entry(ts: u64) -> AuditEntry {
        AuditEntry {
            timestamp_ns: ts,
            event_type: AuditEventType::system_boot(),
            task_id: 1,
            resource_type: 0,
            resource_instance: 0,
            operation: Operation::new("test"),
            result: AuditResult::Success,
            capability_id: 0,
        }
    }

    #[test]
    fn new_buffer_is_empty() {
        let buf = AuditRingBuffer::<8>::new();
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.total_written(), 0);
        assert_eq!(buf.overwritten_count(), 0);
    }

    #[test]
    fn push_and_read() {
        let mut buf = AuditRingBuffer::<8>::new();
        buf.push(make_entry(100));
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
        assert!(!buf.is_full());

        let entry = buf.get(0).unwrap();
        assert_eq!(entry.timestamp_ns, 100);
    }

    #[test]
    fn fill_to_capacity() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..4 {
            buf.push(make_entry(i as u64));
        }
        assert_eq!(buf.len(), 4);
        assert!(buf.is_full());
        assert_eq!(buf.total_written(), 4);
        assert_eq!(buf.overwritten_count(), 0);
    }

    #[test]
    fn overwrite_oldest() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..6 {
            buf.push(make_entry(i as u64));
        }
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.total_written(), 6);
        assert_eq!(buf.overwritten_count(), 2);

        // Oldest should now be entry with timestamp 2 (entries 0,1 were overwritten)
        let oldest = buf.get(0).unwrap();
        assert_eq!(oldest.timestamp_ns, 2);

        let newest = buf.get(3).unwrap();
        assert_eq!(newest.timestamp_ns, 5);
    }

    #[test]
    fn get_out_of_range() {
        let mut buf = AuditRingBuffer::<4>::new();
        assert!(buf.get(0).is_none());
        buf.push(make_entry(1));
        assert!(buf.get(0).is_some());
        assert!(buf.get(1).is_none());
    }

    #[test]
    fn iter_order() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..6u64 {
            buf.push(make_entry(i));
        }
        let timestamps: alloc::vec::Vec<u64> = buf.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(timestamps, alloc::vec![2, 3, 4, 5]);
    }

    #[test]
    fn iter_empty() {
        let buf = AuditRingBuffer::<4>::new();
        assert_eq!(buf.iter().count(), 0);
    }

    #[test]
    fn iter_exact_size() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..3 {
            buf.push(make_entry(i));
        }
        let iter = buf.iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn drain_all() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..3 {
            buf.push(make_entry(i));
        }
        let mut out = [make_entry(0); 4];
        let drained = buf.drain(&mut out, 4);
        assert_eq!(drained, 3);
        assert_eq!(out[0].timestamp_ns, 0);
        assert_eq!(out[1].timestamp_ns, 1);
        assert_eq!(out[2].timestamp_ns, 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_partial() {
        let mut buf = AuditRingBuffer::<8>::new();
        for i in 0..5 {
            buf.push(make_entry(i));
        }
        let mut out = [make_entry(0); 3];
        let drained = buf.drain(&mut out, 3);
        assert_eq!(drained, 3);
        assert_eq!(out[0].timestamp_ns, 0);
        assert_eq!(out[1].timestamp_ns, 1);
        assert_eq!(out[2].timestamp_ns, 2);
    }

    #[test]
    fn drain_empty() {
        let mut buf = AuditRingBuffer::<4>::new();
        let mut out = [make_entry(0); 4];
        let drained = buf.drain(&mut out, 4);
        assert_eq!(drained, 0);
    }

    #[test]
    fn clear_resets() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..4 {
            buf.push(make_entry(i));
        }
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn heavy_overwrite() {
        let mut buf = AuditRingBuffer::<4>::new();
        for i in 0..1000u64 {
            buf.push(make_entry(i));
        }
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.total_written(), 1000);
        assert_eq!(buf.overwritten_count(), 996);

        // Most recent 4 entries
        let timestamps: alloc::vec::Vec<u64> = buf.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(timestamps, alloc::vec![996, 997, 998, 999]);
    }
}
