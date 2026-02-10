// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared memory transport -- lock-free SPSC ring buffers.
//!
//! Provides a single-producer single-consumer (SPSC) ring buffer for
//! zero-copy intra-kernel messaging. The ring buffer uses atomic
//! head/tail indices with acquire/release ordering to ensure correct
//! visibility between the producer and consumer without locks.
//!
//! The buffer capacity is a power of two so that index wrapping can be
//! performed with a bitmask instead of a modulo operation.

#![allow(dead_code)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::IpcError;

/// Number of slots in the ring buffer (must be a power of 2).
pub const RING_BUFFER_CAPACITY: usize = 64;

/// Maximum size of a single message in bytes.
pub const MAX_MSG_SIZE: usize = 4096;

/// Bitmask for wrapping indices (RING_BUFFER_CAPACITY - 1).
const INDEX_MASK: usize = RING_BUFFER_CAPACITY - 1;

/// A single slot in the ring buffer.
///
/// Each slot holds a fixed-size byte array and the actual length of the
/// message stored in it. Slots are default-initialised to zero length.
#[derive(Clone)]
pub struct RingSlot {
    /// Message data (fixed-size, only `len` bytes are valid).
    data: [u8; MAX_MSG_SIZE],
    /// Number of valid bytes in `data`.
    len: usize,
}

impl Default for RingSlot {
    fn default() -> Self {
        Self {
            data: [0u8; MAX_MSG_SIZE],
            len: 0,
        }
    }
}

/// Lock-free SPSC ring buffer for zero-copy intra-kernel messaging.
///
/// The producer calls [`push`](ShmRingBuffer::push) to write messages and
/// the consumer calls [`pop`](ShmRingBuffer::pop) to read them. The ring
/// buffer is designed for exactly one producer thread and one consumer
/// thread; using multiple producers or consumers simultaneously is
/// undefined behaviour.
///
/// # Memory ordering
///
/// - **push**: reads `tail` with `Acquire`, writes `head` with `Release`.
/// - **pop**: reads `head` with `Acquire`, writes `tail` with `Release`.
///
/// This ensures that slot data written by the producer is visible to the
/// consumer before the consumer sees the updated head, and slot data read
/// by the consumer is no longer referenced before the producer sees the
/// updated tail.
pub struct ShmRingBuffer {
    /// Fixed-size array of message slots, wrapped in `UnsafeCell` for
    /// interior mutability. The SPSC protocol guarantees that at most one
    /// thread writes to any given slot at a time.
    slots: [UnsafeCell<RingSlot>; RING_BUFFER_CAPACITY],
    /// Write position (advanced by the producer).
    head: AtomicUsize,
    /// Read position (advanced by the consumer).
    tail: AtomicUsize,
}

// SAFETY: The SPSC ring buffer protocol guarantees that the producer and
// consumer never access the same slot simultaneously. The producer only
// writes to the slot at `head`, and the consumer only reads from the slot
// at `tail`. Atomic ordering on head/tail ensures proper synchronisation.
unsafe impl Send for ShmRingBuffer {}
unsafe impl Sync for ShmRingBuffer {}

impl ShmRingBuffer {
    /// Create a new, empty ring buffer.
    ///
    /// All slots are zero-initialised and both head and tail start at 0.
    pub fn new() -> Self {
        // Build the slots array with Default initialisation.
        // We cannot use `[UnsafeCell::new(RingSlot::default()); N]` because
        // UnsafeCell is not Copy, so we use array::from_fn.
        let slots = core::array::from_fn(|_| UnsafeCell::new(RingSlot::default()));
        Self {
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Push a message into the ring buffer (producer side).
    ///
    /// # Errors
    ///
    /// - [`IpcError::MessageTooLarge`] if `data.len() > MAX_MSG_SIZE`.
    /// - [`IpcError::BufferFull`] if the ring buffer has no free slots.
    pub fn push(&self, data: &[u8]) -> Result<(), IpcError> {
        if data.len() > MAX_MSG_SIZE {
            return Err(IpcError::MessageTooLarge);
        }

        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Full when head is one full lap ahead of tail.
        if head.wrapping_sub(tail) >= RING_BUFFER_CAPACITY {
            return Err(IpcError::BufferFull);
        }

        let idx = head & INDEX_MASK;

        // SAFETY: We are the sole producer so no concurrent write to this
        // slot. The consumer will not read it until we update head below
        // (with Release ordering). UnsafeCell provides the interior
        // mutability needed for this shared-reference write.
        let slot = unsafe { &mut *self.slots[idx].get() };
        slot.data[..data.len()].copy_from_slice(data);
        slot.len = data.len();

        // Release: make slot data visible before consumer sees updated head.
        self.head.store(head.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Pop a message from the ring buffer into `buf` (consumer side).
    ///
    /// Returns the number of bytes written into `buf`.
    ///
    /// # Errors
    ///
    /// - [`IpcError::NotFound`] if the ring buffer is empty.
    /// - [`IpcError::BufferTooSmall`] if `buf` is smaller than the message.
    pub fn pop(&self, buf: &mut [u8]) -> Result<usize, IpcError> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return Err(IpcError::NotFound);
        }

        let idx = tail & INDEX_MASK;

        // SAFETY: We are the sole consumer so no concurrent read to this
        // slot. The producer will not write to it until we update tail below
        // (with Release ordering).
        let slot = unsafe { &*self.slots[idx].get() };
        let msg_len = slot.len;

        if buf.len() < msg_len {
            return Err(IpcError::BufferTooSmall);
        }

        buf[..msg_len].copy_from_slice(&slot.data[..msg_len]);

        // Release: ensure we are done reading before producer sees free slot.
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        Ok(msg_len)
    }

    /// Check whether the ring buffer is empty.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Check whether the ring buffer is full.
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail) >= RING_BUFFER_CAPACITY
    }

    /// Return the number of messages currently in the ring buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer_is_empty() {
        let rb = ShmRingBuffer::new();
        assert!(rb.is_empty());
        assert!(!rb.is_full());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn push_and_pop_single_message() {
        let rb = ShmRingBuffer::new();
        let data = [1u8, 2, 3, 4, 5];
        rb.push(&data).unwrap();

        assert!(!rb.is_empty());
        assert_eq!(rb.len(), 1);

        let mut buf = [0u8; MAX_MSG_SIZE];
        let n = rb.pop(&mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf[..5], &[1, 2, 3, 4, 5]);
        assert!(rb.is_empty());
    }

    #[test]
    fn fifo_ordering() {
        let rb = ShmRingBuffer::new();
        for i in 0u8..10 {
            rb.push(&[i]).unwrap();
        }
        assert_eq!(rb.len(), 10);

        for i in 0u8..10 {
            let mut buf = [0u8; MAX_MSG_SIZE];
            let n = rb.pop(&mut buf).unwrap();
            assert_eq!(n, 1);
            assert_eq!(buf[0], i);
        }
    }

    #[test]
    fn full_buffer_returns_error() {
        let rb = ShmRingBuffer::new();
        for i in 0..RING_BUFFER_CAPACITY {
            rb.push(&[i as u8]).unwrap();
        }
        assert!(rb.is_full());
        assert_eq!(rb.len(), RING_BUFFER_CAPACITY);

        let result = rb.push(&[0xFF]);
        assert_eq!(result, Err(IpcError::BufferFull));
    }

    #[test]
    fn empty_buffer_pop_returns_error() {
        let rb = ShmRingBuffer::new();
        let mut buf = [0u8; MAX_MSG_SIZE];
        let result = rb.pop(&mut buf);
        assert_eq!(result, Err(IpcError::NotFound));
    }

    #[test]
    fn message_too_large_error() {
        let rb = ShmRingBuffer::new();
        let oversized = [0u8; MAX_MSG_SIZE + 1];
        let result = rb.push(&oversized);
        assert_eq!(result, Err(IpcError::MessageTooLarge));
    }

    #[test]
    fn buffer_too_small_for_pop() {
        let rb = ShmRingBuffer::new();
        let data = [0xABu8; 100];
        rb.push(&data).unwrap();

        let mut tiny_buf = [0u8; 50];
        let result = rb.pop(&mut tiny_buf);
        assert_eq!(result, Err(IpcError::BufferTooSmall));

        // Message should still be in the buffer (not consumed).
        assert_eq!(rb.len(), 1);
    }

    #[test]
    fn wrap_around_behavior() {
        let rb = ShmRingBuffer::new();

        // Fill and drain the entire buffer to advance head/tail past capacity.
        for _ in 0..RING_BUFFER_CAPACITY {
            rb.push(&[0xAA]).unwrap();
        }
        for _ in 0..RING_BUFFER_CAPACITY {
            let mut buf = [0u8; MAX_MSG_SIZE];
            rb.pop(&mut buf).unwrap();
        }
        assert!(rb.is_empty());

        // Now push more messages -- these wrap around the internal array.
        for i in 0u8..10 {
            rb.push(&[i]).unwrap();
        }
        assert_eq!(rb.len(), 10);

        for i in 0u8..10 {
            let mut buf = [0u8; MAX_MSG_SIZE];
            let n = rb.pop(&mut buf).unwrap();
            assert_eq!(n, 1);
            assert_eq!(buf[0], i);
        }
        assert!(rb.is_empty());
    }

    #[test]
    fn length_tracks_push_and_pop() {
        let rb = ShmRingBuffer::new();
        assert_eq!(rb.len(), 0);

        rb.push(&[1]).unwrap();
        assert_eq!(rb.len(), 1);

        rb.push(&[2]).unwrap();
        assert_eq!(rb.len(), 2);

        let mut buf = [0u8; MAX_MSG_SIZE];
        rb.pop(&mut buf).unwrap();
        assert_eq!(rb.len(), 1);

        rb.pop(&mut buf).unwrap();
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn is_empty_transitions() {
        let rb = ShmRingBuffer::new();
        assert!(rb.is_empty());

        rb.push(&[1]).unwrap();
        assert!(!rb.is_empty());

        let mut buf = [0u8; MAX_MSG_SIZE];
        rb.pop(&mut buf).unwrap();
        assert!(rb.is_empty());
    }

    #[test]
    fn is_full_transitions() {
        let rb = ShmRingBuffer::new();
        assert!(!rb.is_full());

        for i in 0..RING_BUFFER_CAPACITY {
            rb.push(&[i as u8]).unwrap();
        }
        assert!(rb.is_full());

        let mut buf = [0u8; MAX_MSG_SIZE];
        rb.pop(&mut buf).unwrap();
        assert!(!rb.is_full());
    }

    #[test]
    fn push_max_size_message() {
        let rb = ShmRingBuffer::new();
        let big = [0xFFu8; MAX_MSG_SIZE];
        rb.push(&big).unwrap();

        let mut buf = [0u8; MAX_MSG_SIZE];
        let n = rb.pop(&mut buf).unwrap();
        assert_eq!(n, MAX_MSG_SIZE);
        assert_eq!(buf, big);
    }

    #[test]
    fn push_zero_length_message() {
        let rb = ShmRingBuffer::new();
        rb.push(&[]).unwrap();

        assert_eq!(rb.len(), 1);

        let mut buf = [0u8; MAX_MSG_SIZE];
        let n = rb.pop(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn interleaved_push_pop() {
        let rb = ShmRingBuffer::new();

        // Interleave single push/pop pairs to exercise various head/tail
        // positions without leaving items between rounds.
        for round in 0u8..50 {
            rb.push(&[round]).unwrap();
            assert_eq!(rb.len(), 1);

            let mut buf = [0u8; MAX_MSG_SIZE];
            let n = rb.pop(&mut buf).unwrap();
            assert_eq!(n, 1);
            assert_eq!(buf[0], round);
            assert!(rb.is_empty());
        }

        // Now interleave with two pushes followed by two pops.
        for round in 0u8..20 {
            rb.push(&[round]).unwrap();
            rb.push(&[round + 100]).unwrap();
            assert_eq!(rb.len(), 2);

            let mut buf = [0u8; MAX_MSG_SIZE];
            rb.pop(&mut buf).unwrap();
            assert_eq!(buf[0], round);

            rb.pop(&mut buf).unwrap();
            assert_eq!(buf[0], round + 100);
            assert!(rb.is_empty());
        }
    }

    #[test]
    fn multiple_wrap_arounds() {
        let rb = ShmRingBuffer::new();

        // Do three full cycles through the buffer.
        for cycle in 0u8..3 {
            for i in 0..RING_BUFFER_CAPACITY {
                rb.push(&[cycle, i as u8]).unwrap();
            }
            assert!(rb.is_full());

            for i in 0..RING_BUFFER_CAPACITY {
                let mut buf = [0u8; MAX_MSG_SIZE];
                let n = rb.pop(&mut buf).unwrap();
                assert_eq!(n, 2);
                assert_eq!(buf[0], cycle);
                assert_eq!(buf[1], i as u8);
            }
            assert!(rb.is_empty());
        }
    }
}
