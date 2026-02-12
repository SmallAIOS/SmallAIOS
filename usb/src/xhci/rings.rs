// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! xHCI ring management — Command Ring, Transfer Rings, and Event Ring.
//!
//! Rings are circular buffers of TRBs shared between the host driver and
//! the xHCI controller. The driver is the producer for Command/Transfer Rings
//! and the consumer for Event Rings.

use super::trb::Trb;

/// Default number of TRBs per ring (including the Link TRB at the end).
pub const DEFAULT_RING_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Producer Ring (Command Ring / Transfer Ring)
// ---------------------------------------------------------------------------

/// A producer ring (used for Command Ring and per-endpoint Transfer Rings).
///
/// The driver enqueues TRBs and the controller dequeues them.
/// A Link TRB at the end wraps back to the start with cycle bit toggle.
pub struct ProducerRing {
    /// TRB storage.
    trbs: [Trb; DEFAULT_RING_SIZE],
    /// Current enqueue index.
    enqueue_index: usize,
    /// Current Producer Cycle State (PCS).
    cycle_state: bool,
    /// Number of TRBs enqueued since last wrap (for stats).
    enqueue_count: u64,
}

impl Default for ProducerRing {
    fn default() -> Self {
        Self::new()
    }
}

impl ProducerRing {
    /// Create a new empty producer ring.
    pub fn new() -> Self {
        let mut ring = Self {
            trbs: [Trb::zeroed(); DEFAULT_RING_SIZE],
            enqueue_index: 0,
            cycle_state: true, // PCS starts at 1
            enqueue_count: 0,
        };
        // Place a Link TRB at the last position.
        // ring_base would be the physical address; using 0 as placeholder.
        ring.trbs[DEFAULT_RING_SIZE - 1] = Trb::link(0, true, !ring.cycle_state);
        ring
    }

    /// Enqueue a TRB onto the ring.
    ///
    /// Returns the index at which the TRB was placed, or None if the ring
    /// would overwrite the Link TRB (ring full).
    pub fn enqueue(&mut self, mut trb: Trb) -> Option<usize> {
        // Don't write over the Link TRB.
        if self.enqueue_index >= DEFAULT_RING_SIZE - 1 {
            // Wrap around: advance past the Link TRB.
            self.wrap();
        }

        // Set the cycle bit to match current PCS.
        trb.set_cycle_bit(self.cycle_state);

        let idx = self.enqueue_index;
        self.trbs[idx] = trb;
        self.enqueue_index += 1;
        self.enqueue_count += 1;

        Some(idx)
    }

    /// Handle ring wrap-around.
    fn wrap(&mut self) {
        // Update the Link TRB's cycle bit to current PCS.
        self.trbs[DEFAULT_RING_SIZE - 1].set_cycle_bit(self.cycle_state);

        // Toggle cycle state.
        self.cycle_state = !self.cycle_state;
        self.enqueue_index = 0;
    }

    /// Get the current enqueue index.
    pub fn enqueue_index(&self) -> usize {
        self.enqueue_index
    }

    /// Get the current Producer Cycle State.
    pub fn cycle_state(&self) -> bool {
        self.cycle_state
    }

    /// Get a reference to a TRB at the given index.
    pub fn get(&self, index: usize) -> Option<&Trb> {
        self.trbs.get(index)
    }

    /// Get the total number of TRBs enqueued.
    pub fn enqueue_count(&self) -> u64 {
        self.enqueue_count
    }

    /// Return the usable capacity (excluding the Link TRB).
    pub fn capacity(&self) -> usize {
        DEFAULT_RING_SIZE - 1
    }
}

// ---------------------------------------------------------------------------
// Consumer Ring (Event Ring)
// ---------------------------------------------------------------------------

/// An event ring (controller is the producer, driver is the consumer).
///
/// The driver reads events and advances the dequeue pointer.
pub struct EventRing {
    /// TRB storage.
    trbs: [Trb; DEFAULT_RING_SIZE],
    /// Current dequeue index.
    dequeue_index: usize,
    /// Current Consumer Cycle State (CCS).
    cycle_state: bool,
    /// Number of events dequeued (for stats).
    dequeue_count: u64,
}

impl Default for EventRing {
    fn default() -> Self {
        Self::new()
    }
}

impl EventRing {
    /// Create a new empty event ring.
    pub fn new() -> Self {
        Self {
            trbs: [Trb::zeroed(); DEFAULT_RING_SIZE],
            dequeue_index: 0,
            cycle_state: true, // CCS starts at 1
            dequeue_count: 0,
        }
    }

    /// Check if there is a pending event (cycle bit matches CCS).
    pub fn has_event(&self) -> bool {
        self.trbs[self.dequeue_index].cycle_bit() == self.cycle_state
    }

    /// Dequeue the next event TRB, if available.
    pub fn dequeue(&mut self) -> Option<Trb> {
        if !self.has_event() {
            return None;
        }

        let trb = self.trbs[self.dequeue_index];
        self.dequeue_index += 1;
        self.dequeue_count += 1;

        if self.dequeue_index >= DEFAULT_RING_SIZE {
            self.dequeue_index = 0;
            self.cycle_state = !self.cycle_state;
        }

        Some(trb)
    }

    /// Push an event into the ring (for testing — simulates controller writing).
    #[cfg(test)]
    pub fn push_event(&mut self, mut trb: Trb, at_index: usize) {
        trb.set_cycle_bit(self.cycle_state);
        self.trbs[at_index] = trb;
    }

    /// Get the current dequeue index.
    pub fn dequeue_index(&self) -> usize {
        self.dequeue_index
    }

    /// Get the current Consumer Cycle State.
    pub fn cycle_state(&self) -> bool {
        self.cycle_state
    }

    /// Get the total number of events dequeued.
    pub fn dequeue_count(&self) -> u64 {
        self.dequeue_count
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::trb::TrbType;
    use super::*;

    #[test]
    fn test_producer_ring_new() {
        let ring = ProducerRing::new();
        assert_eq!(ring.enqueue_index(), 0);
        assert!(ring.cycle_state());
        assert_eq!(ring.capacity(), DEFAULT_RING_SIZE - 1);
    }

    #[test]
    fn test_producer_ring_enqueue() {
        let mut ring = ProducerRing::new();
        let trb = Trb::normal(0x1000, 64, false, true); // cycle bit will be overwritten
        let idx = ring.enqueue(trb).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(ring.enqueue_index(), 1);
        assert_eq!(ring.enqueue_count(), 1);

        // Verify cycle bit was set.
        let stored = ring.get(0).unwrap();
        assert!(stored.cycle_bit());
    }

    #[test]
    fn test_producer_ring_multiple_enqueue() {
        let mut ring = ProducerRing::new();
        for i in 0..10 {
            let trb = Trb::normal(i * 0x1000, 64, false, false);
            let idx = ring.enqueue(trb).unwrap();
            assert_eq!(idx, i as usize);
        }
        assert_eq!(ring.enqueue_index(), 10);
        assert_eq!(ring.enqueue_count(), 10);
    }

    #[test]
    fn test_producer_ring_wrap_around() {
        let mut ring = ProducerRing::new();
        // Fill all usable slots (0..254), index becomes 255 (the Link TRB slot).
        for _ in 0..DEFAULT_RING_SIZE - 1 {
            let trb = Trb::normal(0, 0, false, false);
            ring.enqueue(trb);
        }
        assert_eq!(ring.enqueue_index(), DEFAULT_RING_SIZE - 1);
        assert!(ring.cycle_state()); // Not yet toggled

        // Next enqueue should trigger wrap: Link TRB at 255 → reset to 0 → write at 0.
        let trb = Trb::normal(0xBEEF, 0, false, false);
        ring.enqueue(trb);
        assert_eq!(ring.enqueue_index(), 1); // Written at 0, advanced to 1
        assert!(!ring.cycle_state()); // Cycle toggled after wrap
    }

    #[test]
    fn test_producer_ring_link_trb() {
        let ring = ProducerRing::new();
        let link = ring.get(DEFAULT_RING_SIZE - 1).unwrap();
        assert_eq!(link.trb_type(), Some(TrbType::Link));
    }

    #[test]
    fn test_event_ring_empty() {
        let ring = EventRing::new();
        assert!(!ring.has_event());
        assert_eq!(ring.dequeue_index(), 0);
    }

    #[test]
    fn test_event_ring_dequeue() {
        let mut ring = EventRing::new();

        // Simulate controller writing an event.
        let event = Trb {
            parameter: 0xDEAD,
            status: (1u32 << 24), // Success
            control: Trb::make_control(TrbType::CommandCompletionEvent, true, 0),
        };
        ring.push_event(event, 0);

        assert!(ring.has_event());
        let dequeued = ring.dequeue().unwrap();
        assert_eq!(dequeued.parameter, 0xDEAD);
        assert_eq!(ring.dequeue_count(), 1);
        assert_eq!(ring.dequeue_index(), 1);
    }

    #[test]
    fn test_event_ring_no_event_when_cycle_mismatch() {
        let ring = EventRing::new();
        // All TRBs are zeroed (cycle bit = 0), CCS = 1 → no match → no event.
        assert!(!ring.has_event());
    }

    #[test]
    fn test_event_ring_wrap_around() {
        let mut ring = EventRing::new();

        // Fill all slots with events.
        for i in 0..DEFAULT_RING_SIZE {
            let mut event = Trb::zeroed();
            event.parameter = i as u64;
            ring.push_event(event, i);
        }

        // Dequeue all.
        for i in 0..DEFAULT_RING_SIZE {
            let trb = ring.dequeue().unwrap();
            assert_eq!(trb.parameter, i as u64);
        }

        // After wrap, cycle state should have toggled.
        assert!(!ring.cycle_state());
        assert_eq!(ring.dequeue_index(), 0);
    }
}
