// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! FIFO Channel Engine for GM20B (Tegra X1 / T210).
//!
//! The FIFO engine manages GPU command submission channels. Each channel
//! has an associated GPFIFO (GPU push-buffer FIFO) that holds pointers
//! to command buffers for the GPU to execute.

#![allow(dead_code)]

use crate::GpuError;

const MAX_CHANNELS: usize = 128;
const MAX_GPFIFO_ENTRIES: usize = 1024;
const FIFO_RUNLIST_BASE: u32 = 0x2270;

/// State machine for an individual FIFO channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChannelState {
    Free,
    Allocated,
    Bound,
    Active,
    Error,
}

/// A single GPFIFO entry (8 bytes) pointing to a command pushbuffer segment.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpfifoEntry {
    /// bits [31:2] = pushbuffer address >> 2, bit[0] = entry valid
    pub addr_lo: u32,
    /// bits [7:0] = addr[39:32], bits [30:10] = length in dwords
    pub addr_hi_len: u32,
}

impl GpfifoEntry {
    /// Create a new GPFIFO entry from a pushbuffer address and length in dwords.
    pub fn new(addr: u64, len_dwords: u32) -> Self {
        let addr_lo = ((addr >> 2) << 2) as u32 | 1; // set valid bit
        let addr_hi = ((addr >> 32) & 0xFF) as u32;
        let len_field = (len_dwords & 0x1F_FFFF) << 10; // bits [30:10]
        let addr_hi_len = addr_hi | len_field;

        Self {
            addr_lo,
            addr_hi_len,
        }
    }

    /// Check if this entry is valid (bit 0 of addr_lo).
    pub fn is_valid(&self) -> bool {
        self.addr_lo & 1 != 0
    }
}

/// Represents a single GPU command channel.
#[derive(Clone, Debug)]
pub struct FifoChannel {
    pub id: u32,
    pub state: ChannelState,
    pub instance_block_addr: u64,
    pub pushbuffer_addr: u64,
    pub gpfifo_entries: u32,
}

impl FifoChannel {
    fn new_free(id: u32) -> Self {
        Self {
            id,
            state: ChannelState::Free,
            instance_block_addr: 0,
            pushbuffer_addr: 0,
            gpfifo_entries: 0,
        }
    }
}

/// FIFO engine managing command submission channels.
pub struct FifoEngine {
    channels: [FifoChannel; MAX_CHANNELS],
    next_id: u32,
}

impl Default for FifoEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FifoEngine {
    /// Create a new FIFO engine with all channels free.
    pub fn new() -> Self {
        let channels = core::array::from_fn(|i| FifoChannel::new_free(i as u32));
        Self {
            channels,
            next_id: 0,
        }
    }

    /// Allocate a free channel, returning its ID.
    pub fn allocate_channel(&mut self) -> Result<u32, GpuError> {
        for ch in self.channels.iter_mut() {
            if ch.state == ChannelState::Free {
                ch.state = ChannelState::Allocated;
                self.next_id = self.next_id.wrapping_add(1);
                return Ok(ch.id);
            }
        }
        Err(GpuError::QueueFull)
    }

    /// Bind an allocated channel to the GR engine.
    pub fn bind_to_gr(&mut self, channel_id: u32) -> Result<(), GpuError> {
        let ch = self.get_channel_mut(channel_id)?;
        if ch.state != ChannelState::Allocated {
            return Err(GpuError::InvalidState);
        }
        ch.state = ChannelState::Bound;
        Ok(())
    }

    /// Activate a bound channel for command submission.
    pub fn activate(&mut self, channel_id: u32) -> Result<(), GpuError> {
        let ch = self.get_channel_mut(channel_id)?;
        if ch.state != ChannelState::Bound {
            return Err(GpuError::InvalidState);
        }
        ch.state = ChannelState::Active;
        Ok(())
    }

    /// Submit a GPFIFO entry to an active channel.
    pub fn submit(&self, channel_id: u32, entry: &GpfifoEntry) -> Result<(), GpuError> {
        let ch = self.get_channel(channel_id)?;
        if ch.state != ChannelState::Active {
            return Err(GpuError::InvalidState);
        }
        if !entry.is_valid() {
            return Err(GpuError::InvalidConfig);
        }
        // In mock/stub mode: entry is validated but not sent to hardware
        Ok(())
    }

    /// Free a channel, returning it to the pool regardless of current state.
    pub fn free_channel(&mut self, channel_id: u32) -> Result<(), GpuError> {
        let ch = self.get_channel_mut(channel_id)?;
        ch.state = ChannelState::Free;
        ch.instance_block_addr = 0;
        ch.pushbuffer_addr = 0;
        ch.gpfifo_entries = 0;
        Ok(())
    }

    /// Count of channels currently in Active state.
    pub fn active_channel_count(&self) -> usize {
        self.channels
            .iter()
            .filter(|ch| ch.state == ChannelState::Active)
            .count()
    }

    fn get_channel(&self, id: u32) -> Result<&FifoChannel, GpuError> {
        let idx = id as usize;
        if idx >= MAX_CHANNELS {
            return Err(GpuError::NotFound);
        }
        Ok(&self.channels[idx])
    }

    fn get_channel_mut(&mut self, id: u32) -> Result<&mut FifoChannel, GpuError> {
        let idx = id as usize;
        if idx >= MAX_CHANNELS {
            return Err(GpuError::NotFound);
        }
        Ok(&mut self.channels[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpfifo_entry_new() {
        let entry = GpfifoEntry::new(0x1000_0000, 64);
        assert!(entry.is_valid());
        // Check address bits are preserved (lower 2 bits of addr_lo used for valid)
        assert_eq!(entry.addr_lo & !3, 0x1000_0000);
    }

    #[test]
    fn test_gpfifo_entry_valid_bit() {
        let entry = GpfifoEntry::new(0x2000, 32);
        assert!(entry.is_valid());

        let invalid = GpfifoEntry {
            addr_lo: 0,
            addr_hi_len: 0,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_gpfifo_entry_high_address() {
        let addr: u64 = 0x1_0000_0000; // bit 32 set
        let entry = GpfifoEntry::new(addr, 16);
        assert_eq!(entry.addr_hi_len & 0xFF, 1); // addr[39:32] = 1
    }

    #[test]
    fn test_gpfifo_entry_length() {
        let entry = GpfifoEntry::new(0x1000, 512);
        let len = (entry.addr_hi_len >> 10) & 0x1F_FFFF;
        assert_eq!(len, 512);
    }

    #[test]
    fn test_fifo_engine_new() {
        let engine = FifoEngine::new();
        assert_eq!(engine.active_channel_count(), 0);
    }

    #[test]
    fn test_allocate_channel() {
        let mut engine = FifoEngine::new();
        let id = engine.allocate_channel().unwrap();
        assert_eq!(id, 0);

        let id2 = engine.allocate_channel().unwrap();
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_channel_full_lifecycle() {
        let mut engine = FifoEngine::new();

        // Allocate
        let id = engine.allocate_channel().unwrap();
        assert_eq!(engine.active_channel_count(), 0);

        // Bind
        engine.bind_to_gr(id).unwrap();
        assert_eq!(engine.active_channel_count(), 0);

        // Activate
        engine.activate(id).unwrap();
        assert_eq!(engine.active_channel_count(), 1);

        // Submit
        let entry = GpfifoEntry::new(0x1000, 32);
        engine.submit(id, &entry).unwrap();

        // Free
        engine.free_channel(id).unwrap();
        assert_eq!(engine.active_channel_count(), 0);
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut engine = FifoEngine::new();
        let id = engine.allocate_channel().unwrap();

        // Can't activate without binding
        assert_eq!(engine.activate(id), Err(GpuError::InvalidState));

        // Can't submit without activating
        let entry = GpfifoEntry::new(0x1000, 32);
        assert_eq!(engine.submit(id, &entry), Err(GpuError::InvalidState));

        // Can't bind twice
        engine.bind_to_gr(id).unwrap();
        assert_eq!(engine.bind_to_gr(id), Err(GpuError::InvalidState));
    }

    #[test]
    fn test_submit_invalid_entry() {
        let mut engine = FifoEngine::new();
        let id = engine.allocate_channel().unwrap();
        engine.bind_to_gr(id).unwrap();
        engine.activate(id).unwrap();

        let invalid = GpfifoEntry {
            addr_lo: 0,
            addr_hi_len: 0,
        };
        assert_eq!(engine.submit(id, &invalid), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_channel_not_found() {
        let mut engine = FifoEngine::new();
        assert_eq!(engine.bind_to_gr(200), Err(GpuError::NotFound));
        assert_eq!(engine.activate(200), Err(GpuError::NotFound));
        assert_eq!(engine.free_channel(200), Err(GpuError::NotFound));
    }

    #[test]
    fn test_allocate_all_channels() {
        let mut engine = FifoEngine::new();
        for _ in 0..MAX_CHANNELS {
            engine.allocate_channel().unwrap();
        }
        // Next allocation should fail
        assert_eq!(engine.allocate_channel(), Err(GpuError::QueueFull));
    }

    #[test]
    fn test_free_and_reallocate() {
        let mut engine = FifoEngine::new();
        let id = engine.allocate_channel().unwrap();
        engine.bind_to_gr(id).unwrap();
        engine.activate(id).unwrap();

        // Free from active state
        engine.free_channel(id).unwrap();

        // Re-allocate the same slot
        let id2 = engine.allocate_channel().unwrap();
        assert_eq!(id2, id);
    }

    #[test]
    fn test_multiple_active_channels() {
        let mut engine = FifoEngine::new();

        for _ in 0..5 {
            let id = engine.allocate_channel().unwrap();
            engine.bind_to_gr(id).unwrap();
            engine.activate(id).unwrap();
        }

        assert_eq!(engine.active_channel_count(), 5);
    }
}
