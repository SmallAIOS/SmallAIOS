// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AXI DMA controller driver.
//!
//! Supports simple (single-buffer) DMA and scatter-gather DMA modes.
//! Descriptor chains are built in-memory with source, destination, length,
//! and next-descriptor pointers.

use core::fmt;

/// Maximum number of scatter-gather descriptors per chain.
pub const MAX_SG_DESCRIPTORS: usize = 64;

/// Maximum single DMA transfer length (16 MiB).
pub const MAX_TRANSFER_LEN: u32 = 16 * 1024 * 1024;

/// DMA transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaDirection {
    /// Device (FPGA) to system memory (MM2S read channel).
    DeviceToMemory,
    /// System memory to device (FPGA) (S2MM write channel).
    MemoryToDevice,
}

/// DMA channel state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaChannelState {
    /// Channel is idle, ready for a new transfer.
    Idle,
    /// Transfer is in progress.
    Running,
    /// Transfer completed successfully.
    Complete,
    /// Transfer encountered an error (bus error, decode error, timeout).
    Error,
    /// Channel is halted (needs reset).
    Halted,
}

/// DMA transfer error information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaError {
    /// Channel is not in idle state.
    ChannelBusy,
    /// Invalid channel number.
    InvalidChannel,
    /// Transfer length is zero or exceeds maximum.
    InvalidLength,
    /// Source or destination address is null.
    InvalidAddress,
    /// Too many scatter-gather descriptors.
    TooManyDescriptors,
    /// No descriptors provided for scatter-gather transfer.
    NoDescriptors,
    /// Bus error during DMA transfer.
    BusError {
        /// Faulting address (if available).
        fault_addr: u64,
    },
    /// Decode error (no slave at target address).
    DecodeError {
        /// Faulting address.
        fault_addr: u64,
    },
    /// Transfer timeout.
    Timeout,
    /// Descriptor chain is malformed.
    InvalidDescriptor,
}

impl fmt::Display for DmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DmaError::ChannelBusy => write!(f, "DMA channel busy"),
            DmaError::InvalidChannel => write!(f, "invalid DMA channel"),
            DmaError::InvalidLength => write!(f, "invalid transfer length"),
            DmaError::InvalidAddress => write!(f, "invalid address"),
            DmaError::TooManyDescriptors => write!(f, "too many SG descriptors"),
            DmaError::NoDescriptors => write!(f, "no descriptors provided"),
            DmaError::BusError { fault_addr } => {
                write!(f, "DMA bus error at 0x{fault_addr:x}")
            }
            DmaError::DecodeError { fault_addr } => {
                write!(f, "DMA decode error at 0x{fault_addr:x}")
            }
            DmaError::Timeout => write!(f, "DMA transfer timeout"),
            DmaError::InvalidDescriptor => write!(f, "invalid descriptor"),
        }
    }
}

/// Single DMA transfer descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaDescriptor {
    /// Source physical address.
    pub src_addr: u64,
    /// Destination physical address.
    pub dst_addr: u64,
    /// Transfer length in bytes.
    pub length: u32,
}

impl DmaDescriptor {
    /// Validate the descriptor.
    pub fn validate(&self) -> Result<(), DmaError> {
        if self.src_addr == 0 {
            return Err(DmaError::InvalidAddress);
        }
        if self.dst_addr == 0 {
            return Err(DmaError::InvalidAddress);
        }
        if self.length == 0 || self.length > MAX_TRANSFER_LEN {
            return Err(DmaError::InvalidLength);
        }
        Ok(())
    }
}

/// Scatter-gather descriptor chain.
#[derive(Debug)]
pub struct SgDescriptorChain {
    /// Descriptors in the chain.
    descriptors: [DmaDescriptor; MAX_SG_DESCRIPTORS],
    /// Number of active descriptors.
    count: usize,
}

impl Default for SgDescriptorChain {
    fn default() -> Self {
        Self::new()
    }
}

impl SgDescriptorChain {
    /// Create an empty descriptor chain.
    pub fn new() -> Self {
        Self {
            descriptors: [DmaDescriptor {
                src_addr: 0,
                dst_addr: 0,
                length: 0,
            }; MAX_SG_DESCRIPTORS],
            count: 0,
        }
    }

    /// Add a descriptor to the chain.
    pub fn push(&mut self, desc: DmaDescriptor) -> Result<(), DmaError> {
        desc.validate()?;
        if self.count >= MAX_SG_DESCRIPTORS {
            return Err(DmaError::TooManyDescriptors);
        }
        self.descriptors[self.count] = desc;
        self.count += 1;
        Ok(())
    }

    /// Number of descriptors in the chain.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get a descriptor by index.
    pub fn get(&self, index: usize) -> Option<&DmaDescriptor> {
        if index < self.count {
            Some(&self.descriptors[index])
        } else {
            None
        }
    }

    /// Total bytes across all descriptors.
    pub fn total_bytes(&self) -> u64 {
        let mut total = 0u64;
        for i in 0..self.count {
            total += self.descriptors[i].length as u64;
        }
        total
    }

    /// Clear the chain.
    pub fn clear(&mut self) {
        self.count = 0;
    }
}

/// DMA transfer token returned after starting a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaToken {
    /// Channel number.
    pub channel: u8,
    /// Transfer ID (unique within the channel).
    pub transfer_id: u32,
}

/// DMA transfer status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaStatus {
    /// Channel state.
    pub state: DmaChannelState,
    /// Number of descriptors completed (for SG mode).
    pub descriptors_completed: u16,
    /// Total bytes transferred so far.
    pub bytes_transferred: u64,
}

/// DMA channel state.
#[derive(Debug)]
struct DmaChannel {
    /// Current state.
    state: DmaChannelState,
    /// Current transfer ID.
    transfer_id: u32,
    /// Descriptors completed in current transfer.
    descriptors_completed: u16,
    /// Bytes transferred in current transfer.
    bytes_transferred: u64,
}

impl DmaChannel {
    fn new() -> Self {
        Self {
            state: DmaChannelState::Idle,
            transfer_id: 0,
            descriptors_completed: 0,
            bytes_transferred: 0,
        }
    }

    fn reset(&mut self) {
        self.state = DmaChannelState::Idle;
        self.descriptors_completed = 0;
        self.bytes_transferred = 0;
    }
}

/// Maximum number of DMA channels.
const MAX_CHANNELS: usize = 4;

/// AXI DMA controller driver.
///
/// Manages multiple DMA channels supporting both simple and scatter-gather
/// transfer modes.
#[derive(Debug)]
pub struct DmaController {
    /// MMIO base address of the DMA controller.
    base_addr: u64,
    /// DMA channels.
    channels: [DmaChannel; MAX_CHANNELS],
    /// Number of configured channels.
    num_channels: u8,
}

impl DmaController {
    /// Create a new DMA controller.
    ///
    /// `base_addr` is the MMIO base of the DMA controller registers.
    /// `num_channels` is the number of available channels (max 4).
    pub fn new(base_addr: u64, num_channels: u8) -> Result<Self, DmaError> {
        if base_addr == 0 {
            return Err(DmaError::InvalidAddress);
        }
        let nc = num_channels.min(MAX_CHANNELS as u8);
        let channels = [
            DmaChannel::new(),
            DmaChannel::new(),
            DmaChannel::new(),
            DmaChannel::new(),
        ];
        Ok(Self {
            base_addr,
            channels,
            num_channels: nc,
        })
    }

    /// Base address of the DMA controller.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Number of available channels.
    pub fn num_channels(&self) -> u8 {
        self.num_channels
    }

    /// Start a simple (single-buffer) DMA transfer.
    pub fn start_simple(
        &mut self,
        channel: u8,
        src_addr: u64,
        dst_addr: u64,
        length: u32,
    ) -> Result<DmaToken, DmaError> {
        let ch = self.get_channel_mut(channel)?;
        if ch.state == DmaChannelState::Running {
            return Err(DmaError::ChannelBusy);
        }

        let desc = DmaDescriptor {
            src_addr,
            dst_addr,
            length,
        };
        desc.validate()?;

        ch.transfer_id += 1;
        ch.state = DmaChannelState::Running;
        ch.descriptors_completed = 0;
        ch.bytes_transferred = 0;

        // In a real driver: program SA, DA, LENGTH, CONTROL registers and start
        // Stub: immediately mark as complete
        ch.descriptors_completed = 1;
        ch.bytes_transferred = length as u64;
        ch.state = DmaChannelState::Complete;

        Ok(DmaToken {
            channel,
            transfer_id: ch.transfer_id,
        })
    }

    /// Start a scatter-gather DMA transfer.
    pub fn start_scatter_gather(
        &mut self,
        channel: u8,
        chain: &SgDescriptorChain,
    ) -> Result<DmaToken, DmaError> {
        if chain.is_empty() {
            return Err(DmaError::NoDescriptors);
        }

        let ch = self.get_channel_mut(channel)?;
        if ch.state == DmaChannelState::Running {
            return Err(DmaError::ChannelBusy);
        }

        ch.transfer_id += 1;
        ch.state = DmaChannelState::Running;
        ch.descriptors_completed = 0;
        ch.bytes_transferred = 0;

        // In a real driver: build hardware descriptor ring in DMA-accessible memory,
        // program CURDESC and TAILDESC registers, start the SG engine.
        // Stub: immediately mark all descriptors as complete.
        ch.descriptors_completed = chain.len() as u16;
        ch.bytes_transferred = chain.total_bytes();
        ch.state = DmaChannelState::Complete;

        Ok(DmaToken {
            channel,
            transfer_id: ch.transfer_id,
        })
    }

    /// Poll the status of a DMA transfer.
    pub fn poll(&self, token: &DmaToken) -> Result<DmaStatus, DmaError> {
        let ch = self.get_channel(token.channel)?;
        Ok(DmaStatus {
            state: ch.state,
            descriptors_completed: ch.descriptors_completed,
            bytes_transferred: ch.bytes_transferred,
        })
    }

    /// Reset a DMA channel (halt and return to idle).
    pub fn reset_channel(&mut self, channel: u8) -> Result<(), DmaError> {
        let ch = self.get_channel_mut(channel)?;
        ch.reset();
        Ok(())
    }

    /// Halt a running DMA channel.
    pub fn halt_channel(&mut self, channel: u8) -> Result<(), DmaError> {
        let ch = self.get_channel_mut(channel)?;
        if ch.state == DmaChannelState::Running {
            ch.state = DmaChannelState::Halted;
        }
        Ok(())
    }

    /// Get channel state.
    pub fn channel_state(&self, channel: u8) -> Result<DmaChannelState, DmaError> {
        Ok(self.get_channel(channel)?.state)
    }

    fn get_channel(&self, channel: u8) -> Result<&DmaChannel, DmaError> {
        if channel >= self.num_channels {
            return Err(DmaError::InvalidChannel);
        }
        Ok(&self.channels[channel as usize])
    }

    fn get_channel_mut(&mut self, channel: u8) -> Result<&mut DmaChannel, DmaError> {
        if channel >= self.num_channels {
            return Err(DmaError::InvalidChannel);
        }
        Ok(&mut self.channels[channel as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    // -- DmaDescriptor -------------------------------------------------------

    #[test]
    fn test_descriptor_valid() {
        let desc = DmaDescriptor {
            src_addr: 0x1000,
            dst_addr: 0x2000,
            length: 4096,
        };
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_descriptor_null_src() {
        let desc = DmaDescriptor {
            src_addr: 0,
            dst_addr: 0x2000,
            length: 4096,
        };
        assert_eq!(desc.validate(), Err(DmaError::InvalidAddress));
    }

    #[test]
    fn test_descriptor_null_dst() {
        let desc = DmaDescriptor {
            src_addr: 0x1000,
            dst_addr: 0,
            length: 4096,
        };
        assert_eq!(desc.validate(), Err(DmaError::InvalidAddress));
    }

    #[test]
    fn test_descriptor_zero_length() {
        let desc = DmaDescriptor {
            src_addr: 0x1000,
            dst_addr: 0x2000,
            length: 0,
        };
        assert_eq!(desc.validate(), Err(DmaError::InvalidLength));
    }

    #[test]
    fn test_descriptor_excess_length() {
        let desc = DmaDescriptor {
            src_addr: 0x1000,
            dst_addr: 0x2000,
            length: MAX_TRANSFER_LEN + 1,
        };
        assert_eq!(desc.validate(), Err(DmaError::InvalidLength));
    }

    #[test]
    fn test_descriptor_max_length() {
        let desc = DmaDescriptor {
            src_addr: 0x1000,
            dst_addr: 0x2000,
            length: MAX_TRANSFER_LEN,
        };
        assert!(desc.validate().is_ok());
    }

    // -- SgDescriptorChain ---------------------------------------------------

    #[test]
    fn test_sg_chain_new() {
        let chain = SgDescriptorChain::new();
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
        assert_eq!(chain.total_bytes(), 0);
    }

    #[test]
    fn test_sg_chain_push() {
        let mut chain = SgDescriptorChain::new();
        chain
            .push(DmaDescriptor {
                src_addr: 0x1000,
                dst_addr: 0x2000,
                length: 512,
            })
            .unwrap();
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
        assert_eq!(chain.total_bytes(), 512);
    }

    #[test]
    fn test_sg_chain_push_multiple() {
        let mut chain = SgDescriptorChain::new();
        for i in 1..=5 {
            chain
                .push(DmaDescriptor {
                    src_addr: i * 0x1000,
                    dst_addr: i * 0x2000,
                    length: 256,
                })
                .unwrap();
        }
        assert_eq!(chain.len(), 5);
        assert_eq!(chain.total_bytes(), 1280);
    }

    #[test]
    fn test_sg_chain_push_invalid_descriptor() {
        let mut chain = SgDescriptorChain::new();
        let result = chain.push(DmaDescriptor {
            src_addr: 0,
            dst_addr: 0x2000,
            length: 512,
        });
        assert_eq!(result, Err(DmaError::InvalidAddress));
    }

    #[test]
    fn test_sg_chain_get() {
        let mut chain = SgDescriptorChain::new();
        chain
            .push(DmaDescriptor {
                src_addr: 0x1000,
                dst_addr: 0x2000,
                length: 100,
            })
            .unwrap();
        let desc = chain.get(0).unwrap();
        assert_eq!(desc.src_addr, 0x1000);
        assert_eq!(desc.dst_addr, 0x2000);
        assert_eq!(desc.length, 100);
        assert!(chain.get(1).is_none());
    }

    #[test]
    fn test_sg_chain_clear() {
        let mut chain = SgDescriptorChain::new();
        chain
            .push(DmaDescriptor {
                src_addr: 0x1000,
                dst_addr: 0x2000,
                length: 512,
            })
            .unwrap();
        chain.clear();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_sg_chain_overflow() {
        let mut chain = SgDescriptorChain::new();
        for i in 0..MAX_SG_DESCRIPTORS {
            chain
                .push(DmaDescriptor {
                    src_addr: (i as u64 + 1) * 0x1000,
                    dst_addr: (i as u64 + 1) * 0x2000,
                    length: 64,
                })
                .unwrap();
        }
        let result = chain.push(DmaDescriptor {
            src_addr: 0xFF000,
            dst_addr: 0xFF000,
            length: 64,
        });
        assert_eq!(result, Err(DmaError::TooManyDescriptors));
    }

    // -- DmaController -------------------------------------------------------

    #[test]
    fn test_controller_new() {
        let ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        assert_eq!(ctrl.base_addr(), 0x4060_0000);
        assert_eq!(ctrl.num_channels(), 2);
    }

    #[test]
    fn test_controller_new_invalid_addr() {
        let err = DmaController::new(0, 2).unwrap_err();
        assert_eq!(err, DmaError::InvalidAddress);
    }

    #[test]
    fn test_controller_simple_transfer() {
        let mut ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        let token = ctrl.start_simple(0, 0x1000, 0x2000, 4096).unwrap();
        assert_eq!(token.channel, 0);

        let status = ctrl.poll(&token).unwrap();
        assert_eq!(status.state, DmaChannelState::Complete);
        assert_eq!(status.bytes_transferred, 4096);
        assert_eq!(status.descriptors_completed, 1);
    }

    #[test]
    fn test_controller_simple_invalid_channel() {
        let mut ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        let err = ctrl.start_simple(2, 0x1000, 0x2000, 4096).unwrap_err();
        assert_eq!(err, DmaError::InvalidChannel);
    }

    #[test]
    fn test_controller_sg_transfer() {
        let mut ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        ctrl.reset_channel(0).unwrap();

        let mut chain = SgDescriptorChain::new();
        chain
            .push(DmaDescriptor {
                src_addr: 0x1000,
                dst_addr: 0x2000,
                length: 1024,
            })
            .unwrap();
        chain
            .push(DmaDescriptor {
                src_addr: 0x3000,
                dst_addr: 0x4000,
                length: 2048,
            })
            .unwrap();

        let token = ctrl.start_scatter_gather(0, &chain).unwrap();
        let status = ctrl.poll(&token).unwrap();
        assert_eq!(status.state, DmaChannelState::Complete);
        assert_eq!(status.descriptors_completed, 2);
        assert_eq!(status.bytes_transferred, 3072);
    }

    #[test]
    fn test_controller_sg_empty_chain() {
        let mut ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        let chain = SgDescriptorChain::new();
        let err = ctrl.start_scatter_gather(0, &chain).unwrap_err();
        assert_eq!(err, DmaError::NoDescriptors);
    }

    #[test]
    fn test_controller_reset_channel() {
        let mut ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        ctrl.start_simple(0, 0x1000, 0x2000, 256).unwrap();
        ctrl.reset_channel(0).unwrap();
        assert_eq!(ctrl.channel_state(0).unwrap(), DmaChannelState::Idle);
    }

    #[test]
    fn test_controller_halt_channel() {
        let mut ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        // Halt an idle channel does nothing harmful
        ctrl.halt_channel(0).unwrap();
        assert_eq!(ctrl.channel_state(0).unwrap(), DmaChannelState::Idle);
    }

    #[test]
    fn test_controller_invalid_channel_state() {
        let ctrl = DmaController::new(0x4060_0000, 2).unwrap();
        assert_eq!(ctrl.channel_state(5), Err(DmaError::InvalidChannel));
    }

    // -- DmaError Display ----------------------------------------------------

    #[test]
    fn test_dma_error_display() {
        assert_eq!(format!("{}", DmaError::ChannelBusy), "DMA channel busy");
        assert_eq!(format!("{}", DmaError::InvalidChannel), "invalid DMA channel");
        assert_eq!(format!("{}", DmaError::InvalidLength), "invalid transfer length");
        assert_eq!(format!("{}", DmaError::InvalidAddress), "invalid address");
        assert_eq!(format!("{}", DmaError::TooManyDescriptors), "too many SG descriptors");
        assert_eq!(format!("{}", DmaError::NoDescriptors), "no descriptors provided");
        assert_eq!(
            format!("{}", DmaError::BusError { fault_addr: 0xDEAD }),
            "DMA bus error at 0xdead"
        );
        assert_eq!(
            format!("{}", DmaError::DecodeError { fault_addr: 0xBEEF }),
            "DMA decode error at 0xbeef"
        );
        assert_eq!(format!("{}", DmaError::Timeout), "DMA transfer timeout");
        assert_eq!(format!("{}", DmaError::InvalidDescriptor), "invalid descriptor");
    }

    // -- DmaDirection --------------------------------------------------------

    #[test]
    fn test_dma_direction_variants() {
        assert_ne!(DmaDirection::DeviceToMemory, DmaDirection::MemoryToDevice);
    }
}
