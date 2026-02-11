// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SDMA (System DMA) engine for host<->device transfers.
//!
//! Manages asynchronous memory transfers between host RAM and device VRAM
//! (and device-to-device copies). AMD GPUs use SDMA engines (typically 2
//! per GPU) for async data movement. Each transfer goes through a strict
//! state machine: `Pending -> InProgress -> Completed | Failed`.

use alloc::vec::Vec;

use crate::GpuError;

/// Direction of a DMA transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DmaDirection {
    /// Host RAM -> Device VRAM.
    HostToDevice,
    /// Device VRAM -> Host RAM.
    DeviceToHost,
    /// Device VRAM -> Device VRAM (peer or same GPU).
    DeviceToDevice,
}

/// State of a DMA transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DmaStatus {
    /// Submitted but not yet started by the SDMA engine.
    Pending,
    /// Currently being executed by the SDMA engine.
    InProgress,
    /// Successfully finished.
    Completed,
    /// Terminated with an error.
    Failed,
}

/// Unique identifier for a DMA transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

/// A single DMA transfer descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct DmaTransfer {
    /// Unique transfer identifier.
    pub id: TransferId,
    /// Direction of the transfer.
    pub direction: DmaDirection,
    /// Source address (host-physical or GPU-virtual, depending on direction).
    pub src_addr: u64,
    /// Destination address.
    pub dst_addr: u64,
    /// Transfer size in bytes.
    pub size: u64,
    /// Current status.
    pub status: DmaStatus,
}

/// Maximum number of tracked transfers.
pub const MAX_TRANSFERS: usize = 256;

/// Maximum size of a single DMA transfer (256 MiB).
pub const MAX_TRANSFER_SIZE: u64 = 256 * 1024 * 1024;

/// SDMA engine controller.
#[derive(Clone, Debug, PartialEq)]
pub struct SdmaEngine {
    /// All tracked transfer descriptors.
    transfers: Vec<DmaTransfer>,
    /// Monotonically increasing id generator.
    next_id: u64,
    /// Running total of successfully transferred bytes.
    bytes_transferred: u64,
    /// Whether the engine is active (ready to accept work).
    active: bool,
    /// SDMA engine index (AMD GPUs typically have 2 SDMA engines).
    engine_index: u32,
}

impl SdmaEngine {
    /// Create a new, active SDMA engine with no pending work.
    pub fn new(engine_index: u32) -> Self {
        Self {
            transfers: Vec::new(),
            next_id: 1,
            bytes_transferred: 0,
            active: true,
            engine_index,
        }
    }

    /// Submit a new transfer request.
    ///
    /// The transfer is created in [`DmaStatus::Pending`] state.
    pub fn submit(
        &mut self,
        direction: DmaDirection,
        src: u64,
        dst: u64,
        size: u64,
    ) -> Result<TransferId, GpuError> {
        if size == 0 {
            return Err(GpuError::DmaError);
        }
        if size > MAX_TRANSFER_SIZE {
            return Err(GpuError::TransferTooLarge);
        }
        if self.transfers.len() >= MAX_TRANSFERS {
            return Err(GpuError::QueueFull);
        }

        let id = TransferId(self.next_id);
        self.next_id += 1;

        self.transfers.push(DmaTransfer {
            id,
            direction,
            src_addr: src,
            dst_addr: dst,
            size,
            status: DmaStatus::Pending,
        });

        Ok(id)
    }

    /// Start a pending transfer (Pending -> InProgress).
    pub fn start(&mut self, id: TransferId) -> Result<(), GpuError> {
        let xfer = self.find_mut(id)?;
        if xfer.status != DmaStatus::Pending {
            return Err(GpuError::InvalidState);
        }
        xfer.status = DmaStatus::InProgress;
        Ok(())
    }

    /// Mark an in-progress transfer as completed (InProgress -> Completed).
    pub fn complete(&mut self, id: TransferId) -> Result<(), GpuError> {
        let xfer = self.find_mut(id)?;
        if xfer.status != DmaStatus::InProgress {
            return Err(GpuError::InvalidState);
        }
        xfer.status = DmaStatus::Completed;
        self.bytes_transferred += xfer.size;
        Ok(())
    }

    /// Mark an in-progress transfer as failed (InProgress -> Failed).
    pub fn fail(&mut self, id: TransferId) -> Result<(), GpuError> {
        let xfer = self.find_mut(id)?;
        if xfer.status != DmaStatus::InProgress {
            return Err(GpuError::InvalidState);
        }
        xfer.status = DmaStatus::Failed;
        Ok(())
    }

    /// Query the current status of a transfer.
    pub fn status(&self, id: TransferId) -> Result<&DmaStatus, GpuError> {
        let xfer = self
            .transfers
            .iter()
            .find(|t| t.id == id)
            .ok_or(GpuError::NotFound)?;
        Ok(&xfer.status)
    }

    /// Cancel a pending transfer.
    pub fn cancel(&mut self, id: TransferId) -> Result<(), GpuError> {
        let idx = self
            .transfers
            .iter()
            .position(|t| t.id == id)
            .ok_or(GpuError::NotFound)?;

        if self.transfers[idx].status != DmaStatus::Pending {
            return Err(GpuError::InvalidState);
        }

        self.transfers.remove(idx);
        Ok(())
    }

    /// Total bytes successfully transferred since engine creation.
    pub fn total_bytes_transferred(&self) -> u64 {
        self.bytes_transferred
    }

    /// Number of transfers currently in [`DmaStatus::Pending`] state.
    pub fn pending_count(&self) -> usize {
        self.transfers
            .iter()
            .filter(|t| t.status == DmaStatus::Pending)
            .count()
    }

    /// Number of transfers currently in [`DmaStatus::InProgress`] state.
    pub fn active_count(&self) -> usize {
        self.transfers
            .iter()
            .filter(|t| t.status == DmaStatus::InProgress)
            .count()
    }

    /// SDMA engine index.
    pub fn engine_index(&self) -> u32 {
        self.engine_index
    }

    // -- internal helpers ---------------------------------------------------

    fn find_mut(&mut self, id: TransferId) -> Result<&mut DmaTransfer, GpuError> {
        self.transfers
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or(GpuError::NotFound)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_unique_ids() {
        let mut eng = SdmaEngine::new(0);
        let id1 = eng
            .submit(DmaDirection::HostToDevice, 0x1000, 0x2000, 4096)
            .unwrap();
        let id2 = eng
            .submit(DmaDirection::DeviceToHost, 0x3000, 0x4000, 8192)
            .unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_start_pending() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
        eng.start(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::InProgress));
    }

    #[test]
    fn test_complete_in_progress() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.complete(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Completed));
    }

    #[test]
    fn test_fail_in_progress() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::DeviceToHost, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.fail(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Failed));
    }

    #[test]
    fn test_cancel_pending() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.cancel(id).unwrap();
        assert_eq!(eng.status(id), Err(GpuError::NotFound));
    }

    #[test]
    fn test_cannot_start_completed() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.complete(id).unwrap();
        assert_eq!(eng.start(id), Err(GpuError::InvalidState));
    }

    #[test]
    fn test_cannot_complete_pending() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        assert_eq!(eng.complete(id), Err(GpuError::InvalidState));
    }

    #[test]
    fn test_zero_size_rejected() {
        let mut eng = SdmaEngine::new(0);
        assert_eq!(
            eng.submit(DmaDirection::HostToDevice, 0, 0, 0),
            Err(GpuError::DmaError)
        );
    }

    #[test]
    fn test_transfer_too_large() {
        let mut eng = SdmaEngine::new(0);
        assert_eq!(
            eng.submit(DmaDirection::HostToDevice, 0, 0, MAX_TRANSFER_SIZE + 1),
            Err(GpuError::TransferTooLarge)
        );
    }

    #[test]
    fn test_bytes_transferred() {
        let mut eng = SdmaEngine::new(0);
        let id1 = eng.submit(DmaDirection::HostToDevice, 0, 0, 1000).unwrap();
        eng.start(id1).unwrap();
        eng.complete(id1).unwrap();

        let id2 = eng.submit(DmaDirection::DeviceToHost, 0, 0, 2000).unwrap();
        eng.start(id2).unwrap();
        eng.complete(id2).unwrap();

        assert_eq!(eng.total_bytes_transferred(), 3000);
    }

    #[test]
    fn test_pending_active_counts() {
        let mut eng = SdmaEngine::new(0);
        let id1 = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        let _id2 = eng.submit(DmaDirection::DeviceToHost, 0, 0, 4096).unwrap();
        assert_eq!(eng.pending_count(), 2);
        assert_eq!(eng.active_count(), 0);

        eng.start(id1).unwrap();
        assert_eq!(eng.pending_count(), 1);
        assert_eq!(eng.active_count(), 1);
    }

    #[test]
    fn test_status_lookup() {
        let mut eng = SdmaEngine::new(0);
        assert_eq!(eng.status(TransferId(999)), Err(GpuError::NotFound));
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
    }

    #[test]
    fn test_direction_types() {
        let mut eng = SdmaEngine::new(0);
        let id_h2d = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        let id_d2h = eng.submit(DmaDirection::DeviceToHost, 0, 0, 4096).unwrap();
        let id_d2d = eng
            .submit(DmaDirection::DeviceToDevice, 0, 0, 4096)
            .unwrap();
        let t1 = eng.transfers.iter().find(|t| t.id == id_h2d).unwrap();
        let t2 = eng.transfers.iter().find(|t| t.id == id_d2h).unwrap();
        let t3 = eng.transfers.iter().find(|t| t.id == id_d2d).unwrap();
        assert_eq!(t1.direction, DmaDirection::HostToDevice);
        assert_eq!(t2.direction, DmaDirection::DeviceToHost);
        assert_eq!(t3.direction, DmaDirection::DeviceToDevice);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut eng = SdmaEngine::new(0);
        let id = eng
            .submit(DmaDirection::HostToDevice, 0xA000, 0xB000, 65536)
            .unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
        eng.start(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::InProgress));
        eng.complete(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Completed));
        assert_eq!(eng.total_bytes_transferred(), 65536);
    }

    #[test]
    fn test_cannot_cancel_in_progress() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        assert_eq!(eng.cancel(id), Err(GpuError::InvalidState));
    }

    #[test]
    fn test_queue_full() {
        let mut eng = SdmaEngine::new(0);
        for _ in 0..MAX_TRANSFERS {
            eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        }
        assert_eq!(
            eng.submit(DmaDirection::HostToDevice, 0, 0, 4096),
            Err(GpuError::QueueFull)
        );
    }

    #[test]
    fn test_failed_no_bytes() {
        let mut eng = SdmaEngine::new(0);
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.fail(id).unwrap();
        assert_eq!(eng.total_bytes_transferred(), 0);
    }

    #[test]
    fn test_max_transfer_size_boundary() {
        let mut eng = SdmaEngine::new(0);
        let id = eng
            .submit(DmaDirection::HostToDevice, 0, 0, MAX_TRANSFER_SIZE)
            .unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
    }

    #[test]
    fn test_engine_index() {
        let eng0 = SdmaEngine::new(0);
        let eng1 = SdmaEngine::new(1);
        assert_eq!(eng0.engine_index(), 0);
        assert_eq!(eng1.engine_index(), 1);
    }
}
