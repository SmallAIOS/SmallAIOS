// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DMA/Copy engine for host<->device transfers.
//!
//! Manages asynchronous memory transfers between host RAM and device VRAM
//! (and device-to-device copies).  Each transfer goes through a strict state
//! machine: `Pending -> InProgress -> Completed | Failed`.  Pending transfers
//! may also be cancelled.

#![allow(dead_code)]

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
    /// Submitted but not yet started by the engine.
    Pending,
    /// Currently being executed by the copy engine.
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

/// DMA / copy-engine controller.
#[derive(Clone, Debug, PartialEq)]
pub struct DmaEngine {
    /// All tracked transfer descriptors.
    transfers: Vec<DmaTransfer>,
    /// Monotonically increasing id generator.
    next_id: u64,
    /// Running total of successfully transferred bytes.
    bytes_transferred: u64,
    /// Whether the engine is active (ready to accept work).
    active: bool,
}

impl Default for DmaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DmaEngine {
    /// Create a new, active DMA engine with no pending work.
    pub fn new() -> Self {
        Self {
            transfers: Vec::new(),
            next_id: 1,
            bytes_transferred: 0,
            active: true,
        }
    }

    /// Submit a new transfer request.
    ///
    /// The transfer is created in [`DmaStatus::Pending`] state.
    /// Returns the unique [`TransferId`] on success.
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

    /// Cancel a pending transfer.  Only [`DmaStatus::Pending`] transfers may
    /// be cancelled.
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

    // 1. Submit transfer returns unique ID.
    #[test]
    fn test_submit_unique_ids() {
        let mut eng = DmaEngine::new();
        let id1 = eng
            .submit(DmaDirection::HostToDevice, 0x1000, 0x2000, 4096)
            .unwrap();
        let id2 = eng
            .submit(DmaDirection::DeviceToHost, 0x3000, 0x4000, 8192)
            .unwrap();
        assert_ne!(id1, id2);
    }

    // 2. Start pending transfer.
    #[test]
    fn test_start_pending() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
        eng.start(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::InProgress));
    }

    // 3. Complete in-progress transfer.
    #[test]
    fn test_complete_in_progress() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.complete(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Completed));
    }

    // 4. Fail in-progress transfer.
    #[test]
    fn test_fail_in_progress() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::DeviceToHost, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.fail(id).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Failed));
    }

    // 5. Cancel pending transfer.
    #[test]
    fn test_cancel_pending() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.cancel(id).unwrap();
        // After cancel the transfer is removed entirely.
        assert_eq!(eng.status(id), Err(GpuError::NotFound));
    }

    // 6. Cannot start completed transfer.
    #[test]
    fn test_cannot_start_completed() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.complete(id).unwrap();
        assert_eq!(eng.start(id), Err(GpuError::InvalidState));
    }

    // 7. Cannot complete pending transfer.
    #[test]
    fn test_cannot_complete_pending() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        assert_eq!(eng.complete(id), Err(GpuError::InvalidState));
    }

    // 8. Zero-size transfer rejected.
    #[test]
    fn test_zero_size_rejected() {
        let mut eng = DmaEngine::new();
        assert_eq!(
            eng.submit(DmaDirection::HostToDevice, 0, 0, 0),
            Err(GpuError::DmaError)
        );
    }

    // 9. Transfer too large rejected.
    #[test]
    fn test_transfer_too_large() {
        let mut eng = DmaEngine::new();
        assert_eq!(
            eng.submit(DmaDirection::HostToDevice, 0, 0, MAX_TRANSFER_SIZE + 1),
            Err(GpuError::TransferTooLarge)
        );
    }

    // 10. Bytes transferred tracking.
    #[test]
    fn test_bytes_transferred() {
        let mut eng = DmaEngine::new();
        let id1 = eng.submit(DmaDirection::HostToDevice, 0, 0, 1000).unwrap();
        eng.start(id1).unwrap();
        eng.complete(id1).unwrap();

        let id2 = eng.submit(DmaDirection::DeviceToHost, 0, 0, 2000).unwrap();
        eng.start(id2).unwrap();
        eng.complete(id2).unwrap();

        assert_eq!(eng.total_bytes_transferred(), 3000);
    }

    // 11. Pending and active counts.
    #[test]
    fn test_pending_active_counts() {
        let mut eng = DmaEngine::new();
        let id1 = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        let _id2 = eng.submit(DmaDirection::DeviceToHost, 0, 0, 4096).unwrap();
        assert_eq!(eng.pending_count(), 2);
        assert_eq!(eng.active_count(), 0);

        eng.start(id1).unwrap();
        assert_eq!(eng.pending_count(), 1);
        assert_eq!(eng.active_count(), 1);
    }

    // 12. Status lookup.
    #[test]
    fn test_status_lookup() {
        let mut eng = DmaEngine::new();
        assert_eq!(eng.status(TransferId(999)), Err(GpuError::NotFound));
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
    }

    // 13. Direction types.
    #[test]
    fn test_direction_types() {
        let mut eng = DmaEngine::new();
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

    // 14. Full lifecycle (Pending -> InProgress -> Completed).
    #[test]
    fn test_full_lifecycle() {
        let mut eng = DmaEngine::new();
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

    // 15. Cannot cancel in-progress transfer.
    #[test]
    fn test_cannot_cancel_in_progress() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        assert_eq!(eng.cancel(id), Err(GpuError::InvalidState));
    }

    // 16. Queue full at MAX_TRANSFERS.
    #[test]
    fn test_queue_full() {
        let mut eng = DmaEngine::new();
        for _ in 0..MAX_TRANSFERS {
            eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        }
        assert_eq!(
            eng.submit(DmaDirection::HostToDevice, 0, 0, 4096),
            Err(GpuError::QueueFull)
        );
    }

    // 17. Failed transfer does not count bytes.
    #[test]
    fn test_failed_no_bytes() {
        let mut eng = DmaEngine::new();
        let id = eng.submit(DmaDirection::HostToDevice, 0, 0, 4096).unwrap();
        eng.start(id).unwrap();
        eng.fail(id).unwrap();
        assert_eq!(eng.total_bytes_transferred(), 0);
    }

    // 18. Max transfer size boundary (exactly MAX_TRANSFER_SIZE is allowed).
    #[test]
    fn test_max_transfer_size_boundary() {
        let mut eng = DmaEngine::new();
        let id = eng
            .submit(DmaDirection::HostToDevice, 0, 0, MAX_TRANSFER_SIZE)
            .unwrap();
        assert_eq!(eng.status(id), Ok(&DmaStatus::Pending));
    }
}
