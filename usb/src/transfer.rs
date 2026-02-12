// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB transfer management — control and bulk transfer helpers.

use crate::UsbError;
use smallaios_kernel::hal::{UsbHostController, UsbSetupPacket, UsbTransferResult};

// ---------------------------------------------------------------------------
// Vendor control transfer helper
// ---------------------------------------------------------------------------

/// Submit a vendor control transfer (commonly used by SDR devices).
///
/// Builds and sends a vendor-type, device-recipient control transfer.
pub fn vendor_control_out(
    hc: &mut dyn UsbHostController,
    slot: u8,
    request: u8,
    value: u16,
    index: u16,
    data: &[u8],
) -> Result<UsbTransferResult, UsbError> {
    let setup = UsbSetupPacket::new(
        crate::request_type::DIR_OUT | crate::request_type::TYPE_VENDOR | crate::request_type::RECIP_DEVICE,
        request,
        value,
        index,
        data.len() as u16,
    );
    // For OUT transfers with data, we need a mutable copy to satisfy the trait.
    // The host controller reads from this buffer; the data isn't modified.
    if data.is_empty() {
        Ok(hc.control_transfer(slot, &setup, None)?)
    } else {
        let mut buf = [0u8; 4096];
        let len = data.len().min(buf.len());
        buf[..len].copy_from_slice(&data[..len]);
        Ok(hc.control_transfer(slot, &setup, Some(&mut buf[..len]))?)
    }
}

/// Submit a vendor control IN transfer (read data from device).
pub fn vendor_control_in(
    hc: &mut dyn UsbHostController,
    slot: u8,
    request: u8,
    value: u16,
    index: u16,
    buf: &mut [u8],
) -> Result<UsbTransferResult, UsbError> {
    let setup = UsbSetupPacket::new(
        crate::request_type::DIR_IN | crate::request_type::TYPE_VENDOR | crate::request_type::RECIP_DEVICE,
        request,
        value,
        index,
        buf.len() as u16,
    );
    Ok(hc.control_transfer(slot, &setup, Some(buf))?)
}

// ---------------------------------------------------------------------------
// Bulk transfer helpers
// ---------------------------------------------------------------------------

/// Submit a bulk IN transfer (device → host).
pub fn bulk_in(
    hc: &mut dyn UsbHostController,
    slot: u8,
    endpoint: u8,
    buf: &mut [u8],
) -> Result<UsbTransferResult, UsbError> {
    // Ensure endpoint address has IN direction bit set.
    let ep_addr = endpoint | 0x80;
    Ok(hc.bulk_transfer(slot, ep_addr, buf)?)
}

/// Submit a bulk OUT transfer (host → device).
pub fn bulk_out(
    hc: &mut dyn UsbHostController,
    slot: u8,
    endpoint: u8,
    data: &mut [u8],
) -> Result<UsbTransferResult, UsbError> {
    // Ensure endpoint address has OUT direction (bit 7 clear).
    let ep_addr = endpoint & 0x7F;
    Ok(hc.bulk_transfer(slot, ep_addr, data)?)
}

// ---------------------------------------------------------------------------
// Endpoint state management
// ---------------------------------------------------------------------------

/// USB endpoint state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointState {
    /// Endpoint is idle, ready for transfers.
    Idle,
    /// Endpoint has an active (pending) transfer.
    Active,
    /// Endpoint is halted due to a STALL or error.
    Halted,
}

/// Tracks the state of an endpoint.
#[derive(Debug, Clone, Copy)]
pub struct EndpointTracker {
    /// Endpoint address.
    pub address: u8,
    /// Current state.
    pub state: EndpointState,
    /// Number of successfully completed transfers.
    pub completed_count: u32,
    /// Number of failed transfers.
    pub error_count: u32,
}

impl EndpointTracker {
    /// Create a new tracker for the given endpoint address.
    pub fn new(address: u8) -> Self {
        Self {
            address,
            state: EndpointState::Idle,
            completed_count: 0,
            error_count: 0,
        }
    }

    /// Record a successful transfer completion.
    pub fn on_complete(&mut self) {
        self.state = EndpointState::Idle;
        self.completed_count = self.completed_count.saturating_add(1);
    }

    /// Record a transfer error.
    pub fn on_error(&mut self) {
        self.error_count = self.error_count.saturating_add(1);
    }

    /// Mark endpoint as halted.
    pub fn on_halt(&mut self) {
        self.state = EndpointState::Halted;
        self.error_count = self.error_count.saturating_add(1);
    }

    /// Clear halt condition (after CLEAR_FEATURE succeeds).
    pub fn on_clear_halt(&mut self) {
        self.state = EndpointState::Idle;
    }

    /// Mark endpoint as having an active transfer.
    pub fn on_submit(&mut self) {
        self.state = EndpointState::Active;
    }
}

/// Clear the halt condition on an endpoint via CLEAR_FEATURE(ENDPOINT_HALT).
pub fn clear_endpoint_halt(
    hc: &mut dyn UsbHostController,
    slot: u8,
    endpoint: u8,
) -> Result<(), UsbError> {
    let setup = crate::enumeration::make_clear_endpoint_halt(endpoint);
    let result = hc.control_transfer(slot, &setup, None)?;
    if result.success {
        Ok(())
    } else if result.stalled {
        Err(UsbError::Stall)
    } else {
        Err(UsbError::TransferError)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_tracker_lifecycle() {
        let mut tracker = EndpointTracker::new(0x81);
        assert_eq!(tracker.state, EndpointState::Idle);
        assert_eq!(tracker.completed_count, 0);

        tracker.on_submit();
        assert_eq!(tracker.state, EndpointState::Active);

        tracker.on_complete();
        assert_eq!(tracker.state, EndpointState::Idle);
        assert_eq!(tracker.completed_count, 1);
    }

    #[test]
    fn test_endpoint_tracker_halt_clear() {
        let mut tracker = EndpointTracker::new(0x02);
        tracker.on_halt();
        assert_eq!(tracker.state, EndpointState::Halted);
        assert_eq!(tracker.error_count, 1);

        tracker.on_clear_halt();
        assert_eq!(tracker.state, EndpointState::Idle);
    }

    #[test]
    fn test_endpoint_tracker_error_count() {
        let mut tracker = EndpointTracker::new(0x83);
        tracker.on_error();
        tracker.on_error();
        tracker.on_error();
        assert_eq!(tracker.error_count, 3);
    }

    #[test]
    fn test_endpoint_state_variants() {
        assert_ne!(EndpointState::Idle, EndpointState::Active);
        assert_ne!(EndpointState::Active, EndpointState::Halted);
    }
}
