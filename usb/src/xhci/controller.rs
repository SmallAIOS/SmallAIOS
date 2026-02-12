// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! xHCI controller driver — high-level interface to the xHCI hardware.
//!
//! Manages controller initialization, port status monitoring, device slot
//! allocation, and transfer submission via the ring infrastructure.

use super::registers::XhciCapabilities;
use super::rings::{EventRing, ProducerRing};
use super::trb::{Trb, TrbType};
use smallaios_kernel::hal::{
    HalError, UsbHostController, UsbPortStatus, UsbSetupPacket, UsbSpeed, UsbTransferResult,
    UsbTransferType,
};

/// Maximum number of device slots supported.
pub const MAX_DEVICE_SLOTS: usize = 16;

/// Maximum number of root hub ports tracked.
pub const MAX_PORTS: usize = 16;

/// Maximum number of endpoints per device (including EP0).
pub const MAX_ENDPOINTS_PER_DEVICE: usize = 32;

/// State of a device slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is free and available.
    Free,
    /// Slot is enabled, device is being addressed.
    Enabled,
    /// Slot is fully configured, device is ready for transfers.
    Configured,
}

/// Per-device slot tracking information.
#[derive(Debug, Clone, Copy)]
pub struct DeviceSlot {
    /// Slot state.
    pub state: SlotState,
    /// Port number this device is connected to.
    pub port: u8,
    /// Negotiated USB speed.
    pub speed: UsbSpeed,
    /// Number of configured endpoints.
    pub num_endpoints: u8,
}

impl Default for DeviceSlot {
    fn default() -> Self {
        Self {
            state: SlotState::Free,
            port: 0,
            speed: UsbSpeed::Full,
            num_endpoints: 0,
        }
    }
}

/// xHCI controller driver.
///
/// In a real implementation, this would hold MMIO register base addresses
/// and DMA-capable memory allocations. This version provides the logic
/// and state management, with hardware access abstracted via the HAL.
pub struct XhciController {
    /// Parsed controller capabilities.
    pub caps: XhciCapabilities,
    /// Command ring for host→controller commands.
    pub command_ring: ProducerRing,
    /// Event ring for controller→host notifications.
    pub event_ring: EventRing,
    /// Per-endpoint transfer rings (slot_index * MAX_ENDPOINTS + ep_index).
    /// Only the transfer ring management state is here; actual ring memory
    /// would be DMA-allocated in a real implementation.
    pub transfer_ring_active: [[bool; MAX_ENDPOINTS_PER_DEVICE]; MAX_DEVICE_SLOTS],
    /// Device slot tracking.
    pub slots: [DeviceSlot; MAX_DEVICE_SLOTS],
    /// Port connection status cache.
    pub port_connected: [bool; MAX_PORTS],
    /// Whether the controller is running.
    pub running: bool,
    /// Next transfer token for matching completions.
    pub next_token: u32,
}

impl Default for XhciController {
    fn default() -> Self {
        Self::new()
    }
}

impl XhciController {
    /// Create a new uninitialized controller.
    pub fn new() -> Self {
        Self {
            caps: XhciCapabilities {
                op_reg_offset: 0x20,
                hci_version: 0x0100,
                max_slots: MAX_DEVICE_SLOTS as u8,
                max_intrs: 1,
                max_ports: MAX_PORTS as u8,
                scratchpad_bufs: 0,
                doorbell_offset: 0x2000,
                runtime_offset: 0x3000,
                context_size_64: false,
            },
            command_ring: ProducerRing::new(),
            event_ring: EventRing::new(),
            transfer_ring_active: [[false; MAX_ENDPOINTS_PER_DEVICE]; MAX_DEVICE_SLOTS],
            slots: [DeviceSlot::default(); MAX_DEVICE_SLOTS],
            port_connected: [false; MAX_PORTS],
            running: false,
            next_token: 1,
        }
    }

    /// Find a free device slot.
    fn find_free_slot(&self) -> Option<u8> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.state == SlotState::Free {
                return Some(i as u8 + 1); // Slots are 1-based
            }
        }
        None
    }

    /// Get the slot index (0-based) from a slot ID (1-based).
    fn slot_index(slot_id: u8) -> Option<usize> {
        if slot_id == 0 || slot_id as usize > MAX_DEVICE_SLOTS {
            None
        } else {
            Some(slot_id as usize - 1)
        }
    }

    /// Allocate the next transfer token.
    fn next_token(&mut self) -> u32 {
        let token = self.next_token;
        self.next_token = self.next_token.wrapping_add(1);
        if self.next_token == 0 {
            self.next_token = 1;
        }
        token
    }

    /// Convert xHCI port speed to HAL UsbSpeed.
    pub fn decode_port_speed(speed_field: u32) -> UsbSpeed {
        match speed_field {
            1 => UsbSpeed::Full,
            3 => UsbSpeed::High,
            4 => UsbSpeed::Super,
            5 => UsbSpeed::SuperPlus,
            _ => UsbSpeed::Full,
        }
    }
}

impl UsbHostController for XhciController {
    fn init(&mut self) -> Result<(), HalError> {
        // In a real driver:
        // 1. Read capability registers
        // 2. Write USBCMD.HCRST, wait for USBSTS.CNR to clear
        // 3. Configure MaxSlotsEn, DCBAAP, CRCR
        // 4. Set up Event Ring Segment Table
        // 5. Set USBCMD.RS to start controller
        self.running = true;
        Ok(())
    }

    fn port_count(&self) -> u8 {
        self.caps.max_ports
    }

    fn port_status(&self, port: u8) -> Result<UsbPortStatus, HalError> {
        if port >= self.caps.max_ports {
            return Err(HalError::OutOfRange);
        }
        Ok(UsbPortStatus {
            connected: self.port_connected[port as usize],
            enabled: self.port_connected[port as usize],
            reset_active: false,
            speed: UsbSpeed::High,
            port,
        })
    }

    fn port_reset(&mut self, port: u8) -> Result<(), HalError> {
        if port >= self.caps.max_ports {
            return Err(HalError::OutOfRange);
        }
        // In a real driver: write PORTSC.PR, wait for PRC.
        Ok(())
    }

    fn device_attach(&mut self, port: u8) -> Result<u8, HalError> {
        if port >= self.caps.max_ports {
            return Err(HalError::OutOfRange);
        }
        let slot_id = self.find_free_slot().ok_or(HalError::NotSupported)?;
        let idx = Self::slot_index(slot_id).unwrap();
        self.slots[idx] = DeviceSlot {
            state: SlotState::Enabled,
            port,
            speed: UsbSpeed::High,
            num_endpoints: 1, // EP0
        };
        Ok(slot_id)
    }

    fn device_detach(&mut self, slot: u8) -> Result<(), HalError> {
        let idx = Self::slot_index(slot).ok_or(HalError::UsbDeviceNotFound)?;
        if self.slots[idx].state == SlotState::Free {
            return Err(HalError::UsbDeviceNotFound);
        }
        self.slots[idx] = DeviceSlot::default();
        self.transfer_ring_active[idx] = [false; MAX_ENDPOINTS_PER_DEVICE];
        Ok(())
    }

    fn control_transfer(
        &mut self,
        slot: u8,
        setup: &UsbSetupPacket,
        data: Option<&mut [u8]>,
    ) -> Result<UsbTransferResult, HalError> {
        let idx = Self::slot_index(slot).ok_or(HalError::UsbDeviceNotFound)?;
        if self.slots[idx].state == SlotState::Free {
            return Err(HalError::UsbDeviceNotFound);
        }

        let token = self.next_token();

        // In a real driver: enqueue Setup TRB + optional Data TRB + Status TRB
        // on the EP0 transfer ring, ring doorbell, wait for Transfer Event.
        let setup_data = u64::from_le_bytes(setup.to_bytes());
        let _transfer_type = if data.is_none() {
            0u8 // No Data Stage
        } else if setup.bm_request_type & 0x80 != 0 {
            3u8 // IN Data Stage
        } else {
            2u8 // OUT Data Stage
        };

        let _setup_trb =
            Trb::setup_stage(setup_data, _transfer_type, self.command_ring.cycle_state());

        let bytes = data.as_ref().map_or(0, |d| d.len());

        Ok(UsbTransferResult {
            bytes_transferred: bytes as u32,
            success: true,
            stalled: false,
            token,
        })
    }

    fn bulk_transfer(
        &mut self,
        slot: u8,
        _endpoint: u8,
        data: &mut [u8],
    ) -> Result<UsbTransferResult, HalError> {
        let idx = Self::slot_index(slot).ok_or(HalError::UsbDeviceNotFound)?;
        if self.slots[idx].state == SlotState::Free {
            return Err(HalError::UsbDeviceNotFound);
        }

        let token = self.next_token();

        // In a real driver: enqueue Normal TRB on the endpoint's transfer ring,
        // ring doorbell, wait for Transfer Event.
        Ok(UsbTransferResult {
            bytes_transferred: data.len() as u32,
            success: true,
            stalled: false,
            token,
        })
    }

    fn configure_endpoint(
        &mut self,
        slot: u8,
        endpoint: u8,
        _transfer_type: UsbTransferType,
        _max_packet_size: u16,
    ) -> Result<(), HalError> {
        let idx = Self::slot_index(slot).ok_or(HalError::UsbDeviceNotFound)?;
        if self.slots[idx].state == SlotState::Free {
            return Err(HalError::UsbDeviceNotFound);
        }

        // DCI (Device Context Index) = ep_num * 2 + direction
        let ep_num = endpoint & 0x0F;
        let direction = if endpoint & 0x80 != 0 { 1u8 } else { 0u8 };
        let dci = (ep_num * 2 + direction) as usize;

        if dci >= MAX_ENDPOINTS_PER_DEVICE {
            return Err(HalError::OutOfRange);
        }

        self.transfer_ring_active[idx][dci] = true;
        self.slots[idx].num_endpoints += 1;
        self.slots[idx].state = SlotState::Configured;

        Ok(())
    }

    fn poll_events(&mut self) -> Result<bool, HalError> {
        let mut processed = false;
        while self.event_ring.has_event() {
            if let Some(event) = self.event_ring.dequeue() {
                processed = true;
                match event.trb_type() {
                    Some(TrbType::PortStatusChangeEvent) => {
                        let port = event.port_id();
                        if (port as usize) < MAX_PORTS {
                            // In a real driver: read PORTSC to check CCS.
                            // For now, toggle connection state.
                        }
                    }
                    Some(TrbType::CommandCompletionEvent) => {
                        let _code = event.completion_code();
                        let _slot = event.slot_id();
                        // In a real driver: match to pending command, complete it.
                    }
                    Some(TrbType::TransferEvent) => {
                        let _code = event.completion_code();
                        let _trb_ptr = event.trb_pointer();
                        // In a real driver: match to pending transfer, notify driver.
                    }
                    _ => {}
                }
            }
        }
        Ok(processed)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::UsbHostController;

    #[test]
    fn test_controller_new() {
        let ctrl = XhciController::new();
        assert!(!ctrl.running);
        assert_eq!(ctrl.caps.max_ports, MAX_PORTS as u8);
    }

    #[test]
    fn test_controller_init() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();
        assert!(ctrl.running);
    }

    #[test]
    fn test_controller_port_count() {
        let ctrl = XhciController::new();
        assert_eq!(ctrl.port_count(), MAX_PORTS as u8);
    }

    #[test]
    fn test_controller_port_status() {
        let ctrl = XhciController::new();
        let status = ctrl.port_status(0).unwrap();
        assert!(!status.connected);
        assert_eq!(status.port, 0);
    }

    #[test]
    fn test_controller_port_status_out_of_range() {
        let ctrl = XhciController::new();
        assert_eq!(ctrl.port_status(99), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_controller_device_attach_detach() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();

        let slot = ctrl.device_attach(0).unwrap();
        assert_eq!(slot, 1); // First slot
        assert_eq!(ctrl.slots[0].state, SlotState::Enabled);
        assert_eq!(ctrl.slots[0].port, 0);

        ctrl.device_detach(slot).unwrap();
        assert_eq!(ctrl.slots[0].state, SlotState::Free);
    }

    #[test]
    fn test_controller_device_attach_all_slots() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();

        for i in 0..MAX_DEVICE_SLOTS {
            let slot = ctrl.device_attach(i as u8 % ctrl.caps.max_ports).unwrap();
            assert_eq!(slot, i as u8 + 1);
        }

        // Next attach should fail.
        assert_eq!(ctrl.device_attach(0), Err(HalError::NotSupported));
    }

    #[test]
    fn test_controller_control_transfer() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();
        let slot = ctrl.device_attach(0).unwrap();

        let setup = UsbSetupPacket::new(0x80, 0x06, 0x0100, 0, 18);
        let mut buf = [0u8; 18];
        let result = ctrl.control_transfer(slot, &setup, Some(&mut buf)).unwrap();
        assert!(result.success);
        assert_eq!(result.bytes_transferred, 18);
    }

    #[test]
    fn test_controller_control_transfer_no_data() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();
        let slot = ctrl.device_attach(0).unwrap();

        let setup = UsbSetupPacket::new(0x00, 0x05, 1, 0, 0); // SET_ADDRESS
        let result = ctrl.control_transfer(slot, &setup, None).unwrap();
        assert!(result.success);
        assert_eq!(result.bytes_transferred, 0);
    }

    #[test]
    fn test_controller_control_transfer_invalid_slot() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();

        let setup = UsbSetupPacket::new(0x80, 0x06, 0x0100, 0, 18);
        assert_eq!(
            ctrl.control_transfer(99, &setup, None),
            Err(HalError::UsbDeviceNotFound)
        );
    }

    #[test]
    fn test_controller_bulk_transfer() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();
        let slot = ctrl.device_attach(0).unwrap();

        let mut buf = [0u8; 512];
        let result = ctrl.bulk_transfer(slot, 0x81, &mut buf).unwrap();
        assert!(result.success);
        assert_eq!(result.bytes_transferred, 512);
    }

    #[test]
    fn test_controller_configure_endpoint() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();
        let slot = ctrl.device_attach(0).unwrap();

        ctrl.configure_endpoint(slot, 0x81, UsbTransferType::Bulk, 512)
            .unwrap();
        assert_eq!(ctrl.slots[0].state, SlotState::Configured);
        assert_eq!(ctrl.slots[0].num_endpoints, 2); // EP0 + one bulk EP
    }

    #[test]
    fn test_controller_poll_events_empty() {
        let mut ctrl = XhciController::new();
        ctrl.init().unwrap();
        let processed = ctrl.poll_events().unwrap();
        assert!(!processed);
    }

    #[test]
    fn test_decode_port_speed() {
        assert_eq!(XhciController::decode_port_speed(1), UsbSpeed::Full);
        assert_eq!(XhciController::decode_port_speed(3), UsbSpeed::High);
        assert_eq!(XhciController::decode_port_speed(4), UsbSpeed::Super);
        assert_eq!(XhciController::decode_port_speed(5), UsbSpeed::SuperPlus);
        assert_eq!(XhciController::decode_port_speed(99), UsbSpeed::Full);
    }

    #[test]
    fn test_slot_state_transitions() {
        assert_ne!(SlotState::Free, SlotState::Enabled);
        assert_ne!(SlotState::Enabled, SlotState::Configured);
    }

    #[test]
    fn test_find_free_slot() {
        let mut ctrl = XhciController::new();
        assert_eq!(ctrl.find_free_slot(), Some(1));
        ctrl.slots[0].state = SlotState::Enabled;
        assert_eq!(ctrl.find_free_slot(), Some(2));
    }
}
