// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test (Task 2.17): xHCI init -> enumerate USB device -> bulk transfer.
//!
//! Exercises the xHCI controller through its `UsbHostController` trait
//! implementation, using the built-in mock MMIO (the controller state-machine
//! already operates without real hardware).

#![cfg(feature = "xhci")]

extern crate alloc;

use smallaios_kernel::hal::{UsbHostController, UsbSpeed, UsbTransferType};
use smallaios_usb::descriptor::{
    DeviceDescriptor, CONFIG_DESCRIPTOR_MIN_LEN, DEVICE_DESCRIPTOR_LEN,
};
use smallaios_usb::enumeration::{
    make_get_config_descriptor, make_get_device_descriptor, make_set_address,
    make_set_configuration, AddressAllocator,
};
use smallaios_usb::registry::{DeviceMatch, DeviceRegistry};
use smallaios_usb::transfer::{EndpointState, EndpointTracker};
use smallaios_usb::xhci::controller::{SlotState, XhciController, MAX_DEVICE_SLOTS, MAX_PORTS};
use smallaios_usb::xhci::registers::XhciCapabilities;
use smallaios_usb::xhci::rings::{EventRing, ProducerRing, DEFAULT_RING_SIZE};
use smallaios_usb::xhci::trb::{CompletionCode, Trb, TrbType};

// ---------------------------------------------------------------------------
// 1. Controller initialization sequence
// ---------------------------------------------------------------------------

#[test]
fn xhci_init_creates_running_controller() {
    let mut ctrl = XhciController::new();
    assert!(
        !ctrl.running,
        "controller should not be running before init"
    );
    ctrl.init().expect("init should succeed");
    assert!(ctrl.running, "controller should be running after init");
}

#[test]
fn xhci_init_default_capabilities() {
    let ctrl = XhciController::new();
    assert_eq!(ctrl.caps.hci_version, 0x0100);
    assert_eq!(ctrl.caps.max_slots, MAX_DEVICE_SLOTS as u8);
    assert_eq!(ctrl.caps.max_ports, MAX_PORTS as u8);
    assert!(!ctrl.caps.context_size_64);
}

#[test]
fn xhci_capability_parsing_from_raw_registers() {
    // Simulate reading capability registers from mock MMIO
    let caplength_hciversion = 0x0110_0020; // version 1.1, caplength 0x20
    let hcsparams1 = (4u32 << 24) | (2 << 8) | 32; // 4 ports, 2 interrupters, 32 slots
    let hcsparams2 = 0;
    let hccparams1 = 0x04; // CSZ bit set (64-byte contexts)
    let dboff = 0x2000;
    let rtsoff = 0x3000;

    let caps = XhciCapabilities::parse(
        caplength_hciversion,
        hcsparams1,
        hcsparams2,
        hccparams1,
        dboff,
        rtsoff,
    );

    assert_eq!(caps.op_reg_offset, 0x20);
    assert_eq!(caps.hci_version, 0x0110);
    assert_eq!(caps.max_slots, 32);
    assert_eq!(caps.max_intrs, 2);
    assert_eq!(caps.max_ports, 4);
    assert!(caps.context_size_64);
    assert_eq!(caps.doorbell_offset, 0x2000);
    assert_eq!(caps.runtime_offset, 0x3000);
}

// ---------------------------------------------------------------------------
// 2. Mock port status change and device enumeration
// ---------------------------------------------------------------------------

#[test]
fn xhci_port_status_reports_disconnected_initially() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    for port in 0..ctrl.port_count() {
        let status = ctrl.port_status(port).unwrap();
        assert!(!status.connected);
        assert_eq!(status.port, port);
    }
}

#[test]
fn xhci_port_reset_succeeds_for_valid_port() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    assert!(ctrl.port_reset(0).is_ok());
}

#[test]
fn xhci_port_reset_fails_for_out_of_range_port() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    assert!(ctrl.port_reset(99).is_err());
}

#[test]
fn xhci_device_attach_allocates_slot() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    // Simulate a connection on port 0.
    ctrl.port_connected[0] = true;

    // Reset port, then attach device.
    ctrl.port_reset(0).unwrap();
    let slot = ctrl.device_attach(0).unwrap();

    assert_eq!(slot, 1, "first slot should be 1 (1-based)");
    assert_eq!(ctrl.slots[0].state, SlotState::Enabled);
    assert_eq!(ctrl.slots[0].port, 0);
    assert_eq!(ctrl.slots[0].speed, UsbSpeed::High);
}

#[test]
fn xhci_full_enumeration_sequence() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    ctrl.port_connected[0] = true;

    // Step 1: Port reset.
    ctrl.port_reset(0).unwrap();

    // Step 2: Attach device (allocates slot, assigns address).
    let slot = ctrl.device_attach(0).unwrap();
    assert_eq!(ctrl.slots[(slot - 1) as usize].state, SlotState::Enabled);

    // Step 3: SET_ADDRESS (control transfer, no data stage).
    let setup_addr = make_set_address(slot);
    let result = ctrl.control_transfer(slot, &setup_addr, None).unwrap();
    assert!(result.success);

    // Step 4: GET_DESCRIPTOR(Device).
    let setup_dev = make_get_device_descriptor();
    let mut dev_buf = [0u8; DEVICE_DESCRIPTOR_LEN];
    let result = ctrl
        .control_transfer(slot, &setup_dev, Some(&mut dev_buf))
        .unwrap();
    assert!(result.success);
    assert_eq!(result.bytes_transferred as usize, DEVICE_DESCRIPTOR_LEN);

    // Step 5: GET_DESCRIPTOR(Configuration) - header.
    let setup_cfg = make_get_config_descriptor(CONFIG_DESCRIPTOR_MIN_LEN as u16);
    let mut cfg_buf = [0u8; CONFIG_DESCRIPTOR_MIN_LEN];
    let result = ctrl
        .control_transfer(slot, &setup_cfg, Some(&mut cfg_buf))
        .unwrap();
    assert!(result.success);
    assert_eq!(result.bytes_transferred as usize, CONFIG_DESCRIPTOR_MIN_LEN);

    // Step 6: SET_CONFIGURATION.
    let setup_setcfg = make_set_configuration(1);
    let result = ctrl.control_transfer(slot, &setup_setcfg, None).unwrap();
    assert!(result.success);
}

#[test]
fn xhci_multiple_device_enumeration() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    let mut slots = alloc::vec::Vec::new();
    for port in 0..4u8 {
        ctrl.port_connected[port as usize] = true;
        ctrl.port_reset(port).unwrap();
        let slot = ctrl.device_attach(port).unwrap();
        slots.push(slot);
    }

    // All slots should be unique and sequential.
    assert_eq!(slots, alloc::vec![1, 2, 3, 4]);

    // Each slot should be in Enabled state.
    for &slot in &slots {
        let idx = (slot - 1) as usize;
        assert_eq!(ctrl.slots[idx].state, SlotState::Enabled);
    }
}

// ---------------------------------------------------------------------------
// 3. Bulk transfer enqueue and completion
// ---------------------------------------------------------------------------

#[test]
fn xhci_bulk_transfer_succeeds_on_enabled_slot() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    ctrl.port_connected[0] = true;
    ctrl.port_reset(0).unwrap();
    let slot = ctrl.device_attach(0).unwrap();

    let mut buf = [0u8; 512];
    let result = ctrl.bulk_transfer(slot, 0x81, &mut buf).unwrap();
    assert!(result.success);
    assert_eq!(result.bytes_transferred, 512);
    assert!(!result.stalled);
}

#[test]
fn xhci_bulk_transfer_fails_on_invalid_slot() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    let mut buf = [0u8; 64];
    assert!(ctrl.bulk_transfer(99, 0x81, &mut buf).is_err());
}

#[test]
fn xhci_bulk_transfer_fails_on_free_slot() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    // Slot 1 is free (no device attached).
    let mut buf = [0u8; 64];
    assert!(ctrl.bulk_transfer(1, 0x81, &mut buf).is_err());
}

#[test]
fn xhci_configure_endpoint_transitions_to_configured() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    let slot = ctrl.device_attach(0).unwrap();

    // Configure a bulk IN endpoint.
    ctrl.configure_endpoint(slot, 0x81, UsbTransferType::Bulk, 512)
        .unwrap();
    assert_eq!(ctrl.slots[0].state, SlotState::Configured);
    assert_eq!(ctrl.slots[0].num_endpoints, 2); // EP0 + bulk IN

    // Configure a bulk OUT endpoint.
    ctrl.configure_endpoint(slot, 0x02, UsbTransferType::Bulk, 512)
        .unwrap();
    assert_eq!(ctrl.slots[0].num_endpoints, 3);
}

#[test]
fn xhci_endpoint_tracker_records_bulk_transfer_lifecycle() {
    let mut tracker = EndpointTracker::new(0x81);
    assert_eq!(tracker.state, EndpointState::Idle);

    // Submit transfer.
    tracker.on_submit();
    assert_eq!(tracker.state, EndpointState::Active);

    // Complete transfer.
    tracker.on_complete();
    assert_eq!(tracker.state, EndpointState::Idle);
    assert_eq!(tracker.completed_count, 1);
    assert_eq!(tracker.error_count, 0);
}

// ---------------------------------------------------------------------------
// 4. Command ring and event ring integration
// ---------------------------------------------------------------------------

#[test]
fn xhci_command_ring_enqueue_and_wrap() {
    let mut ring = ProducerRing::new();
    assert_eq!(ring.capacity(), DEFAULT_RING_SIZE - 1);

    // Enqueue several TRBs.
    for i in 0..10 {
        let trb = Trb::normal(i * 0x1000, 64, false, false);
        let idx = ring.enqueue(trb).unwrap();
        assert_eq!(idx, i as usize);
    }
    assert_eq!(ring.enqueue_count(), 10);
    assert!(ring.cycle_state()); // Still in first cycle.
}

#[test]
fn xhci_event_ring_initially_empty() {
    let event_ring = EventRing::new();

    // A fresh event ring should have no pending events (cycle bit mismatch).
    assert!(!event_ring.has_event());
    assert_eq!(event_ring.dequeue_index(), 0);
    assert!(event_ring.cycle_state());
    assert_eq!(event_ring.dequeue_count(), 0);
}

#[test]
fn xhci_trb_transfer_event_field_extraction() {
    // Verify TRB field extraction for transfer events without requiring push_event
    // (which is #[cfg(test)] only within the crate).
    let mut trb = Trb::zeroed();
    trb.parameter = 0x1000; // TRB pointer
    trb.status = 1u32 << 24; // Success
    trb.control = Trb::make_control(TrbType::TransferEvent, true, 1 << 24); // slot 1

    assert_eq!(trb.trb_type(), Some(TrbType::TransferEvent));
    assert_eq!(trb.completion_code(), CompletionCode::Success);
    assert_eq!(trb.slot_id(), 1);
    assert_eq!(trb.trb_pointer(), 0x1000);
    assert!(trb.cycle_bit());
}

// ---------------------------------------------------------------------------
// 5. Device detach and cleanup
// ---------------------------------------------------------------------------

#[test]
fn xhci_device_detach_frees_slot() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    let slot = ctrl.device_attach(0).unwrap();
    assert_eq!(ctrl.slots[0].state, SlotState::Enabled);

    ctrl.device_detach(slot).unwrap();
    assert_eq!(ctrl.slots[0].state, SlotState::Free);

    // Bulk transfer should now fail on the freed slot.
    let mut buf = [0u8; 64];
    assert!(ctrl.bulk_transfer(slot, 0x81, &mut buf).is_err());
}

#[test]
fn xhci_slot_reuse_after_detach() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    let slot1 = ctrl.device_attach(0).unwrap();
    ctrl.device_detach(slot1).unwrap();

    // Attaching a new device should reuse slot 1.
    let slot2 = ctrl.device_attach(1).unwrap();
    assert_eq!(slot2, 1, "freed slot should be reused");
}

// ---------------------------------------------------------------------------
// 6. Registry integration with xHCI enumeration
// ---------------------------------------------------------------------------

#[test]
fn xhci_enumerate_then_register_in_registry() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    ctrl.port_connected[0] = true;

    ctrl.port_reset(0).unwrap();
    let slot = ctrl.device_attach(0).unwrap();

    // Build a mock DeviceDescriptor matching HackRF VID/PID.
    let desc = DeviceDescriptor {
        bcd_usb: 0x0200,
        device_class: 0xFF,
        device_sub_class: 0,
        device_protocol: 0,
        max_packet_size_0: 64,
        vendor_id: 0x1D50,
        product_id: 0x6089,
        bcd_device: 0x0100,
        manufacturer_index: 1,
        product_index: 2,
        serial_number_index: 3,
        num_configurations: 1,
    };

    let mut registry = DeviceRegistry::new();
    let _idx = registry.register(slot, 0, &desc).unwrap();
    assert_eq!(registry.active_count(), 1);

    // Find by VID/PID.
    let found = registry.find_by_vid_pid(0x1D50, 0x6089).unwrap();
    assert_eq!(found.slot, slot);
    assert_eq!(found.port, 0);

    // Match criteria.
    let criteria = DeviceMatch::VidPid {
        vendor_id: 0x1D50,
        product_id: 0x6089,
    };
    assert!(DeviceRegistry::matches(found, &criteria));
}

// ---------------------------------------------------------------------------
// 7. Address allocator integration
// ---------------------------------------------------------------------------

#[test]
fn xhci_address_allocator_tracks_enumerated_devices() {
    let mut alloc = AddressAllocator::new();
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();

    // Enumerate 3 devices.
    for port in 0..3u8 {
        ctrl.port_connected[port as usize] = true;
        ctrl.port_reset(port).unwrap();
        let _slot = ctrl.device_attach(port).unwrap();
        let addr = alloc.allocate().unwrap();
        assert_eq!(addr, port + 1);
        assert!(alloc.is_used(addr));
    }

    // Release one.
    alloc.release(2);
    assert!(!alloc.is_used(2));

    // Re-allocate should return the freed address.
    let addr = alloc.allocate().unwrap();
    assert_eq!(addr, 2);
}

// ---------------------------------------------------------------------------
// 8. Poll events (empty)
// ---------------------------------------------------------------------------

#[test]
fn xhci_poll_events_returns_false_when_empty() {
    let mut ctrl = XhciController::new();
    ctrl.init().unwrap();
    let processed = ctrl.poll_events().unwrap();
    assert!(!processed);
}

// ---------------------------------------------------------------------------
// 9. Port speed decoding
// ---------------------------------------------------------------------------

#[test]
fn xhci_port_speed_decode_all_variants() {
    assert_eq!(XhciController::decode_port_speed(1), UsbSpeed::Full);
    assert_eq!(XhciController::decode_port_speed(3), UsbSpeed::High);
    assert_eq!(XhciController::decode_port_speed(4), UsbSpeed::Super);
    assert_eq!(XhciController::decode_port_speed(5), UsbSpeed::SuperPlus);
    assert_eq!(
        XhciController::decode_port_speed(0),
        UsbSpeed::Full,
        "unknown speed defaults to Full"
    );
}
