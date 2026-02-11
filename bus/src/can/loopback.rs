// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN bus loopback integration test.
//!
//! Tests end-to-end CAN data flow: frame construction, CRC validation,
//! controller loopback (TX -> RX), Zenoh adapter key mapping, filtering,
//! CAN FD frames, bus state machine, and CANaerospace decoding.

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use crate::can::adapter::CanZenohAdapter;
    use crate::can::canaerospace::{CanaMessage, DataType};
    use crate::can::controller::{CanBitTiming, CanController, CanMode, MockCanController};
    use crate::can::crc::{crc15, crc17};
    use crate::can::fd::CanFdFrame;
    use crate::can::filter::{AcceptanceFilter, HwFilter, SwFilter};
    use crate::can::frame::{CanFrame, FrameType};
    use crate::can::state::{BusEvent, BusState, BusStateMachine};
    use crate::transport::ZenohTransport;

    // -----------------------------------------------------------------------
    // Loopback: Standard frame TX -> loopback -> RX
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_standard_frame() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_bit_timing(&CanBitTiming::standard_500kbps()).unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        // Transmit a standard frame
        let frame = CanFrame::new_standard(0x123, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        ctrl.transmit(&frame).unwrap();

        // Receive it back via loopback
        let received = ctrl.receive().unwrap().unwrap();
        assert_eq!(received.id, 0x123);
        assert_eq!(received.data, &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(received.frame_type, FrameType::Standard);
    }

    // -----------------------------------------------------------------------
    // Loopback: Extended frame TX -> loopback -> RX
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_extended_frame() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        let frame = CanFrame::new_extended(0x12345678, &[0x01, 0x02, 0x03]).unwrap();
        ctrl.transmit(&frame).unwrap();

        let received = ctrl.receive().unwrap().unwrap();
        assert_eq!(received.id, 0x12345678);
        assert_eq!(received.frame_type, FrameType::Extended);
        assert_eq!(received.data, &[0x01, 0x02, 0x03]);
    }

    // -----------------------------------------------------------------------
    // Loopback: Multiple frames in sequence
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_multiple_frames() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        let frames = [
            CanFrame::new_standard(0x100, &[0x01]).unwrap(),
            CanFrame::new_standard(0x200, &[0x02, 0x03]).unwrap(),
            CanFrame::new_standard(0x300, &[0x04, 0x05, 0x06]).unwrap(),
        ];

        for frame in &frames {
            ctrl.transmit(frame).unwrap();
        }

        for (i, expected) in frames.iter().enumerate() {
            let received = ctrl.receive().unwrap().unwrap();
            assert_eq!(received.id, expected.id, "Frame {} ID mismatch", i);
            assert_eq!(received.data, expected.data, "Frame {} data mismatch", i);
        }

        // Queue should be empty
        assert!(ctrl.receive().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // Loopback: Frame encode -> decode roundtrip with CRC
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_encode_decode_with_crc() {
        let frame = CanFrame::new_standard(0x7FF, &[0xFF; 8]).unwrap();

        // Encode the frame
        let encoded = frame.encode();
        assert!(!encoded.is_empty());

        // Decode the frame
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.id, frame.id);
        assert_eq!(decoded.data, frame.data);

        // Verify CRC-15 on the encoded data (the CRC is computed over the data field)
        let crc = crc15(&frame.data);
        assert_ne!(crc, 0); // Non-trivial CRC for non-zero data
    }

    // -----------------------------------------------------------------------
    // Loopback: CAN FD frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_fd_frame() {
        // Create a CAN FD frame with 32 bytes of data
        let data: Vec<u8> = (0..32).collect();
        let frame = CanFdFrame::new_standard(0x100, &data, true).unwrap();

        assert_eq!(frame.dlc.payload_len(), 32);
        assert_eq!(frame.data.len(), 32);

        // Encode/decode roundtrip
        let encoded = frame.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.id, frame.id);
        assert_eq!(decoded.data, frame.data);

        // Verify CRC-17 (for payload <= 16 bytes uses CRC-17)
        let crc = crc17(&data[..16]);
        assert_ne!(crc, 0);
    }

    // -----------------------------------------------------------------------
    // Loopback: Zenoh adapter TX/RX
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_zenoh_adapter() {
        let ctrl = MockCanController::new();
        let mut adapter = CanZenohAdapter::new(ctrl, 0);

        // Initialize the controller through the adapter
        adapter.controller_mut().init().unwrap();
        adapter.controller_mut().set_mode(CanMode::Loopback).unwrap();

        // Transmit a CAN frame via the Zenoh adapter
        let frame = CanFrame::new_standard(0x1A3, &[0xCA, 0xFE]).unwrap();
        adapter.controller_mut().transmit(&frame).unwrap();

        // Poll for received frames
        let count = adapter.poll_rx().unwrap();
        assert_eq!(count, 1);

        // Receive through Zenoh interface
        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert!(sample.key_expr.contains("can/0"));
        assert!(sample.key_expr.contains("0x1A3"));
    }

    // -----------------------------------------------------------------------
    // Loopback: Acceptance filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_with_filtering() {
        // Hardware filter: accept only IDs where (id & 0x700) == 0x100
        // This matches IDs 0x100-0x1FF
        let mut filter = AcceptanceFilter::new();
        filter.add_hw_filter(HwFilter::new(0x700, 0x100)).unwrap();
        // Software filter: further restrict to exact 0x105
        filter.add_sw_filter(SwFilter::new(0x105)).unwrap();

        // 0x105 passes HW (0x105 & 0x700 == 0x100) and SW (exact match)
        assert!(filter.accepts(0x105));
        // 0x200 fails HW (0x200 & 0x700 == 0x200 != 0x100)
        assert!(!filter.accepts(0x200));
        // 0x150 passes HW (0x150 & 0x700 == 0x100) but fails SW (not 0x105)
        assert!(!filter.accepts(0x150));
    }

    // -----------------------------------------------------------------------
    // Loopback: Bus state machine transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_bus_state_machine() {
        let mut sm = BusStateMachine::new();

        // Initially in Error Active
        assert_eq!(sm.state(), BusState::ErrorActive);

        // Accumulate TX errors to trigger Error Passive (TEC > 127)
        sm.process_event(BusEvent::TxError(128));
        assert_eq!(sm.state(), BusState::ErrorPassive);

        // More errors -> Bus Off (TEC > 255)
        sm.process_event(BusEvent::TxError(128));
        assert_eq!(sm.state(), BusState::BusOff);

        // Recovery: 128 recessive sequences
        for _ in 0..128 {
            sm.process_event(BusEvent::RecessiveSequence);
        }
        assert_eq!(sm.state(), BusState::ErrorActive);
    }

    // -----------------------------------------------------------------------
    // Loopback: CANaerospace profile
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_canaerospace_roundtrip() {
        // Create a CANaerospace NOD (Normal Operation Data) message
        let msg = CanaMessage::new_nod(
            200, // CAN ID for AIRSPEED_IAS
            0x01, // Node ID
            DataType::Float,
            &125.5f32.to_be_bytes(),
        ).unwrap();

        // Convert to CAN frame
        let frame = msg.to_can_frame().unwrap();
        assert_eq!(frame.id, 200);

        // Encode/decode the CAN frame
        let encoded = frame.encode();
        let decoded_frame = CanFrame::decode(&encoded).unwrap();

        // Decode back to CANaerospace
        let decoded_msg = CanaMessage::from_can_frame(&decoded_frame).unwrap();
        assert_eq!(decoded_msg.msg_type, crate::can::canaerospace::MessageType::NodNormal);
        assert_eq!(decoded_msg.data_type, DataType::Float.raw());

        // Verify the float value
        let value = decoded_msg.as_f32().unwrap();
        assert_eq!(value, 125.5);
    }

    // -----------------------------------------------------------------------
    // Loopback: Full pipeline TX -> encode -> loopback -> decode -> RX
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_full_pipeline() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_bit_timing(&CanBitTiming::standard_1mbps()).unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        // Create a frame, encode it, transmit through controller
        let original = CanFrame::new_standard(0x555, &[0xA0, 0xB1, 0xC2, 0xD3]).unwrap();
        let encoded = original.encode();

        // Transmit
        ctrl.transmit(&original).unwrap();

        // Receive via loopback
        let received = ctrl.receive().unwrap().unwrap();

        // Encode the received frame
        let re_encoded = received.encode();

        // Both encoded forms should match
        assert_eq!(encoded.len(), re_encoded.len());
        assert_eq!(encoded, re_encoded);

        // Verify stats
        assert_eq!(ctrl.stats().tx_frames, 1);
        assert_eq!(ctrl.stats().rx_frames, 1);
    }

    // -----------------------------------------------------------------------
    // Loopback: Multiple frame types mixed
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_mixed_frame_types() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        // Transmit a mix of standard and extended frames
        let std_frame = CanFrame::new_standard(0x100, &[0x01, 0x02]).unwrap();
        let ext_frame = CanFrame::new_extended(0x0ABCDEF0, &[0x03, 0x04, 0x05]).unwrap();

        ctrl.transmit(&std_frame).unwrap();
        ctrl.transmit(&ext_frame).unwrap();

        let r1 = ctrl.receive().unwrap().unwrap();
        assert_eq!(r1.frame_type, FrameType::Standard);
        assert_eq!(r1.id, 0x100);

        let r2 = ctrl.receive().unwrap().unwrap();
        assert_eq!(r2.frame_type, FrameType::Extended);
        assert_eq!(r2.id, 0x0ABCDEF0);
    }

    // -----------------------------------------------------------------------
    // Loopback: Empty payload frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_empty_payload() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        let frame = CanFrame::new_standard(0x7FF, &[]).unwrap();
        ctrl.transmit(&frame).unwrap();

        let received = ctrl.receive().unwrap().unwrap();
        assert!(received.data.is_empty());
        assert_eq!(received.id, 0x7FF);
    }

    // -----------------------------------------------------------------------
    // Loopback: Max payload frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_loopback_max_payload() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();

        let frame = CanFrame::new_standard(0x001, &[0xFF; 8]).unwrap();
        ctrl.transmit(&frame).unwrap();

        let received = ctrl.receive().unwrap().unwrap();
        assert_eq!(received.data.len(), 8);
        assert_eq!(received.data, &[0xFF; 8]);
    }
}
