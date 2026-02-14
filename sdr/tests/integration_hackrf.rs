// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test (Task 5.17): Configure HackRF -> start RX -> receive IQ
//! samples.
//!
//! Exercises the HackRF driver state machine and IQ parsing pipeline using
//! mock USB vendor control transfer encodings (no real hardware required).

#![cfg(feature = "hackrf")]

extern crate alloc;

use smallaios_sdr::hackrf::{
    detect_hackrf, encode_amp_enable, encode_board_id_read, encode_board_rev_read,
    encode_init_sweep, encode_reset, encode_sample_rate, encode_set_freq, encode_set_lna_gain,
    encode_set_transceiver_mode, encode_set_txvga_gain, encode_set_vga_gain,
    encode_version_string_read, parse_iq_8bit, parse_sweep_header, HackRfDevice,
    CONCURRENT_TRANSFERS, HACKRF_ONE_BOARD_ID, HACKRF_PID, HACKRF_VID, LNA_GAIN_MAX, RX_ENDPOINT,
    SWEEP_HEADER_SIZE, TRANSFER_BUFFER_SIZE, TXVGA_GAIN_MAX, TX_ENDPOINT, VGA_GAIN_MAX,
};
use smallaios_sdr::{IqFormat, SdrConfig, SdrError, TransceiverMode};

// ---------------------------------------------------------------------------
// 1. Device detection
// ---------------------------------------------------------------------------

#[test]
fn hackrf_device_detection_valid() {
    assert!(detect_hackrf(HACKRF_VID, HACKRF_PID));
}

#[test]
fn hackrf_device_detection_wrong_ids() {
    assert!(!detect_hackrf(0x0456, 0xB673)); // PlutoSDR, not HackRF
    assert!(!detect_hackrf(0x0000, 0x0000));
    assert!(!detect_hackrf(HACKRF_VID, 0x0000));
    assert!(!detect_hackrf(0x0000, HACKRF_PID));
}

// ---------------------------------------------------------------------------
// 2. Full configuration sequence (freq, gain, sample rate)
// ---------------------------------------------------------------------------

#[test]
fn hackrf_full_configuration_sequence() {
    let mut dev = HackRfDevice::new();

    // Step 1: Set board ID (simulates board ID read response).
    dev.set_board_id(HACKRF_ONE_BOARD_ID);
    assert!(dev.verify_board_id().is_ok());
    assert_eq!(dev.board_id(), HACKRF_ONE_BOARD_ID);

    // Step 2: Configure frequency.
    let ((bm, _br, _, _), freq_data) = encode_set_freq(915_000_000);
    assert_eq!(bm, 0x40); // Vendor OUT
    let mhz = u32::from_le_bytes([freq_data[0], freq_data[1], freq_data[2], freq_data[3]]);
    let hz = u32::from_le_bytes([freq_data[4], freq_data[5], freq_data[6], freq_data[7]]);
    assert_eq!(mhz, 915);
    assert_eq!(hz, 0);

    // Step 3: Configure sample rate.
    let ((_, _, _, _), rate_data) = encode_sample_rate(20_000_000);
    let rate = u32::from_le_bytes([rate_data[0], rate_data[1], rate_data[2], rate_data[3]]);
    let div = u32::from_le_bytes([rate_data[4], rate_data[5], rate_data[6], rate_data[7]]);
    assert_eq!(rate, 20_000_000);
    assert_eq!(div, 1);

    // Step 4: Configure LNA gain.
    let (_, _, wv, _) = encode_set_lna_gain(24).unwrap();
    assert_eq!(wv, 24); // 24 is a multiple of 8

    // Step 5: Configure VGA gain.
    let (_, _, wv, _) = encode_set_vga_gain(20).unwrap();
    assert_eq!(wv, 20); // 20 is a multiple of 2

    // Step 6: Enable amplifier.
    let (_, _, wv, _) = encode_amp_enable(true);
    assert_eq!(wv, 1);

    // Step 7: Apply configuration to device state.
    dev.set_config(SdrConfig {
        frequency_hz: 915_000_000,
        sample_rate_hz: 20_000_000,
        bandwidth_hz: 0,
    });
    assert!(dev.is_configured());
    assert_eq!(dev.config().frequency_hz, 915_000_000);
    assert_eq!(dev.config().sample_rate_hz, 20_000_000);
}

// ---------------------------------------------------------------------------
// 3. RX streaming start and IQ sample reception
// ---------------------------------------------------------------------------

#[test]
fn hackrf_rx_streaming_lifecycle() {
    let mut dev = HackRfDevice::new();
    dev.set_board_id(HACKRF_ONE_BOARD_ID);
    dev.set_config(SdrConfig {
        frequency_hz: 433_000_000,
        sample_rate_hz: 10_000_000,
        bandwidth_hz: 0,
    });

    // Start RX mode.
    dev.set_mode(TransceiverMode::Rx).unwrap();
    assert_eq!(dev.mode(), TransceiverMode::Rx);

    // Verify transceiver mode encoding.
    let (bm, _br, wv, _) = encode_set_transceiver_mode(TransceiverMode::Rx);
    assert_eq!(bm, 0x40);
    assert_eq!(wv, 1); // RECEIVE = 1

    // Start streaming.
    dev.start_streaming().unwrap();
    assert!(dev.is_streaming());
}

#[test]
fn hackrf_rx_streaming_requires_configuration() {
    let mut dev = HackRfDevice::new();
    assert!(!dev.is_configured());

    // Should fail without configuration.
    assert_eq!(
        dev.set_mode(TransceiverMode::Rx),
        Err(SdrError::NotConfigured)
    );
}

#[test]
fn hackrf_iq_sample_reception_and_parsing() {
    // Simulate receiving IQ data from a bulk transfer.
    // HackRF uses 8-bit signed interleaved I/Q format.
    let raw_samples: alloc::vec::Vec<u8> = (0..256)
        .map(|i| {
            // Generate a sine wave pattern
            let phase = (i as f32) * core::f32::consts::PI * 2.0 / 128.0;
            let value = (phase.sin() * 100.0) as i8;
            value as u8
        })
        .collect();

    let mut output = [(0.0f32, 0.0f32); 128];
    let count = parse_iq_8bit(&raw_samples, &mut output);
    assert_eq!(count, 128);

    // All values should be in [-1.0, 1.0] range.
    for &(i, q) in &output[..count] {
        assert!((-1.0..=1.0).contains(&i), "I value {} out of range", i);
        assert!((-1.0..=1.0).contains(&q), "Q value {} out of range", q);
    }

    // Verify IQ format metadata.
    assert_eq!(IqFormat::Signed8.bytes_per_sample(), 2);
}

#[test]
fn hackrf_iq_parsing_known_values() {
    // Specific known values: I=0, Q=0 -> (0.0, 0.0)
    let data = [0u8, 0, 127, 128]; // (0,0), (127/128, -128/128)
    let mut output = [(0.0f32, 0.0f32); 2];
    let count = parse_iq_8bit(&data, &mut output);
    assert_eq!(count, 2);
    assert!((output[0].0 - 0.0).abs() < 0.01);
    assert!((output[0].1 - 0.0).abs() < 0.01);
    assert!((output[1].0 - (127.0 / 128.0)).abs() < 0.01);
    assert!((output[1].1 - (-1.0)).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// 4. Half-duplex enforcement
// ---------------------------------------------------------------------------

#[test]
fn hackrf_half_duplex_prevents_simultaneous_tx_rx() {
    let mut dev = HackRfDevice::new();
    dev.set_config(SdrConfig {
        frequency_hz: 915_000_000,
        sample_rate_hz: 20_000_000,
        bandwidth_hz: 0,
    });

    // Start RX.
    dev.set_mode(TransceiverMode::Rx).unwrap();

    // Try to switch to TX without stopping first.
    assert_eq!(
        dev.set_mode(TransceiverMode::Tx),
        Err(SdrError::HalfDuplexViolation)
    );

    // Stop RX first.
    dev.stop_streaming();
    assert_eq!(dev.mode(), TransceiverMode::Off);

    // Now TX should succeed.
    dev.set_mode(TransceiverMode::Tx).unwrap();
    assert_eq!(dev.mode(), TransceiverMode::Tx);
}

// ---------------------------------------------------------------------------
// 5. Gain range validation
// ---------------------------------------------------------------------------

#[test]
fn hackrf_gain_validation_lna() {
    assert!(encode_set_lna_gain(0).is_ok());
    assert!(encode_set_lna_gain(LNA_GAIN_MAX).is_ok());
    assert_eq!(
        encode_set_lna_gain(LNA_GAIN_MAX + 1),
        Err(SdrError::InvalidParameter)
    );

    // Step rounding: 10 rounds down to 8.
    let (_, _, wv, _) = encode_set_lna_gain(10).unwrap();
    assert_eq!(wv, 8);
}

#[test]
fn hackrf_gain_validation_vga() {
    assert!(encode_set_vga_gain(0).is_ok());
    assert!(encode_set_vga_gain(VGA_GAIN_MAX).is_ok());
    assert_eq!(
        encode_set_vga_gain(VGA_GAIN_MAX + 1),
        Err(SdrError::InvalidParameter)
    );

    // Step rounding: 31 rounds down to 30.
    let (_, _, wv, _) = encode_set_vga_gain(31).unwrap();
    assert_eq!(wv, 30);
}

#[test]
fn hackrf_gain_validation_txvga() {
    assert!(encode_set_txvga_gain(0).is_ok());
    assert!(encode_set_txvga_gain(TXVGA_GAIN_MAX).is_ok());
    assert_eq!(
        encode_set_txvga_gain(TXVGA_GAIN_MAX + 1),
        Err(SdrError::InvalidParameter)
    );
}

// ---------------------------------------------------------------------------
// 6. Sweep mode
// ---------------------------------------------------------------------------

#[test]
fn hackrf_sweep_mode_encoding_and_header_parsing() {
    // Encode sweep parameters.
    let ((bm, _br, _, _), sweep_data) = encode_init_sweep(100, 3000, 1_000_000);
    assert_eq!(bm, 0x40);
    let start = u16::from_le_bytes([sweep_data[0], sweep_data[1]]);
    let stop = u16::from_le_bytes([sweep_data[2], sweep_data[3]]);
    assert_eq!(start, 100);
    assert_eq!(stop, 3000);

    // Simulate receiving a sweep header.
    let mut header_data = [0u8; 12];
    let freq: u64 = 915_000_000;
    let samples: u16 = 8192;
    header_data[0..8].copy_from_slice(&freq.to_le_bytes());
    header_data[8..10].copy_from_slice(&samples.to_le_bytes());

    let (f, s) = parse_sweep_header(&header_data).unwrap();
    assert_eq!(f, 915_000_000);
    assert_eq!(s, 8192);
}

#[test]
fn hackrf_sweep_header_too_short() {
    let data = [0u8; 5];
    assert_eq!(parse_sweep_header(&data), Err(SdrError::ProtocolError));
}

#[test]
fn hackrf_sweep_mode_state_tracking() {
    let mut dev = HackRfDevice::new();
    dev.set_config(SdrConfig {
        frequency_hz: 0,
        sample_rate_hz: 20_000_000,
        bandwidth_hz: 0,
    });

    assert!(!dev.is_sweep_active());
    dev.set_sweep_active(true);
    assert!(dev.is_sweep_active());

    dev.stop_streaming();
    assert!(!dev.is_sweep_active(), "sweep should be cleared on stop");
}

// ---------------------------------------------------------------------------
// 7. Device cleanup and reset
// ---------------------------------------------------------------------------

#[test]
fn hackrf_device_reset_clears_all_state() {
    let mut dev = HackRfDevice::new();
    dev.set_board_id(HACKRF_ONE_BOARD_ID);
    dev.set_config(SdrConfig {
        frequency_hz: 915_000_000,
        sample_rate_hz: 20_000_000,
        bandwidth_hz: 0,
    });
    dev.set_mode(TransceiverMode::Rx).unwrap();
    dev.start_streaming().unwrap();

    // Verify encoding of reset command.
    let (bm, _br, _, _) = encode_reset();
    assert_eq!(bm, 0x40);

    // Perform reset.
    dev.reset();
    assert_eq!(dev.mode(), TransceiverMode::Off);
    assert!(!dev.is_configured());
    assert!(!dev.is_streaming());
    assert_eq!(dev.config().frequency_hz, 0);
    assert_eq!(dev.config().sample_rate_hz, 0);
}

// ---------------------------------------------------------------------------
// 8. Board info queries
// ---------------------------------------------------------------------------

#[test]
fn hackrf_board_info_query_encoding() {
    // Board ID read.
    let (bm, br, _, _) = encode_board_id_read();
    assert_eq!(bm, 0xC0); // Vendor IN
    assert_eq!(br, 14); // BOARD_ID_READ

    // Version string read.
    let (bm, br, _, _) = encode_version_string_read();
    assert_eq!(bm, 0xC0);
    assert_eq!(br, 15); // VERSION_STRING_READ

    // Board rev read.
    let (bm, br, _, _) = encode_board_rev_read();
    assert_eq!(bm, 0xC0);
    assert_eq!(br, 31); // BOARD_REV_READ
}

#[test]
fn hackrf_board_id_verification() {
    let mut dev = HackRfDevice::new();

    // Without setting board ID, it defaults to 0.
    assert_eq!(dev.board_id(), 0);
    assert_eq!(dev.verify_board_id(), Err(SdrError::DeviceNotFound));

    // Set correct board ID.
    dev.set_board_id(HACKRF_ONE_BOARD_ID);
    assert!(dev.verify_board_id().is_ok());

    // Wrong board ID.
    dev.set_board_id(0xFF);
    assert_eq!(dev.verify_board_id(), Err(SdrError::DeviceNotFound));
}

// ---------------------------------------------------------------------------
// 9. Frequency encoding edge cases
// ---------------------------------------------------------------------------

#[test]
fn hackrf_frequency_encoding_fractional_mhz() {
    let (_, data) = encode_set_freq(2_437_500_000);
    let mhz = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let hz = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    assert_eq!(mhz, 2437);
    assert_eq!(hz, 500_000);
}

#[test]
fn hackrf_frequency_encoding_minimum() {
    let (_, data) = encode_set_freq(1_000_000); // 1 MHz
    let mhz = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let hz = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    assert_eq!(mhz, 1);
    assert_eq!(hz, 0);
}

// ---------------------------------------------------------------------------
// 10. Constants verification
// ---------------------------------------------------------------------------

#[test]
fn hackrf_endpoint_addresses() {
    assert_eq!(RX_ENDPOINT, 0x81);
    assert_eq!(TX_ENDPOINT, 0x02);
    assert_eq!(CONCURRENT_TRANSFERS, 4);
    assert_eq!(TRANSFER_BUFFER_SIZE, 262_144);
    assert_eq!(SWEEP_HEADER_SIZE, 10);
}
