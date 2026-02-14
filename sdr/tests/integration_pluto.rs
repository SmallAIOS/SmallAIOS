// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test (Task 6.18): Configure PlutoSDR -> open buffer -> stream
//! IQ samples.
//!
//! Exercises the ADALM-PLUTO SDR driver, including IIOD protocol encoding/
//! decoding, AD9363 parameter validation, device state machine, and 16-bit
//! IQ sample parsing. Uses mock USB bulk endpoints (no real hardware).

#![cfg(feature = "pluto")]

extern crate alloc;

use smallaios_sdr::pluto::{
    attribute, channel, detect_pluto, device, encode_close, encode_open, encode_print, encode_read,
    encode_readbuf, encode_timeout, encode_write, encode_writebuf, find_iio_interface,
    format_bandwidth, format_frequency, format_gain, format_sample_rate, map_iiod_error,
    parse_iq_16bit, parse_response, validate_frequency, validate_gain, PlutoDevice,
    AD9363_FREQ_MAX_HZ, AD9363_FREQ_MIN_HZ, AD9363_GAIN_MAX_DB, AD9363_GAIN_MIN_DB,
    DEFAULT_TIMEOUT_MS, IQ_CHANNEL_MASK, PLUTO_PID, PLUTO_VID, VENDOR_INTERFACE_CLASS,
};
use smallaios_sdr::{IqFormat, SdrError};

// ---------------------------------------------------------------------------
// 1. Device detection
// ---------------------------------------------------------------------------

#[test]
fn pluto_device_detection_valid() {
    assert!(detect_pluto(PLUTO_VID, PLUTO_PID));
}

#[test]
fn pluto_device_detection_wrong_ids() {
    assert!(!detect_pluto(0x1D50, 0x6089)); // HackRF, not Pluto
    assert!(!detect_pluto(PLUTO_VID, 0x0000));
    assert!(!detect_pluto(0x0000, PLUTO_PID));
}

#[test]
fn pluto_iio_interface_discovery() {
    // Typical Pluto composite device: CDC(0x02), MSC(0x08), DFU(0xFE), IIO(0xFF)
    let classes = [0x02, 0x08, 0xFE, VENDOR_INTERFACE_CLASS];
    let iio_idx = find_iio_interface(&classes);
    assert_eq!(iio_idx, Some(3));
}

#[test]
fn pluto_iio_interface_not_present() {
    let classes = [0x02, 0x08, 0xFE];
    assert_eq!(find_iio_interface(&classes), None);
}

// ---------------------------------------------------------------------------
// 2. IIOD PRINT command for context discovery
// ---------------------------------------------------------------------------

#[test]
fn pluto_iiod_print_command() {
    let mut buf = [0u8; 64];
    let len = encode_print(&mut buf).unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "PRINT\n");
}

#[test]
fn pluto_iiod_print_context_discovery_flow() {
    let mut dev = PlutoDevice::new(3);
    assert!(!dev.context_discovered());

    // Encode the PRINT command.
    let mut buf = [0u8; 64];
    let len = encode_print(&mut buf).unwrap();
    assert_eq!(&buf[..len], b"PRINT\n");

    // Simulate a successful PRINT response (positive byte count means XML data follows).
    let response_data = b"512\n<context>...</context>";
    let (code, data_offset) = parse_response(response_data).unwrap();
    assert_eq!(code, 512);
    assert!(data_offset > 0);

    // Mark context as discovered.
    dev.set_context_discovered();
    assert!(dev.context_discovered());
}

// ---------------------------------------------------------------------------
// 3. Frequency/gain/bandwidth attribute configuration
// ---------------------------------------------------------------------------

#[test]
fn pluto_full_configuration_sequence() {
    let mut dev = PlutoDevice::new(3);
    dev.set_context_discovered();

    // Step 1: Set RX frequency (2.4 GHz).
    dev.set_rx_frequency(2_400_000_000).unwrap();
    assert_eq!(dev.rx_frequency(), 2_400_000_000);

    // Encode the write command for frequency.
    let freq_str = format_frequency(2_400_000_000);
    assert_eq!(freq_str.as_str(), "2400000000");

    let mut buf = [0u8; 128];
    let len = encode_write(
        &mut buf,
        device::AD9361_PHY,
        false,
        channel::ALTVOLTAGE0,
        attribute::FREQUENCY,
        freq_str.as_str(),
    )
    .unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert!(cmd.starts_with("WRITE ad9361-phy OUTPUT altvoltage0 frequency"));
    assert!(cmd.ends_with("2400000000"));

    // Step 2: Set TX frequency.
    dev.set_tx_frequency(2_400_000_000).unwrap();
    assert_eq!(dev.tx_frequency(), 2_400_000_000);

    // Step 3: Set sample rate (2.5 MSPS).
    dev.set_sample_rate(2_500_000);
    assert_eq!(dev.sample_rate(), 2_500_000);

    let rate_str = format_sample_rate(2_500_000);
    assert_eq!(rate_str.as_str(), "2500000");

    // Step 4: Set gain (manual mode, 30 dB).
    dev.set_gain(30).unwrap();
    assert_eq!(dev.gain(), 30);
    assert!(dev.is_gain_manual());

    let gain_str = format_gain(30);
    assert_eq!(gain_str.as_str(), "30");

    // Step 5: Set bandwidth.
    dev.set_bandwidth(2_000_000);
    assert_eq!(dev.bandwidth(), 2_000_000);

    let bw_str = format_bandwidth(2_000_000);
    assert_eq!(bw_str.as_str(), "2000000");

    // Verify device is configured.
    assert!(dev.is_configured());
}

#[test]
fn pluto_frequency_validation() {
    assert!(validate_frequency(AD9363_FREQ_MIN_HZ).is_ok());
    assert!(validate_frequency(AD9363_FREQ_MAX_HZ).is_ok());
    assert!(validate_frequency(2_400_000_000).is_ok());
    assert_eq!(
        validate_frequency(AD9363_FREQ_MIN_HZ - 1),
        Err(SdrError::InvalidParameter)
    );
    assert_eq!(
        validate_frequency(AD9363_FREQ_MAX_HZ + 1),
        Err(SdrError::InvalidParameter)
    );
    assert_eq!(validate_frequency(0), Err(SdrError::InvalidParameter));
}

#[test]
fn pluto_gain_validation() {
    assert!(validate_gain(AD9363_GAIN_MIN_DB).is_ok());
    assert!(validate_gain(AD9363_GAIN_MAX_DB).is_ok());
    assert!(validate_gain(0).is_ok());
    assert_eq!(
        validate_gain(AD9363_GAIN_MIN_DB - 1),
        Err(SdrError::InvalidParameter)
    );
    assert_eq!(
        validate_gain(AD9363_GAIN_MAX_DB + 1),
        Err(SdrError::InvalidParameter)
    );
}

#[test]
fn pluto_negative_gain_format() {
    let fb = format_gain(-1);
    assert_eq!(fb.as_str(), "-1");
}

// ---------------------------------------------------------------------------
// 4. IIOD READ command encoding
// ---------------------------------------------------------------------------

#[test]
fn pluto_iiod_read_rx_frequency() {
    let mut buf = [0u8; 128];
    let len = encode_read(
        &mut buf,
        device::AD9361_PHY,
        true,
        channel::ALTVOLTAGE0,
        attribute::FREQUENCY,
    )
    .unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "READ ad9361-phy INPUT altvoltage0 frequency\n");
}

#[test]
fn pluto_iiod_read_gain_mode() {
    let mut buf = [0u8; 128];
    let len = encode_read(
        &mut buf,
        device::AD9361_PHY,
        true,
        channel::VOLTAGE0,
        attribute::GAIN_CONTROL_MODE,
    )
    .unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "READ ad9361-phy INPUT voltage0 gain_control_mode\n");
}

// ---------------------------------------------------------------------------
// 5. Buffer streaming (OPEN, READBUF, CLOSE)
// ---------------------------------------------------------------------------

#[test]
fn pluto_buffer_streaming_command_sequence() {
    let sample_count: u32 = 65536;

    // Step 1: OPEN command.
    let mut buf = [0u8; 128];
    let len = encode_open(
        &mut buf,
        device::CF_AD9361_LPC,
        sample_count,
        IQ_CHANNEL_MASK,
    )
    .unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "OPEN cf-ad9361-lpc 65536 3\n");

    // Step 2: READBUF command.
    let byte_count = sample_count * 4; // 16-bit I + 16-bit Q = 4 bytes per sample
    let mut buf = [0u8; 128];
    let len = encode_readbuf(&mut buf, device::CF_AD9361_LPC, byte_count).unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "READBUF cf-ad9361-lpc 262144\n");

    // Step 3: CLOSE command.
    let mut buf = [0u8; 64];
    let len = encode_close(&mut buf, device::CF_AD9361_LPC).unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "CLOSE cf-ad9361-lpc\n");
}

#[test]
fn pluto_device_streaming_state() {
    let mut dev = PlutoDevice::new(3);
    dev.set_rx_frequency(2_400_000_000).unwrap();
    dev.set_sample_rate(2_500_000);

    assert!(!dev.is_rx_streaming());
    dev.set_rx_streaming(true);
    assert!(dev.is_rx_streaming());

    // Full-duplex: TX can run simultaneously.
    dev.set_tx_streaming(true);
    assert!(dev.is_tx_streaming());
    assert!(dev.is_rx_streaming());

    // Stop RX only.
    dev.set_rx_streaming(false);
    assert!(!dev.is_rx_streaming());
    assert!(dev.is_tx_streaming());
}

// ---------------------------------------------------------------------------
// 6. IIOD WRITEBUF for TX streaming
// ---------------------------------------------------------------------------

#[test]
fn pluto_writebuf_encoding() {
    let tx_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    let mut buf = [0u8; 128];
    let len = encode_writebuf(&mut buf, device::CF_AD9361_DDS, &tx_data).unwrap();

    let header_end = len - tx_data.len();
    let header = core::str::from_utf8(&buf[..header_end]).unwrap();
    assert!(header.starts_with("WRITEBUF cf-ad9361-dds-core-lpc 8\n"));
    assert_eq!(&buf[header_end..len], &tx_data);
}

// ---------------------------------------------------------------------------
// 7. IIOD TIMEOUT configuration
// ---------------------------------------------------------------------------

#[test]
fn pluto_timeout_configuration() {
    let mut dev = PlutoDevice::new(3);
    assert_eq!(dev.timeout(), DEFAULT_TIMEOUT_MS);

    let mut buf = [0u8; 64];
    let len = encode_timeout(&mut buf, 10000).unwrap();
    let cmd = core::str::from_utf8(&buf[..len]).unwrap();
    assert_eq!(cmd, "TIMEOUT 10000\n");

    dev.set_timeout(10000);
    assert_eq!(dev.timeout(), 10000);
}

// ---------------------------------------------------------------------------
// 8. IIOD response parsing
// ---------------------------------------------------------------------------

#[test]
fn pluto_iiod_response_positive() {
    let data = b"4096\nIQ data here";
    let (code, offset) = parse_response(data).unwrap();
    assert_eq!(code, 4096);
    assert_eq!(offset, 5);
    assert_eq!(&data[offset..], b"IQ data here");
}

#[test]
fn pluto_iiod_response_negative_error() {
    let data = b"-22\n";
    let (code, offset) = parse_response(data).unwrap();
    assert_eq!(code, -22);
    assert_eq!(offset, 4);

    let err = map_iiod_error(code);
    assert_eq!(err, SdrError::InvalidParameter);
}

#[test]
fn pluto_iiod_response_zero() {
    let data = b"0\n";
    let (code, _) = parse_response(data).unwrap();
    assert_eq!(code, 0);
}

#[test]
fn pluto_iiod_error_mapping() {
    assert_eq!(map_iiod_error(-22), SdrError::InvalidParameter);
    assert_eq!(map_iiod_error(-16), SdrError::DeviceBusy);
    assert_eq!(map_iiod_error(-110), SdrError::Timeout);
    assert_eq!(map_iiod_error(-19), SdrError::DeviceNotFound);
    assert_eq!(map_iiod_error(-999), SdrError::DeviceError(-999));
}

// ---------------------------------------------------------------------------
// 9. IQ sample parsing (16-bit signed)
// ---------------------------------------------------------------------------

#[test]
fn pluto_iq_16bit_sample_parsing() {
    // Two complex samples: (16384, -16384), (32767, -32768)
    let mut input = [0u8; 8];
    input[0..2].copy_from_slice(&16384i16.to_le_bytes());
    input[2..4].copy_from_slice(&(-16384i16).to_le_bytes());
    input[4..6].copy_from_slice(&32767i16.to_le_bytes());
    input[6..8].copy_from_slice(&(-32768i16).to_le_bytes());

    let mut output = [(0.0f32, 0.0f32); 2];
    let count = parse_iq_16bit(&input, &mut output);
    assert_eq!(count, 2);

    // 16384/32768 = 0.5
    assert!((output[0].0 - 0.5).abs() < 0.001);
    assert!((output[0].1 - (-0.5)).abs() < 0.001);
    // 32767/32768 ~ 1.0
    assert!((output[1].0 - 1.0).abs() < 0.001);
    // -32768/32768 = -1.0
    assert!((output[1].1 - (-1.0)).abs() < 0.001);

    assert_eq!(IqFormat::Signed16.bytes_per_sample(), 4);
}

#[test]
fn pluto_iq_16bit_empty_input() {
    let input: &[u8] = &[];
    let mut output = [(0.0f32, 0.0f32); 4];
    let count = parse_iq_16bit(input, &mut output);
    assert_eq!(count, 0);
}

#[test]
fn pluto_iq_16bit_output_smaller_than_input() {
    let mut input = [0u8; 16]; // 4 samples
    for i in 0..4 {
        let val = (i as i16 * 1000).to_le_bytes();
        input[i * 4..i * 4 + 2].copy_from_slice(&val);
        input[i * 4 + 2..i * 4 + 4].copy_from_slice(&val);
    }
    let mut output = [(0.0f32, 0.0f32); 2]; // Only room for 2
    let count = parse_iq_16bit(&input, &mut output);
    assert_eq!(count, 2);
}

// ---------------------------------------------------------------------------
// 10. Full end-to-end flow simulation
// ---------------------------------------------------------------------------

#[test]
fn pluto_end_to_end_rx_flow() {
    let mut dev = PlutoDevice::new(3);

    // 1. Discover context.
    dev.set_context_discovered();
    assert!(dev.context_discovered());

    // 2. Configure RX frequency.
    dev.set_rx_frequency(2_400_000_000).unwrap();

    // 3. Configure sample rate.
    dev.set_sample_rate(2_500_000);

    // 4. Configure gain.
    dev.set_gain(30).unwrap();

    // 5. Configure bandwidth.
    dev.set_bandwidth(2_000_000);

    // 6. Verify configured.
    assert!(dev.is_configured());

    // 7. Start RX streaming.
    dev.set_rx_streaming(true);
    assert!(dev.is_rx_streaming());

    // 8. Simulate receiving IQ data (100 complex samples).
    let mut raw_iq = alloc::vec![0u8; 400]; // 100 samples * 4 bytes
    for i in 0..100 {
        let phase = (i as f32) * core::f32::consts::PI * 2.0 / 100.0;
        let i_val = (phase.sin() * 16000.0) as i16;
        let q_val = (phase.cos() * 16000.0) as i16;
        raw_iq[i * 4..i * 4 + 2].copy_from_slice(&i_val.to_le_bytes());
        raw_iq[i * 4 + 2..i * 4 + 4].copy_from_slice(&q_val.to_le_bytes());
    }

    let mut output = [(0.0f32, 0.0f32); 100];
    let count = parse_iq_16bit(&raw_iq, &mut output);
    assert_eq!(count, 100);

    // 9. Verify all values are in valid range.
    for &(i, q) in &output[..count] {
        assert!((-1.0..=1.0).contains(&i), "I value {} out of range", i);
        assert!((-1.0..=1.0).contains(&q), "Q value {} out of range", q);
    }

    // 10. Stop streaming.
    dev.set_rx_streaming(false);
    assert!(!dev.is_rx_streaming());
}
