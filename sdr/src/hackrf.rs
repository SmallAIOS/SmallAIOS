//! HackRF One SDR driver.
//!
//! Implements device detection, vendor control transfers, RF configuration,
//! and bulk IQ streaming for the HackRF One (VID 0x1D50, PID 0x6089).

use crate::{SdrConfig, SdrError, TransceiverMode};

/// HackRF USB Vendor ID.
pub const HACKRF_VID: u16 = 0x1D50;
/// HackRF USB Product ID.
pub const HACKRF_PID: u16 = 0x6089;
/// Expected board ID for HackRF One.
pub const HACKRF_ONE_BOARD_ID: u8 = 2;

/// RX bulk endpoint address.
pub const RX_ENDPOINT: u8 = 0x81;
/// TX bulk endpoint address.
pub const TX_ENDPOINT: u8 = 0x02;

/// Number of concurrent bulk transfers for streaming.
pub const CONCURRENT_TRANSFERS: usize = 4;
/// Size of each bulk transfer buffer in bytes.
pub const TRANSFER_BUFFER_SIZE: usize = 262_144;

/// Vendor request codes.
pub mod request {
    pub const SET_TRANSCEIVER_MODE: u8 = 1;
    pub const SAMPLE_RATE_SET: u8 = 6;
    pub const BOARD_ID_READ: u8 = 14;
    pub const VERSION_STRING_READ: u8 = 15;
    pub const SET_FREQ: u8 = 16;
    pub const AMP_ENABLE: u8 = 17;
    pub const SET_LNA_GAIN: u8 = 19;
    pub const SET_VGA_GAIN: u8 = 20;
    pub const SET_TXVGA_GAIN: u8 = 21;
    pub const INIT_SWEEP: u8 = 26;
    pub const RESET: u8 = 30;
    pub const BOARD_REV_READ: u8 = 31;
}

/// Transceiver mode values for SET_TRANSCEIVER_MODE.
pub mod transceiver {
    pub const OFF: u16 = 0;
    pub const RECEIVE: u16 = 1;
    pub const TRANSMIT: u16 = 2;
    pub const SWEEP: u16 = 5;
}

/// LNA gain range: 0-40 dB in 8 dB steps.
pub const LNA_GAIN_MIN: u16 = 0;
pub const LNA_GAIN_MAX: u16 = 40;
pub const LNA_GAIN_STEP: u16 = 8;

/// VGA gain range: 0-62 dB in 2 dB steps.
pub const VGA_GAIN_MIN: u16 = 0;
pub const VGA_GAIN_MAX: u16 = 62;
pub const VGA_GAIN_STEP: u16 = 2;

/// TX VGA gain range: 0-47 dB in 1 dB steps.
pub const TXVGA_GAIN_MIN: u16 = 0;
pub const TXVGA_GAIN_MAX: u16 = 47;

/// Sweep mode header size in bytes.
pub const SWEEP_HEADER_SIZE: usize = 10;

/// Encodes a vendor control OUT request for HackRF.
///
/// Returns (bmRequestType, bRequest, wValue, wIndex).
pub fn encode_vendor_out(brequest: u8, wvalue: u16, windex: u16) -> (u8, u8, u16, u16) {
    // bmRequestType: host-to-device, vendor, device
    (0x40, brequest, wvalue, windex)
}

/// Encodes a vendor control IN request for HackRF.
///
/// Returns (bmRequestType, bRequest, wValue, wIndex).
pub fn encode_vendor_in(brequest: u8, wvalue: u16, windex: u16) -> (u8, u8, u16, u16) {
    // bmRequestType: device-to-host, vendor, device
    (0xC0, brequest, wvalue, windex)
}

/// Detects whether a USB device is a HackRF One.
pub fn detect_hackrf(vid: u16, pid: u16) -> bool {
    vid == HACKRF_VID && pid == HACKRF_PID
}

/// Encodes SET_TRANSCEIVER_MODE command.
pub fn encode_set_transceiver_mode(mode: TransceiverMode) -> (u8, u8, u16, u16) {
    let value = match mode {
        TransceiverMode::Off => transceiver::OFF,
        TransceiverMode::Rx => transceiver::RECEIVE,
        TransceiverMode::Tx => transceiver::TRANSMIT,
    };
    encode_vendor_out(request::SET_TRANSCEIVER_MODE, value, 0)
}

/// Encodes SET_FREQ command. Frequency is sent as data payload (8 bytes, little-endian).
/// Returns the control request tuple and the 8-byte frequency payload.
pub fn encode_set_freq(freq_hz: u64) -> ((u8, u8, u16, u16), [u8; 8]) {
    // HackRF sends frequency as two 32-bit values: MHz part and Hz remainder
    let mhz = (freq_hz / 1_000_000) as u32;
    let hz = (freq_hz % 1_000_000) as u32;
    let mut data = [0u8; 8];
    data[0..4].copy_from_slice(&mhz.to_le_bytes());
    data[4..8].copy_from_slice(&hz.to_le_bytes());
    (encode_vendor_out(request::SET_FREQ, 0, 0), data)
}

/// Encodes SAMPLE_RATE_SET command. Sends frequency and divider as data payload.
/// Returns control request and 8-byte payload (sample_rate_hz, divider=1).
pub fn encode_sample_rate(sample_rate_hz: u32) -> ((u8, u8, u16, u16), [u8; 8]) {
    let mut data = [0u8; 8];
    data[0..4].copy_from_slice(&sample_rate_hz.to_le_bytes());
    // Divider = 1 (no division)
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    (encode_vendor_out(request::SAMPLE_RATE_SET, 0, 0), data)
}

/// Validates and encodes SET_LNA_GAIN command.
pub fn encode_set_lna_gain(gain_db: u16) -> Result<(u8, u8, u16, u16), SdrError> {
    if gain_db > LNA_GAIN_MAX {
        return Err(SdrError::InvalidParameter);
    }
    // Round to nearest valid step
    let rounded = (gain_db / LNA_GAIN_STEP) * LNA_GAIN_STEP;
    Ok(encode_vendor_out(request::SET_LNA_GAIN, rounded, 0))
}

/// Validates and encodes SET_VGA_GAIN command.
pub fn encode_set_vga_gain(gain_db: u16) -> Result<(u8, u8, u16, u16), SdrError> {
    if gain_db > VGA_GAIN_MAX {
        return Err(SdrError::InvalidParameter);
    }
    let rounded = (gain_db / VGA_GAIN_STEP) * VGA_GAIN_STEP;
    Ok(encode_vendor_out(request::SET_VGA_GAIN, rounded, 0))
}

/// Validates and encodes SET_TXVGA_GAIN command.
pub fn encode_set_txvga_gain(gain_db: u16) -> Result<(u8, u8, u16, u16), SdrError> {
    if gain_db > TXVGA_GAIN_MAX {
        return Err(SdrError::InvalidParameter);
    }
    Ok(encode_vendor_out(request::SET_TXVGA_GAIN, gain_db, 0))
}

/// Encodes AMP_ENABLE command.
pub fn encode_amp_enable(enable: bool) -> (u8, u8, u16, u16) {
    encode_vendor_out(request::AMP_ENABLE, if enable { 1 } else { 0 }, 0)
}

/// Encodes BOARD_ID_READ command.
pub fn encode_board_id_read() -> (u8, u8, u16, u16) {
    encode_vendor_in(request::BOARD_ID_READ, 0, 0)
}

/// Encodes VERSION_STRING_READ command.
pub fn encode_version_string_read() -> (u8, u8, u16, u16) {
    encode_vendor_in(request::VERSION_STRING_READ, 0, 0)
}

/// Encodes BOARD_REV_READ command.
pub fn encode_board_rev_read() -> (u8, u8, u16, u16) {
    encode_vendor_in(request::BOARD_REV_READ, 0, 0)
}

/// Encodes RESET command.
pub fn encode_reset() -> (u8, u8, u16, u16) {
    encode_vendor_out(request::RESET, 0, 0)
}

/// Encodes INIT_SWEEP command parameters into a data payload.
/// Returns the control request tuple and sweep config data.
pub fn encode_init_sweep(
    freq_start_mhz: u16,
    freq_stop_mhz: u16,
    step_width_hz: u32,
) -> ((u8, u8, u16, u16), [u8; 10]) {
    let mut data = [0u8; 10];
    data[0..2].copy_from_slice(&freq_start_mhz.to_le_bytes());
    data[2..4].copy_from_slice(&freq_stop_mhz.to_le_bytes());
    // num_ranges = 1
    data[4..6].copy_from_slice(&1u16.to_le_bytes());
    data[6..10].copy_from_slice(&step_width_hz.to_le_bytes());
    (encode_vendor_out(request::INIT_SWEEP, 0, 0), data)
}

/// Parses 8-bit signed IQ data into (I, Q) pairs as f32 normalized to [-1.0, 1.0].
/// Input: interleaved I,Q,I,Q bytes (signed 8-bit).
/// Returns number of samples parsed into the output buffer.
pub fn parse_iq_8bit(input: &[u8], output: &mut [(f32, f32)]) -> usize {
    let sample_count = input.len() / 2;
    let count = sample_count.min(output.len());
    for i in 0..count {
        let i_val = input[i * 2] as i8;
        let q_val = input[i * 2 + 1] as i8;
        output[i] = (i_val as f32 / 128.0, q_val as f32 / 128.0);
    }
    count
}

/// Parses a sweep mode header from the beginning of a bulk transfer buffer.
/// Returns (frequency_hz, sample_count) or error.
pub fn parse_sweep_header(data: &[u8]) -> Result<(u64, u16), SdrError> {
    if data.len() < SWEEP_HEADER_SIZE {
        return Err(SdrError::ProtocolError);
    }
    let freq_hz = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]);
    let sample_count = u16::from_le_bytes([data[8], data[9]]);
    Ok((freq_hz, sample_count))
}

/// HackRF One device state.
#[derive(Debug)]
pub struct HackRfDevice {
    /// Current operating mode.
    mode: TransceiverMode,
    /// Whether the device has been configured.
    configured: bool,
    /// Current configuration.
    config: SdrConfig,
    /// Number of active streaming transfers.
    active_transfers: u8,
    /// Whether sweep mode is active.
    sweep_active: bool,
    /// Board ID read from device.
    board_id: u8,
}

impl HackRfDevice {
    /// Creates a new HackRF device instance.
    pub fn new() -> Self {
        Self {
            mode: TransceiverMode::Off,
            configured: false,
            config: SdrConfig {
                frequency_hz: 0,
                sample_rate_hz: 0,
                bandwidth_hz: 0,
            },
            active_transfers: 0,
            sweep_active: false,
            board_id: 0,
        }
    }

    /// Sets the board ID after reading from device.
    pub fn set_board_id(&mut self, id: u8) {
        self.board_id = id;
    }

    /// Returns the board ID.
    pub fn board_id(&self) -> u8 {
        self.board_id
    }

    /// Verifies this is a HackRF One (board ID = 2).
    pub fn verify_board_id(&self) -> Result<(), SdrError> {
        if self.board_id != HACKRF_ONE_BOARD_ID {
            return Err(SdrError::DeviceNotFound);
        }
        Ok(())
    }

    /// Sets RF configuration.
    pub fn set_config(&mut self, config: SdrConfig) {
        self.config = config;
        self.configured = true;
    }

    /// Returns current configuration.
    pub fn config(&self) -> &SdrConfig {
        &self.config
    }

    /// Returns true if device is configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Returns current transceiver mode.
    pub fn mode(&self) -> TransceiverMode {
        self.mode
    }

    /// Checks half-duplex constraint and sets new mode.
    pub fn set_mode(&mut self, mode: TransceiverMode) -> Result<(), SdrError> {
        if !self.configured && mode != TransceiverMode::Off {
            return Err(SdrError::NotConfigured);
        }
        // HackRF is half-duplex - if currently streaming, must stop first
        if self.mode != TransceiverMode::Off && mode != TransceiverMode::Off && self.mode != mode {
            return Err(SdrError::HalfDuplexViolation);
        }
        self.mode = mode;
        if mode == TransceiverMode::Off {
            self.active_transfers = 0;
            self.sweep_active = false;
        }
        Ok(())
    }

    /// Returns true if currently streaming.
    pub fn is_streaming(&self) -> bool {
        self.mode != TransceiverMode::Off && self.active_transfers > 0
    }

    /// Starts streaming with concurrent transfers.
    pub fn start_streaming(&mut self) -> Result<(), SdrError> {
        if self.mode == TransceiverMode::Off {
            return Err(SdrError::NotConfigured);
        }
        self.active_transfers = CONCURRENT_TRANSFERS as u8;
        Ok(())
    }

    /// Stops all streaming transfers.
    pub fn stop_streaming(&mut self) {
        self.active_transfers = 0;
        self.mode = TransceiverMode::Off;
        self.sweep_active = false;
    }

    /// Activates sweep mode.
    pub fn set_sweep_active(&mut self, active: bool) {
        self.sweep_active = active;
    }

    /// Returns true if sweep mode is active.
    pub fn is_sweep_active(&self) -> bool {
        self.sweep_active
    }

    /// Performs device reset (returns to unconfigured state).
    pub fn reset(&mut self) {
        self.mode = TransceiverMode::Off;
        self.configured = false;
        self.active_transfers = 0;
        self.sweep_active = false;
        self.config = SdrConfig {
            frequency_hz: 0,
            sample_rate_hz: 0,
            bandwidth_hz: 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_detect_hackrf() {
        assert!(detect_hackrf(HACKRF_VID, HACKRF_PID));
        assert!(!detect_hackrf(0x1234, 0x5678));
        assert!(!detect_hackrf(HACKRF_VID, 0x0000));
    }

    #[test]
    fn test_vendor_request_encoding() {
        // Board ID read (vendor IN)
        let (bm, br, wv, wi) = encode_board_id_read();
        assert_eq!(bm, 0xC0); // device-to-host, vendor
        assert_eq!(br, request::BOARD_ID_READ);
        assert_eq!(wv, 0);
        assert_eq!(wi, 0);

        // Version string read (vendor IN)
        let (bm, br, _wv, _wi) = encode_version_string_read();
        assert_eq!(bm, 0xC0);
        assert_eq!(br, request::VERSION_STRING_READ);

        // Board rev read
        let (bm, br, _wv, _wi) = encode_board_rev_read();
        assert_eq!(bm, 0xC0);
        assert_eq!(br, request::BOARD_REV_READ);

        // Reset (vendor OUT)
        let (bm, br, _wv, _wi) = encode_reset();
        assert_eq!(bm, 0x40); // host-to-device, vendor
        assert_eq!(br, request::RESET);
    }

    #[test]
    fn test_set_transceiver_mode_encoding() {
        let (bm, br, wv, _wi) = encode_set_transceiver_mode(TransceiverMode::Off);
        assert_eq!(bm, 0x40);
        assert_eq!(br, request::SET_TRANSCEIVER_MODE);
        assert_eq!(wv, transceiver::OFF);

        let (_, _, wv, _) = encode_set_transceiver_mode(TransceiverMode::Rx);
        assert_eq!(wv, transceiver::RECEIVE);

        let (_, _, wv, _) = encode_set_transceiver_mode(TransceiverMode::Tx);
        assert_eq!(wv, transceiver::TRANSMIT);
    }

    #[test]
    fn test_set_freq_encoding() {
        let ((bm, br, _, _), data) = encode_set_freq(915_000_000);
        assert_eq!(bm, 0x40);
        assert_eq!(br, request::SET_FREQ);
        let mhz = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let hz = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(mhz, 915);
        assert_eq!(hz, 0);

        // Non-round frequency
        let (_, data) = encode_set_freq(433_500_000);
        let mhz = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let hz = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(mhz, 433);
        assert_eq!(hz, 500_000);
    }

    #[test]
    fn test_sample_rate_encoding() {
        let ((bm, br, _, _), data) = encode_sample_rate(20_000_000);
        assert_eq!(bm, 0x40);
        assert_eq!(br, request::SAMPLE_RATE_SET);
        let rate = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let div = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(rate, 20_000_000);
        assert_eq!(div, 1);
    }

    #[test]
    fn test_gain_range_validation() {
        // LNA: 0-40 in steps of 8
        assert!(encode_set_lna_gain(0).is_ok());
        assert!(encode_set_lna_gain(40).is_ok());
        assert!(encode_set_lna_gain(41).is_err());
        // Should round to step
        let (_, _, wv, _) = encode_set_lna_gain(10).unwrap();
        assert_eq!(wv, 8); // Rounded down to nearest 8

        // VGA: 0-62 in steps of 2
        assert!(encode_set_vga_gain(0).is_ok());
        assert!(encode_set_vga_gain(62).is_ok());
        assert!(encode_set_vga_gain(63).is_err());
        let (_, _, wv, _) = encode_set_vga_gain(31).unwrap();
        assert_eq!(wv, 30); // Rounded down to nearest 2

        // TXVGA: 0-47
        assert!(encode_set_txvga_gain(0).is_ok());
        assert!(encode_set_txvga_gain(47).is_ok());
        assert!(encode_set_txvga_gain(48).is_err());
    }

    #[test]
    fn test_amp_enable_encoding() {
        let (bm, br, wv, _) = encode_amp_enable(true);
        assert_eq!(bm, 0x40);
        assert_eq!(br, request::AMP_ENABLE);
        assert_eq!(wv, 1);

        let (_, _, wv, _) = encode_amp_enable(false);
        assert_eq!(wv, 0);
    }

    #[test]
    fn test_iq_8bit_parsing() {
        // Two samples: I=64, Q=-64, I=127, Q=-128
        let input: &[u8] = &[64, 192, 127, 128]; // 192 = -64 as u8, 128 = -128 as u8
        let mut output = [(0.0f32, 0.0f32); 2];
        let count = parse_iq_8bit(input, &mut output);
        assert_eq!(count, 2);
        assert!((output[0].0 - 0.5).abs() < 0.01); // 64/128 = 0.5
        assert!((output[0].1 - (-0.5)).abs() < 0.01); // -64/128 = -0.5
        assert!((output[1].0 - 0.9921875).abs() < 0.01); // 127/128
        assert!((output[1].1 - (-1.0)).abs() < 0.01); // -128/128 = -1.0
    }

    #[test]
    fn test_iq_8bit_empty() {
        let input: &[u8] = &[];
        let mut output = [(0.0f32, 0.0f32); 2];
        let count = parse_iq_8bit(input, &mut output);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_iq_8bit_output_smaller_than_input() {
        let input: &[u8] = &[10, 20, 30, 40, 50, 60];
        let mut output = [(0.0f32, 0.0f32); 1];
        let count = parse_iq_8bit(input, &mut output);
        assert_eq!(count, 1); // Only first sample fits
    }

    #[test]
    fn test_sweep_header_parsing() {
        let mut data = [0u8; 12];
        let freq: u64 = 915_000_000;
        let samples: u16 = 8192;
        data[0..8].copy_from_slice(&freq.to_le_bytes());
        data[8..10].copy_from_slice(&samples.to_le_bytes());
        let (f, s) = parse_sweep_header(&data).unwrap();
        assert_eq!(f, 915_000_000);
        assert_eq!(s, 8192);
    }

    #[test]
    fn test_sweep_header_too_short() {
        let data = [0u8; 5];
        assert_eq!(parse_sweep_header(&data), Err(SdrError::ProtocolError));
    }

    #[test]
    fn test_init_sweep_encoding() {
        let ((bm, br, _, _), data) = encode_init_sweep(100, 3000, 1_000_000);
        assert_eq!(bm, 0x40);
        assert_eq!(br, request::INIT_SWEEP);
        let start = u16::from_le_bytes([data[0], data[1]]);
        let stop = u16::from_le_bytes([data[2], data[3]]);
        let ranges = u16::from_le_bytes([data[4], data[5]]);
        let step = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
        assert_eq!(start, 100);
        assert_eq!(stop, 3000);
        assert_eq!(ranges, 1);
        assert_eq!(step, 1_000_000);
    }

    #[test]
    fn test_hackrf_device_lifecycle() {
        let mut dev = HackRfDevice::new();
        assert_eq!(dev.mode(), TransceiverMode::Off);
        assert!(!dev.is_configured());
        assert!(!dev.is_streaming());

        // Can't start RX without configuring
        assert_eq!(dev.set_mode(TransceiverMode::Rx), Err(SdrError::NotConfigured));

        // Configure
        dev.set_config(SdrConfig {
            frequency_hz: 915_000_000,
            sample_rate_hz: 20_000_000,
            bandwidth_hz: 0,
        });
        assert!(dev.is_configured());

        // Start RX
        assert!(dev.set_mode(TransceiverMode::Rx).is_ok());
        assert!(dev.start_streaming().is_ok());
        assert!(dev.is_streaming());

        // Half-duplex: can't switch to TX while in RX
        assert_eq!(dev.set_mode(TransceiverMode::Tx), Err(SdrError::HalfDuplexViolation));

        // Stop streaming first
        dev.stop_streaming();
        assert!(!dev.is_streaming());
        assert_eq!(dev.mode(), TransceiverMode::Off);

        // Now can switch to TX
        assert!(dev.set_mode(TransceiverMode::Tx).is_ok());
    }

    #[test]
    fn test_hackrf_device_reset() {
        let mut dev = HackRfDevice::new();
        dev.set_board_id(HACKRF_ONE_BOARD_ID);
        dev.set_config(SdrConfig {
            frequency_hz: 915_000_000,
            sample_rate_hz: 20_000_000,
            bandwidth_hz: 0,
        });
        let _ = dev.set_mode(TransceiverMode::Rx);
        let _ = dev.start_streaming();

        dev.reset();
        assert_eq!(dev.mode(), TransceiverMode::Off);
        assert!(!dev.is_configured());
        assert!(!dev.is_streaming());
    }

    #[test]
    fn test_board_id_verification() {
        let mut dev = HackRfDevice::new();
        dev.set_board_id(HACKRF_ONE_BOARD_ID);
        assert!(dev.verify_board_id().is_ok());

        dev.set_board_id(0xFF);
        assert_eq!(dev.verify_board_id(), Err(SdrError::DeviceNotFound));
    }

    #[test]
    fn test_sweep_mode() {
        let mut dev = HackRfDevice::new();
        dev.set_config(SdrConfig {
            frequency_hz: 0,
            sample_rate_hz: 20_000_000,
            bandwidth_hz: 0,
        });
        assert!(!dev.is_sweep_active());
        dev.set_sweep_active(true);
        assert!(dev.is_sweep_active());

        // Stop streaming clears sweep
        dev.stop_streaming();
        assert!(!dev.is_sweep_active());
    }
}
