//! ADALM-PLUTO SDR driver.
//!
//! Implements device detection, IIOD text protocol, AD9363 RF configuration,
//! and IQ streaming for the Analog Devices ADALM-PLUTO (VID 0x0456, PID 0xB673).

use crate::SdrError;

/// ADALM-PLUTO USB Vendor ID (Analog Devices).
pub const PLUTO_VID: u16 = 0x0456;
/// ADALM-PLUTO USB Product ID.
pub const PLUTO_PID: u16 = 0xB673;

/// Vendor-specific interface class used for IIO.
pub const VENDOR_INTERFACE_CLASS: u8 = 0xFF;

/// AD9363 minimum frequency in Hz (325 MHz).
pub const AD9363_FREQ_MIN_HZ: u64 = 325_000_000;
/// AD9363 maximum frequency in Hz (3.8 GHz).
pub const AD9363_FREQ_MAX_HZ: u64 = 3_800_000_000;

/// AD9363 minimum hardware gain in dB (approximately -1).
pub const AD9363_GAIN_MIN_DB: i32 = -1;
/// AD9363 maximum hardware gain in dB (approximately 73).
pub const AD9363_GAIN_MAX_DB: i32 = 73;

/// Default IIOD timeout in milliseconds.
pub const DEFAULT_TIMEOUT_MS: u32 = 5000;

/// IIO channel mask for I+Q (both channels enabled).
pub const IQ_CHANNEL_MASK: u32 = 3;

/// IIO device names.
pub mod device {
    /// AD9361 physical layer device (RX/TX configuration).
    pub const AD9361_PHY: &str = "ad9361-phy";
    /// AD9361 data capture device (streaming).
    pub const CF_AD9361_LPC: &str = "cf-ad9361-lpc";
    /// AD9361 data output device (TX streaming).
    pub const CF_AD9361_DDS: &str = "cf-ad9361-dds-core-lpc";
}

/// IIO channel names.
pub mod channel {
    pub const VOLTAGE0: &str = "voltage0";
    pub const VOLTAGE1: &str = "voltage1";
    pub const ALTVOLTAGE0: &str = "altvoltage0";
    pub const ALTVOLTAGE1: &str = "altvoltage1";
}

/// IIO attribute names.
pub mod attribute {
    pub const FREQUENCY: &str = "frequency";
    pub const SAMPLING_FREQUENCY: &str = "sampling_frequency";
    pub const GAIN_CONTROL_MODE: &str = "gain_control_mode";
    pub const HARDWAREGAIN: &str = "hardwaregain";
    pub const RF_BANDWIDTH: &str = "rf_bandwidth";
}

/// Gain control modes.
pub mod gain_mode {
    pub const MANUAL: &str = "manual";
    pub const SLOW_ATTACK: &str = "slow_attack";
    pub const FAST_ATTACK: &str = "fast_attack";
    pub const HYBRID: &str = "hybrid";
}

/// IIOD protocol command types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IiodCommand {
    Print,
    Read,
    Write,
    Open,
    Close,
    ReadBuf,
    WriteBuf,
    Timeout,
}

/// Encodes an IIOD PRINT command to discover available devices.
/// Format: "PRINT\n"
pub fn encode_print(buf: &mut [u8]) -> Result<usize, SdrError> {
    let cmd = b"PRINT\n";
    if buf.len() < cmd.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[..cmd.len()].copy_from_slice(cmd);
    Ok(cmd.len())
}

/// Encodes an IIOD READ command.
/// Format: "READ <device> <is_debug> <channel> <attribute>\n"
pub fn encode_read(
    buf: &mut [u8],
    device: &str,
    is_input: bool,
    channel_name: &str,
    attr: &str,
) -> Result<usize, SdrError> {
    let dir = if is_input { "INPUT" } else { "OUTPUT" };
    // Manual formatting since we're no_std
    let mut pos = 0;
    let parts: &[&[u8]] = &[
        b"READ ",
        device.as_bytes(),
        b" ",
        dir.as_bytes(),
        b" ",
        channel_name.as_bytes(),
        b" ",
        attr.as_bytes(),
        b"\n",
    ];
    for part in parts {
        if pos + part.len() > buf.len() {
            return Err(SdrError::InvalidParameter);
        }
        buf[pos..pos + part.len()].copy_from_slice(part);
        pos += part.len();
    }
    Ok(pos)
}

/// Encodes an IIOD WRITE command with value.
/// Format: "WRITE <device> <direction> <channel> <attribute> <byte_count>\n<value>"
pub fn encode_write(
    buf: &mut [u8],
    device: &str,
    is_input: bool,
    channel_name: &str,
    attr: &str,
    value: &str,
) -> Result<usize, SdrError> {
    let dir = if is_input { "INPUT" } else { "OUTPUT" };
    let value_bytes = value.as_bytes();
    let mut pos = 0;

    // Write header
    let parts: &[&[u8]] = &[
        b"WRITE ",
        device.as_bytes(),
        b" ",
        dir.as_bytes(),
        b" ",
        channel_name.as_bytes(),
        b" ",
        attr.as_bytes(),
        b" ",
    ];
    for part in parts {
        if pos + part.len() > buf.len() {
            return Err(SdrError::InvalidParameter);
        }
        buf[pos..pos + part.len()].copy_from_slice(part);
        pos += part.len();
    }

    // Write byte count as decimal
    let byte_count = value_bytes.len();
    let count_str = format_u32(byte_count as u32);
    let count_bytes = count_str.as_bytes();
    if pos + count_bytes.len() + 1 + value_bytes.len() > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[pos..pos + count_bytes.len()].copy_from_slice(count_bytes);
    pos += count_bytes.len();
    buf[pos] = b'\n';
    pos += 1;

    // Write value
    buf[pos..pos + value_bytes.len()].copy_from_slice(value_bytes);
    pos += value_bytes.len();

    Ok(pos)
}

/// Encodes an IIOD OPEN command.
/// Format: "OPEN <device> <sample_count> <channel_mask>\n"
pub fn encode_open(
    buf: &mut [u8],
    device: &str,
    sample_count: u32,
    channel_mask: u32,
) -> Result<usize, SdrError> {
    let mut pos = 0;
    let parts: &[&[u8]] = &[b"OPEN ", device.as_bytes(), b" "];
    for part in parts {
        if pos + part.len() > buf.len() {
            return Err(SdrError::InvalidParameter);
        }
        buf[pos..pos + part.len()].copy_from_slice(part);
        pos += part.len();
    }

    let sc = format_u32(sample_count);
    let sc_b = sc.as_bytes();
    if pos + sc_b.len() > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[pos..pos + sc_b.len()].copy_from_slice(sc_b);
    pos += sc_b.len();

    buf[pos] = b' ';
    pos += 1;

    let cm = format_u32(channel_mask);
    let cm_b = cm.as_bytes();
    if pos + cm_b.len() + 1 > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[pos..pos + cm_b.len()].copy_from_slice(cm_b);
    pos += cm_b.len();

    buf[pos] = b'\n';
    pos += 1;

    Ok(pos)
}

/// Encodes an IIOD CLOSE command.
/// Format: "CLOSE <device>\n"
pub fn encode_close(buf: &mut [u8], device: &str) -> Result<usize, SdrError> {
    let parts: &[&[u8]] = &[b"CLOSE ", device.as_bytes(), b"\n"];
    let mut pos = 0;
    for part in parts {
        if pos + part.len() > buf.len() {
            return Err(SdrError::InvalidParameter);
        }
        buf[pos..pos + part.len()].copy_from_slice(part);
        pos += part.len();
    }
    Ok(pos)
}

/// Encodes an IIOD READBUF command.
/// Format: "READBUF <device> <bytes>\n"
pub fn encode_readbuf(buf: &mut [u8], device: &str, byte_count: u32) -> Result<usize, SdrError> {
    let mut pos = 0;
    let parts: &[&[u8]] = &[b"READBUF ", device.as_bytes(), b" "];
    for part in parts {
        if pos + part.len() > buf.len() {
            return Err(SdrError::InvalidParameter);
        }
        buf[pos..pos + part.len()].copy_from_slice(part);
        pos += part.len();
    }

    let bc = format_u32(byte_count);
    let bc_b = bc.as_bytes();
    if pos + bc_b.len() + 1 > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[pos..pos + bc_b.len()].copy_from_slice(bc_b);
    pos += bc_b.len();

    buf[pos] = b'\n';
    pos += 1;

    Ok(pos)
}

/// Encodes an IIOD WRITEBUF command.
/// Format: "WRITEBUF <device> <bytes>\n<data>"
pub fn encode_writebuf(
    buf: &mut [u8],
    device: &str,
    data: &[u8],
) -> Result<usize, SdrError> {
    let mut pos = 0;
    let parts: &[&[u8]] = &[b"WRITEBUF ", device.as_bytes(), b" "];
    for part in parts {
        if pos + part.len() > buf.len() {
            return Err(SdrError::InvalidParameter);
        }
        buf[pos..pos + part.len()].copy_from_slice(part);
        pos += part.len();
    }

    let bc = format_u32(data.len() as u32);
    let bc_b = bc.as_bytes();
    if pos + bc_b.len() + 1 + data.len() > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[pos..pos + bc_b.len()].copy_from_slice(bc_b);
    pos += bc_b.len();

    buf[pos] = b'\n';
    pos += 1;

    buf[pos..pos + data.len()].copy_from_slice(data);
    pos += data.len();

    Ok(pos)
}

/// Encodes an IIOD TIMEOUT command.
/// Format: "TIMEOUT <milliseconds>\n"
pub fn encode_timeout(buf: &mut [u8], timeout_ms: u32) -> Result<usize, SdrError> {
    let mut pos = 0;
    let prefix = b"TIMEOUT ";
    if pos + prefix.len() > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[..prefix.len()].copy_from_slice(prefix);
    pos += prefix.len();

    let ms = format_u32(timeout_ms);
    let ms_b = ms.as_bytes();
    if pos + ms_b.len() + 1 > buf.len() {
        return Err(SdrError::InvalidParameter);
    }
    buf[pos..pos + ms_b.len()].copy_from_slice(ms_b);
    pos += ms_b.len();

    buf[pos] = b'\n';
    pos += 1;

    Ok(pos)
}

/// Parses an IIOD response return code from response data.
/// The return code is the first line as a decimal integer.
/// Returns (return_code, data_offset) where data_offset points past the newline.
pub fn parse_response(data: &[u8]) -> Result<(i32, usize), SdrError> {
    // Find the newline
    let newline_pos = data.iter().position(|&b| b == b'\n');
    let line_end = newline_pos.unwrap_or(data.len());

    if line_end == 0 {
        return Err(SdrError::ProtocolError);
    }

    let line = &data[..line_end];

    // Parse as signed integer
    let (negative, start) = if line[0] == b'-' {
        (true, 1)
    } else {
        (false, 0)
    };

    if start >= line.len() {
        return Err(SdrError::ProtocolError);
    }

    let mut value: i32 = 0;
    for &b in &line[start..] {
        if b < b'0' || b > b'9' {
            return Err(SdrError::ProtocolError);
        }
        value = value.wrapping_mul(10).wrapping_add((b - b'0') as i32);
    }

    if negative {
        value = -value;
    }

    let data_offset = if newline_pos.is_some() {
        line_end + 1
    } else {
        line_end
    };

    Ok((value, data_offset))
}

/// Maps an IIOD negative return code to an SdrError.
pub fn map_iiod_error(code: i32) -> SdrError {
    match code {
        -22 => SdrError::InvalidParameter, // EINVAL
        -16 => SdrError::DeviceBusy,       // EBUSY
        -110 => SdrError::Timeout,         // ETIMEDOUT
        -19 => SdrError::DeviceNotFound,   // ENODEV
        _ => SdrError::DeviceError(code),
    }
}

/// Validates AD9363 frequency range.
pub fn validate_frequency(freq_hz: u64) -> Result<(), SdrError> {
    if freq_hz < AD9363_FREQ_MIN_HZ || freq_hz > AD9363_FREQ_MAX_HZ {
        return Err(SdrError::InvalidParameter);
    }
    Ok(())
}

/// Validates AD9363 hardware gain range.
pub fn validate_gain(gain_db: i32) -> Result<(), SdrError> {
    if gain_db < AD9363_GAIN_MIN_DB || gain_db > AD9363_GAIN_MAX_DB {
        return Err(SdrError::InvalidParameter);
    }
    Ok(())
}

/// Detects whether a USB device is an ADALM-PLUTO.
pub fn detect_pluto(vid: u16, pid: u16) -> bool {
    vid == PLUTO_VID && pid == PLUTO_PID
}

/// Identifies the IIO vendor-specific interface from a list of interface classes.
/// Returns the interface index if found.
pub fn find_iio_interface(interface_classes: &[u8]) -> Option<usize> {
    interface_classes
        .iter()
        .position(|&class| class == VENDOR_INTERFACE_CLASS)
}

/// Formats a u64 value as a decimal string for IIOD attribute writes.
/// Returns a fixed-size buffer with the formatted value.
pub fn format_frequency(freq_hz: u64) -> FormatBuf {
    format_u64(freq_hz)
}

/// Formats a u32 value as a decimal string for IIOD attribute writes.
pub fn format_sample_rate(rate_hz: u32) -> FormatBuf {
    let mut buf = FormatBuf::new();
    let s = format_u32(rate_hz);
    let bytes = s.as_bytes();
    buf.buf[..bytes.len()].copy_from_slice(bytes);
    buf.len = bytes.len();
    buf
}

/// Formats an i32 gain value as a decimal string.
pub fn format_gain(gain_db: i32) -> FormatBuf {
    format_i32(gain_db)
}

/// Formats a u32 bandwidth value.
pub fn format_bandwidth(bw_hz: u32) -> FormatBuf {
    format_sample_rate(bw_hz)
}

/// Parses 16-bit signed IQ data into (I, Q) pairs as f32 normalized to [-1.0, 1.0].
/// Input: interleaved I,Q,I,Q as 16-bit little-endian signed integers.
/// Returns number of samples parsed.
pub fn parse_iq_16bit(input: &[u8], output: &mut [(f32, f32)]) -> usize {
    let sample_count = input.len() / 4; // 4 bytes per complex sample
    let count = sample_count.min(output.len());
    for idx in 0..count {
        let i_val = i16::from_le_bytes([input[idx * 4], input[idx * 4 + 1]]);
        let q_val = i16::from_le_bytes([input[idx * 4 + 2], input[idx * 4 + 3]]);
        output[idx] = (i_val as f32 / 32768.0, q_val as f32 / 32768.0);
    }
    count
}

/// Fixed-size format buffer for no_std number formatting.
#[derive(Debug)]
pub struct FormatBuf {
    buf: [u8; 24],
    len: usize,
}

impl FormatBuf {
    fn new() -> Self {
        Self {
            buf: [0u8; 24],
            len: 0,
        }
    }

    /// Returns the formatted string as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// Returns the formatted string as str.
    pub fn as_str(&self) -> &str {
        // Safety: we only write ASCII digits and '-'
        unsafe { core::str::from_utf8_unchecked(&self.buf[..self.len]) }
    }
}

/// Small fixed-size string for u32 formatting (no alloc).
struct SmallStr {
    buf: [u8; 12],
    len: usize,
}

impl SmallStr {
    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

fn format_u32(mut val: u32) -> SmallStr {
    let mut s = SmallStr {
        buf: [0u8; 12],
        len: 0,
    };
    if val == 0 {
        s.buf[0] = b'0';
        s.len = 1;
        return s;
    }
    let mut digits = [0u8; 10];
    let mut n = 0;
    while val > 0 {
        digits[n] = b'0' + (val % 10) as u8;
        val /= 10;
        n += 1;
    }
    for i in 0..n {
        s.buf[i] = digits[n - 1 - i];
    }
    s.len = n;
    s
}

fn format_u64(mut val: u64) -> FormatBuf {
    let mut fb = FormatBuf::new();
    if val == 0 {
        fb.buf[0] = b'0';
        fb.len = 1;
        return fb;
    }
    let mut digits = [0u8; 20];
    let mut n = 0;
    while val > 0 {
        digits[n] = b'0' + (val % 10) as u8;
        val /= 10;
        n += 1;
    }
    for i in 0..n {
        fb.buf[i] = digits[n - 1 - i];
    }
    fb.len = n;
    fb
}

fn format_i32(val: i32) -> FormatBuf {
    let mut fb = FormatBuf::new();
    if val < 0 {
        fb.buf[0] = b'-';
        let s = format_u32((-val) as u32);
        let bytes = s.as_bytes();
        fb.buf[1..1 + bytes.len()].copy_from_slice(bytes);
        fb.len = 1 + bytes.len();
    } else {
        let s = format_u32(val as u32);
        let bytes = s.as_bytes();
        fb.buf[..bytes.len()].copy_from_slice(bytes);
        fb.len = bytes.len();
    }
    fb
}

/// PlutoSDR device state.
#[derive(Debug)]
pub struct PlutoDevice {
    /// Whether IIO context has been discovered.
    context_discovered: bool,
    /// Current RX frequency in Hz.
    rx_freq_hz: u64,
    /// Current TX frequency in Hz.
    tx_freq_hz: u64,
    /// Current sample rate in Hz.
    sample_rate_hz: u32,
    /// Current gain in dB.
    gain_db: i32,
    /// Current gain control mode index.
    gain_mode_manual: bool,
    /// Current RF bandwidth in Hz.
    bandwidth_hz: u32,
    /// Whether RX streaming is active.
    rx_streaming: bool,
    /// Whether TX streaming is active.
    tx_streaming: bool,
    /// IIO interface index on the USB composite device.
    iio_interface: u8,
    /// IIOD timeout in ms.
    timeout_ms: u32,
}

impl PlutoDevice {
    /// Creates a new PlutoSDR device instance.
    pub fn new(iio_interface: u8) -> Self {
        Self {
            context_discovered: false,
            rx_freq_hz: 0,
            tx_freq_hz: 0,
            sample_rate_hz: 0,
            gain_db: 0,
            gain_mode_manual: true,
            bandwidth_hz: 0,
            rx_streaming: false,
            tx_streaming: false,
            iio_interface,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Returns the IIO interface index.
    pub fn iio_interface(&self) -> u8 {
        self.iio_interface
    }

    /// Marks context as discovered.
    pub fn set_context_discovered(&mut self) {
        self.context_discovered = true;
    }

    /// Returns true if context has been discovered.
    pub fn context_discovered(&self) -> bool {
        self.context_discovered
    }

    /// Sets RX frequency with validation.
    pub fn set_rx_frequency(&mut self, freq_hz: u64) -> Result<(), SdrError> {
        validate_frequency(freq_hz)?;
        self.rx_freq_hz = freq_hz;
        Ok(())
    }

    /// Returns current RX frequency.
    pub fn rx_frequency(&self) -> u64 {
        self.rx_freq_hz
    }

    /// Sets TX frequency with validation.
    pub fn set_tx_frequency(&mut self, freq_hz: u64) -> Result<(), SdrError> {
        validate_frequency(freq_hz)?;
        self.tx_freq_hz = freq_hz;
        Ok(())
    }

    /// Returns current TX frequency.
    pub fn tx_frequency(&self) -> u64 {
        self.tx_freq_hz
    }

    /// Sets sample rate.
    pub fn set_sample_rate(&mut self, rate_hz: u32) {
        self.sample_rate_hz = rate_hz;
    }

    /// Returns current sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Sets hardware gain with validation.
    pub fn set_gain(&mut self, gain_db: i32) -> Result<(), SdrError> {
        validate_gain(gain_db)?;
        self.gain_db = gain_db;
        self.gain_mode_manual = true;
        Ok(())
    }

    /// Returns current gain.
    pub fn gain(&self) -> i32 {
        self.gain_db
    }

    /// Sets gain control mode.
    pub fn set_gain_mode_manual(&mut self, manual: bool) {
        self.gain_mode_manual = manual;
    }

    /// Returns true if gain mode is manual.
    pub fn is_gain_manual(&self) -> bool {
        self.gain_mode_manual
    }

    /// Sets RF bandwidth.
    pub fn set_bandwidth(&mut self, bw_hz: u32) {
        self.bandwidth_hz = bw_hz;
    }

    /// Returns current RF bandwidth.
    pub fn bandwidth(&self) -> u32 {
        self.bandwidth_hz
    }

    /// Sets IIOD timeout.
    pub fn set_timeout(&mut self, timeout_ms: u32) {
        self.timeout_ms = timeout_ms;
    }

    /// Returns current timeout.
    pub fn timeout(&self) -> u32 {
        self.timeout_ms
    }

    /// Marks RX streaming as active/inactive.
    pub fn set_rx_streaming(&mut self, active: bool) {
        self.rx_streaming = active;
    }

    /// Returns true if RX streaming.
    pub fn is_rx_streaming(&self) -> bool {
        self.rx_streaming
    }

    /// Marks TX streaming as active/inactive.
    pub fn set_tx_streaming(&mut self, active: bool) {
        self.tx_streaming = active;
    }

    /// Returns true if TX streaming.
    pub fn is_tx_streaming(&self) -> bool {
        self.tx_streaming
    }

    /// Returns true if device is configured (frequency + sample rate set).
    pub fn is_configured(&self) -> bool {
        self.rx_freq_hz > 0 && self.sample_rate_hz > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_pluto() {
        assert!(detect_pluto(PLUTO_VID, PLUTO_PID));
        assert!(!detect_pluto(0x1234, 0x5678));
        assert!(!detect_pluto(PLUTO_VID, 0x0000));
    }

    #[test]
    fn test_find_iio_interface() {
        // Typical Pluto composite: CDC(0x02), MSC(0x08), DFU(0xFE), IIO(0xFF)
        let classes = [0x02, 0x08, 0xFE, 0xFF];
        assert_eq!(find_iio_interface(&classes), Some(3));

        // No vendor-specific interface
        let classes = [0x02, 0x08, 0xFE];
        assert_eq!(find_iio_interface(&classes), None);
    }

    #[test]
    fn test_encode_print() {
        let mut buf = [0u8; 64];
        let len = encode_print(&mut buf).unwrap();
        assert_eq!(&buf[..len], b"PRINT\n");
    }

    #[test]
    fn test_encode_read() {
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
    fn test_encode_write() {
        let mut buf = [0u8; 128];
        let len = encode_write(
            &mut buf,
            device::AD9361_PHY,
            false,
            channel::ALTVOLTAGE0,
            attribute::FREQUENCY,
            "2400000000",
        )
        .unwrap();
        let cmd = core::str::from_utf8(&buf[..len]).unwrap();
        assert!(cmd.starts_with("WRITE ad9361-phy OUTPUT altvoltage0 frequency 10\n"));
        assert!(cmd.ends_with("2400000000"));
    }

    #[test]
    fn test_encode_open() {
        let mut buf = [0u8; 128];
        let len = encode_open(&mut buf, device::CF_AD9361_LPC, 65536, IQ_CHANNEL_MASK).unwrap();
        let cmd = core::str::from_utf8(&buf[..len]).unwrap();
        assert_eq!(cmd, "OPEN cf-ad9361-lpc 65536 3\n");
    }

    #[test]
    fn test_encode_close() {
        let mut buf = [0u8; 64];
        let len = encode_close(&mut buf, device::CF_AD9361_LPC).unwrap();
        let cmd = core::str::from_utf8(&buf[..len]).unwrap();
        assert_eq!(cmd, "CLOSE cf-ad9361-lpc\n");
    }

    #[test]
    fn test_encode_readbuf() {
        let mut buf = [0u8; 64];
        let len = encode_readbuf(&mut buf, device::CF_AD9361_LPC, 262144).unwrap();
        let cmd = core::str::from_utf8(&buf[..len]).unwrap();
        assert_eq!(cmd, "READBUF cf-ad9361-lpc 262144\n");
    }

    #[test]
    fn test_encode_writebuf() {
        let mut buf = [0u8; 128];
        let data = [0x01, 0x02, 0x03, 0x04];
        let len = encode_writebuf(&mut buf, device::CF_AD9361_DDS, &data).unwrap();
        let header = core::str::from_utf8(&buf[..len - 4]).unwrap();
        assert!(header.starts_with("WRITEBUF cf-ad9361-dds-core-lpc 4\n"));
        assert_eq!(&buf[len - 4..len], &data);
    }

    #[test]
    fn test_encode_timeout() {
        let mut buf = [0u8; 64];
        let len = encode_timeout(&mut buf, 5000).unwrap();
        let cmd = core::str::from_utf8(&buf[..len]).unwrap();
        assert_eq!(cmd, "TIMEOUT 5000\n");
    }

    #[test]
    fn test_parse_response_positive() {
        let data = b"4096\nHello";
        let (code, offset) = parse_response(data).unwrap();
        assert_eq!(code, 4096);
        assert_eq!(offset, 5); // past the newline
        assert_eq!(&data[offset..], b"Hello");
    }

    #[test]
    fn test_parse_response_negative() {
        let data = b"-22\n";
        let (code, offset) = parse_response(data).unwrap();
        assert_eq!(code, -22);
        assert_eq!(offset, 4);
    }

    #[test]
    fn test_parse_response_zero() {
        let data = b"0\n";
        let (code, _) = parse_response(data).unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn test_map_iiod_errors() {
        assert_eq!(map_iiod_error(-22), SdrError::InvalidParameter);
        assert_eq!(map_iiod_error(-16), SdrError::DeviceBusy);
        assert_eq!(map_iiod_error(-110), SdrError::Timeout);
        assert_eq!(map_iiod_error(-19), SdrError::DeviceNotFound);
        assert_eq!(map_iiod_error(-999), SdrError::DeviceError(-999));
    }

    #[test]
    fn test_frequency_validation() {
        assert!(validate_frequency(325_000_000).is_ok());
        assert!(validate_frequency(3_800_000_000).is_ok());
        assert!(validate_frequency(2_400_000_000).is_ok());
        assert!(validate_frequency(324_999_999).is_err());
        assert!(validate_frequency(3_800_000_001).is_err());
        assert!(validate_frequency(0).is_err());
    }

    #[test]
    fn test_gain_validation() {
        assert!(validate_gain(-1).is_ok());
        assert!(validate_gain(0).is_ok());
        assert!(validate_gain(73).is_ok());
        assert!(validate_gain(-2).is_err());
        assert!(validate_gain(74).is_err());
    }

    #[test]
    fn test_format_frequency() {
        let fb = format_frequency(2_400_000_000);
        assert_eq!(fb.as_str(), "2400000000");
    }

    #[test]
    fn test_format_gain() {
        let fb = format_gain(30);
        assert_eq!(fb.as_str(), "30");

        let fb = format_gain(-1);
        assert_eq!(fb.as_str(), "-1");

        let fb = format_gain(0);
        assert_eq!(fb.as_str(), "0");
    }

    #[test]
    fn test_iq_16bit_parsing() {
        // Two samples: I=16384, Q=-16384, I=32767, Q=-32768
        let mut input = [0u8; 8];
        input[0..2].copy_from_slice(&16384i16.to_le_bytes());
        input[2..4].copy_from_slice(&(-16384i16).to_le_bytes());
        input[4..6].copy_from_slice(&32767i16.to_le_bytes());
        input[6..8].copy_from_slice(&(-32768i16).to_le_bytes());

        let mut output = [(0.0f32, 0.0f32); 2];
        let count = parse_iq_16bit(&input, &mut output);
        assert_eq!(count, 2);
        assert!((output[0].0 - 0.5).abs() < 0.001);
        assert!((output[0].1 - (-0.5)).abs() < 0.001);
        assert!((output[1].0 - 1.0).abs() < 0.001);
        assert!((output[1].1 - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_pluto_device_lifecycle() {
        let mut dev = PlutoDevice::new(3);
        assert_eq!(dev.iio_interface(), 3);
        assert!(!dev.context_discovered());
        assert!(!dev.is_configured());

        dev.set_context_discovered();
        assert!(dev.context_discovered());

        // Set RX frequency
        assert!(dev.set_rx_frequency(2_400_000_000).is_ok());
        assert_eq!(dev.rx_frequency(), 2_400_000_000);

        // Set TX frequency
        assert!(dev.set_tx_frequency(2_400_000_000).is_ok());
        assert_eq!(dev.tx_frequency(), 2_400_000_000);

        // Invalid frequency
        assert!(dev.set_rx_frequency(100_000_000).is_err());

        // Set sample rate
        dev.set_sample_rate(2_500_000);
        assert_eq!(dev.sample_rate(), 2_500_000);
        assert!(dev.is_configured());

        // Set gain
        assert!(dev.set_gain(30).is_ok());
        assert_eq!(dev.gain(), 30);
        assert!(dev.is_gain_manual());

        // Invalid gain
        assert!(dev.set_gain(100).is_err());

        // Set bandwidth
        dev.set_bandwidth(2_000_000);
        assert_eq!(dev.bandwidth(), 2_000_000);

        // Full-duplex: both RX and TX can be active
        dev.set_rx_streaming(true);
        dev.set_tx_streaming(true);
        assert!(dev.is_rx_streaming());
        assert!(dev.is_tx_streaming());
    }

    #[test]
    fn test_pluto_timeout() {
        let mut dev = PlutoDevice::new(3);
        assert_eq!(dev.timeout(), DEFAULT_TIMEOUT_MS);
        dev.set_timeout(10000);
        assert_eq!(dev.timeout(), 10000);
    }
}
