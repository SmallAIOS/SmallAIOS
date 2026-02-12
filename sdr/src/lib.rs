//! SmallAIOS SDR driver crate.
//!
//! Provides drivers for HackRF One and ADALM-PLUTO SDR devices,
//! plus an IQ-to-ONNX inference pipeline.

#![no_std]

#[cfg(test)]
extern crate alloc;

#[cfg(feature = "hackrf")]
pub mod hackrf;

#[cfg(feature = "pluto")]
pub mod pluto;

#[cfg(feature = "iq-pipeline")]
pub mod pipeline;

/// SDR error types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdrError {
    /// USB transfer failed.
    UsbError,
    /// Device not found or wrong VID/PID.
    DeviceNotFound,
    /// Invalid parameter (frequency, gain, sample rate out of range).
    InvalidParameter,
    /// Device is busy (e.g., already streaming).
    DeviceBusy,
    /// Device not configured (must set frequency/sample rate first).
    NotConfigured,
    /// Half-duplex violation (HackRF cannot TX and RX simultaneously).
    HalfDuplexViolation,
    /// Protocol error (malformed response).
    ProtocolError,
    /// Timeout waiting for device response.
    Timeout,
    /// Buffer overflow (ring buffer overwrite).
    BufferOverflow,
    /// Device returned an error code.
    DeviceError(i32),
}

impl From<smallaios_usb::UsbError> for SdrError {
    fn from(_: smallaios_usb::UsbError) -> Self {
        SdrError::UsbError
    }
}

/// Transceiver mode common to SDR devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransceiverMode {
    Off,
    Rx,
    Tx,
}

/// IQ sample formats supported by SDR devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IqFormat {
    /// 8-bit signed interleaved I/Q (HackRF). 2 bytes per sample.
    Signed8,
    /// 16-bit signed interleaved I/Q (PlutoSDR). 4 bytes per sample.
    Signed16,
}

impl IqFormat {
    /// Bytes per complex sample.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            IqFormat::Signed8 => 2,
            IqFormat::Signed16 => 4,
        }
    }
}

/// Common SDR device configuration.
#[derive(Debug, Clone, Copy)]
pub struct SdrConfig {
    /// Center frequency in Hz.
    pub frequency_hz: u64,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Bandwidth in Hz (0 = auto).
    pub bandwidth_hz: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iq_format_bytes() {
        assert_eq!(IqFormat::Signed8.bytes_per_sample(), 2);
        assert_eq!(IqFormat::Signed16.bytes_per_sample(), 4);
    }

    #[test]
    fn test_sdr_error_from_usb() {
        let usb_err = smallaios_usb::UsbError::TransferError;
        let sdr_err: SdrError = usb_err.into();
        assert_eq!(sdr_err, SdrError::UsbError);
    }

    #[test]
    fn test_transceiver_mode() {
        assert_ne!(TransceiverMode::Off, TransceiverMode::Rx);
        assert_ne!(TransceiverMode::Rx, TransceiverMode::Tx);
    }
}
