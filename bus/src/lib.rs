// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Safety-Critical Bus Protocols
//!
//! Unified transport abstraction for deterministic bus protocols used in
//! automotive, aviation, defense, and space domains. Each protocol implements
//! the [`ZenohTransport`] trait, mapping protocol-native addressing to Zenoh
//! key expressions for transport-agnostic messaging.
//!
//! # Feature Flags
//!
//! - `can` (default) — CAN 2.0A/B and CAN FD frame codecs
//! - `arinc429` — ARINC 429 word codec
//! - `arinc664` — ARINC 664 (AFDX) Virtual Link support
//! - `mil1553` — MIL-STD-1553B command/response protocol
//! - `spacewire` — SpaceWire packet codec and link state machine
//! - `ccsds` — CCSDS Space Packet protocol
//! - `dds` — OMG DDS DCPS/RTPS

#![no_std]

extern crate alloc;

#[cfg(feature = "can")]
pub mod can;

#[cfg(feature = "arinc429")]
pub mod arinc429;

#[cfg(feature = "arinc664")]
pub mod arinc664;

#[cfg(feature = "mil1553")]
pub mod mil1553;

#[cfg(feature = "spacewire")]
pub mod spacewire;

#[cfg(feature = "ccsds")]
pub mod ccsds;

#[cfg(feature = "dds")]
pub mod dds;

#[cfg(feature = "fpga")]
pub mod fpga;

pub mod transport;

use core::fmt;

/// Bus protocol error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusError {
    /// Frame data is shorter than the minimum required length.
    FrameTooShort,
    /// Header or control fields are invalid or malformed.
    InvalidHeader,
    /// Computed CRC does not match the frame CRC.
    CrcMismatch,
    /// Provided buffer is too small for the requested operation.
    BufferTooSmall,
    /// Payload exceeds the maximum allowed size for this frame type.
    PayloadTooLarge,
    /// Identifier value is out of range for the frame type.
    InvalidId,
    /// Data length code is invalid or unsupported.
    InvalidDlc,
    /// The requested feature or protocol variant is not supported.
    NotSupported,
    /// Parity check failed on a received word.
    ParityError,
    /// Value is out of the valid range for encoding.
    OutOfRange,
    /// Invalid BCD digit (nibble value > 9).
    InvalidBcd,
    /// Bus controller is in Bus Off state and cannot transmit.
    BusOff,
    /// Filter table is at capacity.
    FilterTableFull,
    /// Scheduler slot table is at capacity.
    SchedulerFull,
    /// Invalid label value.
    InvalidLabel,
    /// Invalid APID value (> 0x7FF).
    InvalidApid,
    /// QoS policies are incompatible.
    QosIncompatible,
    /// Invalid domain ID.
    InvalidDomain,
    /// Security authentication or access control failure.
    SecurityError,
    /// Timeout waiting for an operation to complete.
    Timeout,
    /// Frame structure is invalid (e.g. misaligned code blocks).
    InvalidFrame,
}

impl fmt::Display for BusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BusError::FrameTooShort => write!(f, "frame too short"),
            BusError::InvalidHeader => write!(f, "invalid header"),
            BusError::CrcMismatch => write!(f, "CRC mismatch"),
            BusError::BufferTooSmall => write!(f, "buffer too small"),
            BusError::PayloadTooLarge => write!(f, "payload too large"),
            BusError::InvalidId => write!(f, "invalid identifier"),
            BusError::InvalidDlc => write!(f, "invalid DLC"),
            BusError::NotSupported => write!(f, "not supported"),
            BusError::ParityError => write!(f, "parity error"),
            BusError::OutOfRange => write!(f, "value out of range"),
            BusError::InvalidBcd => write!(f, "invalid BCD digit"),
            BusError::BusOff => write!(f, "bus off"),
            BusError::FilterTableFull => write!(f, "filter table full"),
            BusError::SchedulerFull => write!(f, "scheduler full"),
            BusError::InvalidLabel => write!(f, "invalid label"),
            BusError::InvalidApid => write!(f, "invalid APID"),
            BusError::QosIncompatible => write!(f, "QoS incompatible"),
            BusError::InvalidDomain => write!(f, "invalid domain"),
            BusError::SecurityError => write!(f, "security error"),
            BusError::Timeout => write!(f, "timeout"),
            BusError::InvalidFrame => write!(f, "invalid frame"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_bus_error_display() {
        assert_eq!(format!("{}", BusError::FrameTooShort), "frame too short");
        assert_eq!(format!("{}", BusError::CrcMismatch), "CRC mismatch");
        assert_eq!(format!("{}", BusError::PayloadTooLarge), "payload too large");
        assert_eq!(format!("{}", BusError::InvalidId), "invalid identifier");
        assert_eq!(format!("{}", BusError::InvalidDlc), "invalid DLC");
        assert_eq!(format!("{}", BusError::NotSupported), "not supported");
        assert_eq!(format!("{}", BusError::ParityError), "parity error");
        assert_eq!(format!("{}", BusError::OutOfRange), "value out of range");
        assert_eq!(format!("{}", BusError::InvalidBcd), "invalid BCD digit");
        assert_eq!(format!("{}", BusError::BusOff), "bus off");
        assert_eq!(format!("{}", BusError::FilterTableFull), "filter table full");
        assert_eq!(format!("{}", BusError::SchedulerFull), "scheduler full");
        assert_eq!(format!("{}", BusError::InvalidLabel), "invalid label");
        assert_eq!(format!("{}", BusError::InvalidApid), "invalid APID");
        assert_eq!(format!("{}", BusError::QosIncompatible), "QoS incompatible");
        assert_eq!(format!("{}", BusError::InvalidDomain), "invalid domain");
        assert_eq!(format!("{}", BusError::SecurityError), "security error");
        assert_eq!(format!("{}", BusError::Timeout), "timeout");
        assert_eq!(format!("{}", BusError::InvalidFrame), "invalid frame");
    }

    #[test]
    fn test_bus_error_debug() {
        assert_eq!(format!("{:?}", BusError::FrameTooShort), "FrameTooShort");
        assert_eq!(format!("{:?}", BusError::InvalidHeader), "InvalidHeader");
    }

    #[test]
    fn test_bus_error_eq() {
        assert_eq!(BusError::CrcMismatch, BusError::CrcMismatch);
        assert_ne!(BusError::CrcMismatch, BusError::InvalidHeader);
    }

    #[test]
    fn test_bus_error_clone() {
        let err = BusError::PayloadTooLarge;
        let cloned = err;
        assert_eq!(err, cloned);
    }
}
