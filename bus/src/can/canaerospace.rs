// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CANaerospace profile for civil aviation applications.
//!
//! CANaerospace is a CAN-based communication standard for airborne systems,
//! defined in ARINC 825 and widely used in general aviation, UAVs, and
//! experimental aircraft. It defines:
//!
//! - **Node IDs** (0-255): Unique address per device on the bus
//! - **Message types**: Normal Operation Data (NOD), Emergency Event Data (EED),
//!   Node Service (NSS), Test/Maintenance (TSD), and User-Defined (UDD)
//! - **Service channels**: Node identification, module configuration,
//!   and data distribution
//! - **Data types**: Standard parameter formats (float32, int16/32, etc.)
//!
//! # CAN ID Layout (11-bit or 29-bit)
//!
//! Standard (11-bit) CAN ID encodes the message type and service code.
//! The 4-byte payload carries the CANaerospace data with:
//! - Byte 0: Node ID of sender
//! - Byte 1: Data type code
//! - Byte 2: Service code
//! - Byte 3: Message code / data byte 0

use super::frame::CanFrame;
use crate::BusError;
use alloc::vec::Vec;

/// Maximum CANaerospace node ID.
pub const MAX_NODE_ID: u8 = 255;

/// CANaerospace message type, determined by CAN ID range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Emergency Event Data (CAN ID 0-127).
    Emergency,
    /// Normal Operation Data — high priority (CAN ID 128-199).
    NodHigh,
    /// Normal Operation Data — normal priority (CAN ID 200-299).
    NodNormal,
    /// Node Service — high priority (CAN ID 300-1799).
    NssHigh,
    /// User-Defined Data (CAN ID 1800-1899).
    UserDefined,
    /// Normal Operation Data — low priority (CAN ID 1900-1999).
    NodLow,
    /// Debug/Test Service Data (CAN ID 2000-2031).
    TestService,
    /// Node Service — low priority (CAN ID 2000-2031 for some; not used here).
    NssLow,
}

impl MessageType {
    /// Determine the message type from a CAN identifier.
    pub fn from_can_id(can_id: u32) -> Result<Self, BusError> {
        match can_id {
            0..=127 => Ok(MessageType::Emergency),
            128..=199 => Ok(MessageType::NodHigh),
            200..=299 => Ok(MessageType::NodNormal),
            300..=1799 => Ok(MessageType::NssHigh),
            1800..=1899 => Ok(MessageType::UserDefined),
            1900..=1999 => Ok(MessageType::NodLow),
            2000..=2031 => Ok(MessageType::TestService),
            _ => Err(BusError::InvalidId),
        }
    }
}

/// CANaerospace data type codes (subset of common types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    /// No data (0 bytes).
    NoData = 0x00,
    /// Error / status (4 bytes).
    Error = 0x01,
    /// Float 32-bit (IEEE 754).
    Float = 0x02,
    /// Signed 32-bit integer.
    Long = 0x03,
    /// Unsigned 32-bit integer.
    ULong = 0x04,
    /// Signed 16-bit integer (Bshort).
    BShort = 0x05,
    /// Unsigned 16-bit integer (Bushort).
    BUShort = 0x06,
    /// Signed 16-bit integer (short, low 2 bytes).
    Short = 0x07,
    /// Unsigned 16-bit integer.
    UShort = 0x08,
    /// 8-bit signed integer (Bchar).
    BChar = 0x09,
    /// 8-bit unsigned integer (Buchar).
    BUChar = 0x0A,
    /// 2-byte short (short2, high 2 bytes).
    Short2 = 0x0B,
    /// Unsigned 2-byte short.
    UShort2 = 0x0C,
    /// 4-byte string / character data.
    Char4 = 0x0D,
    /// Boolean (1 byte used, 3 padding).
    Boolean = 0x0E,
    /// Memory address / pointer (32-bit).
    MemId = 0x0F,
}

impl DataType {
    /// Create from raw byte value.
    pub fn from_raw(val: u8) -> Option<Self> {
        match val {
            0x00 => Some(DataType::NoData),
            0x01 => Some(DataType::Error),
            0x02 => Some(DataType::Float),
            0x03 => Some(DataType::Long),
            0x04 => Some(DataType::ULong),
            0x05 => Some(DataType::BShort),
            0x06 => Some(DataType::BUShort),
            0x07 => Some(DataType::Short),
            0x08 => Some(DataType::UShort),
            0x09 => Some(DataType::BChar),
            0x0A => Some(DataType::BUChar),
            0x0B => Some(DataType::Short2),
            0x0C => Some(DataType::UShort2),
            0x0D => Some(DataType::Char4),
            0x0E => Some(DataType::Boolean),
            0x0F => Some(DataType::MemId),
            _ => None,
        }
    }

    /// Returns the raw byte value.
    pub fn raw(self) -> u8 {
        self as u8
    }
}

/// Node Service (NSS) codes for CANaerospace service channel requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceCode {
    /// Identification request/response (IDS).
    Identification = 0x00,
    /// Node synchronization (NSS).
    NodeSync = 0x01,
    /// Data download service (DDS).
    DataDownload = 0x02,
    /// Data upload service (DUS).
    DataUpload = 0x03,
    /// Module configuration (MCS).
    ModuleConfig = 0x04,
    /// State transmission request (STS).
    StateTransmission = 0x05,
    /// Filter setting (FSS).
    FilterSetting = 0x06,
    /// Test control (TCS).
    TestControl = 0x07,
}

impl ServiceCode {
    /// Create from raw byte value.
    pub fn from_raw(val: u8) -> Option<Self> {
        match val {
            0x00 => Some(ServiceCode::Identification),
            0x01 => Some(ServiceCode::NodeSync),
            0x02 => Some(ServiceCode::DataDownload),
            0x03 => Some(ServiceCode::DataUpload),
            0x04 => Some(ServiceCode::ModuleConfig),
            0x05 => Some(ServiceCode::StateTransmission),
            0x06 => Some(ServiceCode::FilterSetting),
            0x07 => Some(ServiceCode::TestControl),
            _ => None,
        }
    }

    /// Returns the raw byte value.
    pub fn raw(self) -> u8 {
        self as u8
    }
}

/// A CANaerospace message decoded from a CAN frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CanaMessage {
    /// CAN identifier (determines message type and parameter).
    pub can_id: u32,
    /// Message type (derived from CAN ID range).
    pub msg_type: MessageType,
    /// Sender's node ID.
    pub node_id: u8,
    /// Data type code.
    pub data_type: u8,
    /// Service code / message code.
    pub service_code: u8,
    /// Message code (byte 3 of payload).
    pub message_code: u8,
    /// Remaining data bytes (bytes 4-7, if DLC > 4).
    pub data: Vec<u8>,
}

impl CanaMessage {
    /// Create a new CANaerospace Normal Operation Data (NOD) message.
    pub fn new_nod(
        can_id: u32,
        node_id: u8,
        data_type: DataType,
        data: &[u8],
    ) -> Result<Self, BusError> {
        let msg_type = MessageType::from_can_id(can_id)?;
        if data.len() > 4 {
            return Err(BusError::PayloadTooLarge);
        }
        Ok(Self {
            can_id,
            msg_type,
            node_id,
            data_type: data_type.raw(),
            service_code: 0,
            message_code: 0,
            data: Vec::from(data),
        })
    }

    /// Create a new Node Service (NSS) message.
    pub fn new_nss(
        can_id: u32,
        node_id: u8,
        service_code: ServiceCode,
        message_code: u8,
        data: &[u8],
    ) -> Result<Self, BusError> {
        let msg_type = MessageType::from_can_id(can_id)?;
        if data.len() > 4 {
            return Err(BusError::PayloadTooLarge);
        }
        Ok(Self {
            can_id,
            msg_type,
            node_id,
            data_type: DataType::NoData.raw(),
            service_code: service_code.raw(),
            message_code,
            data: Vec::from(data),
        })
    }

    /// Encode the CANaerospace message into a CAN frame.
    ///
    /// The CAN frame payload is:
    /// - Byte 0: Node ID
    /// - Byte 1: Data Type
    /// - Byte 2: Service Code
    /// - Byte 3: Message Code
    /// - Bytes 4-7: Data (0-4 bytes)
    pub fn to_can_frame(&self) -> Result<CanFrame, BusError> {
        let mut payload = Vec::with_capacity(8);
        payload.push(self.node_id);
        payload.push(self.data_type);
        payload.push(self.service_code);
        payload.push(self.message_code);
        payload.extend_from_slice(&self.data);

        CanFrame::new_standard(self.can_id, &payload)
    }

    /// Decode a CAN frame into a CANaerospace message.
    pub fn from_can_frame(frame: &CanFrame) -> Result<Self, BusError> {
        if frame.data.len() < 4 {
            return Err(BusError::FrameTooShort);
        }

        let msg_type = MessageType::from_can_id(frame.id)?;

        Ok(Self {
            can_id: frame.id,
            msg_type,
            node_id: frame.data[0],
            data_type: frame.data[1],
            service_code: frame.data[2],
            message_code: frame.data[3],
            data: Vec::from(&frame.data[4..]),
        })
    }

    /// Extract a float32 value from the data bytes (big-endian).
    pub fn as_f32(&self) -> Option<f32> {
        if self.data.len() >= 4 {
            Some(f32::from_be_bytes([
                self.data[0],
                self.data[1],
                self.data[2],
                self.data[3],
            ]))
        } else {
            None
        }
    }

    /// Extract a u32 value from the data bytes (big-endian).
    pub fn as_u32(&self) -> Option<u32> {
        if self.data.len() >= 4 {
            Some(u32::from_be_bytes([
                self.data[0],
                self.data[1],
                self.data[2],
                self.data[3],
            ]))
        } else {
            None
        }
    }

    /// Extract an i16 value from the first 2 data bytes (big-endian).
    pub fn as_i16(&self) -> Option<i16> {
        if self.data.len() >= 2 {
            Some(i16::from_be_bytes([self.data[0], self.data[1]]))
        } else {
            None
        }
    }
}

/// Standard CANaerospace parameters (common IDs from ARINC 825).
pub mod params {
    /// Indicated airspeed (knots, float32). CAN ID 200.
    pub const AIRSPEED_IAS: u32 = 200;
    /// True airspeed (knots, float32). CAN ID 201.
    pub const AIRSPEED_TAS: u32 = 201;
    /// Barometric altitude (feet, float32). CAN ID 203.
    pub const ALTITUDE_BARO: u32 = 203;
    /// GPS altitude (feet, float32). CAN ID 204.
    pub const ALTITUDE_GPS: u32 = 204;
    /// Heading (degrees, float32). CAN ID 206.
    pub const HEADING: u32 = 206;
    /// Pitch angle (degrees, float32). CAN ID 207.
    pub const PITCH: u32 = 207;
    /// Roll angle (degrees, float32). CAN ID 208.
    pub const ROLL: u32 = 208;
    /// Yaw rate (deg/s, float32). CAN ID 209.
    pub const YAW_RATE: u32 = 209;
    /// Body X acceleration (g, float32). CAN ID 210.
    pub const ACCEL_X: u32 = 210;
    /// Body Y acceleration (g, float32). CAN ID 211.
    pub const ACCEL_Y: u32 = 211;
    /// Body Z acceleration (g, float32). CAN ID 212.
    pub const ACCEL_Z: u32 = 212;
    /// Engine RPM (rpm, float32). CAN ID 300.
    pub const ENGINE_RPM: u32 = 300;
    /// Oil pressure (psi, float32). CAN ID 301.
    pub const OIL_PRESSURE: u32 = 301;
    /// Oil temperature (deg C, float32). CAN ID 302.
    pub const OIL_TEMP: u32 = 302;
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MessageType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_message_type_emergency() {
        assert_eq!(MessageType::from_can_id(0), Ok(MessageType::Emergency));
        assert_eq!(MessageType::from_can_id(127), Ok(MessageType::Emergency));
    }

    #[test]
    fn test_message_type_nod_high() {
        assert_eq!(MessageType::from_can_id(128), Ok(MessageType::NodHigh));
        assert_eq!(MessageType::from_can_id(199), Ok(MessageType::NodHigh));
    }

    #[test]
    fn test_message_type_nod_normal() {
        assert_eq!(MessageType::from_can_id(200), Ok(MessageType::NodNormal));
        assert_eq!(MessageType::from_can_id(299), Ok(MessageType::NodNormal));
    }

    #[test]
    fn test_message_type_nss_high() {
        assert_eq!(MessageType::from_can_id(300), Ok(MessageType::NssHigh));
        assert_eq!(MessageType::from_can_id(1799), Ok(MessageType::NssHigh));
    }

    #[test]
    fn test_message_type_user_defined() {
        assert_eq!(MessageType::from_can_id(1800), Ok(MessageType::UserDefined));
        assert_eq!(MessageType::from_can_id(1899), Ok(MessageType::UserDefined));
    }

    #[test]
    fn test_message_type_nod_low() {
        assert_eq!(MessageType::from_can_id(1900), Ok(MessageType::NodLow));
        assert_eq!(MessageType::from_can_id(1999), Ok(MessageType::NodLow));
    }

    #[test]
    fn test_message_type_test_service() {
        assert_eq!(MessageType::from_can_id(2000), Ok(MessageType::TestService));
        assert_eq!(MessageType::from_can_id(2031), Ok(MessageType::TestService));
    }

    #[test]
    fn test_message_type_invalid() {
        assert_eq!(MessageType::from_can_id(2032), Err(BusError::InvalidId));
        assert_eq!(MessageType::from_can_id(0x7FF), Err(BusError::InvalidId));
    }

    // -----------------------------------------------------------------------
    // DataType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_data_type_roundtrip() {
        let types = [
            DataType::NoData,
            DataType::Error,
            DataType::Float,
            DataType::Long,
            DataType::ULong,
            DataType::BShort,
            DataType::BUShort,
            DataType::Short,
            DataType::UShort,
            DataType::BChar,
            DataType::BUChar,
            DataType::Short2,
            DataType::UShort2,
            DataType::Char4,
            DataType::Boolean,
            DataType::MemId,
        ];
        for dt in types {
            let raw = dt.raw();
            let decoded = DataType::from_raw(raw).unwrap();
            assert_eq!(decoded, dt);
        }
    }

    #[test]
    fn test_data_type_unknown() {
        assert!(DataType::from_raw(0xFF).is_none());
        assert!(DataType::from_raw(0x10).is_none());
    }

    // -----------------------------------------------------------------------
    // ServiceCode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_code_roundtrip() {
        let codes = [
            ServiceCode::Identification,
            ServiceCode::NodeSync,
            ServiceCode::DataDownload,
            ServiceCode::DataUpload,
            ServiceCode::ModuleConfig,
            ServiceCode::StateTransmission,
            ServiceCode::FilterSetting,
            ServiceCode::TestControl,
        ];
        for sc in codes {
            let raw = sc.raw();
            let decoded = ServiceCode::from_raw(raw).unwrap();
            assert_eq!(decoded, sc);
        }
    }

    #[test]
    fn test_service_code_unknown() {
        assert!(ServiceCode::from_raw(0xFF).is_none());
        assert!(ServiceCode::from_raw(0x08).is_none());
    }

    // -----------------------------------------------------------------------
    // CanaMessage NOD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_nod_airspeed() {
        let speed: f32 = 125.5;
        let data = speed.to_be_bytes();
        let msg = CanaMessage::new_nod(params::AIRSPEED_IAS, 1, DataType::Float, &data).unwrap();
        assert_eq!(msg.msg_type, MessageType::NodNormal);
        assert_eq!(msg.node_id, 1);
        assert_eq!(msg.as_f32(), Some(125.5));
    }

    #[test]
    fn test_new_nod_altitude() {
        let alt: f32 = 35000.0;
        let data = alt.to_be_bytes();
        let msg = CanaMessage::new_nod(params::ALTITUDE_BARO, 2, DataType::Float, &data).unwrap();
        assert_eq!(msg.as_f32(), Some(35000.0));
    }

    #[test]
    fn test_new_nod_emergency() {
        let msg = CanaMessage::new_nod(0, 5, DataType::Error, &[0x00, 0x00, 0x00, 0x01]).unwrap();
        assert_eq!(msg.msg_type, MessageType::Emergency);
        assert_eq!(msg.as_u32(), Some(1));
    }

    #[test]
    fn test_new_nod_invalid_id() {
        assert_eq!(
            CanaMessage::new_nod(2032, 1, DataType::Float, &[0; 4]),
            Err(BusError::InvalidId)
        );
    }

    #[test]
    fn test_new_nod_payload_too_large() {
        assert_eq!(
            CanaMessage::new_nod(200, 1, DataType::Float, &[0; 5]),
            Err(BusError::PayloadTooLarge)
        );
    }

    // -----------------------------------------------------------------------
    // CanaMessage NSS tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_nss_identification() {
        let msg = CanaMessage::new_nss(300, 10, ServiceCode::Identification, 0x00, &[]).unwrap();
        assert_eq!(msg.msg_type, MessageType::NssHigh);
        assert_eq!(msg.service_code, ServiceCode::Identification.raw());
    }

    #[test]
    fn test_new_nss_data_download() {
        let msg = CanaMessage::new_nss(
            500,
            20,
            ServiceCode::DataDownload,
            0x01,
            &[0xAA, 0xBB, 0xCC, 0xDD],
        )
        .unwrap();
        assert_eq!(msg.service_code, ServiceCode::DataDownload.raw());
        assert_eq!(msg.as_u32(), Some(0xAABBCCDD));
    }

    // -----------------------------------------------------------------------
    // CAN frame encode/decode roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_nod_float() {
        let speed: f32 = 250.75;
        let msg = CanaMessage::new_nod(
            params::AIRSPEED_TAS,
            1,
            DataType::Float,
            &speed.to_be_bytes(),
        )
        .unwrap();

        let frame = msg.to_can_frame().unwrap();
        assert_eq!(frame.id, params::AIRSPEED_TAS);
        assert_eq!(frame.dlc(), 8); // 4 header + 4 data

        let decoded = CanaMessage::from_can_frame(&frame).unwrap();
        assert_eq!(decoded.node_id, 1);
        assert_eq!(decoded.data_type, DataType::Float.raw());
        assert_eq!(decoded.as_f32(), Some(250.75));
    }

    #[test]
    fn test_roundtrip_nod_short_data() {
        let msg = CanaMessage::new_nod(params::HEADING, 3, DataType::Short, &180i16.to_be_bytes())
            .unwrap();

        let frame = msg.to_can_frame().unwrap();
        let decoded = CanaMessage::from_can_frame(&frame).unwrap();
        assert_eq!(decoded.as_i16(), Some(180));
    }

    #[test]
    fn test_roundtrip_nss() {
        let msg =
            CanaMessage::new_nss(400, 15, ServiceCode::ModuleConfig, 0x42, &[0x01, 0x02]).unwrap();

        let frame = msg.to_can_frame().unwrap();
        let decoded = CanaMessage::from_can_frame(&frame).unwrap();
        assert_eq!(decoded.node_id, 15);
        assert_eq!(decoded.service_code, ServiceCode::ModuleConfig.raw());
        assert_eq!(decoded.message_code, 0x42);
        assert_eq!(&decoded.data, &[0x01, 0x02]);
    }

    #[test]
    fn test_roundtrip_empty_data() {
        let msg = CanaMessage::new_nod(params::ROLL, 7, DataType::NoData, &[]).unwrap();

        let frame = msg.to_can_frame().unwrap();
        assert_eq!(frame.dlc(), 4); // 4 header only

        let decoded = CanaMessage::from_can_frame(&frame).unwrap();
        assert_eq!(decoded.node_id, 7);
        assert!(decoded.data.is_empty());
    }

    // -----------------------------------------------------------------------
    // from_can_frame error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_can_frame_too_short() {
        let frame = CanFrame::new_standard(200, &[0x01, 0x02, 0x03]).unwrap();
        assert_eq!(
            CanaMessage::from_can_frame(&frame),
            Err(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_from_can_frame_invalid_id() {
        // CAN ID 2032+ is invalid for CANaerospace
        let frame = CanFrame::new_standard(0x7FF, &[0; 4]).unwrap();
        assert_eq!(
            CanaMessage::from_can_frame(&frame),
            Err(BusError::InvalidId)
        );
    }

    // -----------------------------------------------------------------------
    // Data extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_as_f32_none_for_short_data() {
        let msg = CanaMessage::new_nod(200, 1, DataType::Short, &[0x00, 0x01]).unwrap();
        assert!(msg.as_f32().is_none());
    }

    #[test]
    fn test_as_u32_none_for_short_data() {
        let msg = CanaMessage::new_nod(200, 1, DataType::BChar, &[0x42]).unwrap();
        assert!(msg.as_u32().is_none());
    }

    #[test]
    fn test_as_i16_none_for_short_data() {
        let msg = CanaMessage::new_nod(200, 1, DataType::BChar, &[0x42]).unwrap();
        assert!(msg.as_i16().is_none());
    }

    #[test]
    fn test_as_i16_negative() {
        let msg = CanaMessage::new_nod(200, 1, DataType::Short, &(-100i16).to_be_bytes()).unwrap();
        assert_eq!(msg.as_i16(), Some(-100));
    }

    // -----------------------------------------------------------------------
    // Parameter constants tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_standard_params_in_nod_range() {
        let nod_params = [
            params::AIRSPEED_IAS,
            params::AIRSPEED_TAS,
            params::ALTITUDE_BARO,
            params::ALTITUDE_GPS,
            params::HEADING,
            params::PITCH,
            params::ROLL,
            params::YAW_RATE,
            params::ACCEL_X,
            params::ACCEL_Y,
            params::ACCEL_Z,
        ];
        for &id in &nod_params {
            assert_eq!(MessageType::from_can_id(id), Ok(MessageType::NodNormal));
        }
    }

    #[test]
    fn test_engine_params_in_nss_range() {
        let engine_params = [params::ENGINE_RPM, params::OIL_PRESSURE, params::OIL_TEMP];
        for &id in &engine_params {
            assert_eq!(MessageType::from_can_id(id), Ok(MessageType::NssHigh));
        }
    }
}
