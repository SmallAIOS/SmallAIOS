// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN bus protocol implementation.
//!
//! Clean-room implementation from ISO 11898 and BOSCH CAN 2.0 specification.
//! Supports:
//! - CAN 2.0A standard frames (11-bit ID, 0-8 byte payload)
//! - CAN 2.0B extended frames (29-bit ID)
//! - CAN FD frames (up to 64-byte payload, BRS, ESI)
//!
//! All frame types use the appropriate CRC per specification:
//! - CAN 2.0A/B: CRC-15 with polynomial 0x4599
//! - CAN FD (payload <= 16 bytes): CRC-17 with polynomial 0x3685B
//! - CAN FD (payload > 16 bytes): CRC-21 with polynomial 0x302899

pub mod adapter;
pub mod axi_can;
pub mod canaerospace;
pub mod controller;
pub mod crc;
pub mod fd;
pub mod filter;
pub mod frame;
pub mod loopback;
pub mod mcp2515;
pub mod state;

pub use adapter::CanZenohAdapter;
pub use canaerospace::{CanaMessage, DataType, MessageType, ServiceCode};
pub use controller::{CanBitTiming, CanController, CanMode, CanStats, MockCanController};
pub use crc::{crc15, crc17, crc21};
pub use fd::{CanFdFrame, FdDlc};
pub use filter::{AcceptanceFilter, HwFilter, SwFilter};
pub use frame::{CanFrame, FrameType};
pub use state::{BusEvent, BusState, BusStateMachine, StateChange};
