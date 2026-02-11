// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B Military Avionics Data Bus.
//!
//! Implements command, status, and data word codecs, bus controller (BC)
//! message sequencing, and remote terminal (RT) response state machine.

pub mod adapter;
pub mod bc;
pub mod dual_bus;
pub mod hw;
pub mod mode_code;
pub mod rt;
pub mod word;

pub use adapter::Mil1553ZenohAdapter;
pub use bc::{BcMessage, BcSequencer, MessageType};
pub use dual_bus::{BusHealth, DualBusManager};
pub use mode_code::{ModeCode, ModeCodeHandler, ModeCodeResult};
pub use rt::{RtState, RtTerminal};
pub use word::{CommandWord, DataWord, StatusWord, SyncType};
