// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 avionics data bus protocol implementation.
//!
//! Clean-room implementation from the publicly available ARINC 429
//! specification. Supports:
//! - 32-bit word encode/decode (label, SDI, data, SSM, parity)
//! - BNR (Binary) data format
//! - BCD (Binary Coded Decimal) data format
//! - Discrete data words
//! - Label-based receive filtering
//! - Fixed-rate transmit scheduling
//!
//! # ARINC 429 Word Format (32 bits)
//!
//! ```text
//! Bit 32 (MSB): Parity (odd)
//! Bits 31-30:   SSM (Sign/Status Matrix)
//! Bits 29-11:   Data field (19 bits)
//! Bits 10-9:    SDI (Source/Destination Identifier)
//! Bits 8-1:     Label (8 bits, transmitted LSB first / reversed bit order)
//! ```
//!
//! Note: ARINC 429 uses 1-based bit numbering (bit 1 = LSB, bit 32 = MSB).
//! In our u32 representation, bit 1 maps to bit 0 (LSB) of the u32.

pub mod adapter;
pub mod bcd;
pub mod bnr;
pub mod channel;
pub mod discrete;
pub mod filter;
pub mod hw;
pub mod scheduler;
pub mod word;

pub use adapter::Arinc429ZenohAdapter;
pub use bcd::BcdCodec;
pub use bnr::BnrCodec;
pub use channel::{BusSpeed, ChannelConfig};
pub use discrete::DiscreteWord;
pub use filter::LabelFilter;
pub use scheduler::TxScheduler;
pub use word::{Arinc429Word, Label, Sdi, Ssm};
