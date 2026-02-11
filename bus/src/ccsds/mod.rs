// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS Space Packet Protocol (CCSDS 133.0-B).
//!
//! Implements Space Packet encode/decode with primary header, TM/TC framing,
//! and APID-based routing.

pub mod adapter;
pub mod cltu;
pub mod packet;
pub mod router;
pub mod tc_frame;
pub mod test_vectors;
pub mod tm_frame;

pub use adapter::CcsdsZenohAdapter;
pub use cltu::{cltu_encode, cltu_decode};
pub use packet::{PacketType, SeqFlags, SpacePacket};
pub use router::ApidRouter;
pub use tc_frame::TcFrame;
pub use tm_frame::TmFrame;
