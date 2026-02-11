// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire (ECSS-E-ST-50-12C) packet codec and link state machine.
//!
//! Implements SpaceWire packet encode/decode with EOP/EEP markers,
//! the 6-state link interface state machine, time-code distribution,
//! and RMAP (Remote Memory Access Protocol).

pub mod adapter;
pub mod hw;
pub mod link;
pub mod packet;
pub mod rmap;
pub mod speed;
pub mod timecode;

pub use adapter::SpwZenohAdapter;
pub use link::{LinkEvent, LinkState, LinkStateMachine};
pub use packet::{EopMarker, SpaceWirePacket};
pub use rmap::{RmapCommand, RmapCommandType, RmapReply};
pub use speed::LinkSpeed;
pub use timecode::TimeCodeManager;
