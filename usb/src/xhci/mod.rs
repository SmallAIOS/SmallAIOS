// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! xHCI (eXtensible Host Controller Interface) driver.
//!
//! Implements USB 3.x host controller support via PCIe-discovered xHCI
//! controllers. Handles device context management, command/transfer/event
//! rings, and port management.

pub mod trb;
pub mod rings;
pub mod registers;
pub mod controller;
