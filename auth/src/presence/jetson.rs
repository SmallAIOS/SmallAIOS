// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AArch64 / Jetson Tegra234 GPIO presence provider.
//!
//! Tegra234 routes GPIO through the Pinmux + GPIO controller; line ids
//! follow the Jetson Orin pinout numbering. The actual MMIO read is
//! stubbed for v1 (the line-id field is recorded so the audit log can
//! identify which line was consulted).

use super::PhysicalPresenceProvider;

/// Jetson / AArch64 GPIO presence provider.
pub struct PresenceJetsonGpio {
    assert: bool,
    line_id: u32,
}

impl PresenceJetsonGpio {
    /// Construct a provider for `line_id` with the given asserted state.
    pub const fn new(line_id: u32, assert: bool) -> Self {
        Self { assert, line_id }
    }

    /// Borrow the GPIO line id.
    pub fn line_id(&self) -> u32 {
        self.line_id
    }
}

impl PhysicalPresenceProvider for PresenceJetsonGpio {
    fn name(&self) -> &'static str {
        "jetson-gpio"
    }

    fn asserted(&self) -> bool {
        // Production: read Tegra234 GPIO controller MMIO.
        self.assert
    }
}
