// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! I2S/TDM audio peripheral drivers.
//!
//! Provides platform-specific I2S controller implementations.
//! All drivers implement the [`I2sController`] trait from `smallaios_kernel::hal`.

#![allow(dead_code)]

pub use smallaios_kernel::hal::{
    AudioBuffer, HalError, I2sBitDepth, I2sConfig, I2sController, I2sIrqSource, I2sMode, I2sRole,
};
