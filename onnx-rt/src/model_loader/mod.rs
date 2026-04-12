// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace-style model loader for SmallAIOS.
//!
//! Loads models distributed in the safetensors format (single-file
//! or sharded with a `model.safetensors.index.json` manifest). The
//! loader memory-maps each shard and exposes zero-copy views into
//! tensor payloads.
//!
//! This module is gated behind the `safetensors` feature and is only
//! compiled when `std` is available, since it relies on `std::fs` and
//! `memmap2::Mmap` for file access.

use alloc::string::String;
use core::fmt;

pub mod config;
pub mod safetensors;
pub mod sharded;

pub use config::{GemmaConfig, ModelArchitecture};
pub use safetensors::{SafetensorsFile, TensorEntry, TensorView};
pub use sharded::MultiShardSafetensors;

/// Errors returned by the safetensors model loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    /// Underlying filesystem / mmap failure.
    Io(String),
    /// The safetensors header (length prefix or JSON body) is malformed.
    InvalidHeader(String),
    /// A tensor entry declares a dtype we do not currently map to
    /// `onnx-rt` `DataType`.
    UnsupportedDtype(String),
    /// A lookup was attempted for a tensor name that does not exist
    /// in the file or in any shard of a multi-shard model.
    TensorNotFound(String),
    /// A sharded index referenced a shard file that could not be
    /// resolved at load time.
    MissingShard(String),
    /// A HuggingFace `config.json` failed to parse or validate
    /// (missing required field, wrong type, or inconsistent values).
    InvalidConfig(String),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoaderError::Io(msg) => write!(f, "safetensors io error: {msg}"),
            LoaderError::InvalidHeader(msg) => write!(f, "invalid safetensors header: {msg}"),
            LoaderError::UnsupportedDtype(dt) => write!(f, "unsupported safetensors dtype: {dt}"),
            LoaderError::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            LoaderError::MissingShard(name) => write!(f, "missing safetensors shard: {name}"),
            LoaderError::InvalidConfig(msg) => write!(f, "invalid config.json: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LoaderError {}
