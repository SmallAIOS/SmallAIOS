// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Container Interface
//!
//! Top-level entry point that wires all subsystems together:
//! - Configuration loading (TOML + environment variables)
//! - Boot sequence orchestration
//! - Health check and readiness endpoints
//! - Prometheus metrics exporter
//! - Minimal HTTP handler for Kubernetes probes
//! - Graceful shutdown (SIGTERM handling)

#![no_std]
#![allow(dead_code)]

extern crate alloc;

pub mod boot;
pub mod config;
pub mod health;
pub mod image;
pub mod metrics;
pub mod shutdown;

use alloc::string::String;
use core::fmt;

/// Errors returned by the container subsystem.
#[derive(Clone, Debug, PartialEq)]
pub enum ContainerError {
    /// Configuration is invalid.
    InvalidConfig(String),
    /// Boot sequence failed.
    BootFailed(String),
    /// Boot sequence already completed.
    AlreadyReady,
    /// Named resource not found.
    NotFound(String),
    /// Registration table is full.
    TableFull,
    /// Shutdown already initiated.
    AlreadyInitiated,
    /// Shutdown sequence failed.
    ShutdownFailed(String),
    /// OCI image manifest is invalid.
    InvalidManifest(String),
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            ContainerError::BootFailed(msg) => write!(f, "boot failed: {}", msg),
            ContainerError::AlreadyReady => write!(f, "boot sequence already completed"),
            ContainerError::NotFound(name) => write!(f, "not found: {}", name),
            ContainerError::TableFull => write!(f, "registration table full"),
            ContainerError::AlreadyInitiated => write!(f, "shutdown already initiated"),
            ContainerError::ShutdownFailed(msg) => write!(f, "shutdown failed: {}", msg),
            ContainerError::InvalidManifest(msg) => write!(f, "invalid manifest: {}", msg),
        }
    }
}
