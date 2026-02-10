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
