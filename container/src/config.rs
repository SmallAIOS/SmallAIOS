// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Configuration loading and management.

#![allow(dead_code)]

use alloc::string::String;

use crate::ContainerError;

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Logging severity level.
#[derive(Clone, Debug, PartialEq)]
pub enum LogLevel {
    /// Only fatal / unrecoverable errors.
    Error,
    /// Warnings that may need attention.
    Warn,
    /// Informational messages (default).
    Info,
    /// Verbose developer diagnostics.
    Debug,
    /// Maximum verbosity — every internal event.
    Trace,
}

impl LogLevel {
    /// Parse a log level from a string (case-insensitive).
    ///
    /// Returns `None` if the string does not match any known level.
    pub fn parse(s: &str) -> Option<LogLevel> {
        // Manual ASCII-lowercase comparison to avoid pulling in Unicode tables.
        let mut buf = [0u8; 8];
        let len = s.len().min(buf.len());
        for (i, &b) in s.as_bytes()[..len].iter().enumerate() {
            buf[i] = if b.is_ascii_uppercase() { b | 0x20 } else { b };
        }
        let lower = core::str::from_utf8(&buf[..len]).ok()?;
        match lower {
            "error" => Some(LogLevel::Error),
            "warn" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NetworkConfig
// ---------------------------------------------------------------------------

/// Network listener configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct NetworkConfig {
    /// Address to bind the listener to.
    pub listen_addr: String,
    /// TCP port number.
    pub listen_port: u16,
    /// Whether TLS is enabled.
    pub enable_tls: bool,
}

// ---------------------------------------------------------------------------
// InferenceConfig
// ---------------------------------------------------------------------------

/// ONNX inference engine configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceConfig {
    /// Maximum number of loaded ONNX models.
    pub max_models: u32,
    /// Maximum batch size per inference request.
    pub max_batch_size: u32,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Whether GPU acceleration is enabled.
    pub enable_gpu: bool,
}

// ---------------------------------------------------------------------------
// MetricsConfig
// ---------------------------------------------------------------------------

/// Prometheus-compatible metrics exporter configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled.
    pub enabled: bool,
    /// TCP port for the metrics endpoint.
    pub port: u16,
    /// HTTP path for scraping (e.g. "/metrics").
    pub path: String,
}

// ---------------------------------------------------------------------------
// ContainerConfig
// ---------------------------------------------------------------------------

/// Top-level container configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct ContainerConfig {
    /// Human-readable container name.
    pub name: String,
    /// Crate version string.
    pub version: &'static str,
    /// Active log level.
    pub log_level: LogLevel,
    /// Network settings.
    pub network: NetworkConfig,
    /// Inference engine settings.
    pub inference: InferenceConfig,
    /// Metrics exporter settings.
    pub metrics: MetricsConfig,
    /// Maximum heap memory in megabytes.
    pub max_memory_mb: u64,
    /// Number of worker tasks.
    pub num_workers: u32,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        ContainerConfig {
            name: String::from("smallaios"),
            version: "0.1.0",
            log_level: LogLevel::Info,
            network: NetworkConfig {
                listen_addr: String::from("0.0.0.0"),
                listen_port: 8080,
                enable_tls: false,
            },
            inference: InferenceConfig {
                max_models: 16,
                max_batch_size: 32,
                timeout_ms: 30_000,
                enable_gpu: false,
            },
            metrics: MetricsConfig {
                enabled: true,
                port: 9090,
                path: String::from("/metrics"),
            },
            max_memory_mb: 512,
            num_workers: 4,
        }
    }
}

impl ContainerConfig {
    /// Enable GPU acceleration.
    pub fn with_gpu(&mut self) -> &mut Self {
        self.inference.enable_gpu = true;
        self
    }

    /// Enable TLS on the network listener.
    pub fn with_tls(&mut self) -> &mut Self {
        self.network.enable_tls = true;
        self
    }

    /// Validate the configuration, returning an error describing the first
    /// invalid field found.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.network.listen_port == 0 {
            return Err(ContainerError::InvalidConfig(String::from(
                "listen_port must be > 0",
            )));
        }
        if self.inference.max_models == 0 {
            return Err(ContainerError::InvalidConfig(String::from(
                "max_models must be > 0",
            )));
        }
        if self.num_workers == 0 {
            return Err(ContainerError::InvalidConfig(String::from(
                "num_workers must be > 0",
            )));
        }
        if self.max_memory_mb == 0 {
            return Err(ContainerError::InvalidConfig(String::from(
                "max_memory_mb must be > 0",
            )));
        }
        Ok(())
    }
}

/// Override a single configuration field from an environment-variable-style
/// key/value pair.
///
/// Recognised keys:
/// - `SMALLAIOS_PORT` — sets `network.listen_port`
/// - `SMALLAIOS_LOG_LEVEL` — sets `log_level`
/// - `SMALLAIOS_WORKERS` — sets `num_workers`
/// - `SMALLAIOS_GPU` — sets `inference.enable_gpu` ("true" / "false")
/// - `SMALLAIOS_MAX_MEMORY` — sets `max_memory_mb`
///
/// Unknown keys are silently ignored.
pub fn parse_env_override(config: &mut ContainerConfig, key: &str, value: &str) {
    match key {
        "SMALLAIOS_PORT" => {
            if let Some(port) = parse_u16(value) {
                config.network.listen_port = port;
            }
        }
        "SMALLAIOS_LOG_LEVEL" => {
            if let Some(level) = LogLevel::parse(value) {
                config.log_level = level;
            }
        }
        "SMALLAIOS_WORKERS" => {
            if let Some(w) = parse_u32(value) {
                config.num_workers = w;
            }
        }
        "SMALLAIOS_GPU" => {
            config.inference.enable_gpu = value == "true";
        }
        "SMALLAIOS_MAX_MEMORY" => {
            if let Some(m) = parse_u64(value) {
                config.max_memory_mb = m;
            }
        }
        _ => { /* unknown key — ignore */ }
    }
}

// ---------------------------------------------------------------------------
// Tiny integer parsers (no_std — no `str::parse`)
// ---------------------------------------------------------------------------

fn parse_u16(s: &str) -> Option<u16> {
    let mut n: u16 = 0;
    if s.is_empty() {
        return None;
    }
    for &b in s.as_bytes() {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(d as u16)?;
    }
    Some(n)
}

fn parse_u32(s: &str) -> Option<u32> {
    let mut n: u32 = 0;
    if s.is_empty() {
        return None;
    }
    for &b in s.as_bytes() {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(d as u32)?;
    }
    Some(n)
}

fn parse_u64(s: &str) -> Option<u64> {
    let mut n: u64 = 0;
    if s.is_empty() {
        return None;
    }
    for &b in s.as_bytes() {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(d as u64)?;
    }
    Some(n)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Default config values ---------------------------------------------

    #[test]
    fn test_default_config_name() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.name, "smallaios");
    }

    #[test]
    fn test_default_config_version() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.version, "0.1.0");
    }

    #[test]
    fn test_default_config_log_level() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.log_level, LogLevel::Info);
    }

    #[test]
    fn test_default_config_workers() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.num_workers, 4);
    }

    #[test]
    fn test_default_config_max_memory() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.max_memory_mb, 512);
    }

    // -- Validate ----------------------------------------------------------

    #[test]
    fn test_validate_valid_config() {
        let cfg = ContainerConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_zero_port() {
        let mut cfg = ContainerConfig::default();
        cfg.network.listen_port = 0;
        assert_eq!(
            cfg.validate(),
            Err(ContainerError::InvalidConfig(String::from(
                "listen_port must be > 0"
            )))
        );
    }

    #[test]
    fn test_validate_zero_workers() {
        let mut cfg = ContainerConfig::default();
        cfg.num_workers = 0;
        assert_eq!(
            cfg.validate(),
            Err(ContainerError::InvalidConfig(String::from(
                "num_workers must be > 0"
            )))
        );
    }

    #[test]
    fn test_validate_zero_max_models() {
        let mut cfg = ContainerConfig::default();
        cfg.inference.max_models = 0;
        assert_eq!(
            cfg.validate(),
            Err(ContainerError::InvalidConfig(String::from(
                "max_models must be > 0"
            )))
        );
    }

    #[test]
    fn test_validate_zero_max_memory() {
        let mut cfg = ContainerConfig::default();
        cfg.max_memory_mb = 0;
        assert_eq!(
            cfg.validate(),
            Err(ContainerError::InvalidConfig(String::from(
                "max_memory_mb must be > 0"
            )))
        );
    }

    // -- Builder helpers ---------------------------------------------------

    #[test]
    fn test_with_gpu_enables_gpu() {
        let mut cfg = ContainerConfig::default();
        assert!(!cfg.inference.enable_gpu);
        cfg.with_gpu();
        assert!(cfg.inference.enable_gpu);
    }

    #[test]
    fn test_with_tls_enables_tls() {
        let mut cfg = ContainerConfig::default();
        assert!(!cfg.network.enable_tls);
        cfg.with_tls();
        assert!(cfg.network.enable_tls);
    }

    // -- Environment overrides ---------------------------------------------

    #[test]
    fn test_env_override_port() {
        let mut cfg = ContainerConfig::default();
        parse_env_override(&mut cfg, "SMALLAIOS_PORT", "3000");
        assert_eq!(cfg.network.listen_port, 3000);
    }

    #[test]
    fn test_env_override_log_level() {
        let mut cfg = ContainerConfig::default();
        parse_env_override(&mut cfg, "SMALLAIOS_LOG_LEVEL", "debug");
        assert_eq!(cfg.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_env_override_workers() {
        let mut cfg = ContainerConfig::default();
        parse_env_override(&mut cfg, "SMALLAIOS_WORKERS", "8");
        assert_eq!(cfg.num_workers, 8);
    }

    #[test]
    fn test_env_override_gpu() {
        let mut cfg = ContainerConfig::default();
        parse_env_override(&mut cfg, "SMALLAIOS_GPU", "true");
        assert!(cfg.inference.enable_gpu);
    }

    #[test]
    fn test_env_override_max_memory() {
        let mut cfg = ContainerConfig::default();
        parse_env_override(&mut cfg, "SMALLAIOS_MAX_MEMORY", "2048");
        assert_eq!(cfg.max_memory_mb, 2048);
    }

    #[test]
    fn test_env_override_unknown_key_ignored() {
        let mut cfg = ContainerConfig::default();
        let before = cfg.clone();
        parse_env_override(&mut cfg, "UNKNOWN_KEY", "whatever");
        assert_eq!(cfg, before);
    }

    // -- LogLevel::from_str ------------------------------------------------

    #[test]
    fn test_log_level_from_str_all_variants() {
        assert_eq!(LogLevel::parse("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("trace"), Some(LogLevel::Trace));
    }

    #[test]
    fn test_log_level_from_str_case_insensitive() {
        assert_eq!(LogLevel::parse("ERROR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("Warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("Debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("TRACE"), Some(LogLevel::Trace));
    }

    #[test]
    fn test_log_level_from_str_invalid() {
        assert_eq!(LogLevel::parse("verbose"), None);
        assert_eq!(LogLevel::parse(""), None);
        assert_eq!(LogLevel::parse("123"), None);
    }

    // -- Sub-config defaults -----------------------------------------------

    #[test]
    fn test_network_config_defaults() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.network.listen_addr, "0.0.0.0");
        assert_eq!(cfg.network.listen_port, 8080);
        assert!(!cfg.network.enable_tls);
    }

    #[test]
    fn test_inference_config_defaults() {
        let cfg = ContainerConfig::default();
        assert_eq!(cfg.inference.max_models, 16);
        assert_eq!(cfg.inference.max_batch_size, 32);
        assert_eq!(cfg.inference.timeout_ms, 30_000);
        assert!(!cfg.inference.enable_gpu);
    }

    #[test]
    fn test_metrics_config_defaults() {
        let cfg = ContainerConfig::default();
        assert!(cfg.metrics.enabled);
        assert_eq!(cfg.metrics.port, 9090);
        assert_eq!(cfg.metrics.path, "/metrics");
    }

    // -- Builder chaining --------------------------------------------------

    #[test]
    fn test_builder_chaining() {
        let mut cfg = ContainerConfig::default();
        cfg.with_gpu().with_tls();
        assert!(cfg.inference.enable_gpu);
        assert!(cfg.network.enable_tls);
    }

    // -- Env override GPU false -------------------------------------------

    #[test]
    fn test_env_override_gpu_false() {
        let mut cfg = ContainerConfig::default();
        cfg.inference.enable_gpu = true;
        parse_env_override(&mut cfg, "SMALLAIOS_GPU", "false");
        assert!(!cfg.inference.enable_gpu);
    }
}
