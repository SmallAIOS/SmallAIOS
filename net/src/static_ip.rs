// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Static IP configuration for SmallAIOS.
//!
//! Provides compile-time and runtime static IP address configuration.
//! When the `static-ip` feature is enabled, a static configuration can
//! be applied to the network interface, bypassing DHCP entirely.
//!
//! Also provides [`InterfaceConfig`] for choosing between static and
//! DHCP configuration at runtime.

use core::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// StaticIpConfig
// ---------------------------------------------------------------------------

/// Static IPv4 (and optional IPv6) network configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticIpConfig {
    /// IPv4 address.
    pub ipv4_addr: [u8; 4],
    /// Subnet mask.
    pub subnet_mask: [u8; 4],
    /// Default gateway.
    pub gateway: [u8; 4],
    /// DNS server address.
    pub dns: [u8; 4],
    /// Optional IPv6 address (128-bit).
    pub ipv6_addr: Option<[u8; 16]>,
}

impl StaticIpConfig {
    /// Create a new static IPv4 configuration.
    pub const fn new(
        ipv4_addr: [u8; 4],
        subnet_mask: [u8; 4],
        gateway: [u8; 4],
        dns: [u8; 4],
    ) -> Self {
        StaticIpConfig {
            ipv4_addr,
            subnet_mask,
            gateway,
            dns,
            ipv6_addr: None,
        }
    }

    /// Create a new static configuration with both IPv4 and IPv6 addresses.
    pub const fn with_ipv6(
        ipv4_addr: [u8; 4],
        subnet_mask: [u8; 4],
        gateway: [u8; 4],
        dns: [u8; 4],
        ipv6_addr: [u8; 16],
    ) -> Self {
        StaticIpConfig {
            ipv4_addr,
            subnet_mask,
            gateway,
            dns,
            ipv6_addr: Some(ipv6_addr),
        }
    }

    /// Validate that the configuration has non-zero addresses.
    pub fn is_valid(&self) -> bool {
        // IP address must not be all-zeros or broadcast
        self.ipv4_addr != [0, 0, 0, 0]
            && self.ipv4_addr != [255, 255, 255, 255]
            && self.subnet_mask != [0, 0, 0, 0]
    }
}

// ---------------------------------------------------------------------------
// Compile-time defaults (behind feature flag)
// ---------------------------------------------------------------------------

/// Default static IPv4 address (192.168.1.100).
///
/// Override at compile time by providing a custom [`StaticIpConfig`].
#[cfg(feature = "static-ip")]
pub const DEFAULT_IPV4_ADDR: [u8; 4] = [192, 168, 1, 100];

/// Default subnet mask (255.255.255.0).
#[cfg(feature = "static-ip")]
pub const DEFAULT_SUBNET_MASK: [u8; 4] = [255, 255, 255, 0];

/// Default gateway (192.168.1.1).
#[cfg(feature = "static-ip")]
pub const DEFAULT_GATEWAY: [u8; 4] = [192, 168, 1, 1];

/// Default DNS server (8.8.8.8).
#[cfg(feature = "static-ip")]
pub const DEFAULT_DNS: [u8; 4] = [8, 8, 8, 8];

/// Compile-time default static IP configuration.
#[cfg(feature = "static-ip")]
pub const DEFAULT_STATIC_CONFIG: StaticIpConfig = StaticIpConfig::new(
    DEFAULT_IPV4_ADDR,
    DEFAULT_SUBNET_MASK,
    DEFAULT_GATEWAY,
    DEFAULT_DNS,
);

// ---------------------------------------------------------------------------
// InterfaceConfig
// ---------------------------------------------------------------------------

/// Network interface configuration mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceConfig {
    /// Use a static IP configuration.
    Static(StaticIpConfig),
    /// Use DHCP to obtain an IP address dynamically.
    Dhcp,
}

/// Resolve the interface configuration.
///
/// If a static configuration is provided, it takes precedence over DHCP.
/// Returns [`InterfaceConfig::Static`] if `static_cfg` is `Some`, otherwise
/// returns [`InterfaceConfig::Dhcp`].
pub fn resolve_config(static_cfg: Option<StaticIpConfig>) -> InterfaceConfig {
    match static_cfg {
        Some(cfg) => InterfaceConfig::Static(cfg),
        None => InterfaceConfig::Dhcp,
    }
}

// ---------------------------------------------------------------------------
// Runtime storage (global static state)
// ---------------------------------------------------------------------------

/// Whether a static configuration has been applied.
static CONFIG_APPLIED: AtomicBool = AtomicBool::new(false);

/// Global storage for the applied static IPv4 configuration.
///
/// Protected by `CONFIG_APPLIED` flag. In a real kernel this would use
/// proper synchronization; here we use simple atomic flag + fixed storage.
static mut STORED_CONFIG: StaticIpConfig = StaticIpConfig {
    ipv4_addr: [0; 4],
    subnet_mask: [0; 4],
    gateway: [0; 4],
    dns: [0; 4],
    ipv6_addr: None,
};

/// Whether an IPv6 address has been applied.
static IPV6_APPLIED: AtomicBool = AtomicBool::new(false);

/// Global storage for the applied static IPv6 address.
static mut STORED_IPV6: [u8; 16] = [0u8; 16];

/// Apply a static IPv4 configuration.
///
/// Stores the configuration in global state for later retrieval.
/// In a full kernel, this would configure the network interface hardware.
pub fn apply_static_config(cfg: &StaticIpConfig) {
    // SAFETY: Single-threaded kernel context. In production, this would be
    // protected by a spinlock or run during init before threading starts.
    unsafe {
        STORED_CONFIG = *cfg;
    }
    CONFIG_APPLIED.store(true, Ordering::Release);
}

/// Apply a static IPv6 address.
///
/// Stores the address in global state for later retrieval.
pub fn apply_static_ipv6(addr: [u8; 16]) {
    // SAFETY: Same single-threaded assumption as apply_static_config.
    unsafe {
        STORED_IPV6 = addr;
    }
    IPV6_APPLIED.store(true, Ordering::Release);
}

/// Retrieve the currently applied static configuration, if any.
pub fn get_applied_config() -> Option<StaticIpConfig> {
    if CONFIG_APPLIED.load(Ordering::Acquire) {
        // SAFETY: Protected by atomic flag.
        Some(unsafe { STORED_CONFIG })
    } else {
        None
    }
}

/// Retrieve the currently applied static IPv6 address, if any.
pub fn get_applied_ipv6() -> Option<[u8; 16]> {
    if IPV6_APPLIED.load(Ordering::Acquire) {
        // SAFETY: Protected by atomic flag.
        Some(unsafe { STORED_IPV6 })
    } else {
        None
    }
}

/// Clear the applied configuration (useful for testing or reconfiguration).
pub fn clear_config() {
    CONFIG_APPLIED.store(false, Ordering::Release);
    IPV6_APPLIED.store(false, Ordering::Release);
}

// ---------------------------------------------------------------------------
// StaticIpError
// ---------------------------------------------------------------------------

/// Errors for static IP configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticIpError {
    /// The provided configuration is invalid.
    InvalidConfig,
    /// A configuration has already been applied.
    AlreadyConfigured,
}

impl core::fmt::Display for StaticIpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StaticIpError::InvalidConfig => write!(f, "static-ip: invalid configuration"),
            StaticIpError::AlreadyConfigured => write!(f, "static-ip: already configured"),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Reset global state between tests
    fn reset_state() {
        clear_config();
    }

    // -----------------------------------------------------------------------
    // StaticIpConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_static_config_new() {
        let cfg = StaticIpConfig::new(
            [192, 168, 1, 100],
            [255, 255, 255, 0],
            [192, 168, 1, 1],
            [8, 8, 8, 8],
        );
        assert_eq!(cfg.ipv4_addr, [192, 168, 1, 100]);
        assert_eq!(cfg.subnet_mask, [255, 255, 255, 0]);
        assert_eq!(cfg.gateway, [192, 168, 1, 1]);
        assert_eq!(cfg.dns, [8, 8, 8, 8]);
        assert_eq!(cfg.ipv6_addr, None);
    }

    #[test]
    fn test_static_config_with_ipv6() {
        let ipv6 = [0xFE, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let cfg = StaticIpConfig::with_ipv6(
            [10, 0, 0, 50],
            [255, 255, 255, 0],
            [10, 0, 0, 1],
            [8, 8, 4, 4],
            ipv6,
        );
        assert_eq!(cfg.ipv4_addr, [10, 0, 0, 50]);
        assert_eq!(cfg.ipv6_addr, Some(ipv6));
    }

    #[test]
    fn test_static_config_validation() {
        let valid = StaticIpConfig::new(
            [192, 168, 1, 100],
            [255, 255, 255, 0],
            [192, 168, 1, 1],
            [8, 8, 8, 8],
        );
        assert!(valid.is_valid());

        let zero_ip = StaticIpConfig::new(
            [0, 0, 0, 0],
            [255, 255, 255, 0],
            [192, 168, 1, 1],
            [8, 8, 8, 8],
        );
        assert!(!zero_ip.is_valid());

        let broadcast = StaticIpConfig::new(
            [255, 255, 255, 255],
            [255, 255, 255, 0],
            [192, 168, 1, 1],
            [8, 8, 8, 8],
        );
        assert!(!broadcast.is_valid());

        let zero_mask = StaticIpConfig::new(
            [192, 168, 1, 100],
            [0, 0, 0, 0],
            [192, 168, 1, 1],
            [8, 8, 8, 8],
        );
        assert!(!zero_mask.is_valid());
    }

    // -----------------------------------------------------------------------
    // Compile-time defaults
    // -----------------------------------------------------------------------

    #[cfg(feature = "static-ip")]
    #[test]
    fn test_from_compiled_defaults() {
        assert_eq!(DEFAULT_IPV4_ADDR, [192, 168, 1, 100]);
        assert_eq!(DEFAULT_SUBNET_MASK, [255, 255, 255, 0]);
        assert_eq!(DEFAULT_GATEWAY, [192, 168, 1, 1]);
        assert_eq!(DEFAULT_DNS, [8, 8, 8, 8]);
        assert!(DEFAULT_STATIC_CONFIG.is_valid());
    }

    // -----------------------------------------------------------------------
    // InterfaceConfig / resolve_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_config_static_takes_precedence() {
        let cfg = StaticIpConfig::new(
            [10, 0, 0, 50],
            [255, 255, 255, 0],
            [10, 0, 0, 1],
            [8, 8, 8, 8],
        );
        let result = resolve_config(Some(cfg));
        assert_eq!(result, InterfaceConfig::Static(cfg));
    }

    #[test]
    fn test_resolve_config_dhcp_fallback() {
        let result = resolve_config(None);
        assert_eq!(result, InterfaceConfig::Dhcp);
    }

    #[test]
    fn test_interface_config_eq() {
        let cfg = StaticIpConfig::new(
            [10, 0, 0, 1],
            [255, 255, 255, 0],
            [10, 0, 0, 1],
            [8, 8, 8, 8],
        );
        assert_eq!(InterfaceConfig::Static(cfg), InterfaceConfig::Static(cfg));
        assert_ne!(InterfaceConfig::Static(cfg), InterfaceConfig::Dhcp);
        assert_eq!(InterfaceConfig::Dhcp, InterfaceConfig::Dhcp);
    }

    // -----------------------------------------------------------------------
    // apply / get
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_and_get_static_config() {
        reset_state();

        assert!(get_applied_config().is_none());

        let cfg = StaticIpConfig::new(
            [172, 16, 0, 10],
            [255, 255, 0, 0],
            [172, 16, 0, 1],
            [1, 1, 1, 1],
        );
        apply_static_config(&cfg);

        let stored = get_applied_config();
        assert!(stored.is_some());
        assert_eq!(stored.unwrap(), cfg);

        reset_state();
    }

    #[test]
    fn test_apply_and_get_ipv6() {
        reset_state();

        assert!(get_applied_ipv6().is_none());

        let ipv6 = [0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        apply_static_ipv6(ipv6);

        let stored = get_applied_ipv6();
        assert!(stored.is_some());
        assert_eq!(stored.unwrap(), ipv6);

        reset_state();
    }

    #[test]
    fn test_clear_config() {
        reset_state();

        let cfg = StaticIpConfig::new(
            [10, 0, 0, 1],
            [255, 255, 255, 0],
            [10, 0, 0, 1],
            [8, 8, 8, 8],
        );
        apply_static_config(&cfg);
        apply_static_ipv6([0xFE, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);

        assert!(get_applied_config().is_some());
        assert!(get_applied_ipv6().is_some());

        clear_config();

        assert!(get_applied_config().is_none());
        assert!(get_applied_ipv6().is_none());
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn test_static_ip_error_display() {
        assert_eq!(
            alloc::format!("{}", StaticIpError::InvalidConfig),
            "static-ip: invalid configuration"
        );
        assert_eq!(
            alloc::format!("{}", StaticIpError::AlreadyConfigured),
            "static-ip: already configured"
        );
    }

    #[test]
    fn test_static_ip_error_eq() {
        assert_eq!(StaticIpError::InvalidConfig, StaticIpError::InvalidConfig);
        assert_ne!(
            StaticIpError::InvalidConfig,
            StaticIpError::AlreadyConfigured
        );
    }

    // -----------------------------------------------------------------------
    // Config clone and copy
    // -----------------------------------------------------------------------

    #[test]
    fn test_static_config_clone() {
        let cfg = StaticIpConfig::new(
            [192, 168, 0, 1],
            [255, 255, 255, 0],
            [192, 168, 0, 1],
            [8, 8, 8, 8],
        );
        let cloned = cfg;
        assert_eq!(cfg, cloned);
    }

    #[test]
    fn test_static_config_debug() {
        let cfg = StaticIpConfig::new(
            [10, 0, 0, 1],
            [255, 255, 255, 0],
            [10, 0, 0, 1],
            [8, 8, 8, 8],
        );
        let debug_str = alloc::format!("{:?}", cfg);
        assert!(debug_str.contains("StaticIpConfig"));
        assert!(debug_str.contains("ipv4_addr"));
    }
}
