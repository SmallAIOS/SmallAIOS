// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Health check and readiness probe management.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::ContainerError;

/// Maximum number of registered health-check components.
const MAX_COMPONENTS: usize = 32;

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// Aggregate or per-component health classification.
#[derive(Clone, Debug, PartialEq)]
pub enum HealthStatus {
    /// Fully operational.
    Healthy,
    /// Operational but with reduced capacity or non-critical faults.
    Degraded,
    /// Unable to serve traffic.
    Unhealthy,
}

impl HealthStatus {
    /// Return a lowercase JSON-safe label.
    fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
        }
    }
}

// ---------------------------------------------------------------------------
// ReadinessStatus
// ---------------------------------------------------------------------------

/// Whether the container is ready to accept requests.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadinessStatus {
    /// Container is accepting traffic.
    Ready,
    /// Container is not yet ready (still booting or draining).
    NotReady,
}

// ---------------------------------------------------------------------------
// ComponentHealth
// ---------------------------------------------------------------------------

/// Health snapshot for a single registered component.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentHealth {
    /// Component name (e.g. "scheduler", "onnx-runtime").
    pub name: String,
    /// Current status.
    pub status: HealthStatus,
    /// Human-readable message describing the current state.
    pub message: String,
    /// Timestamp (ms) of the last health check for this component.
    pub last_check_ms: u64,
}

// ---------------------------------------------------------------------------
// HealthChecker
// ---------------------------------------------------------------------------

/// Central health-check coordinator.
///
/// Kubernetes-style liveness, readiness, and startup probes are all derived
/// from the state tracked here.
pub struct HealthChecker {
    /// Registered component health entries.
    components: Vec<ComponentHealth>,
    /// Computed overall status (recomputed on each [`check_all`] call).
    overall_status: HealthStatus,
    /// Whether the container has declared itself ready for traffic.
    ready: bool,
    /// Monotonic uptime in milliseconds (set externally).
    uptime_ms: u64,
    /// Total number of [`check_all`] invocations.
    check_count: u64,
}

impl HealthChecker {
    /// Create a new checker with no registered components.
    ///
    /// The initial overall status is [`HealthStatus::Healthy`] (vacuously
    /// true — there is nothing unhealthy yet) and readiness is `false`.
    pub fn new() -> Self {
        HealthChecker {
            components: Vec::new(),
            overall_status: HealthStatus::Healthy,
            ready: false,
            uptime_ms: 0,
            check_count: 0,
        }
    }

    /// Register a new component for health tracking.
    ///
    /// The component starts as [`HealthStatus::Healthy`] with an empty
    /// message.  Returns [`ContainerError::TableFull`] if the maximum number
    /// of components has been reached.
    pub fn register_component(&mut self, name: &str) -> Result<(), ContainerError> {
        if self.components.len() >= MAX_COMPONENTS {
            return Err(ContainerError::TableFull);
        }
        self.components.push(ComponentHealth {
            name: String::from(name),
            status: HealthStatus::Healthy,
            message: String::new(),
            last_check_ms: 0,
        });
        Ok(())
    }

    /// Update the status and message of a previously registered component.
    ///
    /// Returns [`ContainerError::NotFound`] if no component with the given
    /// name exists.
    pub fn update_component(
        &mut self,
        name: &str,
        status: HealthStatus,
        msg: &str,
    ) -> Result<(), ContainerError> {
        for c in &mut self.components {
            if c.name == name {
                c.status = status;
                c.message = String::from(msg);
                return Ok(());
            }
        }
        Err(ContainerError::NotFound(String::from(name)))
    }

    /// Recompute the overall health status from all registered components.
    ///
    /// Rules:
    /// - All healthy  --> Healthy
    /// - Any unhealthy --> Unhealthy
    /// - Otherwise     --> Degraded
    ///
    /// Also increments the internal check counter.
    pub fn check_all(&mut self) -> HealthStatus {
        self.check_count += 1;

        let mut any_unhealthy = false;
        let mut any_degraded = false;

        for c in &self.components {
            match c.status {
                HealthStatus::Unhealthy => {
                    any_unhealthy = true;
                }
                HealthStatus::Degraded => {
                    any_degraded = true;
                }
                HealthStatus::Healthy => {}
            }
        }

        self.overall_status = if any_unhealthy {
            HealthStatus::Unhealthy
        } else if any_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        self.overall_status.clone()
    }

    /// Declare the container as ready (or not ready) for traffic.
    pub fn set_ready(&mut self, ready: bool) {
        self.ready = ready;
    }

    /// Whether the container has been declared ready.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Whether the overall status is [`HealthStatus::Healthy`].
    pub fn is_healthy(&self) -> bool {
        self.overall_status == HealthStatus::Healthy
    }

    /// Set the uptime value (milliseconds) — typically driven by a timer
    /// interrupt or monotonic clock.
    pub fn set_uptime(&mut self, ms: u64) {
        self.uptime_ms = ms;
    }

    /// Current uptime in milliseconds.
    pub fn uptime_ms(&self) -> u64 {
        self.uptime_ms
    }

    /// Number of registered components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Produce a minimal JSON health summary.
    ///
    /// Example: `{"status":"healthy","uptime_ms":123,"components":2,"checks":5}`
    pub fn to_health_json(&self) -> String {
        let mut s = String::from("{\"status\":\"");
        s.push_str(self.overall_status.as_str());
        s.push_str("\",\"uptime_ms\":");
        push_u64(&mut s, self.uptime_ms);
        s.push_str(",\"components\":");
        push_u64(&mut s, self.components.len() as u64);
        s.push_str(",\"checks\":");
        push_u64(&mut s, self.check_count);
        s.push('}');
        s
    }

    /// Produce a minimal JSON readiness response.
    ///
    /// Example: `{"ready":true}`
    pub fn to_readiness_json(&self) -> String {
        let mut s = String::from("{\"ready\":");
        if self.ready {
            s.push_str("true");
        } else {
            s.push_str("false");
        }
        s.push('}');
        s
    }

    /// Produce a minimal JSON liveness response.
    ///
    /// The container is always considered alive as long as this code is
    /// executing.
    ///
    /// Example: `{"alive":true}`
    pub fn to_liveness_json(&self) -> String {
        String::from("{\"alive\":true}")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Append the decimal representation of `n` to `s` without pulling in
/// `core::fmt::Write` (which requires a full formatter).
fn push_u64(s: &mut String, n: u64) {
    if n == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 20]; // max digits for u64
    let mut pos = buf.len();
    let mut val = n;
    while val > 0 {
        pos -= 1;
        buf[pos] = b'0' + (val % 10) as u8;
        val /= 10;
    }
    for &b in &buf[pos..] {
        s.push(b as char);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- New checker defaults ----------------------------------------------

    #[test]
    fn test_new_checker_is_healthy() {
        let hc = HealthChecker::new();
        assert!(hc.is_healthy());
    }

    #[test]
    fn test_new_checker_is_not_ready() {
        let hc = HealthChecker::new();
        assert!(!hc.is_ready());
    }

    // -- Register ----------------------------------------------------------

    #[test]
    fn test_register_component() {
        let mut hc = HealthChecker::new();
        assert!(hc.register_component("scheduler").is_ok());
        assert_eq!(hc.component_count(), 1);
    }

    #[test]
    fn test_register_multiple_components() {
        let mut hc = HealthChecker::new();
        hc.register_component("scheduler").unwrap();
        hc.register_component("onnx-rt").unwrap();
        hc.register_component("network").unwrap();
        assert_eq!(hc.component_count(), 3);
    }

    // -- Update ------------------------------------------------------------

    #[test]
    fn test_update_component_status() {
        let mut hc = HealthChecker::new();
        hc.register_component("scheduler").unwrap();
        assert!(hc
            .update_component("scheduler", HealthStatus::Degraded, "high load")
            .is_ok());
    }

    #[test]
    fn test_update_nonexistent_component_returns_error() {
        let mut hc = HealthChecker::new();
        let result = hc.update_component("ghost", HealthStatus::Healthy, "ok");
        assert_eq!(
            result,
            Err(ContainerError::NotFound(String::from("ghost")))
        );
    }

    // -- check_all ---------------------------------------------------------

    #[test]
    fn test_check_all_healthy() {
        let mut hc = HealthChecker::new();
        hc.register_component("a").unwrap();
        hc.register_component("b").unwrap();
        let status = hc.check_all();
        assert_eq!(status, HealthStatus::Healthy);
    }

    #[test]
    fn test_check_all_degraded() {
        let mut hc = HealthChecker::new();
        hc.register_component("a").unwrap();
        hc.register_component("b").unwrap();
        hc.update_component("b", HealthStatus::Degraded, "slow")
            .unwrap();
        let status = hc.check_all();
        assert_eq!(status, HealthStatus::Degraded);
    }

    #[test]
    fn test_check_all_unhealthy() {
        let mut hc = HealthChecker::new();
        hc.register_component("a").unwrap();
        hc.register_component("b").unwrap();
        hc.update_component("a", HealthStatus::Unhealthy, "crashed")
            .unwrap();
        let status = hc.check_all();
        assert_eq!(status, HealthStatus::Unhealthy);
    }

    // -- Readiness ---------------------------------------------------------

    #[test]
    fn test_set_ready_and_is_ready() {
        let mut hc = HealthChecker::new();
        assert!(!hc.is_ready());
        hc.set_ready(true);
        assert!(hc.is_ready());
        hc.set_ready(false);
        assert!(!hc.is_ready());
    }

    // -- Component count ---------------------------------------------------

    #[test]
    fn test_component_count() {
        let mut hc = HealthChecker::new();
        assert_eq!(hc.component_count(), 0);
        hc.register_component("x").unwrap();
        assert_eq!(hc.component_count(), 1);
    }

    // -- JSON outputs ------------------------------------------------------

    #[test]
    fn test_health_json_format() {
        let mut hc = HealthChecker::new();
        hc.register_component("a").unwrap();
        hc.register_component("b").unwrap();
        hc.set_uptime(123);
        hc.check_all(); // check_count = 1
        hc.check_all(); // check_count = 2
        let json = hc.to_health_json();
        assert_eq!(
            json,
            "{\"status\":\"healthy\",\"uptime_ms\":123,\"components\":2,\"checks\":2}"
        );
    }

    #[test]
    fn test_readiness_json_format_not_ready() {
        let hc = HealthChecker::new();
        assert_eq!(hc.to_readiness_json(), "{\"ready\":false}");
    }

    #[test]
    fn test_readiness_json_format_ready() {
        let mut hc = HealthChecker::new();
        hc.set_ready(true);
        assert_eq!(hc.to_readiness_json(), "{\"ready\":true}");
    }

    #[test]
    fn test_liveness_json_format() {
        let hc = HealthChecker::new();
        assert_eq!(hc.to_liveness_json(), "{\"alive\":true}");
    }

    // -- Uptime tracking ---------------------------------------------------

    #[test]
    fn test_uptime_tracking() {
        let mut hc = HealthChecker::new();
        assert_eq!(hc.uptime_ms(), 0);
        hc.set_uptime(5000);
        assert_eq!(hc.uptime_ms(), 5000);
    }

    // -- Check count -------------------------------------------------------

    #[test]
    fn test_check_count_increments() {
        let mut hc = HealthChecker::new();
        hc.register_component("a").unwrap();
        assert_eq!(hc.check_count, 0);
        hc.check_all();
        assert_eq!(hc.check_count, 1);
        hc.check_all();
        hc.check_all();
        assert_eq!(hc.check_count, 3);
    }

    // -- Table full --------------------------------------------------------

    #[test]
    fn test_table_full_error() {
        let mut hc = HealthChecker::new();
        for i in 0..MAX_COMPONENTS {
            let mut name = String::from("comp-");
            push_u64(&mut name, i as u64);
            hc.register_component(&name).unwrap();
        }
        assert_eq!(hc.component_count(), MAX_COMPONENTS);
        let result = hc.register_component("one-too-many");
        assert_eq!(result, Err(ContainerError::TableFull));
    }

    // -- Unhealthy overrides degraded --------------------------------------

    #[test]
    fn test_unhealthy_overrides_degraded() {
        let mut hc = HealthChecker::new();
        hc.register_component("a").unwrap();
        hc.register_component("b").unwrap();
        hc.register_component("c").unwrap();
        hc.update_component("a", HealthStatus::Degraded, "slow")
            .unwrap();
        hc.update_component("b", HealthStatus::Unhealthy, "down")
            .unwrap();
        let status = hc.check_all();
        assert_eq!(status, HealthStatus::Unhealthy);
    }

    // -- Health JSON with degraded -----------------------------------------

    #[test]
    fn test_health_json_degraded() {
        let mut hc = HealthChecker::new();
        hc.register_component("x").unwrap();
        hc.update_component("x", HealthStatus::Degraded, "slow")
            .unwrap();
        hc.check_all();
        let json = hc.to_health_json();
        assert!(json.contains("\"status\":\"degraded\""));
    }
}
