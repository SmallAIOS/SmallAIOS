// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Built-in service endpoints.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::key_expr::KeyExpr;
use crate::IpcError;

/// Maximum number of endpoints that can be registered.
const MAX_ENDPOINTS: usize = 64;

/// Unique endpoint identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointId(pub u64);

/// The kind of a service endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointKind {
    /// System health probe endpoint.
    Health,
    /// Metrics and telemetry endpoint.
    Metrics,
    /// Model management endpoint (list, load, unload).
    Models,
    /// Inference execution endpoint.
    Inference,
    /// Log streaming endpoint.
    Logs,
    /// Configuration management endpoint.
    Config,
    /// User-defined custom endpoint.
    Custom(String),
}

/// Health status reported by the health endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All subsystems operational.
    Healthy,
    /// Some subsystems degraded but functional.
    Degraded,
    /// Critical failure, system not operational.
    Unhealthy,
}

/// Response from the health endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthResponse {
    /// Current system health status.
    pub status: HealthStatus,
    /// System uptime in seconds.
    pub uptime_secs: u64,
    /// Kernel version string.
    pub version: &'static str,
}

/// A single metric value.
///
/// Note: `f64` does not implement `Eq`, so `MetricValue` only derives
/// `Clone` and `Debug`, with a manual `PartialEq` implementation.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Monotonically increasing counter.
    Counter(u64),
    /// Point-in-time gauge measurement.
    Gauge(f64),
    /// Distribution of observed values.
    Histogram(Vec<f64>),
}

impl PartialEq for MetricValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MetricValue::Counter(a), MetricValue::Counter(b)) => a == b,
            (MetricValue::Gauge(a), MetricValue::Gauge(b)) => a.to_bits() == b.to_bits(),
            (MetricValue::Histogram(a), MetricValue::Histogram(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        }
    }
}

/// A named metric with timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    /// Human-readable metric name (e.g., "inference_count").
    pub name: String,
    /// The metric value.
    pub value: MetricValue,
    /// Timestamp when the metric was recorded (kernel ticks or epoch).
    pub timestamp: u64,
}

/// Information about a loaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Unique model identifier.
    pub id: u32,
    /// Human-readable model name.
    pub name: String,
    /// Whether the model is currently loaded into memory.
    pub loaded: bool,
    /// Model size in bytes.
    pub size_bytes: u64,
}

/// A registered service endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Endpoint {
    /// Unique endpoint identifier.
    pub id: EndpointId,
    /// The kind of service this endpoint provides.
    pub kind: EndpointKind,
    /// Zenoh key expression this endpoint is bound to.
    pub key_expr: KeyExpr,
    /// Whether the endpoint is currently accepting requests.
    pub active: bool,
}

/// Registry of service endpoints.
///
/// Maintains a collection of up to [`MAX_ENDPOINTS`] registered endpoints
/// and assigns monotonically increasing endpoint IDs.
#[derive(Debug, Clone)]
pub struct EndpointRegistry {
    /// Registered endpoints.
    endpoints: Vec<Endpoint>,
    /// Next endpoint ID to assign.
    next_id: u64,
}

impl Default for EndpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointRegistry {
    /// Create a new empty endpoint registry.
    pub fn new() -> Self {
        EndpointRegistry {
            endpoints: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new endpoint of the given kind at the specified key expression.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::TableFull`] if the registry is at capacity.
    pub fn register(
        &mut self,
        kind: EndpointKind,
        key_expr: KeyExpr,
    ) -> Result<EndpointId, IpcError> {
        if self.endpoints.len() >= MAX_ENDPOINTS {
            return Err(IpcError::TableFull);
        }

        let id = EndpointId(self.next_id);
        self.next_id += 1;

        self.endpoints.push(Endpoint {
            id,
            kind,
            key_expr,
            active: true,
        });

        Ok(id)
    }

    /// Unregister an endpoint by its ID.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::NotFound`] if no endpoint with the given ID exists.
    pub fn unregister(&mut self, id: EndpointId) -> Result<(), IpcError> {
        let idx = self
            .endpoints
            .iter()
            .position(|e| e.id == id)
            .ok_or(IpcError::NotFound)?;
        self.endpoints.remove(idx);
        Ok(())
    }

    /// Find the first endpoint matching the given kind.
    pub fn find_by_kind(&self, kind: &EndpointKind) -> Option<&Endpoint> {
        self.endpoints.iter().find(|e| &e.kind == kind)
    }

    /// Returns the number of currently active endpoints.
    pub fn active_count(&self) -> usize {
        self.endpoints.iter().filter(|e| e.active).count()
    }

    /// Returns the total number of registered endpoints.
    pub fn len(&self) -> usize {
        self.endpoints.len()
    }

    /// Returns `true` if no endpoints are registered.
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }

    /// Register the default set of SmallAIOS service endpoints.
    ///
    /// Registers endpoints for: Health, Metrics, Models, Inference, Logs.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::TableFull`] if the registry cannot hold all defaults.
    pub fn register_defaults(&mut self) -> Result<(), IpcError> {
        self.register(
            EndpointKind::Health,
            KeyExpr::new("sai/health").map_err(|_| IpcError::InvalidState)?,
        )?;
        self.register(
            EndpointKind::Metrics,
            KeyExpr::new("sai/metrics").map_err(|_| IpcError::InvalidState)?,
        )?;
        self.register(
            EndpointKind::Models,
            KeyExpr::new("sai/models").map_err(|_| IpcError::InvalidState)?,
        )?;
        self.register(
            EndpointKind::Inference,
            KeyExpr::new("sai/inference").map_err(|_| IpcError::InvalidState)?,
        )?;
        self.register(
            EndpointKind::Logs,
            KeyExpr::new("sai/logs").map_err(|_| IpcError::InvalidState)?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    // === Registration ===

    #[test]
    fn register_endpoint() {
        let mut registry = EndpointRegistry::new();
        let key = KeyExpr::new("sai/test").unwrap();
        let id = registry.register(EndpointKind::Health, key).unwrap();
        assert_eq!(id, EndpointId(1));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn register_multiple_endpoints() {
        let mut registry = EndpointRegistry::new();
        let id1 = registry
            .register(EndpointKind::Health, KeyExpr::new("sai/health").unwrap())
            .unwrap();
        let id2 = registry
            .register(EndpointKind::Metrics, KeyExpr::new("sai/metrics").unwrap())
            .unwrap();
        assert_ne!(id1, id2);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn unregister_endpoint() {
        let mut registry = EndpointRegistry::new();
        let id = registry
            .register(EndpointKind::Health, KeyExpr::new("sai/health").unwrap())
            .unwrap();
        assert_eq!(registry.len(), 1);
        registry.unregister(id).unwrap();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn unregister_nonexistent_returns_not_found() {
        let mut registry = EndpointRegistry::new();
        assert_eq!(
            registry.unregister(EndpointId(999)),
            Err(IpcError::NotFound)
        );
    }

    // === Default endpoints ===

    #[test]
    fn register_defaults_creates_five_endpoints() {
        let mut registry = EndpointRegistry::new();
        registry.register_defaults().unwrap();
        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn register_defaults_all_active() {
        let mut registry = EndpointRegistry::new();
        registry.register_defaults().unwrap();
        assert_eq!(registry.active_count(), 5);
    }

    // === Find by kind ===

    #[test]
    fn find_by_kind_health() {
        let mut registry = EndpointRegistry::new();
        registry.register_defaults().unwrap();
        let ep = registry.find_by_kind(&EndpointKind::Health).unwrap();
        assert_eq!(ep.kind, EndpointKind::Health);
        assert_eq!(ep.key_expr.as_str(), "sai/health");
    }

    #[test]
    fn find_by_kind_not_registered() {
        let registry = EndpointRegistry::new();
        assert!(registry.find_by_kind(&EndpointKind::Config).is_none());
    }

    #[test]
    fn find_by_kind_custom() {
        let mut registry = EndpointRegistry::new();
        registry
            .register(
                EndpointKind::Custom(String::from("my-service")),
                KeyExpr::new("sai/custom/my-service").unwrap(),
            )
            .unwrap();
        let ep = registry
            .find_by_kind(&EndpointKind::Custom(String::from("my-service")))
            .unwrap();
        assert_eq!(ep.kind, EndpointKind::Custom(String::from("my-service")));
    }

    // === Table full ===

    #[test]
    fn table_full_error() {
        let mut registry = EndpointRegistry::new();
        for i in 0..MAX_ENDPOINTS {
            let key = KeyExpr::new(&alloc::format!("sai/ep{}", i)).unwrap();
            registry.register(EndpointKind::Health, key).unwrap();
        }
        assert_eq!(registry.len(), MAX_ENDPOINTS);
        let result = registry.register(EndpointKind::Health, KeyExpr::new("sai/overflow").unwrap());
        assert_eq!(result, Err(IpcError::TableFull));
    }

    // === Active count ===

    #[test]
    fn active_count_after_deactivation() {
        let mut registry = EndpointRegistry::new();
        registry.register_defaults().unwrap();
        assert_eq!(registry.active_count(), 5);
        // Manually deactivate the first endpoint
        registry.endpoints[0].active = false;
        assert_eq!(registry.active_count(), 4);
    }

    // === Data construction ===

    #[test]
    fn health_response_construction() {
        let hr = HealthResponse {
            status: HealthStatus::Healthy,
            uptime_secs: 3600,
            version: "0.1.0",
        };
        assert_eq!(hr.status, HealthStatus::Healthy);
        assert_eq!(hr.uptime_secs, 3600);
        assert_eq!(hr.version, "0.1.0");
    }

    #[test]
    fn metric_counter_construction() {
        let m = Metric {
            name: String::from("inference_count"),
            value: MetricValue::Counter(42),
            timestamp: 1000,
        };
        assert_eq!(m.name, "inference_count");
        assert_eq!(m.value, MetricValue::Counter(42));
    }

    #[test]
    fn metric_gauge_construction() {
        let m = Metric {
            name: String::from("cpu_usage"),
            value: MetricValue::Gauge(0.75),
            timestamp: 2000,
        };
        assert_eq!(m.value, MetricValue::Gauge(0.75));
    }

    #[test]
    fn metric_histogram_construction() {
        let m = Metric {
            name: String::from("latency_ms"),
            value: MetricValue::Histogram(vec![1.0, 2.0, 3.0]),
            timestamp: 3000,
        };
        assert_eq!(m.value, MetricValue::Histogram(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn metric_value_inequality() {
        let counter = MetricValue::Counter(1);
        let gauge = MetricValue::Gauge(1.0);
        assert_ne!(counter, gauge);
    }

    #[test]
    fn model_info_construction() {
        let info = ModelInfo {
            id: 1,
            name: String::from("resnet50"),
            loaded: true,
            size_bytes: 102_400_000,
        };
        assert_eq!(info.id, 1);
        assert_eq!(info.name, "resnet50");
        assert!(info.loaded);
        assert_eq!(info.size_bytes, 102_400_000);
    }

    // === EndpointKind matching ===

    #[test]
    fn endpoint_kind_equality() {
        assert_eq!(EndpointKind::Health, EndpointKind::Health);
        assert_ne!(EndpointKind::Health, EndpointKind::Metrics);
        assert_ne!(
            EndpointKind::Custom(String::from("a")),
            EndpointKind::Custom(String::from("b"))
        );
        assert_eq!(
            EndpointKind::Custom(String::from("x")),
            EndpointKind::Custom(String::from("x"))
        );
    }

    #[test]
    fn empty_registry() {
        let registry = EndpointRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.active_count(), 0);
    }
}
