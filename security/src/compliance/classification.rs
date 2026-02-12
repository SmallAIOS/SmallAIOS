// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Data classification enforcement per NIST SP 800-53 and CSF 2.0.
//!
//! Classification levels:
//! - Public: health metrics, version info
//! - Internal: inference I/O, audit logs, configuration
//! - Restricted: model weights, crypto keys
//!
//! Enforces that data at a higher classification level cannot flow
//! to a lower-classified destination without explicit declassification.

/// Data classification levels (ordered from lowest to highest sensitivity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ClassificationLevel {
    /// Publicly accessible data (health metrics, version).
    Public = 0,
    /// Internal use only (inference I/O, audit logs, config).
    Internal = 1,
    /// Restricted access (model weights, crypto keys).
    Restricted = 2,
}

impl ClassificationLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Internal => "internal",
            Self::Restricted => "restricted",
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Public),
            1 => Some(Self::Internal),
            2 => Some(Self::Restricted),
            _ => None,
        }
    }
}

/// Data types in the system with their classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    /// ONNX model weights.
    ModelWeights = 0,
    /// Inference input/output data.
    InferenceIO = 1,
    /// Audit log entries and batches.
    AuditLogs = 2,
    /// System configuration.
    Configuration = 3,
    /// Cryptographic key material.
    CryptoKeys = 4,
    /// Health and monitoring metrics.
    HealthMetrics = 5,
    /// Version and system information.
    VersionInfo = 6,
    /// IPC message payloads.
    IpcPayloads = 7,
}

impl DataType {
    /// Get the classification level for this data type.
    pub fn classification(&self) -> ClassificationLevel {
        match self {
            Self::ModelWeights => ClassificationLevel::Restricted,
            Self::CryptoKeys => ClassificationLevel::Restricted,
            Self::InferenceIO => ClassificationLevel::Internal,
            Self::AuditLogs => ClassificationLevel::Internal,
            Self::Configuration => ClassificationLevel::Internal,
            Self::IpcPayloads => ClassificationLevel::Internal,
            Self::HealthMetrics => ClassificationLevel::Public,
            Self::VersionInfo => ClassificationLevel::Public,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ModelWeights => "model-weights",
            Self::InferenceIO => "inference-io",
            Self::AuditLogs => "audit-logs",
            Self::Configuration => "configuration",
            Self::CryptoKeys => "crypto-keys",
            Self::HealthMetrics => "health-metrics",
            Self::VersionInfo => "version-info",
            Self::IpcPayloads => "ipc-payloads",
        }
    }
}

/// Destination classification for data flow checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataDestination {
    /// Internal kernel subsystem (same classification or higher).
    KernelInternal = 0,
    /// IPC export (internal classification).
    IpcExport = 1,
    /// Network export (public classification only, unless TLS).
    NetworkExport = 2,
    /// Prometheus metrics endpoint (public only).
    MetricsEndpoint = 3,
    /// Audit log storage (internal).
    AuditStorage = 4,
}

impl DataDestination {
    /// Minimum classification level this destination can handle.
    ///
    /// Data with a classification higher than this is blocked unless
    /// explicitly declassified or encrypted.
    pub fn max_classification(&self) -> ClassificationLevel {
        match self {
            Self::KernelInternal => ClassificationLevel::Restricted,
            Self::AuditStorage => ClassificationLevel::Internal,
            Self::IpcExport => ClassificationLevel::Internal,
            Self::NetworkExport => ClassificationLevel::Public,
            Self::MetricsEndpoint => ClassificationLevel::Public,
        }
    }
}

/// Result of a data flow classification check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationCheckResult {
    /// Data flow is allowed (destination can handle the classification).
    Allowed,
    /// Data flow is blocked (classification exceeds destination capability).
    Blocked {
        data_level: ClassificationLevel,
        dest_max: ClassificationLevel,
    },
}

/// Check whether data of a given type can flow to a destination.
pub fn check_data_flow(
    data_type: DataType,
    destination: DataDestination,
) -> ClassificationCheckResult {
    let data_level = data_type.classification();
    let dest_max = destination.max_classification();

    if data_level <= dest_max {
        ClassificationCheckResult::Allowed
    } else {
        ClassificationCheckResult::Blocked {
            data_level,
            dest_max,
        }
    }
}

/// Classification enforcement tracker.
pub struct ClassificationEnforcer {
    total_checks: u64,
    total_blocks: u64,
}

impl ClassificationEnforcer {
    pub fn new() -> Self {
        Self {
            total_checks: 0,
            total_blocks: 0,
        }
    }

    /// Check and record a data flow.
    pub fn check(
        &mut self,
        data_type: DataType,
        destination: DataDestination,
    ) -> ClassificationCheckResult {
        self.total_checks += 1;
        let result = check_data_flow(data_type, destination);
        if matches!(result, ClassificationCheckResult::Blocked { .. }) {
            self.total_blocks += 1;
        }
        result
    }

    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }

    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }
}

impl Default for ClassificationEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_ordering() {
        assert!(ClassificationLevel::Public < ClassificationLevel::Internal);
        assert!(ClassificationLevel::Internal < ClassificationLevel::Restricted);
    }

    #[test]
    fn model_weights_restricted() {
        assert_eq!(
            DataType::ModelWeights.classification(),
            ClassificationLevel::Restricted
        );
    }

    #[test]
    fn crypto_keys_restricted() {
        assert_eq!(
            DataType::CryptoKeys.classification(),
            ClassificationLevel::Restricted
        );
    }

    #[test]
    fn inference_io_internal() {
        assert_eq!(
            DataType::InferenceIO.classification(),
            ClassificationLevel::Internal
        );
    }

    #[test]
    fn audit_logs_internal() {
        assert_eq!(
            DataType::AuditLogs.classification(),
            ClassificationLevel::Internal
        );
    }

    #[test]
    fn health_metrics_public() {
        assert_eq!(
            DataType::HealthMetrics.classification(),
            ClassificationLevel::Public
        );
    }

    #[test]
    fn public_data_to_metrics_allowed() {
        let result = check_data_flow(DataType::HealthMetrics, DataDestination::MetricsEndpoint);
        assert_eq!(result, ClassificationCheckResult::Allowed);
    }

    #[test]
    fn restricted_data_to_metrics_blocked() {
        let result = check_data_flow(DataType::CryptoKeys, DataDestination::MetricsEndpoint);
        assert!(matches!(result, ClassificationCheckResult::Blocked { .. }));
    }

    #[test]
    fn internal_data_to_network_blocked() {
        let result = check_data_flow(DataType::AuditLogs, DataDestination::NetworkExport);
        assert!(matches!(result, ClassificationCheckResult::Blocked { .. }));
    }

    #[test]
    fn internal_data_to_ipc_allowed() {
        let result = check_data_flow(DataType::AuditLogs, DataDestination::IpcExport);
        assert_eq!(result, ClassificationCheckResult::Allowed);
    }

    #[test]
    fn restricted_data_to_kernel_allowed() {
        let result = check_data_flow(DataType::CryptoKeys, DataDestination::KernelInternal);
        assert_eq!(result, ClassificationCheckResult::Allowed);
    }

    #[test]
    fn restricted_data_to_ipc_blocked() {
        let result = check_data_flow(DataType::ModelWeights, DataDestination::IpcExport);
        assert!(matches!(result, ClassificationCheckResult::Blocked { .. }));
    }

    #[test]
    fn enforcer_tracking() {
        let mut enforcer = ClassificationEnforcer::new();
        enforcer.check(DataType::HealthMetrics, DataDestination::MetricsEndpoint);
        enforcer.check(DataType::CryptoKeys, DataDestination::MetricsEndpoint);
        enforcer.check(DataType::AuditLogs, DataDestination::IpcExport);

        assert_eq!(enforcer.total_checks(), 3);
        assert_eq!(enforcer.total_blocks(), 1);
    }

    #[test]
    fn classification_level_strings() {
        assert_eq!(ClassificationLevel::Public.as_str(), "public");
        assert_eq!(ClassificationLevel::Internal.as_str(), "internal");
        assert_eq!(ClassificationLevel::Restricted.as_str(), "restricted");
    }

    #[test]
    fn data_type_strings() {
        assert_eq!(DataType::ModelWeights.as_str(), "model-weights");
        assert_eq!(DataType::CryptoKeys.as_str(), "crypto-keys");
        assert_eq!(DataType::HealthMetrics.as_str(), "health-metrics");
    }

    #[test]
    fn classification_from_u8() {
        assert_eq!(
            ClassificationLevel::from_u8(0),
            Some(ClassificationLevel::Public)
        );
        assert_eq!(
            ClassificationLevel::from_u8(1),
            Some(ClassificationLevel::Internal)
        );
        assert_eq!(
            ClassificationLevel::from_u8(2),
            Some(ClassificationLevel::Restricted)
        );
        assert_eq!(ClassificationLevel::from_u8(3), None);
    }
}
