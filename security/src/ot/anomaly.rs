// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! OT-specific anomaly detection for inference workloads.
//!
//! Provides:
//! - Model output range validation (configurable upper/lower bounds per tensor)
//! - Inference timing bounds enforcement (min/max per model)
//! - Input data validation (NaN/Inf detection, shape verification)

/// Maximum number of models that can be tracked.
const MAX_MODELS: usize = 16;

/// Maximum dimensions per tensor for shape validation.
const MAX_DIMS: usize = 8;

/// Maximum output tensors per model for range validation.
const MAX_OUTPUTS: usize = 8;

/// Anomaly type detected during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Output value outside configured range.
    OutputOutOfRange,
    /// Inference completed too fast (below minimum timing bound).
    InferenceTooFast,
    /// Inference completed too slow (above maximum timing bound).
    InferenceTooSlow,
    /// NaN detected in input tensor.
    InputNaN,
    /// Infinity detected in input tensor.
    InputInf,
    /// Input tensor shape mismatch.
    ShapeMismatch,
}

impl AnomalyType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OutputOutOfRange => "output-out-of-range",
            Self::InferenceTooFast => "inference-too-fast",
            Self::InferenceTooSlow => "inference-too-slow",
            Self::InputNaN => "input-nan",
            Self::InputInf => "input-inf",
            Self::ShapeMismatch => "shape-mismatch",
        }
    }
}

/// An anomaly detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnomalyReport {
    /// Type of anomaly.
    pub anomaly_type: AnomalyType,
    /// Model identifier.
    pub model_id: u32,
    /// Tensor index (input or output).
    pub tensor_index: u16,
    /// Element index within the tensor (if applicable).
    pub element_index: u32,
}

/// Output range bounds for a single tensor.
#[derive(Debug, Clone, Copy)]
pub struct OutputRangeBounds {
    /// Minimum allowed value (as f32 bits for no_std compatibility).
    pub min_bits: u32,
    /// Maximum allowed value (as f32 bits for no_std compatibility).
    pub max_bits: u32,
    /// Whether bounds are configured.
    pub active: bool,
}

impl OutputRangeBounds {
    pub const fn inactive() -> Self {
        Self {
            min_bits: 0,
            max_bits: 0,
            active: false,
        }
    }

    pub fn new(min: f32, max: f32) -> Self {
        Self {
            min_bits: min.to_bits(),
            max_bits: max.to_bits(),
            active: true,
        }
    }

    pub fn min_val(&self) -> f32 {
        f32::from_bits(self.min_bits)
    }

    pub fn max_val(&self) -> f32 {
        f32::from_bits(self.max_bits)
    }

    /// Check if a value is within bounds.
    pub fn check(&self, value: f32) -> bool {
        if !self.active {
            return true;
        }
        value >= self.min_val() && value <= self.max_val()
    }
}

/// Expected input shape for validation.
#[derive(Debug, Clone, Copy)]
pub struct ExpectedShape {
    /// Dimension values (0 = dynamic/any).
    pub dims: [u32; MAX_DIMS],
    /// Number of active dimensions.
    pub ndim: u8,
    /// Minimum allowed value for dynamic dimensions.
    pub dynamic_min: u32,
    /// Maximum allowed value for dynamic dimensions.
    pub dynamic_max: u32,
}

impl ExpectedShape {
    pub const fn empty() -> Self {
        Self {
            dims: [0; MAX_DIMS],
            ndim: 0,
            dynamic_min: 1,
            dynamic_max: u32::MAX,
        }
    }

    /// Create a fixed shape from a slice of dimensions.
    pub fn fixed(dims: &[u32]) -> Self {
        let mut shape = Self::empty();
        let n = dims.len().min(MAX_DIMS);
        shape.dims[..n].copy_from_slice(&dims[..n]);
        shape.ndim = n as u8;
        shape
    }

    /// Validate an actual shape against this expected shape.
    pub fn validate(&self, actual: &[u32]) -> bool {
        if self.ndim == 0 {
            return true; // No shape constraint
        }
        if actual.len() != self.ndim as usize {
            return false;
        }
        for (i, &actual_dim) in actual.iter().enumerate().take(self.ndim as usize) {
            if self.dims[i] == 0 {
                // Dynamic dimension: check bounds
                if actual_dim < self.dynamic_min || actual_dim > self.dynamic_max {
                    return false;
                }
            } else if self.dims[i] != actual_dim {
                return false;
            }
        }
        true
    }
}

/// Per-model anomaly detection configuration.
#[derive(Debug, Clone, Copy)]
pub struct ModelAnomalyConfig {
    /// Model identifier.
    pub model_id: u32,
    /// Whether this config slot is active.
    pub active: bool,
    /// Minimum inference time (nanoseconds), 0 = no minimum.
    pub min_inference_ns: u64,
    /// Maximum inference time (nanoseconds), 0 = no maximum.
    pub max_inference_ns: u64,
    /// Output range bounds per output tensor.
    pub output_bounds: [OutputRangeBounds; MAX_OUTPUTS],
    /// Number of active output bounds.
    pub num_outputs: u8,
    /// Expected input shape (first input tensor only for simplicity).
    pub expected_input_shape: ExpectedShape,
    /// Whether NaN/Inf detection is enabled (default: true).
    pub nan_inf_detection: bool,
}

impl ModelAnomalyConfig {
    pub const fn empty() -> Self {
        Self {
            model_id: 0,
            active: false,
            min_inference_ns: 0,
            max_inference_ns: 0,
            output_bounds: [OutputRangeBounds::inactive(); MAX_OUTPUTS],
            num_outputs: 0,
            expected_input_shape: ExpectedShape::empty(),
            nan_inf_detection: true,
        }
    }
}

/// OT anomaly detection engine.
pub struct AnomalyDetector {
    configs: [ModelAnomalyConfig; MAX_MODELS],
    /// Total anomalies detected since boot.
    total_anomalies: u64,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            configs: [ModelAnomalyConfig::empty(); MAX_MODELS],
            total_anomalies: 0,
        }
    }

    /// Register anomaly detection config for a model.
    pub fn register_model(&mut self, config: ModelAnomalyConfig) -> bool {
        if let Some(slot) = self.configs.iter_mut().find(|c| !c.active) {
            *slot = config;
            slot.active = true;
            true
        } else {
            false
        }
    }

    /// Remove anomaly detection config for a model.
    pub fn unregister_model(&mut self, model_id: u32) -> bool {
        if let Some(slot) = self
            .configs
            .iter_mut()
            .find(|c| c.active && c.model_id == model_id)
        {
            slot.active = false;
            true
        } else {
            false
        }
    }

    /// Get config for a model.
    pub fn get_config(&self, model_id: u32) -> Option<&ModelAnomalyConfig> {
        self.configs
            .iter()
            .find(|c| c.active && c.model_id == model_id)
    }

    /// Check inference timing bounds.
    ///
    /// Returns an anomaly report if the duration is outside bounds.
    pub fn check_timing(&mut self, model_id: u32, duration_ns: u64) -> Option<AnomalyReport> {
        let config = self
            .configs
            .iter()
            .find(|c| c.active && c.model_id == model_id)?;

        if config.min_inference_ns > 0 && duration_ns < config.min_inference_ns {
            self.total_anomalies += 1;
            return Some(AnomalyReport {
                anomaly_type: AnomalyType::InferenceTooFast,
                model_id,
                tensor_index: 0,
                element_index: 0,
            });
        }
        if config.max_inference_ns > 0 && duration_ns > config.max_inference_ns {
            self.total_anomalies += 1;
            return Some(AnomalyReport {
                anomaly_type: AnomalyType::InferenceTooSlow,
                model_id,
                tensor_index: 0,
                element_index: 0,
            });
        }
        None
    }

    /// Check a single f32 output value against range bounds.
    ///
    /// Returns an anomaly report if the value is out of range.
    pub fn check_output_value(
        &mut self,
        model_id: u32,
        output_index: u16,
        element_index: u32,
        value: f32,
    ) -> Option<AnomalyReport> {
        let config = self
            .configs
            .iter()
            .find(|c| c.active && c.model_id == model_id)?;

        let oi = output_index as usize;
        if oi < config.num_outputs as usize && !config.output_bounds[oi].check(value) {
            self.total_anomalies += 1;
            return Some(AnomalyReport {
                anomaly_type: AnomalyType::OutputOutOfRange,
                model_id,
                tensor_index: output_index,
                element_index,
            });
        }
        None
    }

    /// Check a single f32 input value for NaN/Inf.
    ///
    /// Returns an anomaly report if NaN or Inf is detected.
    pub fn check_input_value(
        &mut self,
        model_id: u32,
        tensor_index: u16,
        element_index: u32,
        value: f32,
    ) -> Option<AnomalyReport> {
        let config = self
            .configs
            .iter()
            .find(|c| c.active && c.model_id == model_id)?;

        if !config.nan_inf_detection {
            return None;
        }

        if value.is_nan() {
            self.total_anomalies += 1;
            return Some(AnomalyReport {
                anomaly_type: AnomalyType::InputNaN,
                model_id,
                tensor_index,
                element_index,
            });
        }
        if value.is_infinite() {
            self.total_anomalies += 1;
            return Some(AnomalyReport {
                anomaly_type: AnomalyType::InputInf,
                model_id,
                tensor_index,
                element_index,
            });
        }
        None
    }

    /// Check input tensor shape against expected shape.
    pub fn check_input_shape(
        &mut self,
        model_id: u32,
        actual_shape: &[u32],
    ) -> Option<AnomalyReport> {
        let config = self
            .configs
            .iter()
            .find(|c| c.active && c.model_id == model_id)?;

        if !config.expected_input_shape.validate(actual_shape) {
            self.total_anomalies += 1;
            return Some(AnomalyReport {
                anomaly_type: AnomalyType::ShapeMismatch,
                model_id,
                tensor_index: 0,
                element_index: 0,
            });
        }
        None
    }

    /// Total anomalies detected since boot.
    pub fn total_anomalies(&self) -> u64 {
        self.total_anomalies
    }

    /// Number of registered models.
    pub fn registered_models(&self) -> usize {
        self.configs.iter().filter(|c| c.active).count()
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(model_id: u32) -> ModelAnomalyConfig {
        ModelAnomalyConfig {
            model_id,
            active: true,
            min_inference_ns: 1_000_000,   // 1ms min
            max_inference_ns: 100_000_000, // 100ms max
            output_bounds: {
                let mut bounds = [OutputRangeBounds::inactive(); MAX_OUTPUTS];
                bounds[0] = OutputRangeBounds::new(0.0, 1.0);
                bounds
            },
            num_outputs: 1,
            expected_input_shape: ExpectedShape::fixed(&[1, 3, 224, 224]),
            nan_inf_detection: true,
        }
    }

    #[test]
    fn anomaly_type_as_str() {
        assert_eq!(
            AnomalyType::OutputOutOfRange.as_str(),
            "output-out-of-range"
        );
        assert_eq!(AnomalyType::InferenceTooFast.as_str(), "inference-too-fast");
        assert_eq!(AnomalyType::InferenceTooSlow.as_str(), "inference-too-slow");
        assert_eq!(AnomalyType::InputNaN.as_str(), "input-nan");
        assert_eq!(AnomalyType::InputInf.as_str(), "input-inf");
        assert_eq!(AnomalyType::ShapeMismatch.as_str(), "shape-mismatch");
    }

    #[test]
    fn register_and_unregister() {
        let mut det = AnomalyDetector::new();
        assert!(det.register_model(make_config(1)));
        assert_eq!(det.registered_models(), 1);
        assert!(det.unregister_model(1));
        assert_eq!(det.registered_models(), 0);
    }

    #[test]
    fn unregister_nonexistent() {
        let mut det = AnomalyDetector::new();
        assert!(!det.unregister_model(99));
    }

    #[test]
    fn timing_within_bounds() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        assert!(det.check_timing(1, 50_000_000).is_none()); // 50ms, within bounds
    }

    #[test]
    fn timing_too_fast() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_timing(1, 500_000).unwrap(); // 0.5ms < 1ms
        assert_eq!(report.anomaly_type, AnomalyType::InferenceTooFast);
        assert_eq!(det.total_anomalies(), 1);
    }

    #[test]
    fn timing_too_slow() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_timing(1, 200_000_000).unwrap(); // 200ms > 100ms
        assert_eq!(report.anomaly_type, AnomalyType::InferenceTooSlow);
    }

    #[test]
    fn timing_unknown_model() {
        let mut det = AnomalyDetector::new();
        assert!(det.check_timing(999, 50_000_000).is_none());
    }

    #[test]
    fn output_in_range() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        assert!(det.check_output_value(1, 0, 0, 0.5).is_none());
    }

    #[test]
    fn output_out_of_range() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_output_value(1, 0, 42, 1.5).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::OutputOutOfRange);
        assert_eq!(report.element_index, 42);
    }

    #[test]
    fn output_below_range() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_output_value(1, 0, 0, -0.1).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::OutputOutOfRange);
    }

    #[test]
    fn output_unconfigured_tensor() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        // Output index 5 has no bounds configured
        assert!(det.check_output_value(1, 5, 0, 999.0).is_none());
    }

    #[test]
    fn input_nan_detected() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_input_value(1, 0, 7, f32::NAN).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::InputNaN);
        assert_eq!(report.element_index, 7);
    }

    #[test]
    fn input_inf_detected() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_input_value(1, 0, 0, f32::INFINITY).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::InputInf);
    }

    #[test]
    fn input_neg_inf_detected() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_input_value(1, 0, 0, f32::NEG_INFINITY).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::InputInf);
    }

    #[test]
    fn input_normal_value_ok() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        assert!(det.check_input_value(1, 0, 0, 0.5).is_none());
    }

    #[test]
    fn nan_inf_detection_disabled() {
        let mut det = AnomalyDetector::new();
        let mut config = make_config(1);
        config.nan_inf_detection = false;
        det.register_model(config);
        assert!(det.check_input_value(1, 0, 0, f32::NAN).is_none());
    }

    #[test]
    fn shape_valid() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        assert!(det.check_input_shape(1, &[1, 3, 224, 224]).is_none());
    }

    #[test]
    fn shape_wrong_dims() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_input_shape(1, &[1, 3, 224]).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::ShapeMismatch);
    }

    #[test]
    fn shape_wrong_values() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        let report = det.check_input_shape(1, &[1, 3, 256, 256]).unwrap();
        assert_eq!(report.anomaly_type, AnomalyType::ShapeMismatch);
    }

    #[test]
    fn shape_no_constraint() {
        let mut det = AnomalyDetector::new();
        let mut config = make_config(1);
        config.expected_input_shape = ExpectedShape::empty();
        det.register_model(config);
        assert!(det.check_input_shape(1, &[1, 2, 3]).is_none());
    }

    #[test]
    fn dynamic_shape_validation() {
        let mut det = AnomalyDetector::new();
        let mut config = make_config(1);
        // dim 0 = dynamic, dim 1 = fixed 3
        config.expected_input_shape = ExpectedShape {
            dims: [0, 3, 0, 0, 0, 0, 0, 0],
            ndim: 2,
            dynamic_min: 1,
            dynamic_max: 32,
        };
        det.register_model(config);

        // Batch size 16, 3 channels -> OK
        assert!(det.check_input_shape(1, &[16, 3]).is_none());
        // Batch size 0 -> below dynamic_min
        assert!(det.check_input_shape(1, &[0, 3]).is_some());
        // Batch size 64 -> above dynamic_max
        assert!(det.check_input_shape(1, &[64, 3]).is_some());
    }

    #[test]
    fn output_range_bounds_check() {
        let bounds = OutputRangeBounds::new(-1.0, 1.0);
        assert!(bounds.check(0.0));
        assert!(bounds.check(-1.0));
        assert!(bounds.check(1.0));
        assert!(!bounds.check(1.1));
        assert!(!bounds.check(-1.1));
    }

    #[test]
    fn output_range_inactive() {
        let bounds = OutputRangeBounds::inactive();
        assert!(bounds.check(f32::MAX));
        assert!(bounds.check(f32::MIN));
    }

    #[test]
    fn total_anomalies_accumulate() {
        let mut det = AnomalyDetector::new();
        det.register_model(make_config(1));
        det.check_timing(1, 500_000); // too fast
        det.check_input_value(1, 0, 0, f32::NAN); // nan
        det.check_output_value(1, 0, 0, 5.0); // out of range
        assert_eq!(det.total_anomalies(), 3);
    }
}
