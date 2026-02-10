// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Inference session API for the SmallAIOS ONNX runtime.
//!
//! Provides the top-level interface for loading ONNX models, configuring
//! execution parameters, and running inference. Each `Session` encapsulates
//! a compiled execution graph and manages input/output validation.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::graph::ExecutionGraph;
use crate::onnx_types::ModelProto;
use crate::optimizer::OptimizationLevel;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during session operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// The model data could not be loaded or parsed.
    ModelLoadFailed(String),
    /// The loaded model is structurally invalid.
    InvalidModel(String),
    /// The model requires an ONNX opset version that is not supported.
    UnsupportedOpset(i64),
    /// An error occurred during inference execution.
    ExecutionFailed(String),
    /// A provided input tensor name or shape does not match the model.
    InvalidInput(String),
    /// A requested output tensor name does not exist in the model.
    InvalidOutput(String),
    /// The operation is defined but not yet implemented.
    NotImplemented,
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::ModelLoadFailed(msg) => {
                write!(f, "model load failed: {}", msg)
            }
            SessionError::InvalidModel(msg) => {
                write!(f, "invalid model: {}", msg)
            }
            SessionError::UnsupportedOpset(version) => {
                write!(f, "unsupported opset version: {}", version)
            }
            SessionError::ExecutionFailed(msg) => {
                write!(f, "execution failed: {}", msg)
            }
            SessionError::InvalidInput(msg) => {
                write!(f, "invalid input: {}", msg)
            }
            SessionError::InvalidOutput(msg) => {
                write!(f, "invalid output: {}", msg)
            }
            SessionError::NotImplemented => {
                write!(f, "session operation not implemented")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Session configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for an inference session.
///
/// Controls optimization, profiling, batching, and threading behavior.
pub struct SessionConfig {
    /// Graph optimization level to apply after model loading.
    pub optimization_level: OptimizationLevel,
    /// Whether to collect profiling data during inference.
    pub enable_profiling: bool,
    /// Maximum batch size for dynamic batching.
    pub max_batch_size: usize,
    /// Number of intra-operator threads for parallel computation.
    pub thread_count: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            enable_profiling: false,
            max_batch_size: 1,
            thread_count: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Inference I/O types
// ---------------------------------------------------------------------------

/// A named tensor provided as input to an inference session.
#[derive(Debug)]
pub struct InferenceInput {
    /// The name of the model input this tensor binds to.
    pub name: String,
    /// The input tensor data.
    pub tensor: Tensor,
}

/// A named tensor produced as output from an inference session.
#[derive(Debug)]
pub struct InferenceOutput {
    /// The name of the model output this tensor corresponds to.
    pub name: String,
    /// The output tensor data.
    pub tensor: Tensor,
}

// ---------------------------------------------------------------------------
// Session identifier
// ---------------------------------------------------------------------------

/// Unique identifier for an inference session.
///
/// Session IDs are monotonically increasing within a runtime instance
/// and can be used for logging, profiling, and resource tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "session-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// An inference session holding a compiled ONNX model graph.
///
/// A session is created with a configuration, then initialized with a
/// parsed model. Once initialized, it can execute inference on input
/// tensors and produce output tensors.
pub struct Session {
    /// Unique session identifier.
    pub id: SessionId,
    /// Session configuration.
    pub config: SessionConfig,
    /// Human-readable model name (from ONNX metadata).
    pub model_name: String,
    /// The compiled execution graph, populated after initialization.
    pub graph: Option<ExecutionGraph>,
    /// Names of expected model inputs, in order.
    pub input_names: Vec<String>,
    /// Names of expected model outputs, in order.
    pub output_names: Vec<String>,
    /// Whether the session has been fully initialized with a model.
    is_initialized: bool,
}

/// ONNX model file magic bytes: `\x08` (field 1, varint wire type).
///
/// A valid ONNX protobuf file starts with the ir_version field
/// (field number 1, wire type 0 = varint), giving a tag byte of 0x08.
const ONNX_MAGIC: u8 = 0x08;

/// Minimum valid ONNX model size in bytes.
///
/// An ONNX model must contain at least the ir_version field (tag + varint)
/// and a graph field. Anything smaller than this is rejected.
const MIN_MODEL_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Top-level model loading
// ---------------------------------------------------------------------------

/// Attempts to load and parse an ONNX model from raw bytes.
///
/// Validates the magic byte and minimum size before parsing.
/// Currently returns `NotImplemented` after validation passes;
/// full protobuf parsing will be added in a later phase.
pub fn load_model(data: &[u8]) -> Result<ModelProto, SessionError> {
    if data.len() < MIN_MODEL_SIZE {
        return Err(SessionError::ModelLoadFailed(String::from(
            "data too small to be a valid ONNX model",
        )));
    }

    if data[0] != ONNX_MAGIC {
        return Err(SessionError::ModelLoadFailed(String::from(
            "invalid ONNX magic byte",
        )));
    }

    // Stub: full protobuf decoding not yet implemented.
    Err(SessionError::NotImplemented)
}

// ---------------------------------------------------------------------------
// Session implementation
// ---------------------------------------------------------------------------

impl Session {
    /// Creates a new uninitialized session with the given configuration.
    ///
    /// The session must be initialized with [`Session::initialize`] before
    /// inference can be run. A unique session ID is assigned using a
    /// monotonic counter.
    pub fn new(config: SessionConfig) -> Self {
        use core::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        Self {
            id: SessionId(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
            config,
            model_name: String::new(),
            graph: None,
            input_names: Vec::new(),
            output_names: Vec::new(),
            is_initialized: false,
        }
    }

    /// Initializes the session with a parsed ONNX model.
    ///
    /// This validates the model structure, builds the execution graph,
    /// applies optimizations, and prepares the session for inference.
    /// Currently returns `NotImplemented`.
    pub fn initialize(&mut self, model: &ModelProto) -> Result<(), SessionError> {
        let _ = model;
        Err(SessionError::NotImplemented)
    }

    /// Runs inference on the provided inputs and returns the outputs.
    ///
    /// The session must be initialized before calling this method.
    /// Input tensors are validated against the model's expected input
    /// names and shapes. Currently returns `NotImplemented` after
    /// validation checks pass.
    pub fn run(&self, inputs: &[InferenceInput]) -> Result<Vec<InferenceOutput>, SessionError> {
        if !self.is_initialized {
            return Err(SessionError::ExecutionFailed(String::from(
                "session not initialized",
            )));
        }

        if inputs.is_empty() {
            return Err(SessionError::InvalidInput(String::from(
                "no inputs provided",
            )));
        }

        // Validate that each input name matches a known model input.
        for input in inputs {
            if !self.input_names.contains(&input.name) {
                return Err(SessionError::InvalidInput(input.name.clone()));
            }
        }

        // Stub: actual graph execution not yet implemented.
        Err(SessionError::NotImplemented)
    }

    /// Returns the names of the model's expected inputs.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Returns the names of the model's expected outputs.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Returns `true` if the session has been initialized with a model.
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;
    use crate::tensor::TensorShape;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

    // ---- Session creation tests ----

    #[test]
    fn test_session_new_defaults() {
        let session = Session::new(SessionConfig::default());
        assert!(!session.is_initialized());
        assert!(session.input_names().is_empty());
        assert!(session.output_names().is_empty());
        assert!(session.graph.is_none());
        assert!(session.model_name.is_empty());
    }

    #[test]
    fn test_session_unique_ids() {
        let s1 = Session::new(SessionConfig::default());
        let s2 = Session::new(SessionConfig::default());
        assert_ne!(s1.id, s2.id);
        // IDs should be monotonically increasing.
        assert!(s2.id.0 > s1.id.0);
    }

    #[test]
    fn test_session_id_display() {
        let id = SessionId(42);
        let s = format!("{}", id);
        assert_eq!(s, "session-42");
    }

    // ---- SessionConfig tests ----

    #[test]
    fn test_config_defaults() {
        let config = SessionConfig::default();
        assert_eq!(config.optimization_level, OptimizationLevel::Basic);
        assert!(!config.enable_profiling);
        assert_eq!(config.max_batch_size, 1);
        assert_eq!(config.thread_count, 1);
    }

    #[test]
    fn test_config_custom() {
        let config = SessionConfig {
            optimization_level: OptimizationLevel::Extended,
            enable_profiling: true,
            max_batch_size: 8,
            thread_count: 4,
        };
        assert_eq!(config.optimization_level, OptimizationLevel::Extended);
        assert!(config.enable_profiling);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.thread_count, 4);
    }

    // ---- load_model tests ----

    #[test]
    fn test_load_model_empty_data() {
        let result = load_model(&[]);
        assert!(result.is_err());
        match result {
            Err(SessionError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("too small"));
            }
            other => panic!("expected ModelLoadFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_load_model_too_small() {
        let data = [0x08, 0x01, 0x00, 0x00];
        let result = load_model(&data);
        assert!(result.is_err());
        match result {
            Err(SessionError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("too small"));
            }
            other => panic!("expected ModelLoadFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_load_model_invalid_magic() {
        let data = [0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = load_model(&data);
        assert!(result.is_err());
        match result {
            Err(SessionError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("magic"));
            }
            other => panic!("expected ModelLoadFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_load_model_valid_header_returns_not_implemented() {
        // Valid magic byte and sufficient size, but stub returns NotImplemented.
        let data = [0x08, 0x07, 0x12, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = load_model(&data);
        assert_eq!(result, Err(SessionError::NotImplemented));
    }

    // ---- Session run tests ----

    #[test]
    fn test_run_without_init_returns_error() {
        let session = Session::new(SessionConfig::default());
        let input = InferenceInput {
            name: String::from("input"),
            tensor: Tensor::new(
                DataType::Float,
                TensorShape::new(vec![1, 3, 224, 224]),
                String::from("input"),
            ),
        };
        let result = session.run(&[input]);
        assert!(result.is_err());
        match result {
            Err(SessionError::ExecutionFailed(msg)) => {
                assert!(msg.contains("not initialized"));
            }
            other => panic!("expected ExecutionFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_session_not_initialized_by_default() {
        let session = Session::new(SessionConfig::default());
        assert!(!session.is_initialized());
    }

    // ---- SessionError Display tests ----

    #[test]
    fn test_session_error_display() {
        assert_eq!(
            format!(
                "{}",
                SessionError::ModelLoadFailed(String::from("bad data"))
            ),
            "model load failed: bad data"
        );
        assert_eq!(
            format!("{}", SessionError::InvalidModel(String::from("no graph"))),
            "invalid model: no graph"
        );
        assert_eq!(
            format!("{}", SessionError::UnsupportedOpset(99)),
            "unsupported opset version: 99"
        );
        assert_eq!(
            format!("{}", SessionError::ExecutionFailed(String::from("oom"))),
            "execution failed: oom"
        );
        assert_eq!(
            format!("{}", SessionError::InvalidInput(String::from("missing x"))),
            "invalid input: missing x"
        );
        assert_eq!(
            format!("{}", SessionError::InvalidOutput(String::from("no y"))),
            "invalid output: no y"
        );
        assert_eq!(
            format!("{}", SessionError::NotImplemented),
            "session operation not implemented"
        );
    }

    // ---- InferenceInput / InferenceOutput tests ----

    #[test]
    fn test_inference_input_construction() {
        let input = InferenceInput {
            name: String::from("images"),
            tensor: Tensor::new(
                DataType::Float,
                TensorShape::new(vec![1, 3, 224, 224]),
                String::from("images"),
            ),
        };
        assert_eq!(input.name, "images");
        assert_eq!(input.tensor.data_type, DataType::Float);
        assert_eq!(input.tensor.shape.ndim(), 4);
    }
}
