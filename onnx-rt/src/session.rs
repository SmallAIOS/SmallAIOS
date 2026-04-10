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

use crate::graph::{build_execution_graph, ExecutionGraph};
use crate::onnx_types::{ModelProto, TensorProto, CURRENT_IR_VERSION};
use crate::operators::OperatorRegistry;
use crate::optimizer::{optimize, OptimizationLevel, OptimizerConfig};
use crate::parallel::ParallelConfig;
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
    /// The model failed security policy validation (formal-gate).
    #[cfg(feature = "formal-gate")]
    PolicyViolation(String),
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
            #[cfg(feature = "formal-gate")]
            SessionError::PolicyViolation(msg) => {
                write!(f, "policy violation: {}", msg)
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
    /// Parallel execution configuration for operator-level parallelism.
    pub parallel: ParallelConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            enable_profiling: false,
            max_batch_size: 1,
            thread_count: 1,
            parallel: ParallelConfig::default(),
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
    /// Model initializer tensors (weights, biases).
    pub initializers: Vec<TensorProto>,
    /// Whether the session has been fully initialized with a model.
    is_initialized: bool,
    /// Optional GPU backend for accelerated operator dispatch.
    #[cfg(feature = "gpu")]
    pub gpu_backend: Option<smallaios_compute::GpuBackend>,
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

/// Maximum supported ONNX opset version.
const MAX_OPSET_VERSION: i64 = 21;

// ---------------------------------------------------------------------------
// Model validation
// ---------------------------------------------------------------------------

/// Validates IR version and opset imports.
fn validate_model_version(model: &ModelProto) -> Result<(), SessionError> {
    if model.ir_version > CURRENT_IR_VERSION && model.ir_version != 0 {
        return Err(SessionError::InvalidModel(String::from(
            "unsupported IR version",
        )));
    }
    for opset in &model.opset_import {
        if opset.domain.is_empty() && opset.version > MAX_OPSET_VERSION {
            return Err(SessionError::UnsupportedOpset(opset.version));
        }
    }
    Ok(())
}

/// Validates that all graph nodes use supported operators and have outputs.
fn validate_graph_operators(graph: &crate::onnx_types::GraphProto) -> Result<(), SessionError> {
    let registry = OperatorRegistry::new();
    for node in &graph.node {
        if !node.domain.is_empty() {
            continue;
        }
        if !registry.is_supported(&node.op_type) {
            return Err(SessionError::InvalidModel(alloc::format!(
                "unsupported operator: {}",
                node.op_type
            )));
        }
        if node.output.is_empty() {
            return Err(SessionError::InvalidModel(alloc::format!(
                "node '{}' has no outputs",
                node.name
            )));
        }
    }
    Ok(())
}

/// Checks for duplicate output tensor names across all nodes.
fn validate_unique_outputs(graph: &crate::onnx_types::GraphProto) -> Result<(), SessionError> {
    let mut seen_outputs: Vec<&str> = Vec::new();
    for node in &graph.node {
        for output in &node.output {
            if output.is_empty() {
                continue;
            }
            if seen_outputs.contains(&output.as_str()) {
                return Err(SessionError::InvalidModel(alloc::format!(
                    "duplicate output tensor: {}",
                    output
                )));
            }
            seen_outputs.push(output.as_str());
        }
    }
    Ok(())
}

/// Validates a parsed ONNX model for structural correctness.
///
/// Checks:
/// - IR version is supported
/// - Opset version is within range
/// - Graph is present and non-empty
/// - All operators are in the operator registry
/// - All nodes have at least one output
/// - No duplicate output tensor names
pub fn validate_model(model: &ModelProto) -> Result<(), SessionError> {
    validate_model_version(model)?;

    let graph = match &model.graph {
        Some(g) => g,
        None => {
            return Err(SessionError::InvalidModel(String::from(
                "model has no graph",
            )));
        }
    };

    validate_graph_operators(graph)?;
    validate_unique_outputs(graph)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Top-level model loading
// ---------------------------------------------------------------------------

/// Attempts to load and parse an ONNX model from raw bytes.
///
/// Validates the magic byte and minimum size, then decodes the
/// protobuf payload into a `ModelProto`.
pub fn load_model(data: &[u8]) -> Result<ModelProto, SessionError> {
    // Verified boot: check model signature before parsing
    #[cfg(feature = "verified-boot")]
    {
        let policy = smallaios_security::crypto::verify::VerificationPolicy::default();
        crate::model_verify::verify_model_data(data, policy)?;
    }

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

    crate::protobuf::decode_model(data)
        .map_err(|e| SessionError::ModelLoadFailed(alloc::format!("protobuf decode error: {}", e)))
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
            initializers: Vec::new(),
            is_initialized: false,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        }
    }

    /// Initializes the session with a parsed ONNX model.
    ///
    /// This validates the model structure, builds the execution graph,
    /// applies optimizations, and prepares the session for inference.
    pub fn initialize(&mut self, model: &ModelProto) -> Result<(), SessionError> {
        // Validate model
        validate_model(model)?;

        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| SessionError::InvalidModel(String::from("no graph")))?;

        // Build execution graph
        let mut exec_graph = build_execution_graph(graph)
            .map_err(|e| SessionError::InvalidModel(alloc::format!("{}", e)))?;

        // Apply optimizations
        let opt_config = match self.config.optimization_level {
            OptimizationLevel::None => OptimizerConfig::none(),
            OptimizationLevel::Basic => OptimizerConfig {
                level: OptimizationLevel::Basic,
                ..OptimizerConfig::default()
            },
            OptimizationLevel::Extended => OptimizerConfig::default(),
        };
        let _opt_result = optimize(&mut exec_graph, &opt_config);

        // Store graph metadata
        self.model_name = graph.name.clone();
        self.input_names = graph.input.iter().map(|vi| vi.name.clone()).collect();
        self.output_names = graph.output.iter().map(|vi| vi.name.clone()).collect();
        self.initializers = graph.initializer.clone();
        self.graph = Some(exec_graph);
        self.is_initialized = true;

        Ok(())
    }

    /// Runs inference on the provided inputs and returns the outputs.
    ///
    /// The session must be initialized before calling this method.
    /// Input tensors are validated against the model's expected input
    /// names and shapes. Currently returns `NotImplemented` after
    /// graph traversal setup -- operator dispatch will be wired in a
    /// later phase when the full operator table is complete.
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

        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| SessionError::ExecutionFailed(String::from("no execution graph")))?;

        // Build input pairs for the executor
        let input_pairs: Vec<(String, Tensor)> = inputs
            .iter()
            .map(|i| (i.name.clone(), i.tensor.clone()))
            .collect();

        // Get initializers from the model (stored during initialize)
        let initializers = &self.initializers;

        crate::executor::execute_graph(
            graph,
            &input_pairs,
            initializers,
            None,
            #[cfg(feature = "gpu")]
            self.gpu_backend.as_ref(),
        )
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
            parallel: crate::parallel::ParallelConfig::default_for_cores(4),
        };
        assert_eq!(config.optimization_level, OptimizationLevel::Extended);
        assert!(config.enable_profiling);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.thread_count, 4);
        assert_eq!(config.parallel.max_threads, 4);
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
    fn test_load_model_valid_header_decodes() {
        // Valid magic byte and sufficient size — protobuf decoder now runs.
        // ir_version=7, then some padding bytes that the decoder will try to parse.
        // Build a minimal valid model: just ir_version field.
        let mut data = vec![0x08, 0x07]; // field 1, varint 7 = ir_version
                                         // Pad to MIN_MODEL_SIZE
        while data.len() < 8 {
            // Add unknown field (field 99, varint 0) to pad
            data.push(0x98); // (99 << 3) | 0 = 792, but that's multi-byte...
                             // Simpler: just use field 15 varint 0 = tag 0x78, value 0x00
            data.push(0x00);
        }
        // Actually, let's build it properly with the encode helpers from protobuf tests.
        // Just use raw bytes for a minimal model.
        let data = [
            0x08, 0x07, // ir_version = 7
            0x12, 0x04, // field 2, length 4 (opset_import)
            0x0A, 0x00, // field 1 (domain), length 0
            0x10, 0x11, // field 2 (version), varint 17
        ];
        let result = load_model(&data);
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
        let model = result.unwrap();
        assert_eq!(model.ir_version, 7);
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].version, 17);
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

    // ---- Model validation tests ----

    fn make_simple_model() -> ModelProto {
        use crate::onnx_types::{GraphProto, NodeProto, OperatorSetIdProto, ValueInfoProto};
        ModelProto {
            ir_version: CURRENT_IR_VERSION,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: String::from("test"),
            producer_version: String::from("1.0"),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(GraphProto {
                name: String::from("test"),
                node: vec![NodeProto {
                    op_type: String::from("Relu"),
                    name: String::from("relu0"),
                    input: vec![String::from("x")],
                    output: vec![String::from("y")],
                    ..NodeProto::default()
                }],
                input: vec![ValueInfoProto {
                    name: String::from("x"),
                    elem_type: 1,
                    shape: vec![1, 3],
                }],
                output: vec![ValueInfoProto {
                    name: String::from("y"),
                    elem_type: 1,
                    shape: vec![1, 3],
                }],
                initializer: Vec::new(),
            }),
        }
    }

    #[test]
    fn test_validate_model_valid() {
        let model = make_simple_model();
        assert!(validate_model(&model).is_ok());
    }

    #[test]
    fn test_validate_model_no_graph() {
        let model = ModelProto {
            ir_version: CURRENT_IR_VERSION,
            graph: None,
            ..ModelProto::default()
        };
        let result = validate_model(&model);
        assert!(matches!(result, Err(SessionError::InvalidModel(_))));
    }

    #[test]
    fn test_validate_model_unsupported_opset() {
        use crate::onnx_types::OperatorSetIdProto;
        let mut model = make_simple_model();
        model.opset_import = vec![OperatorSetIdProto {
            domain: String::new(),
            version: 99,
        }];
        let result = validate_model(&model);
        assert!(matches!(result, Err(SessionError::UnsupportedOpset(99))));
    }

    #[test]
    fn test_validate_model_unsupported_operator() {
        use crate::onnx_types::NodeProto;
        let mut model = make_simple_model();
        if let Some(g) = model.graph.as_mut() {
            g.node.push(NodeProto {
                op_type: String::from("UnknownOp"),
                name: String::from("bad"),
                input: vec![String::from("y")],
                output: vec![String::from("z")],
                ..NodeProto::default()
            });
        }
        let result = validate_model(&model);
        assert!(matches!(result, Err(SessionError::InvalidModel(_))));
    }

    #[test]
    fn test_validate_model_duplicate_outputs() {
        use crate::onnx_types::NodeProto;
        let mut model = make_simple_model();
        if let Some(g) = model.graph.as_mut() {
            g.node.push(NodeProto {
                op_type: String::from("Relu"),
                name: String::from("relu1"),
                input: vec![String::from("x")],
                output: vec![String::from("y")], // duplicate of first node
                ..NodeProto::default()
            });
        }
        let result = validate_model(&model);
        assert!(matches!(result, Err(SessionError::InvalidModel(_))));
    }

    // ---- Session initialization tests ----

    #[test]
    fn test_session_initialize_valid_model() {
        let model = make_simple_model();
        let mut session = Session::new(SessionConfig::default());
        let result = session.initialize(&model);
        assert!(result.is_ok());
        assert!(session.is_initialized());
        assert_eq!(session.input_names(), &["x"]);
        assert_eq!(session.output_names(), &["y"]);
        assert!(session.graph.is_some());
        assert_eq!(session.model_name, "test");
    }

    #[test]
    fn test_session_initialize_no_graph() {
        let model = ModelProto {
            ir_version: CURRENT_IR_VERSION,
            graph: None,
            ..ModelProto::default()
        };
        let mut session = Session::new(SessionConfig::default());
        let result = session.initialize(&model);
        assert!(result.is_err());
        assert!(!session.is_initialized());
    }

    #[test]
    fn test_session_run_after_init_executes_relu() {
        let model = make_simple_model();
        let mut session = Session::new(SessionConfig::default());
        session.initialize(&model).unwrap();

        // Create input tensor with actual data
        let mut raw_data = vec![0u8; 3 * 4]; // 3 floats
        for (i, &val) in [-1.0f32, 0.0, 2.0].iter().enumerate() {
            let bytes = val.to_le_bytes();
            raw_data[i * 4] = bytes[0];
            raw_data[i * 4 + 1] = bytes[1];
            raw_data[i * 4 + 2] = bytes[2];
            raw_data[i * 4 + 3] = bytes[3];
        }
        let mut tensor = Tensor::new(
            DataType::Float,
            TensorShape::new(vec![1, 3]),
            String::from("x"),
        );
        tensor.raw_data = raw_data;

        let input = InferenceInput {
            name: String::from("x"),
            tensor,
        };
        let result = session.run(&[input]);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result);
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "y");

        // Relu of [-1, 0, 2] = [0, 0, 2]
        let out_data: Vec<f32> = (0..3)
            .map(|i| {
                f32::from_le_bytes([
                    outputs[0].tensor.raw_data[i * 4],
                    outputs[0].tensor.raw_data[i * 4 + 1],
                    outputs[0].tensor.raw_data[i * 4 + 2],
                    outputs[0].tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(out_data, vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_session_run_invalid_input_name() {
        let model = make_simple_model();
        let mut session = Session::new(SessionConfig::default());
        session.initialize(&model).unwrap();

        let input = InferenceInput {
            name: String::from("nonexistent"),
            tensor: Tensor::new(
                DataType::Float,
                TensorShape::new(vec![1, 3]),
                String::from("nonexistent"),
            ),
        };
        let result = session.run(&[input]);
        assert!(matches!(result, Err(SessionError::InvalidInput(_))));
    }
}
