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
    /// A `SessionConfig` field is set to an unsupported value (e.g.
    /// `StreamConfig::Overlap { transfer_streams: 5 }` exceeds the
    /// runtime cap).
    InvalidConfig(String),
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
            SessionError::InvalidConfig(msg) => {
                write!(f, "invalid config: {}", msg)
            }
            #[cfg(feature = "formal-gate")]
            SessionError::PolicyViolation(msg) => {
                write!(f, "policy violation: {}", msg)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU configuration
// ---------------------------------------------------------------------------

/// GPU backend type for accelerated operator dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendType {
    /// Apple Metal (macOS / Apple Silicon).
    Metal,
    // Future: Cuda, Vulkan, Rocm, LevelZero
}

/// Configuration for GPU-accelerated inference.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// The GPU backend to use.
    pub backend: GpuBackendType,
}

impl GpuConfig {
    /// Creates a GPU configuration for the Metal backend.
    pub fn metal() -> Self {
        Self {
            backend: GpuBackendType::Metal,
        }
    }
}

// ---------------------------------------------------------------------------
// Session configuration
// ---------------------------------------------------------------------------

/// Type alias for the device-resident initializer cache slot
/// (per-Session, lazy-populated by the hybrid executor).
#[cfg(feature = "cuda")]
pub type DeviceInitCacheSlot = Option<
    alloc::sync::Arc<
        alloc::collections::BTreeMap<
            String,
            alloc::sync::Arc<crate::cuda::gpu_executor::DeviceTensor>,
        >,
    >,
>;

/// Selects how the runtime routes operators to GPU vs CPU when a CUDA
/// runtime is attached.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GpuResidency {
    /// Existing per-op dispatch with host-resident intermediates.
    /// Each GPU-eligible op copies its inputs to device, runs, copies
    /// the output back to host, and the next op starts from host
    /// memory. Default for backward compatibility.
    #[default]
    OpByOp,
    /// Track tensor location; keep activations device-resident across
    /// adjacent GPU-supported ops, only copying back when crossing
    /// into a CPU-only op or producing a graph output.
    Hybrid,
}

/// Controls how the GPU executor maps work onto CUDA streams.
///
/// `SingleStream` (the default) issues every cuDNN / cuBLAS call,
/// every H2D copy, and every D2H copy on the default stream — the
/// behavior since the GPU executor was introduced. `Overlap` allocates
/// dedicated H2D and D2H streams alongside the compute stream so that
/// the next inference's input upload can happen while the current
/// inference's compute runs and its output downloads. Cross-stream
/// ordering is enforced via CUDA events so the host blocks only once
/// per inference at the final D2H synchronize.
///
/// `transfer_streams` is capped at 2; beyond that, contention on the
/// PCIe / NVLink fabric dominates and throughput regresses.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum StreamConfig {
    /// Every operation runs on the default CUDA stream. Default for
    /// backward compatibility — no behavior change for existing
    /// callers.
    #[default]
    SingleStream,
    /// Allocate `transfer_streams` H2D + `transfer_streams` D2H
    /// streams in addition to the compute stream and gate them via
    /// CUDA events. `transfer_streams` is validated `<= 2` at
    /// `Session::new`.
    Overlap { transfer_streams: usize },
}

/// Controls CUDA Graph capture / replay for the hybrid GPU executor.
///
/// `Off` (the default) preserves the existing per-op dispatch path —
/// every cuDNN / cuBLAS launch is host-issued and pays its own
/// per-launch overhead. `Capture` records the op sequence on the first
/// hybrid inference and replays it as a single `cudaGraphLaunch` on
/// subsequent runs, eliminating the ~10–50 µs / op host overhead.
///
/// Only meaningful when paired with [`GpuResidency::Hybrid`]; `OpByOp`
/// dispatch is incompatible with capture because each op crosses the
/// CPU boundary. The runtime silently treats `Capture` as `Off` for
/// `OpByOp` sessions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraphMode {
    /// Disable graph capture. The hybrid executor runs the per-op
    /// dispatch loop on every inference. Default for backward
    /// compatibility — no behavior change for existing callers.
    #[default]
    Off,
    /// Capture the hybrid executor's op sequence on the first
    /// inference of a given input shape and replay it as a single
    /// `cudaGraphLaunch` on subsequent inferences with the same
    /// shape. Falls back to per-op dispatch if capture fails or the
    /// shape changes more often than ~1% of inferences.
    Capture,
}

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
    /// Optional GPU configuration for accelerated dispatch.
    pub gpu_config: Option<GpuConfig>,
    /// GPU residency mode controlling whether intermediate activations
    /// stay device-resident across adjacent GPU-supported ops.
    pub gpu_residency: GpuResidency,
    /// CUDA Graph capture mode for the hybrid executor. Only takes
    /// effect when `gpu_residency == GpuResidency::Hybrid`.
    pub cuda_graph: CudaGraphMode,
    /// CUDA stream layout for the GPU executor. Default is
    /// `SingleStream` (every op on the default stream). `Overlap`
    /// allocates dedicated H2D + D2H streams alongside the compute
    /// stream so adjacent inferences can overlap input upload, GPU
    /// compute, and output download.
    pub stream_config: StreamConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            enable_profiling: false,
            max_batch_size: 1,
            thread_count: 1,
            parallel: ParallelConfig::default(),
            gpu_config: None,
            gpu_residency: GpuResidency::default(),
            cuda_graph: CudaGraphMode::default(),
            stream_config: StreamConfig::default(),
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
// Session kind
// ---------------------------------------------------------------------------

/// Discriminator describing how a [`Session`] was constructed, and which
/// execution path [`Session::run`] dispatches through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    /// Loaded from a `.onnx` protobuf file via [`Session::initialize`].
    /// Executes via the mainline [`crate::executor::execute_graph`] path.
    Onnx,
    /// Loaded via [`Session::from_safetensors`]. Weights are resident on
    /// the GPU and inference dispatches through
    /// [`crate::cuda::execute_graph_gpu_with_weights`] with a persistent
    /// [`crate::cuda::GpuKvCache`] held by the session.
    Safetensors,
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
    /// Optional Metal GPU dispatcher for accelerated operator execution.
    ///
    /// Wrapped in `RefCell` to allow `&self` on `run()` while the
    /// dispatcher mutates its internal cache and kernel state.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub metal_dispatcher: core::cell::RefCell<Option<crate::metal_dispatch::MetalDispatcher>>,
    /// Optional CUDA runtime for real GPU dispatch (container mode).
    /// Shared via Arc so multiple sessions can use the same GPU context.
    #[cfg(feature = "cuda")]
    pub cuda_runtime: Option<alloc::sync::Arc<crate::cuda::CudaRuntime>>,
    /// Optional GPU-resident KV cache for autoregressive decoding.
    ///
    /// Populated by `Session::from_safetensors` (Section 9); `None` for
    /// sessions that thread KV state externally through input tensors.
    /// Interior mutability via `Mutex` so `&Session` can update the cache
    /// across successive `Session::run()` calls.
    #[cfg(feature = "cuda")]
    pub kv_cache: Option<alloc::sync::Arc<std::sync::Mutex<crate::cuda::GpuKvCache>>>,
    /// Pre-loaded GPU weight map for safetensors sessions.
    ///
    /// Populated by [`Session::from_safetensors`] via
    /// [`crate::cuda::initializers_to_gpu`]. Wrapped in `Arc` so
    /// [`Session::run`] (which only has `&self`) can hand it to
    /// [`crate::cuda::execute_graph_gpu_with_weights`] without moving.
    #[cfg(feature = "cuda")]
    pub gpu_weights: Option<
        alloc::sync::Arc<
            alloc::collections::BTreeMap<String, crate::cuda::gpu_executor::DeviceTensor>,
        >,
    >,
    /// Cache of model initializers preloaded to device memory for the
    /// hybrid executor (`GpuResidency::Hybrid`). Populated lazily on
    /// the first `Session::run` call after the session is configured
    /// for hybrid mode and a CUDA runtime is attached. Stored as
    /// `Arc<DeviceTensor>` so the hybrid value map can share buffers
    /// across inferences without device→device memcpy.
    #[cfg(feature = "cuda")]
    pub device_initializer_cache: core::cell::RefCell<DeviceInitCacheSlot>,
    /// Per-Session CUDA Graph cache for the hybrid + capture path.
    /// Populated lazily on the first `Session::run` call when the
    /// session is configured for `GpuResidency::Hybrid` and
    /// `CudaGraphMode::Capture`. The cache holds a dedicated CUDA
    /// stream and one or more captured `cudaGraphExec_t` keyed by
    /// input shape + dtype.
    #[cfg(feature = "cuda")]
    pub cuda_graph_cache: core::cell::RefCell<crate::cuda::graph_cache::CudaGraphCacheSlot>,
    /// Per-Session CUDA stream pool for multi-stream overlap. `None`
    /// when `SessionConfig::stream_config = SingleStream` (default)
    /// or before the first multi-stream inference. Lazily allocated
    /// by [`Session::ensure_stream_pool`] on the first
    /// `Overlap`-mode `run` call. Holds one compute stream + N H2D +
    /// N D2H streams + an event reuse pool.
    #[cfg(feature = "cuda")]
    pub stream_pool: core::cell::RefCell<Option<crate::cuda::streams::StreamPool>>,
    /// Discriminator for the session's origin / dispatch path.
    pub kind: SessionKind,
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
// Gemma KV-cache layer pattern helper
// ---------------------------------------------------------------------------

/// Build the per-layer [`crate::cuda::LayerKind`] pattern for Gemma's
/// mixed sliding-window / global attention stack.
///
/// The final layer and every `sliding_window_pattern`-th layer (counting
/// from 1) is a global-attention layer; all others are sliding-window
/// local attention with the config's `sliding_window` size.
#[cfg(all(feature = "safetensors", feature = "cuda"))]
fn layer_kinds_for_gemma(config: &crate::model_loader::GemmaConfig) -> Vec<crate::cuda::LayerKind> {
    let n = config.num_hidden_layers;
    let pattern = config.sliding_window_pattern.max(1);
    (0..n)
        .map(|i| {
            let is_global = i + 1 == n || (i + 1) % pattern == 0;
            if is_global {
                crate::cuda::LayerKind::Global
            } else {
                crate::cuda::LayerKind::SlidingWindow(config.sliding_window.max(1))
            }
        })
        .collect()
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

        // Initialize Metal dispatcher if configured.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let metal_dispatcher = if config
            .gpu_config
            .as_ref()
            .map(|c| c.backend == GpuBackendType::Metal)
            .unwrap_or(false)
        {
            crate::metal_dispatch::MetalDispatcher::new().ok()
        } else {
            None
        };

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
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal_dispatcher: core::cell::RefCell::new(metal_dispatcher),
            #[cfg(feature = "cuda")]
            cuda_runtime: None,
            #[cfg(feature = "cuda")]
            kv_cache: None,
            #[cfg(feature = "cuda")]
            gpu_weights: None,
            #[cfg(feature = "cuda")]
            device_initializer_cache: core::cell::RefCell::new(None),
            #[cfg(feature = "cuda")]
            cuda_graph_cache: core::cell::RefCell::new(None),
            #[cfg(feature = "cuda")]
            stream_pool: core::cell::RefCell::new(None),
            kind: SessionKind::Onnx,
        }
    }

    /// Lazily allocate the per-Session [`StreamPool`] on the first
    /// multi-stream inference. Validates `transfer_streams <= 2`
    /// (returns [`SessionError::InvalidConfig`] if exceeded).
    /// Returns immediately if the pool is already initialized or if
    /// `stream_config` is `SingleStream`.
    #[cfg(feature = "cuda")]
    pub fn ensure_stream_pool(&self) -> Result<(), SessionError> {
        let transfer_streams = match self.config.stream_config {
            StreamConfig::SingleStream => return Ok(()),
            StreamConfig::Overlap { transfer_streams } => transfer_streams,
        };
        if transfer_streams > 2 {
            return Err(SessionError::InvalidConfig(alloc::format!(
                "transfer_streams must be <= 2 (got {})",
                transfer_streams
            )));
        }
        let mut slot = self.stream_pool.borrow_mut();
        if slot.is_some() {
            return Ok(());
        }
        let pool = crate::cuda::streams::StreamPool::new(transfer_streams)
            .map_err(|e| SessionError::ExecutionFailed(alloc::format!("StreamPool init: {}", e)))?;
        *slot = Some(pool);
        Ok(())
    }

    /// Creates a new session with Metal GPU acceleration enabled.
    ///
    /// This is a convenience constructor that configures the session
    /// to use the Metal backend for supported operators, falling back
    /// to CPU for unsupported ones.
    pub fn with_gpu(gpu_config: GpuConfig) -> Self {
        let config = SessionConfig {
            gpu_config: Some(gpu_config),
            ..SessionConfig::default()
        };
        Self::new(config)
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

        match self.kind {
            SessionKind::Onnx => {
                // Build input pairs for the executor
                let input_pairs: Vec<(String, Tensor)> = inputs
                    .iter()
                    .map(|i| (i.name.clone(), i.tensor.clone()))
                    .collect();

                // Get initializers from the model (stored during initialize)
                let initializers = &self.initializers;

                // Route to the hybrid executor when the user opts in via
                // `SessionConfig::gpu_residency = GpuResidency::Hybrid`
                // and a CUDA runtime is attached.
                #[cfg(feature = "cuda")]
                if matches!(self.config.gpu_residency, GpuResidency::Hybrid) {
                    use alloc::collections::BTreeMap;
                    use alloc::sync::Arc;
                    if let Some(rt) = self.cuda_runtime.as_deref() {
                        // Lazy-populate the device-resident initializer
                        // cache on first hybrid run. Skips re-decoding
                        // every TensorProto and re-uploading every
                        // initializer to device on each inference.
                        let cache = {
                            let mut slot = self.device_initializer_cache.borrow_mut();
                            if slot.is_none() {
                                let mut map: BTreeMap<
                                    String,
                                    Arc<crate::cuda::gpu_executor::DeviceTensor>,
                                > = BTreeMap::new();
                                for init in initializers {
                                    if let Some(host) = crate::executor::tensor_from_proto(init) {
                                        let dev =
                                            crate::cuda::gpu_executor::DeviceTensor::from_host(
                                                &host,
                                            )
                                            .map_err(
                                                |e| {
                                                    SessionError::ExecutionFailed(alloc::format!(
                                                        "device cache init: {}",
                                                        e
                                                    ))
                                                },
                                            )?;
                                        map.insert(init.name.clone(), Arc::new(dev));
                                    }
                                }
                                *slot = Some(Arc::new(map));
                            }
                            slot.as_ref().unwrap().clone()
                        };
                        let mut cache_slot = self.cuda_graph_cache.borrow_mut();
                        // Lazily allocate the multi-stream pool (no-op
                        // on SingleStream config). Errors propagate as
                        // SessionError::InvalidConfig before any
                        // inference work happens.
                        self.ensure_stream_pool()?;
                        let pool_slot = self.stream_pool.borrow();
                        return crate::executor_hybrid::execute_graph_hybrid_with_capture(
                            graph,
                            &input_pairs,
                            initializers,
                            rt,
                            Some(&cache),
                            &mut cache_slot,
                            self.config.cuda_graph,
                            pool_slot.as_ref(),
                        );
                    }
                }

                #[cfg(all(feature = "metal", target_os = "macos"))]
                let mut metal_guard = self.metal_dispatcher.borrow_mut();

                crate::executor::execute_graph(
                    graph,
                    &input_pairs,
                    initializers,
                    None,
                    None,
                    &crate::profile::OperatorBudget::DEFAULT,
                    &crate::profile::NullTimeSource,
                    #[cfg(feature = "gpu")]
                    self.gpu_backend.as_ref(),
                    #[cfg(feature = "cuda")]
                    self.cuda_runtime.as_deref(),
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    metal_guard.as_mut(),
                )
            }
            SessionKind::Safetensors => {
                #[cfg(feature = "cuda")]
                {
                    self.run_safetensors(graph, inputs)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = graph;
                    Err(SessionError::ExecutionFailed(String::from(
                        "safetensors session requires the `cuda` feature",
                    )))
                }
            }
        }
    }

    /// GPU-resident run path for safetensors-backed sessions.
    ///
    /// Converts the caller's host input tensors to [`DeviceTensor`]s,
    /// dispatches through
    /// [`crate::cuda::execute_graph_gpu_with_weights`] with the
    /// preloaded weight map, and copies the outputs back to host
    /// tensors for the [`InferenceOutput`] return value.
    ///
    /// TODO(kv-cache): the KV cache is allocated by
    /// [`Session::from_safetensors`] and locked here to prove the
    /// field is used, but it is **not yet threaded into the attention
    /// dispatch**. The forward pass currently recomputes K/V for every
    /// token as if it were the first. Wiring `KvView` through into the
    /// GQA operator is deferred to a follow-up; see the section 9
    /// design notes.
    #[cfg(feature = "cuda")]
    fn run_safetensors(
        &self,
        graph: &ExecutionGraph,
        inputs: &[InferenceInput],
    ) -> Result<Vec<InferenceOutput>, SessionError> {
        let runtime = self.cuda_runtime.as_ref().ok_or_else(|| {
            SessionError::ExecutionFailed(String::from("safetensors session missing cuda runtime"))
        })?;

        // Convert host inputs -> device tensors.
        let mut input_pairs: Vec<(String, crate::cuda::gpu_executor::DeviceTensor)> =
            Vec::with_capacity(inputs.len());
        for inp in inputs {
            let dev = crate::cuda::tensor_to_device(&inp.tensor, runtime).map_err(|e| {
                SessionError::ExecutionFailed(alloc::format!(
                    "tensor_to_device('{}'): {}",
                    inp.name,
                    e
                ))
            })?;
            input_pairs.push((inp.name.clone(), dev));
        }

        // Lock the KV cache for the duration of the forward pass.
        // §7.4: pass the locked guard into the executor so the dispatcher
        // routes each GroupQueryAttention node to its layer slot via
        // gpu_gqa_with_cache. Resolves the TODO from
        // safetensors-model-loader-v1 §9.
        let mut kv_guard = if let Some(cache) = &self.kv_cache {
            Some(cache.lock().map_err(|_| {
                SessionError::ExecutionFailed(String::from("kv_cache mutex poisoned"))
            })?)
        } else {
            None
        };

        let weights_ref = self.gpu_weights.as_deref();
        let cache_ref: Option<&mut crate::cuda::GpuKvCache> = kv_guard.as_deref_mut();

        let device_outputs = crate::cuda::gpu_executor::execute_graph_gpu_with_weights_and_cache(
            graph,
            &input_pairs,
            &self.initializers,
            weights_ref,
            cache_ref,
            runtime,
        )?;

        // Copy device outputs back to host tensors.
        let mut outputs: Vec<InferenceOutput> = Vec::with_capacity(device_outputs.len());
        for (dev_t, name) in device_outputs.iter().zip(self.output_names.iter()) {
            let host_t = dev_t.to_host().map_err(|e| {
                SessionError::ExecutionFailed(alloc::format!(
                    "device->host copy for '{}': {}",
                    name,
                    e
                ))
            })?;
            outputs.push(InferenceOutput {
                name: name.clone(),
                tensor: host_t,
            });
        }
        Ok(outputs)
    }

    /// Reset the GPU-resident KV cache, if present.
    ///
    /// Called by callers driving a fresh generation sequence so that
    /// the next `run()` call starts at token position 0. No-op for
    /// sessions without a KV cache (e.g. ONNX-backed sessions).
    #[cfg(feature = "cuda")]
    pub fn reset_kv_cache(&self) -> Result<(), SessionError> {
        if let Some(cache) = &self.kv_cache {
            cache
                .lock()
                .map_err(|_| {
                    SessionError::ExecutionFailed(String::from("kv_cache mutex poisoned"))
                })?
                .reset();
        }
        Ok(())
    }

    /// Returns the current committed token position of the KV cache, or
    /// `None` for sessions without a cache (ONNX-backed sessions).
    ///
    /// Equivalent to [`crate::cuda::GpuKvCache::current_position`] on
    /// the locked cache. Used by tests to assert that the dispatcher
    /// is actually appending tokens during the forward pass.
    #[cfg(feature = "cuda")]
    pub fn kv_cache_position(&self) -> Result<Option<usize>, SessionError> {
        match &self.kv_cache {
            Some(cache) => {
                let g = cache.lock().map_err(|_| {
                    SessionError::ExecutionFailed(String::from("kv_cache mutex poisoned"))
                })?;
                Ok(Some(g.current_position()))
            }
            None => Ok(None),
        }
    }

    /// Runs inference and returns a per-operator timing profile alongside
    /// the outputs.
    ///
    /// Uses [`crate::profile::StdTimeSource`] for wall-clock measurement.
    /// Only available when the `std` feature is enabled (container mode).
    ///
    /// If any operator exceeds its hard time limit
    /// ([`crate::profile::BudgetResult::HardLimit`]), execution is aborted
    /// with a [`SessionError::ExecutionFailed`] and the partial profile is
    /// not returned.
    #[cfg(feature = "std")]
    pub fn run_with_profile(
        &self,
        inputs: &[InferenceInput],
    ) -> Result<(Vec<InferenceOutput>, crate::profile::InferenceProfile), SessionError> {
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

        for input in inputs {
            if !self.input_names.contains(&input.name) {
                return Err(SessionError::InvalidInput(input.name.clone()));
            }
        }

        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| SessionError::ExecutionFailed(String::from("no execution graph")))?;

        let input_pairs: Vec<(String, Tensor)> = inputs
            .iter()
            .map(|i| (i.name.clone(), i.tensor.clone()))
            .collect();

        let initializers = &self.initializers;

        let mut profile = crate::profile::InferenceProfile::default();
        let budget = crate::profile::OperatorBudget::DEFAULT;
        let time_source = crate::profile::StdTimeSource::new();

        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut metal_guard = self.metal_dispatcher.borrow_mut();

        let outputs = crate::executor::execute_graph(
            graph,
            &input_pairs,
            initializers,
            None,
            Some(&mut profile),
            &budget,
            &time_source,
            #[cfg(feature = "gpu")]
            self.gpu_backend.as_ref(),
            #[cfg(feature = "cuda")]
            self.cuda_runtime.as_deref(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal_guard.as_mut(),
        )?;
        Ok((outputs, profile))
    }

    /// Load a HuggingFace safetensors model directory and build a
    /// GPU-resident Gemma session.
    ///
    /// `dir` must contain a `config.json` plus either a single
    /// `model.safetensors` file or a `model.safetensors.index.json`
    /// manifest and its referenced shards.
    ///
    /// The signature takes an `Arc<CudaRuntime>` rather than an
    /// `Option` because safetensors sessions are strictly GPU-resident:
    /// a valid, initialized [`crate::cuda::CudaRuntime`] is a
    /// prerequisite for construction (Section 9.5).
    #[cfg(all(feature = "safetensors", feature = "cuda"))]
    pub fn from_safetensors(
        dir: &std::path::Path,
        cuda_runtime: alloc::sync::Arc<crate::cuda::CudaRuntime>,
    ) -> Result<Self, SessionError> {
        use alloc::collections::BTreeMap;
        use alloc::sync::Arc;
        use core::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Mutex;

        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        // 1. Read and parse config.json.
        let config_path = dir.join("config.json");
        let config_json = std::fs::read_to_string(&config_path).map_err(|e| {
            SessionError::InvalidModel(alloc::format!("read {}: {}", config_path.display(), e))
        })?;

        let gemma_config = crate::model_loader::GemmaConfig::from_json(&config_json)
            .map_err(|e| SessionError::InvalidModel(alloc::format!("{}", e)))?;

        // 2. Detect architecture and reject unsupported families.
        let arch = crate::model_loader::ModelArchitecture::detect(&config_json)
            .map_err(|e| SessionError::InvalidModel(alloc::format!("{}", e)))?;
        match arch {
            crate::model_loader::ModelArchitecture::Gemma3
            | crate::model_loader::ModelArchitecture::Gemma4 => {}
            other => {
                return Err(SessionError::InvalidModel(alloc::format!(
                    "unsupported architecture: {:?}",
                    other
                )));
            }
        }

        // 3. Open weights — single file or sharded.
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");
        let built = if index_path.exists() {
            let sharded =
                crate::model_loader::MultiShardSafetensors::open_dir(dir).map_err(|e| {
                    SessionError::InvalidModel(alloc::format!("open sharded safetensors: {}", e))
                })?;
            crate::model_loader::build_gemma_graph(&gemma_config, &sharded)
                .map_err(|e| SessionError::InvalidModel(alloc::format!("{}", e)))?
        } else {
            let file = crate::model_loader::SafetensorsFile::open(&single_path).map_err(|e| {
                SessionError::InvalidModel(alloc::format!("open {}: {}", single_path.display(), e))
            })?;
            crate::model_loader::build_gemma_graph(&gemma_config, &file)
                .map_err(|e| SessionError::InvalidModel(alloc::format!("{}", e)))?
        };

        // 3.5 Compile + register the NVRTC kernels the dispatcher will
        //     need (Gather/Add/Mul/Silu/RMSNorm/Rotary/GQA support
        //     kernels). Idempotent on subsequent sessions sharing the
        //     same runtime — `init_kernels` no-ops on already-registered
        //     entries by overwriting them.
        cuda_runtime
            .init_kernels()
            .map_err(|e| SessionError::ExecutionFailed(alloc::format!("init_kernels: {}", e)))?;

        // 4. Materialize initializers -> GPU weight map.
        let gpu_weights = crate::cuda::initializers_to_gpu(&built.initializers, &cuda_runtime)
            .map_err(|e| {
                SessionError::ExecutionFailed(alloc::format!("initializers_to_gpu: {}", e))
            })?;

        // 5. Allocate the KV cache. Bound max_seq_len at 2048 for now
        //    so VRAM usage stays sane for a 31B-class model.
        let max_seq_len = core::cmp::min(gemma_config.max_position_embeddings, 2048).max(1);
        let layer_kinds = layer_kinds_for_gemma(&gemma_config);
        let kv_cache = crate::cuda::GpuKvCache::allocate(
            &cuda_runtime,
            gemma_config.num_hidden_layers,
            gemma_config.num_key_value_heads,
            gemma_config.head_dim,
            max_seq_len,
            crate::tensor::DataType::BFloat16,
            &layer_kinds,
        )
        .map_err(|e| SessionError::ExecutionFailed(alloc::format!("kv_cache alloc: {}", e)))?;

        // 6. Derive session metadata.
        let model_name = alloc::format!(
            "gemma-{}L-{}d",
            gemma_config.num_hidden_layers,
            gemma_config.hidden_size
        );
        let input_names = built.graph.input_names.clone();
        let output_names = built.graph.output_names.clone();

        Ok(Self {
            id: SessionId(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
            config: SessionConfig::default(),
            model_name,
            graph: Some(built.graph),
            input_names,
            output_names,
            initializers: built.initializers,
            is_initialized: true,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            cuda_runtime: Some(cuda_runtime),
            kv_cache: Some(Arc::new(Mutex::new(kv_cache))),
            gpu_weights: Some(Arc::new(
                gpu_weights
                    .into_iter()
                    .collect::<BTreeMap<String, crate::cuda::gpu_executor::DeviceTensor>>(),
            )),
            device_initializer_cache: core::cell::RefCell::new(None),
            cuda_graph_cache: core::cell::RefCell::new(None),
            stream_pool: core::cell::RefCell::new(None),
            kind: SessionKind::Safetensors,
        })
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

    /// Returns the [`SessionKind`] discriminator identifying how this
    /// session was constructed and which execution path `run()` uses.
    ///
    /// Callers (in particular the `llm-generation` loop from
    /// `llm-api-translation-v1`) MUST branch on this when deciding
    /// whether to thread KV cache through input tensors (ONNX) or rely
    /// on the Session-managed internal KV cache (Safetensors). See
    /// `docs/safetensors-integration.md` for the full contract.
    pub fn kind(&self) -> SessionKind {
        self.kind
    }

    /// Returns `true` when this Session owns its KV cache internally and
    /// callers should NOT thread `past_k`/`past_v` through inputs. This
    /// is the case for safetensors-backed sessions whose forward pass
    /// runs entirely on the GPU using a persistent `GpuKvCache`.
    ///
    /// For ONNX sessions, returns `false`: the caller is expected to
    /// thread KV state through the model's input/output tensors as with
    /// any other ONNX LLM export.
    pub fn manages_kv_cache_internally(&self) -> bool {
        matches!(self.kind, SessionKind::Safetensors)
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
    fn test_session_kind_defaults_to_onnx() {
        let session = Session::new(SessionConfig::default());
        assert_eq!(session.kind(), SessionKind::Onnx);
        assert!(
            !session.manages_kv_cache_internally(),
            "a fresh ONNX session must NOT claim internal KV ownership"
        );
    }

    #[test]
    fn test_session_kind_accessor_matches_field() {
        // Contract: Session::kind() must always return the same value
        // as the public `kind` field. `llm-api-translation-v1`
        // branches on this to decide KV cache threading.
        let session = Session::new(SessionConfig::default());
        assert_eq!(session.kind(), session.kind);
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
            gpu_config: None,
            gpu_residency: GpuResidency::default(),
            cuda_graph: CudaGraphMode::default(),
            stream_config: StreamConfig::default(),
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
            0x42, 0x04, // field 8, length 4 (opset_import)
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
