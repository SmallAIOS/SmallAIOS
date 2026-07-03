// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Execution backend HAL for the SmallAIOS ONNX runtime.
//!
//! This module defines the `ExecutionBackend` trait — a vendor-neutral,
//! `#![no_std]`-compatible abstraction for per-operator dispatch to a
//! concrete execution target. Concrete backends (host CPU today;
//! accelerator-, FPGA-, or future GPU-targeted backends in later
//! changes) implement this trait and are registered with a session via
//! `SessionConfig::backends`.
//!
//! # Design
//!
//! - **Op granularity** — backends opt in or out per operator via
//!   [`ExecutionBackend::can_run`]. Subgraph-level batching is not part
//!   of the v1 contract; a future extension trait may layer it on.
//! - **Static binding** — the runtime walks the graph at session build
//!   and binds each op to the highest-priority backend whose `can_run`
//!   returns true. The hot path never re-runs capability checks.
//! - **Borrowed tensors** — backends receive read access to inputs and
//!   write outputs back through [`TensorEnv`]. Backends MAY allocate
//!   internal device-side memory (DMA buffers, scratchpads), but MUST
//!   NOT allocate runtime-owned tensor buffers.
//! - **Per-op fallback** — a backend may signal recovery to the next
//!   backend in priority order by returning [`ExecError::FallbackToCpu`]
//!   from `dispatch`.
//!
//! See `openspec/changes/fpga-accelerator-hal-v1/design.md` for the
//! decision record behind these choices.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::onnx_types::AttributeProto;
use crate::tensor::Tensor;

mod cpu;

pub use cpu::CpuBackend;

// ---------------------------------------------------------------------------
// Op descriptor
// ---------------------------------------------------------------------------

/// A vendor-neutral, session-build-time view of a single graph
/// operator, sufficient for a backend to decide via [`ExecutionBackend::can_run`]
/// whether it can execute the op and to dispatch it via
/// [`ExecutionBackend::dispatch`].
///
/// `OpDescriptor` carries everything a backend needs to make an
/// op-by-op routing decision: the operator type and its domain, the
/// node's attributes, the number of declared outputs, and a list of
/// per-input metadata (data type and rank when known). It deliberately
/// does NOT hold the runtime tensor data — that is delivered through
/// [`TensorEnv`] at dispatch time.
#[derive(Debug, Clone)]
pub struct OpDescriptor<'a> {
    /// The ONNX operator type (e.g., `"Conv"`, `"Relu"`,
    /// `"GroupQueryAttention"`).
    pub op_type: &'a str,
    /// The operator domain. `""` and `"ai.onnx"` denote the standard
    /// ONNX op set; `"com.microsoft"` denotes the contrib op set.
    pub domain: &'a str,
    /// Human-readable node name (from the ONNX `NodeProto.name`
    /// field). Useful for diagnostics and dispatch logging.
    pub node_name: &'a str,
    /// Operator-specific attributes from the ONNX `NodeProto`.
    pub attributes: &'a [AttributeProto],
    /// Number of declared outputs the runtime expects this op to
    /// produce.
    pub output_count: usize,
    /// Per-input metadata, in the order the operator declares its
    /// inputs. Optional inputs (where the ONNX node binds the empty
    /// string) are represented as `None`.
    pub inputs: &'a [Option<TensorMeta>],
}

/// Compile-time metadata about a tensor (data type and dimensionality
/// when known), without the runtime buffer. Used inside [`OpDescriptor`]
/// to let backends report capability without seeing tensor contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorMeta {
    /// The tensor element data type.
    pub data_type: crate::tensor::DataType,
    /// The rank (number of dimensions). `None` if unknown at session
    /// build time.
    pub rank: Option<usize>,
}

// ---------------------------------------------------------------------------
// Tensor environment
// ---------------------------------------------------------------------------

/// Per-op tensor I/O environment passed to [`ExecutionBackend::dispatch`].
///
/// Provides read access to the operator's input tensors (via
/// [`TensorEnv::input`]) and a way to record output tensors back to the
/// runtime (via [`TensorEnv::set_output`]). The runtime owns the input
/// tensors and the output buffer slots; the backend only borrows them
/// for the duration of the call.
///
/// Backends MAY allocate internal device-side memory inside their own
/// state but MUST NOT use `TensorEnv` to allocate runtime-owned
/// tensors — there is no API on `TensorEnv` to do so. This is an
/// intentional contract that protects the runtime's memory planner.
pub struct TensorEnv<'a> {
    inputs: &'a [Option<&'a Tensor>],
    outputs: &'a mut Vec<Tensor>,
}

impl<'a> TensorEnv<'a> {
    /// Constructs a new tensor environment for a single op dispatch.
    ///
    /// `inputs` is the (possibly-sparse) ordered list of input tensors
    /// the runtime resolved from the value map. `outputs` is the slot
    /// the runtime hands the backend to write its results into; on
    /// successful dispatch the backend SHALL push `output_count`
    /// tensors onto it.
    pub fn new(inputs: &'a [Option<&'a Tensor>], outputs: &'a mut Vec<Tensor>) -> Self {
        Self { inputs, outputs }
    }

    /// Returns the input tensor at `idx`, or `None` if the operator's
    /// declared input at that position is optional and absent.
    pub fn input(&self, idx: usize) -> Option<&Tensor> {
        self.inputs.get(idx).and_then(|opt| *opt)
    }

    /// Returns the full slice of optional input tensors. Backends that
    /// dispatch through legacy helpers consume this whole-slice form.
    pub fn inputs(&self) -> &[Option<&Tensor>] {
        self.inputs
    }

    /// Records a single output tensor into the runtime's output slot.
    /// Most operators produce exactly one tensor; multi-output
    /// operators MUST call this once per output, in the order declared
    /// in the corresponding [`OpDescriptor::output_count`].
    pub fn set_output(&mut self, tensor: Tensor) {
        self.outputs.push(tensor);
    }

    /// Replaces the output slot with a full vector of tensors. Used by
    /// the CPU backend to forward results from operator helpers that
    /// already return `Vec<Tensor>`.
    pub fn set_outputs(&mut self, tensors: Vec<Tensor>) {
        *self.outputs = tensors;
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors a backend may report from [`ExecutionBackend::dispatch`].
///
/// `FallbackToCpu` is the documented signal for "another backend should
/// retry this op"; the runtime walks the priority list on this variant.
/// `BackendUnavailable` is reserved for [`ExecutionBackend::probe`]
/// reporting that the backend's underlying device is absent at session
/// creation. `Internal` carries a backend-specific error string for
/// failures that do not warrant fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecError {
    /// The backend cannot complete this op and the runtime SHOULD retry
    /// with the next-priority backend. The output tensor SHALL be
    /// unmodified.
    FallbackToCpu,
    /// The backend's underlying device or driver is not available
    /// (e.g., a stub backend running on a host without the stub
    /// device). Returned from [`ExecutionBackend::probe`] at session
    /// creation; treated as "skip this backend" by the runtime.
    BackendUnavailable,
    /// A backend-specific failure that does not warrant fallback.
    /// The runtime propagates this to the caller as a fatal inference
    /// error.
    Internal(String),
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecError::FallbackToCpu => write!(f, "fallback to next backend requested"),
            ExecError::BackendUnavailable => write!(f, "backend unavailable"),
            ExecError::Internal(msg) => write!(f, "backend internal error: {}", msg),
        }
    }
}

/// Errors raised by the runtime while interacting with a backend (as
/// opposed to errors raised _by_ the backend; see [`ExecError`]).
///
/// Currently used for session-build-time failures detected by the
/// dispatch table builder. Maps onto session-level errors via
/// `From<BackendError>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// No registered backend reported `can_run = true` for an
    /// operator at session build time.
    NoBackendForOp {
        /// The ONNX operator type.
        op_type: String,
        /// The node's human-readable name (or empty string).
        op_name: String,
    },
    /// All registered backends exhausted their fallback options for an
    /// operator at runtime; the inference call SHALL fail without a
    /// panic.
    DispatchExhausted {
        /// The ONNX operator type that exhausted dispatch.
        op_type: String,
        /// The node's human-readable name (or empty string).
        op_name: String,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::NoBackendForOp { op_type, op_name } => {
                write!(
                    f,
                    "no backend registered can run op '{}' (node '{}')",
                    op_type, op_name
                )
            }
            BackendError::DispatchExhausted { op_type, op_name } => {
                write!(
                    f,
                    "all backends exhausted for op '{}' (node '{}')",
                    op_type, op_name
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A pluggable per-operator execution target.
///
/// The runtime accepts an ordered list of `ExecutionBackend`
/// implementations in [`crate::session::SessionConfig::backends`] and
/// binds each operator in the loaded graph to the highest-priority
/// backend whose [`Self::can_run`] returns true at session build time.
/// On the inference path, the runtime calls [`Self::dispatch`] via the
/// precomputed table; capability checks SHALL NOT be revisited per op
/// at runtime.
///
/// # Object safety
///
/// This trait is intentionally object-safe so that the dispatch table
/// can hold heterogeneous backends behind `Box<dyn ExecutionBackend>`
/// without monomorphization explosion.
///
/// # `no_std`
///
/// All methods, error types, and supporting types in this module are
/// `#![no_std]`-compatible. Backends MAY internally use `std`-only
/// facilities behind their own feature flag, but MUST NOT require
/// `std` for the public surface.
///
/// # Vendor neutrality
///
/// The trait surface and the supporting types in this module SHALL
/// NOT reference vendor-specific concepts (no `.xmodel`, no DPU, no
/// Vitis AI, no CUDA, no Metal, no oneAPI). Concrete backends MAY of
/// course wrap such APIs; that is their concern, not the trait's.
pub trait ExecutionBackend: Send + Sync {
    /// Stable, human-readable name for this backend (e.g. `"cpu"`).
    /// Used in diagnostic logs when fallback fires and in dispatch
    /// table debug printing. SHOULD be ASCII and short.
    fn name(&self) -> &'static str;

    /// Reports whether the backend's underlying device or driver is
    /// available at session creation. Backends that have nothing to
    /// probe (the host CPU, for example) SHOULD return `Ok(())`.
    /// Backends backed by an optional device SHOULD return
    /// `Err(ExecError::BackendUnavailable)` when the device is absent;
    /// the runtime SHALL skip such backends without panic.
    fn probe(&self) -> Result<(), ExecError>;

    /// Reports whether this backend can run the operator described by
    /// `op`. The runtime SHALL call this exclusively at session-build
    /// time and SHALL NOT call it from the inference loop.
    fn can_run(&self, op: &OpDescriptor) -> bool;

    /// Estimated wall-clock execution time, in nanoseconds, for the
    /// operator described by `op`. Used by future cost-based dispatch
    /// policies to pick the cheapest capable backend among multiple
    /// matches; the v1 dispatch policy is strict priority order and
    /// ignores this value, so backends with no real estimate MAY
    /// return `u64::MAX` (the documented sentinel).
    fn estimated_ns(&self, op: &OpDescriptor) -> u64;

    /// Executes a single operator and writes the output tensors back
    /// through `env`.
    ///
    /// On `Ok(())`, the backend SHALL have pushed `op.output_count`
    /// tensors onto `env` via [`TensorEnv::set_output`] (or one call to
    /// [`TensorEnv::set_outputs`]).
    ///
    /// On `Err(ExecError::FallbackToCpu)`, the runtime SHALL retry the
    /// op on the next-priority backend; the output slot SHALL be
    /// unmodified.
    ///
    /// On `Err(ExecError::Internal(_))`, the runtime SHALL propagate
    /// the failure to the caller as a fatal inference error and SHALL
    /// NOT panic.
    fn dispatch(&self, op: &OpDescriptor, env: &mut TensorEnv) -> Result<(), ExecError>;
}

// ---------------------------------------------------------------------------
// Backend list (heapless-friendly wrapper around Vec<Box<dyn ...>>)
// ---------------------------------------------------------------------------

/// A small ordered list of `ExecutionBackend` implementations that the
/// runtime walks in priority order during session build and per-op
/// fallback. Owned by [`crate::session::SessionConfig`].
pub struct BackendList {
    backends: Vec<Box<dyn ExecutionBackend>>,
}

impl BackendList {
    /// Constructs an empty list. Mostly useful for tests; production
    /// callers SHOULD use [`BackendList::default`] (a single
    /// [`CpuBackend`]) or [`BackendList::with_backends`].
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Constructs a list from an explicit ordered slice of boxed
    /// backends. The first backend in the slice is highest-priority.
    pub fn with_backends(backends: Vec<Box<dyn ExecutionBackend>>) -> Self {
        Self { backends }
    }

    /// Appends a backend at the lowest priority.
    pub fn push(&mut self, backend: Box<dyn ExecutionBackend>) {
        self.backends.push(backend);
    }

    /// Returns the registered backends in priority order.
    pub fn as_slice(&self) -> &[Box<dyn ExecutionBackend>] {
        &self.backends
    }

    /// Returns the number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Returns `true` if no backends are registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

impl Default for BackendList {
    /// The default backend list is `[CpuBackend::new()]`, preserving
    /// pre-refactor CPU-only behavior for existing callers.
    fn default() -> Self {
        let mut list = Self::new();
        list.push(Box::new(CpuBackend::new()));
        list
    }
}

impl fmt::Debug for BackendList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendList")
            .field(
                "backends",
                &self
                    .backends
                    .iter()
                    .map(|b| b.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Dispatch table
// ---------------------------------------------------------------------------

/// Per-node binding from `node_index → backend_index` precomputed at
/// session build. The runtime never re-runs `can_run` checks at
/// inference time; it indexes into this table.
///
/// `entries[i]` is the index into [`BackendList::as_slice`] of the
/// backend bound to the i-th node in the graph's storage order, or
/// `None` for nodes the runtime executes outside the trait (control
/// flow ops dispatched through the inner-graph executor).
#[derive(Debug, Clone)]
pub struct DispatchTable {
    entries: Vec<Option<usize>>,
}

impl DispatchTable {
    /// Constructs a dispatch table with a slot for each of `n` nodes,
    /// initially all `None`.
    pub fn new(n: usize) -> Self {
        Self {
            entries: alloc::vec![None; n],
        }
    }

    /// Records the bound backend index for a given node. Panics on
    /// out-of-range `node_index` to match the runtime's contract that
    /// every node has a slot.
    pub fn bind(&mut self, node_index: usize, backend_index: usize) {
        self.entries[node_index] = Some(backend_index);
    }

    /// Looks up the bound backend index for a given node, if any.
    pub fn get(&self, node_index: usize) -> Option<usize> {
        self.entries.get(node_index).copied().flatten()
    }

    /// Returns the number of node slots in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table has no slots.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DataType, TensorShape};
    use alloc::string::ToString;
    use alloc::vec;

    /// Stub backend that claims a configurable subset of ops and can
    /// be configured to error in different ways during dispatch. Used
    /// by tests to validate fallback ordering and `NoBackendForOp`
    /// detection at session build.
    struct StubBackend {
        name_: &'static str,
        supported_ops: &'static [&'static str],
        dispatch_outcome: DispatchOutcome,
    }

    #[derive(Clone, Copy)]
    enum DispatchOutcome {
        Ok,
        Fallback,
        Internal,
    }

    impl ExecutionBackend for StubBackend {
        fn name(&self) -> &'static str {
            self.name_
        }
        fn probe(&self) -> Result<(), ExecError> {
            Ok(())
        }
        fn can_run(&self, op: &OpDescriptor) -> bool {
            self.supported_ops.contains(&op.op_type)
        }
        fn estimated_ns(&self, _op: &OpDescriptor) -> u64 {
            u64::MAX
        }
        fn dispatch(&self, op: &OpDescriptor, env: &mut TensorEnv) -> Result<(), ExecError> {
            match self.dispatch_outcome {
                DispatchOutcome::Ok => {
                    let mut outs = Vec::with_capacity(op.output_count);
                    for _ in 0..op.output_count {
                        outs.push(Tensor::new(
                            DataType::Float,
                            TensorShape::new(vec![1]),
                            String::new(),
                        ));
                    }
                    env.set_outputs(outs);
                    Ok(())
                }
                DispatchOutcome::Fallback => Err(ExecError::FallbackToCpu),
                DispatchOutcome::Internal => Err(ExecError::Internal(String::from("boom"))),
            }
        }
    }

    fn op<'a>(op_type: &'a str) -> OpDescriptor<'a> {
        OpDescriptor {
            op_type,
            domain: "",
            node_name: "",
            attributes: &[],
            output_count: 1,
            inputs: &[],
        }
    }

    #[test]
    fn backend_list_default_is_single_cpu() {
        let list = BackendList::default();
        assert_eq!(list.len(), 1);
        assert_eq!(list.as_slice()[0].name(), "cpu");
    }

    #[test]
    fn backend_list_with_backends_preserves_order() {
        let stub = StubBackend {
            name_: "stub",
            supported_ops: &["MatMul"],
            dispatch_outcome: DispatchOutcome::Ok,
        };
        let list = BackendList::with_backends(vec![
            Box::new(stub) as Box<dyn ExecutionBackend>,
            Box::new(CpuBackend::new()),
        ]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.as_slice()[0].name(), "stub");
        assert_eq!(list.as_slice()[1].name(), "cpu");
    }

    #[test]
    fn dispatch_table_binds_and_lookups() {
        let mut table = DispatchTable::new(3);
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(0), None);
        table.bind(0, 1);
        table.bind(2, 0);
        assert_eq!(table.get(0), Some(1));
        assert_eq!(table.get(1), None);
        assert_eq!(table.get(2), Some(0));
    }

    #[test]
    fn exec_error_display_is_informative() {
        let s = ExecError::FallbackToCpu.to_string();
        assert!(s.contains("fallback"));
        let s = ExecError::BackendUnavailable.to_string();
        assert!(s.contains("unavailable"));
        let s = ExecError::Internal(String::from("xyz")).to_string();
        assert!(s.contains("xyz"));
    }

    #[test]
    fn backend_error_display_includes_names() {
        let s = BackendError::NoBackendForOp {
            op_type: String::from("Foo"),
            op_name: String::from("node-3"),
        }
        .to_string();
        assert!(s.contains("Foo"));
        assert!(s.contains("node-3"));
        let s = BackendError::DispatchExhausted {
            op_type: String::from("Bar"),
            op_name: String::from("node-7"),
        }
        .to_string();
        assert!(s.contains("Bar"));
        assert!(s.contains("node-7"));
    }

    #[test]
    fn stub_can_run_reports_known_and_unknown_ops() {
        let stub = StubBackend {
            name_: "stub",
            supported_ops: &["MatMul"],
            dispatch_outcome: DispatchOutcome::Ok,
        };
        assert!(stub.can_run(&op("MatMul")));
        assert!(!stub.can_run(&op("LayerNormalization")));
    }

    #[test]
    fn stub_dispatch_writes_outputs_on_ok() {
        let stub = StubBackend {
            name_: "stub",
            supported_ops: &["X"],
            dispatch_outcome: DispatchOutcome::Ok,
        };
        let inputs: Vec<Option<&Tensor>> = Vec::new();
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut env = TensorEnv::new(&inputs, &mut outputs);
        let res = stub.dispatch(&op("X"), &mut env);
        assert!(res.is_ok());
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn stub_dispatch_returns_fallback_unmodified() {
        let stub = StubBackend {
            name_: "stub",
            supported_ops: &["X"],
            dispatch_outcome: DispatchOutcome::Fallback,
        };
        let inputs: Vec<Option<&Tensor>> = Vec::new();
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut env = TensorEnv::new(&inputs, &mut outputs);
        let res = stub.dispatch(&op("X"), &mut env);
        assert_eq!(res, Err(ExecError::FallbackToCpu));
        assert!(outputs.is_empty());
    }

    /// Verifies that the trait is in fact object-safe — the spec
    /// scenario "Trait is object-safe" hinges on this compiling.
    #[test]
    fn trait_is_object_safe() {
        let stub: Box<dyn ExecutionBackend> = Box::new(StubBackend {
            name_: "stub",
            supported_ops: &[],
            dispatch_outcome: DispatchOutcome::Ok,
        });
        assert_eq!(stub.name(), "stub");
        let _b: &dyn ExecutionBackend = stub.as_ref();
    }
}
