// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-Session CUDA Graph cache for the hybrid executor.
//!
//! Captures the kernel-launch sequence of a single inference into a
//! `cudaGraph_t`, instantiates it as `cudaGraphExec_t`, and replays
//! that on subsequent inferences with the same input shape +
//! dtype. Eliminates the ~10–50 µs of host-side launch overhead per
//! cuDNN/cuBLAS call (≈170 launches per ResNet-50 inference).
//!
//! The cache is keyed by ([`GraphKey`]) tuple of input-tensor shapes
//! and dtypes. On a cache hit, the replay path memcpys the user input
//! into a persistent device buffer, calls `cudaGraphLaunch`, and
//! memcpys the output back. On a miss, the capture path runs the
//! per-op dispatch loop wrapped in `cudaStreamBeginCapture` /
//! `cudaStreamEndCapture` and stores the resulting `cudaGraphExec_t`.
//!
//! On any failure (capture, instantiation, or replay) the cache
//! marks itself disabled and the executor falls through to the per-op
//! path. Errors do NOT propagate out of `Session::run` for
//! capture-related issues — inference always completes.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::gpu_executor::DeviceTensor;
use super::graph::{CaptureStream, CudaGraph, CudaGraphExec};
use super::memory::DeviceBuffer;
use crate::tensor::DataType;

/// Identifies a captured graph by its input signature. Two
/// inferences with the same input shapes and dtypes produce the same
/// op sequence (no data-dependent control flow in our vision /
/// transformer benches), so they can share a captured graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphKey {
    /// Per-input shape (parallel to the input list passed to `run`).
    pub input_shapes: Vec<Vec<i64>>,
    /// Per-input dtype.
    pub input_dtypes: Vec<DataType>,
}

impl GraphKey {
    pub fn from_inputs(inputs: &[(alloc::string::String, crate::tensor::Tensor)]) -> Self {
        let input_shapes = inputs.iter().map(|(_, t)| t.shape.dims.clone()).collect();
        let input_dtypes = inputs.iter().map(|(_, t)| t.data_type).collect();
        Self {
            input_shapes,
            input_dtypes,
        }
    }
}

/// One captured graph, ready for replay.
///
/// Holds the executable `cudaGraphExec_t`, the persistent device
/// buffers used as graph inputs (must remain at the same address
/// across replays — the graph captured the pointers), and references
/// to every device tensor the captured kernels read from or write to,
/// so the underlying allocations stay alive for the cache's lifetime.
pub struct GraphEntry {
    /// Owning handle for the immutable graph. Kept alive because some
    /// CUDA versions tie the lifetime of the `cudaGraphExec_t` to the
    /// originating `cudaGraph_t`.
    #[allow(dead_code)]
    pub graph: CudaGraph,
    /// Executable form, used for `cudaGraphLaunch`.
    pub graph_exec: CudaGraphExec,
    /// Persistent host→device input buffers. The captured graph
    /// references these pointers; replay memcpys new input bytes
    /// into the same buffers before launching.
    pub input_buffers: Vec<DeviceBuffer>,
    /// Per-input shape / dtype, mirrored from [`GraphKey`] so replay
    /// can validate sizes without touching the cache key itself.
    pub input_shapes: Vec<Vec<i64>>,
    pub input_dtypes: Vec<DataType>,
    /// Output device tensors (one per graph output). The captured
    /// kernels write here; replay memcpys these back to host.
    pub output_tensors: Vec<Arc<DeviceTensor>>,
    /// Output names matching the graph's `output_names`, in order.
    pub output_names: Vec<alloc::string::String>,
    /// Liveness anchor for every intermediate device tensor that the
    /// captured kernels reference. Without this, intermediate
    /// `DeviceBuffer`s would be freed and the graph would launch
    /// against dangling pointers on replay.
    #[allow(dead_code)]
    pub intermediate_keep_alive: Vec<Arc<DeviceTensor>>,
}

/// Per-Session graph cache.
///
/// Owns the dedicated capture stream and the map of captured
/// graphs. Holds counters used to disable capture if the workload
/// keeps invalidating the cache (different shape every inference).
pub struct CudaGraphCache {
    /// Dedicated non-default stream used for both capture and replay.
    pub capture_stream: CaptureStream,
    /// Captured graphs, keyed by input signature.
    pub entries: BTreeMap<GraphKey, GraphEntry>,
    /// How many times we've captured a fresh graph. Increments on
    /// every cache miss that successfully captures.
    pub rebuild_count: u32,
    /// Total number of `Session::run` calls that reached this cache.
    pub inference_count: u32,
    /// Once flipped true, all subsequent inferences skip capture and
    /// go straight to the per-op path. Set when capture fails or the
    /// rebuild rate exceeds 1% after enough samples.
    pub disabled: bool,
}

impl CudaGraphCache {
    /// Create a fresh cache with a new capture stream. Returns an
    /// error only if `cudaStreamCreate` fails — should be rare on a
    /// healthy device.
    pub fn new() -> Result<Self, super::CudaError> {
        Ok(Self {
            capture_stream: CaptureStream::new()?,
            entries: BTreeMap::new(),
            rebuild_count: 0,
            inference_count: 0,
            disabled: false,
        })
    }

    /// Record one successful inference that reached this cache (whether
    /// capture, replay, or fallback) for thrash-detection.
    pub fn note_inference(&mut self) {
        self.inference_count = self.inference_count.saturating_add(1);
    }

    /// Record one fresh graph capture. Bumps the rebuild counter and
    /// re-checks the disable threshold.
    pub fn note_rebuild(&mut self) {
        self.rebuild_count = self.rebuild_count.saturating_add(1);
        self.check_disable_threshold();
    }

    /// After 32 inferences, if rebuilds exceed 1% of inferences, disable
    /// the cache permanently for this Session. This catches workloads
    /// where every input has a different shape — capturing+throwing-away
    /// would be slower than just running per-op.
    fn check_disable_threshold(&mut self) {
        if self.disabled || self.inference_count < 32 {
            return;
        }
        // 1% threshold: rebuild_count * 100 > inference_count.
        if (self.rebuild_count as u64).saturating_mul(100) > (self.inference_count as u64) {
            self.disabled = true;
        }
    }
}

/// Type alias used by [`crate::session::Session`] to avoid
/// `clippy::type_complexity` on the `RefCell<Option<...>>` slot.
pub type CudaGraphCacheSlot = Option<CudaGraphCache>;
