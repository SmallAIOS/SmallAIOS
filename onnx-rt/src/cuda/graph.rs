// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RAII wrappers for CUDA stream + graph resources.
//!
//! [`CaptureStream`] owns a non-default `cudaStream_t` used by the
//! capture / replay path. [`CudaGraph`] owns the immutable
//! `cudaGraph_t` produced by `cudaStreamEndCapture`, and
//! [`CudaGraphExec`] owns the executable `cudaGraphExec_t` produced by
//! `cudaGraphInstantiate`.
//!
//! All three types release their underlying resource in `Drop`. The
//! raw handles are wrapped here with `unsafe impl Send + Sync` so a
//! [`Session`](crate::session::Session) can be embedded in a
//! multi-threaded HTTP request handler.
//!
//! # Safety contract for `Send + Sync`
//!
//! Per the CUDA Runtime API, stream / event / graph handles can only
//! be safely used from a thread that has the **owning device made
//! current** via `cudaSetDevice`. Each host thread has its own
//! "current device" state; HTTP worker pools enter with no device
//! current. Lazy primary-context bind (the default for a single-GPU
//! process) papers over many cases, but the soundness of moving these
//! handles between threads requires an explicit guarantee, not luck.
//!
//! The guarantee `Session` makes:
//!
//! 1. **Single device per `Session`.** A `Session` is constructed
//!    against exactly one [`crate::cuda::CudaRuntime`], which records
//!    one [`crate::cuda::CudaDeviceInfo::device_id`]. The Session
//!    never re-targets across devices.
//! 2. **Re-bind on every cross-thread entry point.** Every public
//!    method on `Session` that may be invoked from an HTTP worker
//!    thread or a `std::thread::spawn` body — `Session::run` and
//!    `Session::ensure_stream_pool` — calls
//!    `crate::cuda::set_device(rt.device.device_id)` *before*
//!    acquiring any of the cached-handle mutexes. The rebind is
//!    cheap (a thread-local pointer write in libcudart) and is the
//!    point at which the Send claim is discharged.
//!
//! Together, these mean the `unsafe impl Send + Sync` is sound for
//! the deployments smallaios supports today (single-GPU host;
//! container-mode HTTP request thread pool). Multi-device hosts
//! would require either a per-`Session` device pin enforced with a
//! thread-pinning newtype, or a dedicated CUDA worker thread — both
//! out of scope for the current milestone. Anyone widening that
//! envelope MUST audit this contract.

use core::ptr;

use super::ffi;
use super::CudaError;

/// A dedicated, non-default CUDA stream used by the capture / replay
/// path. Allocated once per [`crate::cuda::CudaRuntime`] when graph
/// capture mode is enabled, dropped on Session teardown.
pub struct CaptureStream {
    stream: ffi::cudaStream_t,
}

// SAFETY: see module-level safety contract.
unsafe impl Send for CaptureStream {}
unsafe impl Sync for CaptureStream {}

impl CaptureStream {
    /// Create a fresh CUDA stream via `cudaStreamCreate`. The stream
    /// is initially attached to no cuDNN/cuBLAS handle; bind via
    /// [`bind_handles`] before launching captured work.
    pub fn new() -> Result<Self, CudaError> {
        let mut stream: ffi::cudaStream_t = ptr::null_mut();
        let err = unsafe { ffi::cudaStreamCreate(&mut stream) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaStreamCreate",
                code: err,
            });
        }
        Ok(Self { stream })
    }

    /// Returns the raw `cudaStream_t` for FFI calls. Pointer is valid
    /// for the lifetime of the [`CaptureStream`].
    pub fn raw(&self) -> ffi::cudaStream_t {
        self.stream
    }

    /// Block until all queued work on this stream completes.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        let err = unsafe { ffi::cudaStreamSynchronize(self.stream) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaStreamSynchronize",
                code: err,
            });
        }
        Ok(())
    }
}

impl Drop for CaptureStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // Best-effort: synchronize and destroy. We can't surface
            // the error from Drop, so log nothing — the next FFI call
            // will fail loudly if there's actually a problem.
            unsafe {
                let _ = ffi::cudaStreamSynchronize(self.stream);
                let _ = ffi::cudaStreamDestroy(self.stream);
            }
        }
    }
}

/// Owns a `cudaGraph_t` produced by `cudaStreamEndCapture`. The graph
/// is the immutable structural form of the captured op sequence —
/// it must be instantiated into a [`CudaGraphExec`] before launch.
pub struct CudaGraph {
    graph: ffi::cudaGraph_t,
}

// SAFETY: see module-level safety contract.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Wrap a previously-captured graph handle. Caller transfers
    /// ownership; Drop will release the underlying graph.
    ///
    /// # Safety
    /// `graph` must be a non-null handle returned by
    /// `cudaStreamEndCapture` (or another graph-creation API). After
    /// calling this constructor the caller must not call
    /// `cudaGraphDestroy` on the same handle.
    pub unsafe fn from_raw(graph: ffi::cudaGraph_t) -> Self {
        Self { graph }
    }

    pub fn raw(&self) -> ffi::cudaGraph_t {
        self.graph
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            unsafe {
                let _ = ffi::cudaGraphDestroy(self.graph);
            }
        }
    }
}

/// Owns a `cudaGraphExec_t` produced by `cudaGraphInstantiate`. This
/// is the launchable form — `cudaGraphLaunch(exec, stream)` enqueues
/// the entire captured kernel sequence in one host-side call.
pub struct CudaGraphExec {
    graph_exec: ffi::cudaGraphExec_t,
}

// SAFETY: see module-level safety contract.
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    /// Instantiate an immutable [`CudaGraph`] into an executable
    /// graph. Wraps `cudaGraphInstantiate` with `flags = 0`.
    pub fn instantiate(graph: &CudaGraph) -> Result<Self, CudaError> {
        let mut exec: ffi::cudaGraphExec_t = ptr::null_mut();
        let err = unsafe { ffi::cudaGraphInstantiate(&mut exec, graph.raw(), 0) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaGraphInstantiate",
                code: err,
            });
        }
        Ok(Self { graph_exec: exec })
    }

    /// Enqueue the entire captured graph onto `stream` in a single
    /// host-side call. Returns immediately; the caller must
    /// synchronize the stream to observe completion.
    pub fn launch(&self, stream: &CaptureStream) -> Result<(), CudaError> {
        let err = unsafe { ffi::cudaGraphLaunch(self.graph_exec, stream.raw()) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaGraphLaunch",
                code: err,
            });
        }
        Ok(())
    }

    pub fn raw(&self) -> ffi::cudaGraphExec_t {
        self.graph_exec
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.graph_exec.is_null() {
            unsafe {
                let _ = ffi::cudaGraphExecDestroy(self.graph_exec);
            }
        }
    }
}
