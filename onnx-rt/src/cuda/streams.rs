// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Multi-stream orchestration for the GPU executor.
//!
//! Per-Session [`StreamPool`] holds:
//! * one **compute** stream for cuDNN / cuBLAS launches and graph replay
//! * one or more **H2D** streams for asynchronous host→device input
//!   uploads
//! * one or more **D2H** streams for asynchronous device→host output
//!   downloads
//! * an event pool used to express ordering between the streams without
//!   serializing on the host
//!
//! On a single-stream Session (the default) the pool stays `None` and
//! every cuDNN call enqueues onto the default stream. With
//! [`crate::session::StreamConfig::Overlap`], the executor records an
//! event after the H2D copy, has the compute stream wait on it before
//! launching the first kernel, records another event after the last
//! kernel, then has the D2H stream wait on that before issuing the
//! output copy. The host blocks once at the end on
//! `cudaStreamSynchronize(d2h)` rather than per-op.

extern crate alloc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ptr;

use super::ffi;
use super::CudaError;

/// Owned non-default CUDA stream. Created by `cudaStreamCreate`,
/// destroyed by `cudaStreamDestroy` on Drop. Best-effort
/// `cudaStreamSynchronize` runs first to avoid destroying a stream
/// with in-flight work.
pub struct Stream {
    stream: ffi::cudaStream_t,
}

impl Stream {
    pub fn new() -> Result<Self, CudaError> {
        let mut s: ffi::cudaStream_t = ptr::null_mut();
        let err = unsafe { ffi::cudaStreamCreate(&mut s) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaStreamCreate",
                code: err,
            });
        }
        Ok(Self { stream: s })
    }

    /// Raw stream handle. Pointer is valid for the lifetime of `self`.
    pub fn raw(&self) -> ffi::cudaStream_t {
        self.stream
    }

    /// Block the calling thread until all queued work on this stream
    /// has completed.
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

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = ffi::cudaStreamSynchronize(self.stream);
                let _ = ffi::cudaStreamDestroy(self.stream);
            }
        }
    }
}

/// Owned CUDA event. Created by `cudaEventCreate`, destroyed by
/// `cudaEventDestroy` on Drop. Used to express stream-to-stream
/// ordering without host synchronization.
pub struct Event {
    event: ffi::cudaEvent_t,
}

impl Event {
    pub fn new() -> Result<Self, CudaError> {
        let mut e: ffi::cudaEvent_t = ptr::null_mut();
        let err = unsafe { ffi::cudaEventCreate(&mut e) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaEventCreate",
                code: err,
            });
        }
        Ok(Self { event: e })
    }

    /// Capture the state of `stream` into this event. Subsequent
    /// `wait` calls on a different stream will hold off until all
    /// work issued on `stream` before this `record` completes.
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        let err = unsafe { ffi::cudaEventRecord(self.event, stream.raw()) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaEventRecord",
                code: err,
            });
        }
        Ok(())
    }

    /// Make `stream` wait on the event (no-op for the host). Returns
    /// once the event is enqueued; the host does not block.
    pub fn wait(&self, stream: &Stream) -> Result<(), CudaError> {
        let err = unsafe {
            ffi::cudaStreamWaitEvent(stream.raw(), self.event, ffi::CUDA_EVENT_WAIT_DEFAULT)
        };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaStreamWaitEvent",
                code: err,
            });
        }
        Ok(())
    }

    pub fn raw(&self) -> ffi::cudaEvent_t {
        self.event
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                let _ = ffi::cudaEventDestroy(self.event);
            }
        }
    }
}

/// Per-Session bundle of compute + transfer streams + a small event
/// pool for cross-stream ordering.
///
/// `transfer_streams` controls how many H2D / D2H streams are
/// allocated. The runtime currently caps this at 2 (`SessionConfig`
/// validates) — beyond ~2 transfer streams contention dominates and
/// throughput drops. With `transfer_streams = 0` the pool acts as a
/// single-stream guard (compute only); H2D / D2H operations fall
/// back to the default stream behavior.
pub struct StreamPool {
    pub compute: Stream,
    pub h2d: Vec<Stream>,
    pub d2h: Vec<Stream>,
    /// Event reuse pool. Events are cheap but `cudaEventCreate`
    /// still has measurable cost; keep a small free list so the
    /// hot path can `acquire_event` instead of allocating.
    events: RefCell<Vec<Event>>,
}

impl StreamPool {
    /// Allocate `transfer_streams` H2D and `transfer_streams` D2H
    /// streams plus a small starter event pool (4 events). Errors
    /// out if any `cudaStreamCreate` / `cudaEventCreate` fails.
    pub fn new(transfer_streams: usize) -> Result<Self, CudaError> {
        let compute = Stream::new()?;
        let mut h2d = Vec::with_capacity(transfer_streams);
        let mut d2h = Vec::with_capacity(transfer_streams);
        for _ in 0..transfer_streams {
            h2d.push(Stream::new()?);
            d2h.push(Stream::new()?);
        }
        let mut starter = Vec::with_capacity(4);
        for _ in 0..4 {
            starter.push(Event::new()?);
        }
        Ok(Self {
            compute,
            h2d,
            d2h,
            events: RefCell::new(starter),
        })
    }

    /// Acquire an event from the pool, allocating a fresh one if the
    /// pool is empty. Caller must call [`release_event`] when done so
    /// the event can be reused.
    pub fn acquire_event(&self) -> Result<Event, CudaError> {
        if let Some(e) = self.events.borrow_mut().pop() {
            return Ok(e);
        }
        Event::new()
    }

    /// Return an event to the pool for reuse. No-op if the pool is
    /// already at its soft cap (16); the event is dropped instead.
    pub fn release_event(&self, e: Event) {
        let mut pool = self.events.borrow_mut();
        if pool.len() < 16 {
            pool.push(e);
        }
        // Otherwise let `e` drop here, freeing the underlying handle.
    }

    /// Block until all queued work on every stream in the pool
    /// completes. Used on Session drop and at the end of each
    /// inference's overlap path.
    pub fn synchronize_all(&self) -> Result<(), CudaError> {
        self.compute.synchronize()?;
        for s in &self.h2d {
            s.synchronize()?;
        }
        for s in &self.d2h {
            s.synchronize()?;
        }
        Ok(())
    }
}
