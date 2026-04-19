// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident KV cache for autoregressive LLM decoding.
//!
//! Autoregressive generation computes one token at a time. Each forward pass
//! produces a new query for the current position plus new K/V values for every
//! transformer layer. To avoid recomputing K/V for previous positions, the K/V
//! values are cached across forward passes — the "KV cache". For Gemma 4 31B
//! with 60 layers and large context windows this cache is multiple GB and
//! MUST stay resident in GPU VRAM between `Session::run()` calls; host<->device
//! transfers per token would dominate latency.
//!
//! ## Design
//!
//! * One [`LayerCache`] per transformer layer, each owning two
//!   [`DeviceBuffer`]s (K and V) sized for `max_seq_len` tokens.
//! * Per-layer K/V buffers are contiguous with layout
//!   `[max_seq_len, num_kv_heads, head_dim]` in element order (BF16 by default).
//! * Each layer can be tagged as global attention or local (sliding-window)
//!   attention via [`LayerKind`]. The current position is tracked globally
//!   across all layers.
//! * Callers append new K/V for every layer for the current token, then call
//!   [`GpuKvCache::advance_position`] exactly once per forward pass.
//! * [`GpuKvCache::view`] returns a [`KvView`] — raw device pointers plus
//!   metadata — that attention dispatch (Section 9) will pass directly to
//!   the attention kernel. Views are borrows into the underlying buffers and
//!   do NOT allocate.
//!
//! ## Ring buffer vs linear layout
//!
//! This implementation uses a **linear layout** for the initial version. The
//! `view()` for a sliding-window layer simply returns the range
//! `[current_position - window, current_position)` out of the linear buffer.
//! Callers must allocate `max_seq_len >= max generation length`. A true
//! ring-buffer semantics (physically wrapping writes and handling split views
//! across the wrap point) is left as a follow-up. A TODO marker is placed in
//! [`GpuKvCache::append`].

extern crate alloc;

use alloc::vec::Vec;

use super::ffi;
use super::gpu_executor::DeviceTensor;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::tensor::DataType;

// ── Layer kind ──────────────────────────────────────────────────────

/// Attention layer kind for a single transformer layer in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Global attention — attends to all cached tokens.
    Global,
    /// Sliding-window local attention with the given window size.
    /// Attends only to the last `window` tokens.
    SlidingWindow(usize),
}

// ── KvView ──────────────────────────────────────────────────────────

/// View into a per-layer cached K/V region for attention dispatch.
///
/// Points into the underlying [`DeviceBuffer`]s owned by the cache. The
/// pointers are valid as long as the owning [`GpuKvCache`] is alive and has
/// not been reset/reallocated. Callers are responsible for passing these
/// pointers to attention kernels without retaining them past the cache's
/// lifetime.
#[derive(Debug, Clone, Copy)]
pub struct KvView {
    /// Device pointer to the first K element in the view.
    pub k_ptr: *const core::ffi::c_void,
    /// Device pointer to the first V element in the view.
    pub v_ptr: *const core::ffi::c_void,
    /// Number of tokens represented by this view.
    pub token_count: usize,
    /// Byte stride between consecutive tokens in both K and V
    /// (= `num_kv_heads * head_dim * dtype.element_size()`).
    pub stride_bytes: usize,
    /// Heads per token.
    pub num_kv_heads: usize,
    /// Dimensions per head.
    pub head_dim: usize,
    /// Element data type (e.g. BFloat16).
    pub dtype: DataType,
}

// SAFETY: KvView holds raw device pointers. CUDA operations are serialized
// on a single stream and the view is always used with the owning cache
// mutable-locked from a single thread at a time. The pointers are not
// dereferenced by Rust code.
unsafe impl Send for KvView {}
unsafe impl Sync for KvView {}

// ── LayerCache ──────────────────────────────────────────────────────

struct LayerCache {
    /// K buffer shape `[max_seq_len, num_kv_heads, head_dim]`.
    k: DeviceBuffer,
    /// V buffer shape `[max_seq_len, num_kv_heads, head_dim]`.
    v: DeviceBuffer,
    /// Sliding-window size, if this is a local-attention layer.
    sliding_window: Option<usize>,
}

// ── GpuKvCache ──────────────────────────────────────────────────────

/// Persistent, GPU-resident K/V cache for a multi-layer transformer.
pub struct GpuKvCache {
    layers: Vec<LayerCache>,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DataType,
    /// Number of tokens already committed to the cache.
    current_position: usize,
}

impl GpuKvCache {
    /// Pre-allocate a new KV cache for `num_layers` transformer layers.
    ///
    /// Each layer's K and V buffer is sized for
    /// `max_seq_len * num_kv_heads * head_dim * dtype.element_size()` bytes
    /// and zero-initialised via `cudaMemset`. `layer_kinds` must have exactly
    /// `num_layers` entries; it describes whether each layer uses global or
    /// sliding-window attention.
    pub fn allocate(
        _runtime: &CudaRuntime,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DataType,
        layer_kinds: &[LayerKind],
    ) -> Result<Self, CudaError> {
        if layer_kinds.len() != num_layers {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_allocate:layer_kinds_len",
                code: -1,
            });
        }
        if num_layers == 0 || num_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_allocate:zero_dim",
                code: -1,
            });
        }

        let elem = dtype.element_size();
        let per_layer_elems = max_seq_len
            .checked_mul(num_kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or(CudaError::AllocFailed {
                size: usize::MAX,
                code: -1,
            })?;
        let per_layer_bytes = per_layer_elems
            .checked_mul(elem)
            .ok_or(CudaError::AllocFailed {
                size: usize::MAX,
                code: -1,
            })?;

        let mut layers: Vec<LayerCache> = Vec::with_capacity(num_layers);
        for kind in layer_kinds.iter().copied() {
            let k = DeviceBuffer::alloc(per_layer_bytes)?;
            let v = DeviceBuffer::alloc(per_layer_bytes)?;
            // Zero-initialise both buffers so uninitialised tokens never
            // contribute spurious attention values.
            zero_buffer(&k)?;
            zero_buffer(&v)?;
            let sliding_window = match kind {
                LayerKind::Global => None,
                LayerKind::SlidingWindow(w) => Some(w),
            };
            layers.push(LayerCache {
                k,
                v,
                sliding_window,
            });
        }

        Ok(Self {
            layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            dtype,
            current_position: 0,
        })
    }

    /// Number of layers in this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Currently committed token count.
    pub fn current_position(&self) -> usize {
        self.current_position
    }

    /// Maximum number of tokens this cache can hold per layer.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Element data type (e.g. BFloat16).
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Byte stride between consecutive tokens in a layer's K/V buffer.
    fn token_stride_bytes(&self) -> usize {
        self.num_kv_heads * self.head_dim * self.dtype.element_size()
    }

    /// Append new single-token K/V values for `layer_idx` at the current
    /// position. Does NOT advance the position — call [`advance_position`]
    /// once per forward pass, after appending all layers for the token.
    ///
    /// `new_k` and `new_v` must each have shape `[1, num_kv_heads, head_dim]`
    /// (or any shape whose byte size equals one token's worth of K/V bytes).
    ///
    /// # Ring-buffer TODO
    ///
    /// For sliding-window layers, once `current_position >= max_seq_len` we
    /// would normally wrap writes via `current_position % max_seq_len`. The
    /// current implementation uses a strictly linear layout and rejects
    /// writes past `max_seq_len`. Callers must allocate enough capacity.
    pub fn append(
        &mut self,
        layer_idx: usize,
        new_k: &DeviceTensor,
        new_v: &DeviceTensor,
    ) -> Result<(), CudaError> {
        if layer_idx >= self.layers.len() {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_append:layer_idx",
                code: -1,
            });
        }
        if self.current_position >= self.max_seq_len {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_append:position_overflow",
                code: -1,
            });
        }
        let stride = self.token_stride_bytes();
        if new_k.byte_size() != stride || new_v.byte_size() != stride {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_append:tensor_byte_size_mismatch",
                code: -1,
            });
        }
        if new_k.dtype != self.dtype || new_v.dtype != self.dtype {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_append:dtype_mismatch",
                code: -1,
            });
        }

        let layer = &self.layers[layer_idx];
        let offset = self.current_position * stride;

        unsafe {
            let k_dst = (layer.k.as_mut_ptr() as *mut u8).add(offset) as *mut core::ffi::c_void;
            let err = ffi::cudaMemcpy(
                k_dst,
                new_k.buffer.as_ptr(),
                stride,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
            if err != ffi::CUDA_SUCCESS {
                return Err(CudaError::CopyFailed {
                    msg: "kv_cache append K (D2D)",
                    code: err,
                });
            }
            let v_dst = (layer.v.as_mut_ptr() as *mut u8).add(offset) as *mut core::ffi::c_void;
            let err = ffi::cudaMemcpy(
                v_dst,
                new_v.buffer.as_ptr(),
                stride,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
            if err != ffi::CUDA_SUCCESS {
                return Err(CudaError::CopyFailed {
                    msg: "kv_cache append V (D2D)",
                    code: err,
                });
            }
        }

        Ok(())
    }

    /// Advance the committed token position by one. Must be called exactly
    /// once per forward pass after all layers' K/V have been appended.
    pub fn advance_position(&mut self) -> Result<(), CudaError> {
        if self.current_position >= self.max_seq_len {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_advance:position_overflow",
                code: -1,
            });
        }
        self.current_position += 1;
        Ok(())
    }

    /// Return a view over the valid cached K/V region for `layer_idx`.
    ///
    /// For **global** attention layers this covers the full range
    /// `[0, current_position)` — `current_position` tokens.
    ///
    /// For **sliding-window** layers with window `w` this covers
    /// `[max(0, current_position - w), current_position)` —
    /// `min(w, current_position)` tokens.
    ///
    /// Returns raw device pointers (see [`KvView`]). The view does NOT copy.
    pub fn view(&self, layer_idx: usize) -> Result<KvView, CudaError> {
        if layer_idx >= self.layers.len() {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_view:layer_idx",
                code: -1,
            });
        }
        let layer = &self.layers[layer_idx];
        let stride = self.token_stride_bytes();

        let (start, count) = match layer.sliding_window {
            None => (0, self.current_position),
            Some(w) => {
                let count = core::cmp::min(w, self.current_position);
                let start = self.current_position - count;
                (start, count)
            }
        };

        // Even when count == 0 we return pointers to the buffer base. The
        // caller is expected to handle the empty-view case gracefully.
        let k_base = layer.k.as_ptr() as *const u8;
        let v_base = layer.v.as_ptr() as *const u8;
        let k_ptr = unsafe { k_base.add(start * stride) } as *const core::ffi::c_void;
        let v_ptr = unsafe { v_base.add(start * stride) } as *const core::ffi::c_void;

        Ok(KvView {
            k_ptr,
            v_ptr,
            token_count: count,
            stride_bytes: stride,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
        })
    }

    /// View that includes the most recently appended (but not yet
    /// advance-committed) token at slot `current_position`.
    ///
    /// The standard [`view`] returns `[start, current_position)` which
    /// excludes the token just written by [`append`]. Self-attention
    /// needs to attend to the new token's own K/V, so this method
    /// returns a view widened by one slot:
    /// - **global** → `[0, current_position + 1)`
    /// - **sliding-window w** → `[max(0, current_position + 1 - w), current_position + 1)`
    ///
    /// Use this between `append()` and `advance_position()`. After the
    /// executor calls `advance_position` at end-of-token, [`view`]
    /// itself reflects the same window — `view_pending` is only needed
    /// during the in-flight append.
    pub fn view_pending(&self, layer_idx: usize) -> Result<KvView, CudaError> {
        if layer_idx >= self.layers.len() {
            return Err(CudaError::RuntimeError {
                op: "kv_cache_view_pending:layer_idx",
                code: -1,
            });
        }
        let layer = &self.layers[layer_idx];
        let stride = self.token_stride_bytes();
        let pending_position = self.current_position + 1;

        let (start, count) = match layer.sliding_window {
            None => (0, pending_position),
            Some(w) => {
                let count = core::cmp::min(w, pending_position);
                let start = pending_position - count;
                (start, count)
            }
        };

        let k_base = layer.k.as_ptr() as *const u8;
        let v_base = layer.v.as_ptr() as *const u8;
        let k_ptr = unsafe { k_base.add(start * stride) } as *const core::ffi::c_void;
        let v_ptr = unsafe { v_base.add(start * stride) } as *const core::ffi::c_void;

        Ok(KvView {
            k_ptr,
            v_ptr,
            token_count: count,
            stride_bytes: stride,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
        })
    }

    /// Reset the cache for a new generation session.
    ///
    /// Sets the committed position back to zero. Does NOT free or reallocate
    /// the underlying buffers — they are reused for the next session. Stale
    /// bytes are never observed because [`view`] is gated on
    /// `current_position`.
    pub fn reset(&mut self) {
        self.current_position = 0;
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Fill a [`DeviceBuffer`] with zero bytes via `cudaMemset`.
fn zero_buffer(buf: &DeviceBuffer) -> Result<(), CudaError> {
    if buf.size() == 0 {
        return Ok(());
    }
    let err = unsafe { ffi::cudaMemset(buf.as_mut_ptr(), 0, buf.size()) };
    if err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaMemset",
            code: err,
        });
    }
    Ok(())
}
