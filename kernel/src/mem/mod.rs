// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory management subsystem.
//!
//! Layers:
//! 1. Physical memory map (boot-time enumeration)
//! 2. Buddy allocator (page-granularity, 4K-8G)
//! 3. Slab allocator (sub-page kernel objects, 16B-2048B)
//! 4. Tensor memory pool (aligned, reference-counted, GPU-mappable)
