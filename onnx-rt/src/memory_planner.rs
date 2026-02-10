// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory planner for the SmallAIOS ONNX runtime.
//!
//! Analyzes tensor lifetimes across the execution graph and produces
//! a memory plan that assigns buffer offsets to each tensor. The planner
//! supports buffer reuse for non-overlapping lifetimes to minimize peak
//! memory consumption.

use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::ExecutionGraph;

// Available for future use when lifetime computation walks the execution order.
#[allow(unused_imports)]
use crate::graph::NodeIndex;

/// The lifetime of a single tensor within the execution graph.
///
/// Lifetimes are expressed as step indices in the topological execution
/// order. `first_use` is the step where the tensor is first produced or
/// consumed, and `last_use` is the step where it is last consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLifetime {
    /// The tensor name (from the ONNX graph).
    pub tensor_name: String,
    /// Execution step where this tensor is first used (produced).
    pub first_use: usize,
    /// Execution step where this tensor is last consumed.
    pub last_use: usize,
    /// Size of the tensor in bytes.
    pub size_bytes: usize,
}

impl TensorLifetime {
    /// Returns `true` if this tensor's lifetime overlaps with another.
    pub fn overlaps(&self, other: &TensorLifetime) -> bool {
        self.first_use <= other.last_use && other.first_use <= self.last_use
    }

    /// Returns the duration (number of steps) this tensor is live.
    pub fn duration(&self) -> usize {
        if self.last_use >= self.first_use {
            self.last_use - self.first_use + 1
        } else {
            0
        }
    }
}

/// A buffer allocation for a single tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferAllocation {
    /// Unique buffer identifier.
    pub buffer_id: usize,
    /// Byte offset within the memory pool.
    pub offset: usize,
    /// Size of this allocation in bytes.
    pub size: usize,
    /// The tensor name this allocation is for.
    pub tensor_name: String,
}

/// The complete memory plan for an execution graph.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Individual buffer allocations for each tensor.
    pub allocations: Vec<BufferAllocation>,
    /// Peak memory usage in bytes (maximum simultaneously live memory).
    pub peak_memory: usize,
    /// Total number of distinct buffers allocated.
    pub buffer_count: usize,
    /// Number of times a buffer was reused for a different tensor.
    pub reuse_count: usize,
}

impl MemoryPlan {
    /// Creates an empty memory plan.
    pub fn empty() -> Self {
        Self {
            allocations: Vec::new(),
            peak_memory: 0,
            buffer_count: 0,
            reuse_count: 0,
        }
    }
}

/// Configuration for the memory planner.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Whether to reuse buffers for non-overlapping tensor lifetimes.
    pub enable_buffer_reuse: bool,
    /// Memory alignment in bytes for buffer offsets.
    pub alignment: usize,
}

impl Default for PlannerConfig {
    /// Creates the default planner configuration.
    ///
    /// Buffer reuse is enabled and alignment is 64 bytes.
    fn default() -> Self {
        Self {
            enable_buffer_reuse: true,
            alignment: 64,
        }
    }
}

/// Computes the lifetime of each tensor in the execution graph.
///
/// For each node in execution order, records the first and last step
/// where each tensor name appears (as an input or output).
///
/// Stub implementation: returns an empty vector.
pub fn compute_tensor_lifetimes(_graph: &ExecutionGraph) -> Vec<TensorLifetime> {
    // TODO: Walk the execution order, building a map of tensor_name ->
    // (first_use_step, last_use_step, size_bytes). The size must be
    // derived from tensor shape and data type information attached to
    // the graph or model initializers.
    Vec::new()
}

/// Rounds `size` up to the nearest multiple of `alignment`.
fn align_up(size: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return size;
    }
    let mask = alignment - 1;
    (size + mask) & !mask
}

/// Produces a memory plan from a set of tensor lifetimes.
///
/// Uses a simple first-fit allocation strategy:
/// - Each tensor is assigned its own buffer (no reuse in this stub).
/// - Buffers are placed sequentially with the configured alignment.
/// - Peak memory is computed as the sum of all aligned allocations.
///
/// When buffer reuse is fully implemented, non-overlapping lifetimes
/// will share the same physical buffer to reduce peak memory.
pub fn plan_memory(lifetimes: &[TensorLifetime], config: &PlannerConfig) -> MemoryPlan {
    if lifetimes.is_empty() {
        return MemoryPlan::empty();
    }

    let mut allocations: Vec<BufferAllocation> = Vec::with_capacity(lifetimes.len());
    let mut current_offset: usize = 0;

    for (i, lifetime) in lifetimes.iter().enumerate() {
        let aligned_size = align_up(lifetime.size_bytes, config.alignment);

        allocations.push(BufferAllocation {
            buffer_id: i,
            offset: current_offset,
            size: aligned_size,
            tensor_name: lifetime.tensor_name.clone(),
        });

        current_offset += aligned_size;
    }

    let peak_memory = current_offset;
    let buffer_count = allocations.len();

    MemoryPlan {
        allocations,
        peak_memory,
        buffer_count,
        reuse_count: 0, // No reuse in this stub implementation.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    // ---------------------------------------------------------------
    // Empty plan
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_plan() {
        let config = PlannerConfig::default();
        let plan = plan_memory(&[], &config);
        assert!(plan.allocations.is_empty());
        assert_eq!(plan.peak_memory, 0);
        assert_eq!(plan.buffer_count, 0);
        assert_eq!(plan.reuse_count, 0);
    }

    // ---------------------------------------------------------------
    // PlannerConfig defaults
    // ---------------------------------------------------------------

    #[test]
    fn test_planner_config_defaults() {
        let config = PlannerConfig::default();
        assert!(config.enable_buffer_reuse);
        assert_eq!(config.alignment, 64);
    }

    // ---------------------------------------------------------------
    // Single tensor lifetime
    // ---------------------------------------------------------------

    #[test]
    fn test_single_tensor_lifetime() {
        let lifetimes = vec![TensorLifetime {
            tensor_name: "t0".to_string(),
            first_use: 0,
            last_use: 2,
            size_bytes: 1024,
        }];
        let config = PlannerConfig {
            enable_buffer_reuse: false,
            alignment: 64,
        };
        let plan = plan_memory(&lifetimes, &config);
        assert_eq!(plan.allocations.len(), 1);
        assert_eq!(plan.buffer_count, 1);
        assert_eq!(plan.allocations[0].offset, 0);
        assert_eq!(plan.allocations[0].size, 1024); // 1024 is already 64-aligned.
        assert_eq!(plan.peak_memory, 1024);
    }

    // ---------------------------------------------------------------
    // Multiple non-overlapping lifetimes
    // ---------------------------------------------------------------

    #[test]
    fn test_multiple_non_overlapping_lifetimes() {
        let lifetimes = vec![
            TensorLifetime {
                tensor_name: "t0".to_string(),
                first_use: 0,
                last_use: 1,
                size_bytes: 256,
            },
            TensorLifetime {
                tensor_name: "t1".to_string(),
                first_use: 2,
                last_use: 3,
                size_bytes: 512,
            },
        ];
        let config = PlannerConfig::default();
        let plan = plan_memory(&lifetimes, &config);

        // Stub assigns separate buffers.
        assert_eq!(plan.buffer_count, 2);
        assert_eq!(plan.allocations[0].offset, 0);
        assert_eq!(plan.allocations[1].offset, 256); // 256 is 64-aligned.
        assert_eq!(plan.peak_memory, 256 + 512);
        assert_eq!(plan.reuse_count, 0);
    }

    // ---------------------------------------------------------------
    // Peak memory calculation
    // ---------------------------------------------------------------

    #[test]
    fn test_peak_memory_calculation() {
        let lifetimes = vec![
            TensorLifetime {
                tensor_name: "a".to_string(),
                first_use: 0,
                last_use: 5,
                size_bytes: 100,
            },
            TensorLifetime {
                tensor_name: "b".to_string(),
                first_use: 1,
                last_use: 3,
                size_bytes: 200,
            },
            TensorLifetime {
                tensor_name: "c".to_string(),
                first_use: 4,
                last_use: 6,
                size_bytes: 300,
            },
        ];
        let config = PlannerConfig {
            enable_buffer_reuse: false,
            alignment: 64,
        };
        let plan = plan_memory(&lifetimes, &config);

        // Each size aligned to 64: 128 + 256 + 320 = 704.
        assert_eq!(plan.allocations[0].size, 128); // 100 -> 128
        assert_eq!(plan.allocations[1].size, 256); // 200 -> 256
        assert_eq!(plan.allocations[2].size, 320); // 300 -> 320
        assert_eq!(plan.peak_memory, 128 + 256 + 320);
    }

    // ---------------------------------------------------------------
    // TensorLifetime overlaps
    // ---------------------------------------------------------------

    #[test]
    fn test_lifetime_overlaps() {
        let a = TensorLifetime {
            tensor_name: "a".to_string(),
            first_use: 0,
            last_use: 3,
            size_bytes: 100,
        };
        let b = TensorLifetime {
            tensor_name: "b".to_string(),
            first_use: 2,
            last_use: 5,
            size_bytes: 100,
        };
        let c = TensorLifetime {
            tensor_name: "c".to_string(),
            first_use: 4,
            last_use: 6,
            size_bytes: 100,
        };

        assert!(a.overlaps(&b)); // [0,3] and [2,5] overlap.
        assert!(b.overlaps(&a)); // Symmetric.
        assert!(!a.overlaps(&c)); // [0,3] and [4,6] do not overlap.
        assert!(b.overlaps(&c)); // [2,5] and [4,6] overlap.
    }

    // ---------------------------------------------------------------
    // TensorLifetime duration
    // ---------------------------------------------------------------

    #[test]
    fn test_lifetime_duration() {
        let lt = TensorLifetime {
            tensor_name: "t".to_string(),
            first_use: 2,
            last_use: 5,
            size_bytes: 64,
        };
        assert_eq!(lt.duration(), 4); // Steps 2, 3, 4, 5.

        let single_step = TensorLifetime {
            tensor_name: "s".to_string(),
            first_use: 3,
            last_use: 3,
            size_bytes: 32,
        };
        assert_eq!(single_step.duration(), 1);
    }

    // ---------------------------------------------------------------
    // Alignment
    // ---------------------------------------------------------------

    #[test]
    fn test_alignment_rounding() {
        let lifetimes = vec![TensorLifetime {
            tensor_name: "unaligned".to_string(),
            first_use: 0,
            last_use: 0,
            size_bytes: 50, // Not a multiple of 64.
        }];
        let config = PlannerConfig::default();
        let plan = plan_memory(&lifetimes, &config);
        assert_eq!(plan.allocations[0].size, 64); // Rounded up to 64.
        assert_eq!(plan.peak_memory, 64);
    }

    // ---------------------------------------------------------------
    // MemoryPlan::empty
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_plan_empty() {
        let plan = MemoryPlan::empty();
        assert!(plan.allocations.is_empty());
        assert_eq!(plan.peak_memory, 0);
        assert_eq!(plan.buffer_count, 0);
        assert_eq!(plan.reuse_count, 0);
    }

    // ---------------------------------------------------------------
    // compute_tensor_lifetimes stub
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_tensor_lifetimes_stub() {
        let graph = ExecutionGraph::new();
        let lifetimes = compute_tensor_lifetimes(&graph);
        assert!(lifetimes.is_empty());
    }
}
