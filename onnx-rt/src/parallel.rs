// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Thread pool for operator-level data parallelism.
//!
//! Container mode uses `std::thread::scope` for scoped threads.
//! Kernel mode falls back to sequential execution (Phase 2 will
//! use the kernel scheduler's per-core RunQueues).

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

/// Thread pool that distributes work across CPU cores.
///
/// In container mode, uses `std::thread::scope` for zero-overhead scoped
/// threads. Each call to [`parallel_for`](CorePool::parallel_for) splits
/// the given range into chunks and executes them in parallel.
pub struct CorePool {
    num_threads: usize,
}

impl CorePool {
    /// Creates a new `CorePool` with the specified number of threads.
    ///
    /// The thread count is clamped to a minimum of 1.
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
        }
    }

    /// Returns the number of threads in this pool.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Splits `range` into `num_threads` chunks, executes `f` on each chunk
    /// in parallel, and collects results.
    ///
    /// If `num_threads` is 1 or the range is empty, executes sequentially
    /// on the calling thread (no thread spawn overhead).
    pub fn parallel_for<F, R>(&self, range: Range<usize>, f: F) -> Vec<R>
    where
        F: Fn(Range<usize>) -> R + Send + Sync,
        R: Send,
    {
        let len = range.end.saturating_sub(range.start);
        if len == 0 {
            return vec![];
        }
        if self.num_threads <= 1 {
            return vec![f(range)];
        }

        #[cfg(feature = "std")]
        {
            // Container mode: use std::thread::scope for real parallelism
            let effective_threads = self.num_threads.min(len);
            let chunk_size = len.div_ceil(effective_threads);
            let mut results = Vec::new();

            std::thread::scope(|s| {
                let mut handles = Vec::new();
                let mut pos = range.start;
                while pos < range.end {
                    let end = (pos + chunk_size).min(range.end);
                    let chunk_range = pos..end;
                    handles.push(s.spawn(|| f(chunk_range)));
                    pos = end;
                }
                for h in handles {
                    results.push(h.join().unwrap());
                }
            });

            results
        }

        #[cfg(not(feature = "std"))]
        {
            // Kernel mode: sequential fallback (Phase 2 will use RunQueue)
            vec![f(range)]
        }
    }
}

// ---------------------------------------------------------------------------
// ParallelConfig
// ---------------------------------------------------------------------------

/// Configuration for operator-level parallelism.
///
/// Controls the number of threads and per-operator thresholds that
/// determine when parallel execution is worthwhile. Below-threshold
/// operations execute sequentially to avoid fork/join overhead.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of threads for intra-operator parallelism.
    pub max_threads: usize,
    /// GEMM: parallelize when `M * K * N > gemm_threshold`.
    pub gemm_threshold: usize,
    /// Conv: parallelize when `output_channels * H * W > conv_threshold`.
    pub conv_threshold: usize,
    /// Element-wise: parallelize when `num_elements > elementwise_threshold`.
    pub elementwise_threshold: usize,
    /// Reduction: parallelize when `num_elements > reduction_threshold`.
    pub reduction_threshold: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: 1, // sequential by default
            gemm_threshold: 65_536,
            conv_threshold: 16_384,
            elementwise_threshold: 32_768,
            reduction_threshold: 65_536,
        }
    }
}

impl ParallelConfig {
    /// Returns a configuration that disables all parallelism.
    pub fn sequential() -> Self {
        Self {
            max_threads: 1,
            ..Default::default()
        }
    }

    /// Returns a configuration tuned for the given core count.
    ///
    /// Lowers thresholds for higher core counts since fork/join overhead
    /// is amortized across more workers.
    pub fn default_for_cores(n: usize) -> Self {
        let n = n.max(1);
        let divisor = if n >= 8 {
            4
        } else if n >= 4 {
            2
        } else {
            1
        };
        Self {
            max_threads: n,
            gemm_threshold: 65_536 / divisor,
            conv_threshold: 16_384 / divisor,
            elementwise_threshold: 32_768 / divisor,
            reduction_threshold: 65_536 / divisor,
        }
    }

    /// Creates a [`CorePool`] based on this configuration.
    pub fn build_pool(&self) -> CorePool {
        CorePool::new(self.max_threads)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use alloc::vec;

    #[test]
    fn test_core_pool_single_thread() {
        let pool = CorePool::new(1);
        assert_eq!(pool.num_threads(), 1);
        let results = pool.parallel_for(0..10, |r| {
            let mut sum = 0usize;
            for i in r {
                sum += i;
            }
            sum
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 45); // 0+1+..+9
    }

    #[test]
    fn test_core_pool_two_threads() {
        let pool = CorePool::new(2);
        assert_eq!(pool.num_threads(), 2);
        let results = pool.parallel_for(0..10, |r| {
            let mut sum = 0usize;
            for i in r {
                sum += i;
            }
            sum
        });
        let total: usize = results.iter().sum();
        assert_eq!(total, 45);
    }

    #[test]
    fn test_core_pool_four_threads() {
        let pool = CorePool::new(4);
        let results = pool.parallel_for(0..100, |r| {
            let mut sum = 0usize;
            for i in r {
                sum += i;
            }
            sum
        });
        let total: usize = results.iter().sum();
        assert_eq!(total, 4950); // 0+1+..+99
    }

    #[test]
    fn test_core_pool_range_smaller_than_threads() {
        let pool = CorePool::new(8);
        let results = pool.parallel_for(0..3, |r| {
            let mut sum = 0usize;
            for i in r {
                sum += i;
            }
            sum
        });
        let total: usize = results.iter().sum();
        assert_eq!(total, 3); // 0+1+2
                              // Should not spawn more threads than work items
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_core_pool_empty_range() {
        let pool = CorePool::new(4);
        let results: Vec<usize> = pool.parallel_for(0..0, |r| {
            let mut sum = 0usize;
            for i in r {
                sum += i;
            }
            sum
        });
        assert!(results.is_empty());
    }

    #[test]
    fn test_core_pool_min_threads_clamped() {
        let pool = CorePool::new(0);
        assert_eq!(pool.num_threads(), 1);
    }

    #[test]
    fn test_core_pool_collect_vec_results() {
        // Verify parallel_for can collect Vec results (used by parallel GEMM)
        let pool = CorePool::new(2);
        let results = pool.parallel_for(0..4, |r| {
            let items: Vec<usize> = r.collect();
            items
        });
        let mut all: Vec<usize> = Vec::new();
        for chunk in results {
            all.extend(chunk);
        }
        all.sort();
        assert_eq!(all, vec![0, 1, 2, 3]);
    }

    // --- ParallelConfig tests ---

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert_eq!(config.max_threads, 1);
        assert_eq!(config.gemm_threshold, 65_536);
        assert_eq!(config.conv_threshold, 16_384);
        assert_eq!(config.elementwise_threshold, 32_768);
        assert_eq!(config.reduction_threshold, 65_536);
    }

    #[test]
    fn test_parallel_config_sequential() {
        let config = ParallelConfig::sequential();
        assert_eq!(config.max_threads, 1);
    }

    #[test]
    fn test_parallel_config_for_cores() {
        let config = ParallelConfig::default_for_cores(4);
        assert_eq!(config.max_threads, 4);
        // 4 cores => divisor 2
        assert_eq!(config.gemm_threshold, 65_536 / 2);

        let config8 = ParallelConfig::default_for_cores(8);
        assert_eq!(config8.max_threads, 8);
        // 8 cores => divisor 4
        assert_eq!(config8.gemm_threshold, 65_536 / 4);
    }

    #[test]
    fn test_parallel_config_build_pool() {
        let config = ParallelConfig::default_for_cores(4);
        let pool = config.build_pool();
        assert_eq!(pool.num_threads(), 4);
    }
}
