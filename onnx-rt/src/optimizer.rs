// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph optimizer for the SmallAIOS ONNX runtime.
//!
//! Provides optimization passes that transform an `ExecutionGraph` to
//! improve inference performance. Currently implemented as stubs that
//! establish the optimization framework; concrete pass implementations
//! will follow.

use alloc::vec::Vec;

use crate::graph::ExecutionGraph;

// Available for future use when optimization passes inspect the original graph.
#[allow(unused_imports)]
use crate::graph::{ExecutionNode, NodeIndex};
#[allow(unused_imports)]
use crate::onnx_types::GraphProto;

/// Individual optimization passes that can be applied to a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Evaluate constant sub-expressions at compile time.
    ConstantFolding,
    /// Merge compatible adjacent operators (e.g., Conv + Relu).
    OperatorFusion,
    /// Remove nodes whose outputs are never consumed.
    DeadCodeElimination,
    /// Merge duplicate sub-expressions that produce identical results.
    CommonSubexpressionElimination,
}

impl OptimizationPass {
    /// Returns the human-readable name of this pass.
    pub fn name(&self) -> &'static str {
        match self {
            OptimizationPass::ConstantFolding => "constant_folding",
            OptimizationPass::OperatorFusion => "operator_fusion",
            OptimizationPass::DeadCodeElimination => "dead_code_elimination",
            OptimizationPass::CommonSubexpressionElimination => "common_subexpression_elimination",
        }
    }
}

/// Controls the aggressiveness of graph optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization passes are applied.
    None,
    /// Basic optimizations only (dead code elimination).
    Basic,
    /// All available optimizations (DCE + constant folding + fusion).
    Extended,
}

/// Result of running optimization passes on a graph.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The passes that were actually applied.
    pub passes_applied: Vec<OptimizationPass>,
    /// Number of nodes removed by all passes combined.
    pub nodes_removed: usize,
    /// Number of nodes fused by all passes combined.
    pub nodes_fused: usize,
}

impl OptimizationResult {
    /// Creates an empty result with no changes.
    fn empty() -> Self {
        Self {
            passes_applied: Vec::new(),
            nodes_removed: 0,
            nodes_fused: 0,
        }
    }
}

/// Configuration for the graph optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// The optimization level.
    pub level: OptimizationLevel,
    /// Maximum number of optimization iterations.
    pub max_passes: usize,
    /// Whether operator fusion is enabled.
    pub enable_fusion: bool,
    /// Whether constant folding is enabled.
    pub enable_constant_folding: bool,
    /// Whether dead code elimination is enabled.
    pub enable_dce: bool,
}

impl Default for OptimizerConfig {
    /// Creates a default configuration with `Extended` optimization.
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Extended,
            max_passes: 10,
            enable_fusion: true,
            enable_constant_folding: true,
            enable_dce: true,
        }
    }
}

impl OptimizerConfig {
    /// Creates a configuration that disables all optimization.
    pub fn none() -> Self {
        Self {
            level: OptimizationLevel::None,
            max_passes: 0,
            enable_fusion: false,
            enable_constant_folding: false,
            enable_dce: false,
        }
    }
}

/// Optimizes an execution graph according to the given configuration.
///
/// Iterates through optimization passes based on the configured level:
/// - `None`: no passes are run.
/// - `Basic`: dead code elimination only.
/// - `Extended`: dead code elimination, constant folding, and operator fusion.
///
/// Returns an `OptimizationResult` summarizing the transformations applied.
pub fn optimize(graph: &mut ExecutionGraph, config: &OptimizerConfig) -> OptimizationResult {
    let mut result = OptimizationResult::empty();

    match config.level {
        OptimizationLevel::None => {
            // No optimization.
        }
        OptimizationLevel::Basic => {
            if config.enable_dce {
                let removed = dead_code_elimination(graph);
                result.nodes_removed += removed;
                result
                    .passes_applied
                    .push(OptimizationPass::DeadCodeElimination);
            }
        }
        OptimizationLevel::Extended => {
            if config.enable_dce {
                let removed = dead_code_elimination(graph);
                result.nodes_removed += removed;
                result
                    .passes_applied
                    .push(OptimizationPass::DeadCodeElimination);
            }
            if config.enable_constant_folding {
                let folded = constant_folding(graph);
                result.nodes_removed += folded;
                result
                    .passes_applied
                    .push(OptimizationPass::ConstantFolding);
            }
            if config.enable_fusion {
                let fused = operator_fusion(graph);
                result.nodes_fused += fused;
                result.passes_applied.push(OptimizationPass::OperatorFusion);
            }
        }
    }

    result
}

/// Dead code elimination: removes nodes whose outputs are never consumed.
///
/// Stub implementation; returns 0 (no nodes removed).
pub fn dead_code_elimination(_graph: &mut ExecutionGraph) -> usize {
    // TODO: Identify nodes whose outputs are not consumed by any
    // downstream node or graph output, and remove them.
    0
}

/// Constant folding: evaluates constant sub-expressions at compile time.
///
/// Stub implementation; returns 0 (no nodes folded).
pub fn constant_folding(_graph: &mut ExecutionGraph) -> usize {
    // TODO: Identify sub-graphs where all inputs are constant
    // initializers, evaluate them, and replace with constant nodes.
    0
}

/// Operator fusion: merges compatible adjacent operators.
///
/// Stub implementation; returns 0 (no nodes fused).
pub fn operator_fusion(_graph: &mut ExecutionGraph) -> usize {
    // TODO: Identify fusible patterns (e.g., Conv+Relu, Conv+BN)
    // and replace them with single fused operators.
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates a minimal empty `ExecutionGraph`.
    fn empty_graph() -> ExecutionGraph {
        ExecutionGraph::new()
    }

    // ---------------------------------------------------------------
    // OptimizerConfig defaults
    // ---------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.level, OptimizationLevel::Extended);
        assert_eq!(config.max_passes, 10);
        assert!(config.enable_fusion);
        assert!(config.enable_constant_folding);
        assert!(config.enable_dce);
    }

    #[test]
    fn test_config_none() {
        let config = OptimizerConfig::none();
        assert_eq!(config.level, OptimizationLevel::None);
        assert_eq!(config.max_passes, 0);
        assert!(!config.enable_fusion);
        assert!(!config.enable_constant_folding);
        assert!(!config.enable_dce);
    }

    // ---------------------------------------------------------------
    // OptimizationLevel variants
    // ---------------------------------------------------------------

    #[test]
    fn test_optimization_levels_distinct() {
        assert_ne!(OptimizationLevel::None, OptimizationLevel::Basic);
        assert_ne!(OptimizationLevel::Basic, OptimizationLevel::Extended);
        assert_ne!(OptimizationLevel::None, OptimizationLevel::Extended);
    }

    // ---------------------------------------------------------------
    // OptimizationPass names
    // ---------------------------------------------------------------

    #[test]
    fn test_pass_names() {
        assert_eq!(OptimizationPass::ConstantFolding.name(), "constant_folding");
        assert_eq!(OptimizationPass::OperatorFusion.name(), "operator_fusion");
        assert_eq!(
            OptimizationPass::DeadCodeElimination.name(),
            "dead_code_elimination"
        );
        assert_eq!(
            OptimizationPass::CommonSubexpressionElimination.name(),
            "common_subexpression_elimination"
        );
    }

    // ---------------------------------------------------------------
    // Optimize with None level
    // ---------------------------------------------------------------

    #[test]
    fn test_optimize_none_level() {
        let mut graph = empty_graph();
        let config = OptimizerConfig::none();
        let result = optimize(&mut graph, &config);
        assert!(result.passes_applied.is_empty());
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(result.nodes_fused, 0);
    }

    // ---------------------------------------------------------------
    // Optimize with Basic level
    // ---------------------------------------------------------------

    #[test]
    fn test_optimize_basic_level() {
        let mut graph = empty_graph();
        let config = OptimizerConfig {
            level: OptimizationLevel::Basic,
            max_passes: 10,
            enable_fusion: true,
            enable_constant_folding: true,
            enable_dce: true,
        };
        let result = optimize(&mut graph, &config);
        assert_eq!(result.passes_applied.len(), 1);
        assert_eq!(
            result.passes_applied[0],
            OptimizationPass::DeadCodeElimination
        );
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(result.nodes_fused, 0);
    }

    // ---------------------------------------------------------------
    // Optimize with Extended level
    // ---------------------------------------------------------------

    #[test]
    fn test_optimize_extended_level() {
        let mut graph = empty_graph();
        let config = OptimizerConfig::default();
        let result = optimize(&mut graph, &config);
        assert_eq!(result.passes_applied.len(), 3);
        assert_eq!(
            result.passes_applied[0],
            OptimizationPass::DeadCodeElimination
        );
        assert_eq!(result.passes_applied[1], OptimizationPass::ConstantFolding);
        assert_eq!(result.passes_applied[2], OptimizationPass::OperatorFusion);
        // Stubs return 0.
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(result.nodes_fused, 0);
    }

    // ---------------------------------------------------------------
    // Stub pass counts
    // ---------------------------------------------------------------

    #[test]
    fn test_dce_stub_returns_zero() {
        let mut graph = empty_graph();
        assert_eq!(dead_code_elimination(&mut graph), 0);
    }

    #[test]
    fn test_constant_folding_stub_returns_zero() {
        let mut graph = empty_graph();
        assert_eq!(constant_folding(&mut graph), 0);
    }

    #[test]
    fn test_operator_fusion_stub_returns_zero() {
        let mut graph = empty_graph();
        assert_eq!(operator_fusion(&mut graph), 0);
    }

    // ---------------------------------------------------------------
    // Extended with selective disable
    // ---------------------------------------------------------------

    #[test]
    fn test_extended_with_fusion_disabled() {
        let mut graph = empty_graph();
        let config = OptimizerConfig {
            level: OptimizationLevel::Extended,
            max_passes: 10,
            enable_fusion: false,
            enable_constant_folding: true,
            enable_dce: true,
        };
        let result = optimize(&mut graph, &config);
        assert_eq!(result.passes_applied.len(), 2);
        assert!(!result
            .passes_applied
            .contains(&OptimizationPass::OperatorFusion));
    }
}
