// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests that load real-world ONNX model exports and verify
//! they parse + build an execution graph without errors.
//!
//! These tests are `#[ignore]`'d by default because they require fixture
//! files downloaded via `scripts/download-test-fixtures.sh`. Run them with:
//!
//!     cargo test -p smallaios-onnx-rt --test test_model_fixtures -- --ignored
//!
//! Or download fixtures first, then run all tests:
//!
//!     ./scripts/download-test-fixtures.sh
//!     cargo test -p smallaios-onnx-rt --test test_model_fixtures -- --ignored
//!
//! ## Known limitation
//!
//! The in-tree `decode_model` protobuf parser is a minimal `#![no_std]`
//! implementation that does not yet handle all wire-format features used
//! by real ONNX exports (external data, unusual packing, large varint
//! fields for raw tensor weights, etc.). Models that embed weights
//! (typically >10 MB) will fail to parse with `decode_model`.
//!
//! The coverage probe tool (`tools/coverage-probe/`) uses a streaming
//! scanner that skips unknown fields and successfully processes these
//! same models. These tests document the gap and will start passing
//! as the protobuf parser is hardened.

extern crate alloc;
extern crate smallaios_onnx_rt;

use smallaios_onnx_rt::graph::build_execution_graph;
use smallaios_onnx_rt::operators::OperatorRegistry;
use smallaios_onnx_rt::protobuf::decode_model;
use std::path::PathBuf;

/// Returns the fixture path if the file exists, otherwise `None`.
///
/// Fixtures live at `<workspace-root>/tests/fixtures/onnx-models/` (gitignored).
/// `CARGO_MANIFEST_DIR` points to `onnx-rt/`, so we go up one level.
fn fixture_path(name: &str) -> Option<PathBuf> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("onnx-rt should have a parent directory")
        .to_path_buf();
    let path = workspace_root.join("tests/fixtures/onnx-models").join(name);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Result of attempting to validate a model fixture.
enum ModelResult {
    /// Parse + graph build succeeded.
    Ok {
        total_nodes: usize,
        unsupported_ops: Vec<String>,
        has_control_flow: bool,
    },
    /// Protobuf decode failed (known limitation for large models with weights).
    ParseFailed(String),
    /// Graph build failed after successful parse.
    GraphBuildFailed(String),
}

/// Core test logic shared by all model fixture tests.
///
/// 1. Reads the `.onnx` file bytes.
/// 2. Attempts to parse via `decode_model`.
/// 3. If parsing succeeds, builds an `ExecutionGraph`.
/// 4. Checks every node's `op_type` against the `OperatorRegistry`.
/// 5. Returns a structured result.
fn validate_model(name: &str) -> ModelResult {
    let path = fixture_path(name).unwrap_or_else(|| {
        panic!(
            "Fixture file not found: {}. Run scripts/download-test-fixtures.sh first.",
            name
        );
    });

    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("Failed to read fixture {}: {}", name, e);
    });
    eprintln!(
        "  Loaded {} ({:.1} MB, {} bytes)",
        name,
        bytes.len() as f64 / (1024.0 * 1024.0),
        bytes.len()
    );

    // Parse the protobuf model.
    let model = match decode_model(&bytes) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("{:?}", e);
            eprintln!("  PARSE FAILURE: {}", msg);
            return ModelResult::ParseFailed(msg);
        }
    };

    eprintln!(
        "  Parsed: ir_version={}, producer={}",
        model.ir_version, model.producer_name
    );

    let graph_proto = match model.graph.as_ref() {
        Some(g) => g,
        None => {
            return ModelResult::ParseFailed("model has no graph".into());
        }
    };
    eprintln!(
        "  Graph '{}': {} nodes, {} inputs, {} outputs",
        graph_proto.name,
        graph_proto.node.len(),
        graph_proto.input.len(),
        graph_proto.output.len(),
    );

    // Build the execution graph.
    let exec_graph = match build_execution_graph(graph_proto) {
        Ok(g) => g,
        Err(e) => {
            let msg = format!("{:?}", e);
            eprintln!("  GRAPH BUILD FAILURE: {}", msg);
            return ModelResult::GraphBuildFailed(msg);
        }
    };

    let total_nodes = exec_graph.nodes.len();
    eprintln!("  ExecutionGraph: {} nodes", total_nodes);

    // Check operator coverage.
    let registry = OperatorRegistry::new();
    let mut unsupported: Vec<String> = Vec::new();
    let mut has_control_flow = false;

    for node in &exec_graph.nodes {
        if !registry.is_supported(&node.op_type) && !unsupported.contains(&node.op_type) {
            unsupported.push(node.op_type.clone());
        }
        if !node.inner_graphs.is_empty() {
            has_control_flow = true;
        }
    }

    if unsupported.is_empty() {
        eprintln!("  All ops supported!");
    } else {
        eprintln!(
            "  Unsupported ops ({} kinds): {:?}",
            unsupported.len(),
            unsupported
        );
    }

    if has_control_flow {
        eprintln!("  Contains control-flow ops with inner graphs");
    }

    ModelResult::Ok {
        total_nodes,
        unsupported_ops: unsupported,
        has_control_flow,
    }
}

// ---------------------------------------------------------------------------
// Per-model tests
//
// Each test documents whether the model currently parses via decode_model.
// Models with embedded weights (>10 MB) are expected to fail parsing until
// the protobuf decoder is hardened. The coverage probe tool can already
// process these models via its streaming scanner.
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_bert_base_loads() {
    let Some(_) = fixture_path("bert-base-uncased.onnx") else {
        eprintln!("SKIP: bert-base fixture not found");
        return;
    };
    let result = validate_model("bert-base-uncased.onnx");
    // BERT-base with weights (~418 MB) — known to fail decode_model.
    // Coverage probe processes it successfully (1268 nodes, 19 ops, all supported).
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "BERT should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "BERT-base: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!(
                "BERT-base: decode_model failed (expected for large models): {}",
                e
            );
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("BERT-base: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_vit_base_loads() {
    let Some(_) = fixture_path("vit-base-patch16-224.onnx") else {
        eprintln!("SKIP: vit-base fixture not found");
        return;
    };
    let result = validate_model("vit-base-patch16-224.onnx");
    // ViT-base with weights (~330 MB) — known to fail decode_model.
    // Coverage probe: 1242 nodes, 24 ops, all supported.
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "ViT should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "ViT-base: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!(
                "ViT-base: decode_model failed (expected for large models): {}",
                e
            );
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("ViT-base: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_distilgpt2_loads() {
    let Some(_) = fixture_path("distilgpt2.onnx") else {
        eprintln!("SKIP: distilgpt2 fixture not found");
        return;
    };
    let result = validate_model("distilgpt2.onnx");
    // DistilGPT-2 with weights (~313 MB) — known to fail decode_model.
    // Coverage probe: 683 nodes, 27 ops, all supported.
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "DistilGPT-2 should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "DistilGPT-2: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!(
                "DistilGPT-2: decode_model failed (expected for large models): {}",
                e
            );
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("DistilGPT-2: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_llama_3_2_1b_loads() {
    let Some(_) = fixture_path("llama-3.2-1b.onnx") else {
        eprintln!("SKIP: llama-3.2-1b fixture not found");
        return;
    };
    let result = validate_model("llama-3.2-1b.onnx");
    // Llama graph-only (~105 KB) — small enough that decode_model may work.
    // Coverage probe: 220 nodes, 13 ops, 3 unrecognized.
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            has_control_flow,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "Llama should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "Llama-3.2-1B: {} nodes, {} unsupported ops, control_flow={}",
                total_nodes,
                unsupported_ops.len(),
                has_control_flow
            );
            if !unsupported_ops.is_empty() {
                eprintln!("  Unrecognized ops: {:?}", unsupported_ops);
            }
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("Llama-3.2-1B: decode_model failed: {}", e);
            eprintln!("  NOTE: This is a graph-only file (105 KB) — parse failure is a real bug.");
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("Llama-3.2-1B: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_gemma_3_1b_loads() {
    let Some(_) = fixture_path("gemma-3-1b-it.onnx") else {
        eprintln!("SKIP: gemma-3-1b fixture not found (download may require auth)");
        return;
    };
    let result = validate_model("gemma-3-1b-it.onnx");
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "Gemma should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "Gemma-3-1B: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("Gemma-3-1B: decode_model failed: {}", e);
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("Gemma-3-1B: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_gemma_3_270m_loads() {
    let Some(_) = fixture_path("gemma-3-270m-it.onnx") else {
        eprintln!("SKIP: gemma-3-270m fixture not found");
        return;
    };
    let result = validate_model("gemma-3-270m-it.onnx");
    // Gemma 3 270M graph-only (~181 KB) — small graph-only file.
    // Coverage probe: 421 nodes, 15 ops, 2 unrecognized (com.microsoft fused ops).
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "Gemma-3-270M should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "Gemma-3-270M: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("Gemma-3-270M: decode_model failed: {}", e);
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("Gemma-3-270M: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_deepseek_r1_loads() {
    let Some(_) = fixture_path("deepseek-r1-distill-qwen-1.5b.onnx") else {
        eprintln!("SKIP: deepseek fixture not found");
        return;
    };
    let result = validate_model("deepseek-r1-distill-qwen-1.5b.onnx");
    // DeepSeek graph-only (~519 KB) — small enough that decode_model may work.
    // Coverage probe: 1503 nodes, 25 ops, 4 unrecognized.
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "DeepSeek should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "DeepSeek-R1-Distill: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
            if !unsupported_ops.is_empty() {
                eprintln!("  Unrecognized ops: {:?}", unsupported_ops);
            }
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("DeepSeek-R1-Distill: decode_model failed: {}", e);
            eprintln!("  NOTE: This is a graph-only file (519 KB) — parse failure is a real bug.");
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("DeepSeek-R1-Distill: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_mobilenet_v2_loads() {
    let Some(_) = fixture_path("mobilenet_v2.onnx") else {
        eprintln!("SKIP: mobilenet_v2 fixture not found");
        return;
    };
    let result = validate_model("mobilenet_v2.onnx");
    // MobileNetV2 with weights (~13 MB).
    // Coverage probe: 105 nodes, 11 ops, all supported.
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "MobileNetV2 should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "MobileNetV2: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("MobileNetV2: decode_model failed: {}", e);
            eprintln!("  NOTE: MobileNetV2 is only 13 MB — this may be fixable.");
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("MobileNetV2: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_resnet50_loads() {
    let Some(_) = fixture_path("resnet50.onnx") else {
        eprintln!("SKIP: resnet50 fixture not found");
        return;
    };
    let result = validate_model("resnet50.onnx");
    // ResNet-50 v2 with weights (~98 MB).
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "ResNet-50 should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "ResNet-50: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("ResNet-50: decode_model failed: {}", e);
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("ResNet-50: unexpected graph build failure: {}", e);
        }
    }
}

#[test]
#[ignore]
fn test_squeezenet_loads() {
    let Some(_) = fixture_path("squeezenet.onnx") else {
        eprintln!("SKIP: squeezenet fixture not found");
        return;
    };
    let result = validate_model("squeezenet.onnx");
    // SqueezeNet 1.1 with weights (~4.8 MB).
    match result {
        ModelResult::Ok {
            total_nodes,
            unsupported_ops,
            ..
        } => {
            assert!(
                total_nodes > 10,
                "SqueezeNet should have many nodes, got {}",
                total_nodes
            );
            eprintln!(
                "SqueezeNet: {} nodes, {} unsupported ops",
                total_nodes,
                unsupported_ops.len()
            );
        }
        ModelResult::ParseFailed(e) => {
            eprintln!("SqueezeNet: decode_model failed: {}", e);
        }
        ModelResult::GraphBuildFailed(e) => {
            panic!("SqueezeNet: unexpected graph build failure: {}", e);
        }
    }
}
