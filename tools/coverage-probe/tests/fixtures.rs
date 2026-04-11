// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `smallaios-coverage-probe`.
//!
//! Synthesizes small in-memory `ModelProto` values and walks them end
//! to end.

use smallaios_coverage_probe::{walk_model, Inventory, Verdict};
use smallaios_onnx_rt::onnx_types::{GraphProto, ModelProto, NodeProto};

fn model(ops: &[&str]) -> ModelProto {
    ModelProto {
        graph: Some(GraphProto {
            node: ops
                .iter()
                .map(|o| NodeProto {
                    op_type: (*o).to_string(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn end_to_end_bert_like_model_is_loadable_after_phase1() {
    // A very rough BERT-shaped op set: most are implemented in
    // develop, but `Sin` (for positional encoding) is Planned-P1 in
    // the real roadmap.
    let m = model(&[
        "MatMul",
        "Add",
        "LayerNormalization",
        "Sin",
        "MatMul",
        "Softmax",
    ]);
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/onnx-coverage-roadmap.md"
    );
    let inv = Inventory::from_roadmap_file(path).expect("roadmap should parse");
    let report = walk_model(&m, "bert-like.onnx", &inv).unwrap();
    assert_eq!(report.total_nodes, 6);
    assert_eq!(report.distinct_ops, 5);
    // `Sin` is Planned-P1 in the real roadmap.
    assert!(report.counts.planned_p1 >= 1);
    assert_eq!(report.verdict, Verdict::LoadableAfterPhase1);
}

#[test]
fn end_to_end_purely_implemented_model() {
    let m = model(&["MatMul", "Add", "Relu"]);
    let inv = Inventory::from_roadmap_str("");
    let report = walk_model(&m, "basic.onnx", &inv).unwrap();
    assert_eq!(report.verdict, Verdict::LoadableToday);
    assert_eq!(report.counts.implemented, 3);
}
