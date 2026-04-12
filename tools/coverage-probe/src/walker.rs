// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX model walker: consumes a `ModelProto`, tallies op kinds, and
//! optionally flags known attribute-level limitations.

use crate::inventory::{Inventory, OperatorStatus};
use crate::report::{CoverageReport, NodeSummary, StatusCounts, Verdict};
use smallaios_onnx_rt::onnx_types::{AttributeProto, AttributeType, ModelProto, NodeProto};
use smallaios_onnx_rt::protobuf::{decode_node, ProtoDecoder, ProtoError, WireType};
use std::collections::BTreeMap;

/// Error type returned by [`walk_bytes`].
#[derive(Debug)]
pub enum WalkError {
    /// Protobuf decode failed.
    Decode(ProtoError),
    /// The model has no graph.
    EmptyModel,
}

impl std::fmt::Display for WalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalkError::Decode(e) => write!(f, "protobuf decode error: {e:?}"),
            WalkError::EmptyModel => write!(f, "model has no graph"),
        }
    }
}

impl std::error::Error for WalkError {}

/// Parse a raw ONNX model buffer and walk it, producing a full report.
///
/// This uses a streaming scanner that only extracts each node's
/// `op_type` (and, best-effort, attributes for a small set of ops that
/// surface attribute-level concerns). It deliberately avoids fully
/// decoding `TensorProto` initializers and other heavy nested messages
/// so that it remains robust against real-world ONNX exports that use
/// corners of the wire format (groups, unusual packing, external data,
/// non-UTF-8 doc strings, …) our minimal in-tree decoder doesn't handle.
pub fn walk_bytes(
    bytes: &[u8],
    model_label: &str,
    inventory: &Inventory,
) -> Result<CoverageReport, WalkError> {
    // Stream the top-level ModelProto looking only for field 7 = graph.
    let mut dec = ProtoDecoder::new(bytes);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut concerns_by_op: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total_nodes: usize = 0;
    let mut found_graph = false;
    while dec.remaining() > 0 {
        let header = dec.read_tag().map_err(WalkError::Decode)?;
        if header.field_number == 7 && header.wire_type == WireType::LengthDelimited {
            let graph_data = dec.read_length_delimited().map_err(WalkError::Decode)?;
            found_graph = true;
            scan_graph(
                graph_data,
                &mut counts,
                &mut concerns_by_op,
                &mut total_nodes,
            )?;
        } else {
            tolerant_skip(&mut dec, header.wire_type).map_err(WalkError::Decode)?;
        }
    }
    if !found_graph {
        return Err(WalkError::EmptyModel);
    }
    Ok(assemble_report(
        model_label,
        inventory,
        total_nodes,
        counts,
        concerns_by_op,
    ))
}

/// Skip a protobuf field by wire type, tolerating malformed length
/// prefixes by clamping at the remaining buffer length. We prefer to
/// surface real op-coverage findings over bailing out on an oversized
/// doc_string or non-UTF8 producer field.
fn tolerant_skip(dec: &mut ProtoDecoder, wire: WireType) -> Result<(), ProtoError> {
    match wire {
        WireType::Varint => {
            dec.read_varint()?;
        }
        WireType::Fixed64 => {
            if dec.remaining() < 8 {
                return Err(ProtoError::UnexpectedEof);
            }
            // read_fixed64 advances the cursor
            dec.read_fixed64()?;
        }
        WireType::Fixed32 => {
            if dec.remaining() < 4 {
                return Err(ProtoError::UnexpectedEof);
            }
            dec.read_fixed32()?;
        }
        WireType::LengthDelimited => {
            dec.read_length_delimited()?;
        }
    }
    Ok(())
}

/// Scan a GraphProto byte region, recursively descending into nodes,
/// attributes and subgraphs. Only `op_type` is extracted from each
/// node; attributes are best-effort-decoded when the op matches one
/// known to carry attribute-level concerns (Resize, RoiAlign, etc).
fn scan_graph(
    data: &[u8],
    counts: &mut BTreeMap<String, usize>,
    concerns: &mut BTreeMap<String, Vec<String>>,
    total_nodes: &mut usize,
) -> Result<(), WalkError> {
    let mut dec = ProtoDecoder::new(data);
    while dec.remaining() > 0 {
        let header = dec.read_tag().map_err(WalkError::Decode)?;
        match (header.field_number, header.wire_type) {
            (1, WireType::LengthDelimited) => {
                // node
                let node_bytes = dec.read_length_delimited().map_err(WalkError::Decode)?;
                scan_node(node_bytes, counts, concerns, total_nodes)?;
            }
            _ => {
                tolerant_skip(&mut dec, header.wire_type).map_err(WalkError::Decode)?;
            }
        }
    }
    Ok(())
}

/// Scan a NodeProto byte region. Extracts op_type (field 4), and for
/// ops with known attribute-level concerns, attempts to decode the
/// full node and runs the concern checker.
fn scan_node(
    data: &[u8],
    counts: &mut BTreeMap<String, usize>,
    concerns_map: &mut BTreeMap<String, Vec<String>>,
    total_nodes: &mut usize,
) -> Result<(), WalkError> {
    // First pass: just extract op_type and any subgraphs (attributes
    // can contain GraphProto via field 6).
    let mut dec = ProtoDecoder::new(data);
    let mut op_type: Option<String> = None;
    let mut attr_regions: Vec<&[u8]> = Vec::new();
    while dec.remaining() > 0 {
        let header = dec.read_tag().map_err(WalkError::Decode)?;
        match (header.field_number, header.wire_type) {
            (4, WireType::LengthDelimited) => {
                let s = dec.read_length_delimited().map_err(WalkError::Decode)?;
                op_type = Some(String::from_utf8_lossy(s).into_owned());
            }
            (5, WireType::LengthDelimited) => {
                let attr_bytes = dec.read_length_delimited().map_err(WalkError::Decode)?;
                attr_regions.push(attr_bytes);
            }
            _ => tolerant_skip(&mut dec, header.wire_type).map_err(WalkError::Decode)?,
        }
    }
    let op_type = match op_type {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(()), // nameless node — ignore
    };

    *total_nodes += 1;
    *counts.entry(op_type.clone()).or_insert(0) += 1;

    // Attribute concerns: only try to decode attributes if the op is
    // known to have attribute-level limitations.
    let needs_attrs = matches!(
        op_type.as_str(),
        "Resize" | "RoiAlign" | "GridSample" | "Unique" | "ScatterND"
    );

    if needs_attrs {
        // Best-effort: try full decode_node; if it fails we silently
        // drop attribute-level reporting for this node but the op
        // tally is already recorded above.
        if let Ok(node) = decode_node(data) {
            for concern in attribute_concerns_for(&node) {
                concerns_map
                    .entry(op_type.clone())
                    .or_default()
                    .push(concern);
            }
        }
    }

    // Recurse into any subgraph attributes (If, Loop, Scan, …).
    for attr_bytes in attr_regions {
        scan_attribute_for_subgraphs(attr_bytes, counts, concerns_map, total_nodes)?;
    }

    Ok(())
}

/// Scan an AttributeProto for field 6 (g / GraphProto) and field 11
/// (graphs, repeated) and recursively walk any nested subgraphs.
fn scan_attribute_for_subgraphs(
    data: &[u8],
    counts: &mut BTreeMap<String, usize>,
    concerns_map: &mut BTreeMap<String, Vec<String>>,
    total_nodes: &mut usize,
) -> Result<(), WalkError> {
    let mut dec = ProtoDecoder::new(data);
    while dec.remaining() > 0 {
        let header = dec.read_tag().map_err(WalkError::Decode)?;
        match (header.field_number, header.wire_type) {
            (6, WireType::LengthDelimited) => {
                let g = dec.read_length_delimited().map_err(WalkError::Decode)?;
                scan_graph(g, counts, concerns_map, total_nodes)?;
            }
            (11, WireType::LengthDelimited) => {
                let g = dec.read_length_delimited().map_err(WalkError::Decode)?;
                scan_graph(g, counts, concerns_map, total_nodes)?;
            }
            _ => tolerant_skip(&mut dec, header.wire_type).map_err(WalkError::Decode)?,
        }
    }
    Ok(())
}

fn assemble_report(
    model_label: &str,
    inventory: &Inventory,
    total_nodes: usize,
    counts: BTreeMap<String, usize>,
    mut concerns_by_op: BTreeMap<String, Vec<String>>,
) -> CoverageReport {
    let mut per_op: Vec<NodeSummary> = counts
        .into_iter()
        .map(|(op, count)| {
            let status = inventory.status_of(&op);
            let attribute_concerns = concerns_by_op.remove(&op).unwrap_or_default();
            NodeSummary {
                op_type: op,
                count,
                status,
                attribute_concerns,
            }
        })
        .collect();
    per_op.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.op_type.cmp(&b.op_type))
    });

    let mut status_counts = StatusCounts::default();
    for op in &per_op {
        match &op.status {
            OperatorStatus::Implemented => status_counts.implemented += 1,
            OperatorStatus::Planned(p) => match p {
                crate::inventory::Phase::P1 => status_counts.planned_p1 += 1,
                crate::inventory::Phase::P2 => status_counts.planned_p2 += 1,
                crate::inventory::Phase::P3 => status_counts.planned_p3 += 1,
                crate::inventory::Phase::P4 => status_counts.planned_p4 += 1,
            },
            OperatorStatus::DeferredSubsystem(_) => status_counts.deferred += 1,
            OperatorStatus::SkippedTraining
            | OperatorStatus::SkippedDeprecated
            | OperatorStatus::SkippedVendor => status_counts.skipped += 1,
            OperatorStatus::Unrecognized => status_counts.unrecognized += 1,
        }
    }
    let verdict = compute_verdict(&status_counts);
    CoverageReport {
        model_label: model_label.to_string(),
        total_nodes,
        distinct_ops: per_op.len(),
        per_op,
        counts: status_counts,
        verdict,
        doc_drifts: inventory.drifts().to_vec(),
    }
}

/// Walk an already-decoded [`ModelProto`].
pub fn walk_model(
    model: &ModelProto,
    model_label: &str,
    inventory: &Inventory,
) -> Result<CoverageReport, WalkError> {
    let graph = model.graph.as_ref().ok_or(WalkError::EmptyModel)?;

    // Tally op kinds.
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut concerns_by_op: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let total_nodes = graph.node.len();

    for node in &graph.node {
        *counts.entry(node.op_type.clone()).or_insert(0) += 1;
        for concern in attribute_concerns_for(node) {
            concerns_by_op
                .entry(node.op_type.clone())
                .or_default()
                .push(concern);
        }
    }

    // Build per-op summaries.
    let mut per_op: Vec<NodeSummary> = counts
        .into_iter()
        .map(|(op, count)| {
            let status = inventory.status_of(&op);
            let attribute_concerns = concerns_by_op.remove(&op).unwrap_or_default();
            NodeSummary {
                op_type: op,
                count,
                status,
                attribute_concerns,
            }
        })
        .collect();
    // Sort by count descending, then op name ascending.
    per_op.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.op_type.cmp(&b.op_type))
    });

    // Tally statuses.
    let mut counts = StatusCounts::default();
    for op in &per_op {
        match &op.status {
            OperatorStatus::Implemented => counts.implemented += 1,
            OperatorStatus::Planned(p) => match p {
                crate::inventory::Phase::P1 => counts.planned_p1 += 1,
                crate::inventory::Phase::P2 => counts.planned_p2 += 1,
                crate::inventory::Phase::P3 => counts.planned_p3 += 1,
                crate::inventory::Phase::P4 => counts.planned_p4 += 1,
            },
            OperatorStatus::DeferredSubsystem(_) => counts.deferred += 1,
            OperatorStatus::SkippedTraining
            | OperatorStatus::SkippedDeprecated
            | OperatorStatus::SkippedVendor => counts.skipped += 1,
            OperatorStatus::Unrecognized => counts.unrecognized += 1,
        }
    }

    let verdict = compute_verdict(&counts);

    Ok(CoverageReport {
        model_label: model_label.to_string(),
        total_nodes,
        distinct_ops: per_op.len(),
        per_op,
        counts,
        verdict,
        doc_drifts: inventory.drifts().to_vec(),
    })
}

/// Compute the overall verdict from aggregate status counts.
fn compute_verdict(c: &StatusCounts) -> Verdict {
    if c.unrecognized > 0 {
        return Verdict::HasUnrecognizedOps;
    }
    if c.deferred > 0 || c.skipped > 0 {
        // These are effectively unsupported for this model.
        return Verdict::NotLoadable;
    }
    if c.planned_p4 > 0 {
        return Verdict::LoadableAfterPhase4;
    }
    if c.planned_p3 > 0 {
        return Verdict::LoadableAfterPhase3;
    }
    if c.planned_p2 > 0 {
        return Verdict::LoadableAfterPhase2;
    }
    if c.planned_p1 > 0 {
        return Verdict::LoadableAfterPhase1;
    }
    Verdict::LoadableToday
}

/// Flag known attribute-level limitations seeded from the
/// `onnx-rt/src/ops/*` comments. This is a best-effort list — the goal
/// is to surface "your Resize node uses cubic mode, which is not
/// supported" kinds of gotchas before the user runs the model and gets
/// a cryptic error.
///
/// Returns a list of human-readable concern strings.
pub fn attribute_concerns_for(node: &NodeProto) -> Vec<String> {
    let mut concerns = Vec::new();
    match node.op_type.as_str() {
        "Resize" => {
            if let Some(s) = string_attr(node, "mode") {
                if s != "linear" && s != "nearest" {
                    concerns.push(format!(
                        "Resize node `{}` uses mode={s}; only linear and nearest are supported",
                        node.name
                    ));
                }
            }
            if let Some(s) = string_attr(node, "coordinate_transformation_mode") {
                if s != "half_pixel" {
                    concerns.push(format!(
                        "Resize node `{}` uses coordinate_transformation_mode={s}; only half_pixel is supported",
                        node.name
                    ));
                }
            }
            if let Some(f) = float_attr(node, "cubic_coeff_a") {
                if (f - -0.75).abs() > 1e-6 {
                    concerns.push(format!(
                        "Resize node `{}` uses cubic_coeff_a={f}; only -0.75 is supported",
                        node.name
                    ));
                }
            }
            if let Some(i) = int_attr(node, "exclude_outside") {
                if i != 0 {
                    concerns.push(format!(
                        "Resize node `{}` uses exclude_outside={i}; only 0 is supported",
                        node.name
                    ));
                }
            }
            if let Some(f) = float_attr(node, "extrapolation_value") {
                if f != 0.0 {
                    concerns.push(format!(
                        "Resize node `{}` uses extrapolation_value={f}; only 0 is supported",
                        node.name
                    ));
                }
            }
        }
        "RoiAlign" => {
            if let Some(s) = string_attr(node, "mode") {
                if s != "avg" {
                    concerns.push(format!(
                        "RoiAlign node `{}` uses mode={s}; only avg is supported",
                        node.name
                    ));
                }
            }
        }
        "GridSample" => {
            if let Some(s) = string_attr(node, "padding_mode") {
                if s != "zeros" {
                    concerns.push(format!(
                        "GridSample node `{}` uses padding_mode={s}; only zeros is supported",
                        node.name
                    ));
                }
            }
        }
        "Unique" => {
            // Non-flattened mode = axis attribute present.
            if int_attr(node, "axis").is_some() {
                concerns.push(format!(
                    "Unique node `{}` uses axis mode; only flattened (no axis) is supported",
                    node.name
                ));
            }
        }
        "ScatterND" => {
            if let Some(s) = string_attr(node, "reduction") {
                if s != "none" && !s.is_empty() {
                    concerns.push(format!(
                        "ScatterND node `{}` uses reduction={s}; only none is supported",
                        node.name
                    ));
                }
            }
        }
        _ => {}
    }
    concerns
}

fn find_attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

fn string_attr(node: &NodeProto, name: &str) -> Option<String> {
    let a = find_attr(node, name)?;
    if a.attr_type != AttributeType::String {
        return None;
    }
    Some(String::from_utf8_lossy(&a.s).into_owned())
}

fn int_attr(node: &NodeProto, name: &str) -> Option<i64> {
    let a = find_attr(node, name)?;
    if a.attr_type != AttributeType::Int {
        return None;
    }
    Some(a.i)
}

fn float_attr(node: &NodeProto, name: &str) -> Option<f32> {
    let a = find_attr(node, name)?;
    if a.attr_type != AttributeType::Float {
        return None;
    }
    Some(a.f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_onnx_rt::onnx_types::{GraphProto, ModelProto, NodeProto};

    fn node(op: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: format!("{op}_0"),
            ..Default::default()
        }
    }

    fn model_with_ops(ops: &[&str]) -> ModelProto {
        let nodes = ops.iter().map(|o| node(o)).collect();
        ModelProto {
            graph: Some(GraphProto {
                node: nodes,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn empty_model_is_loadable_today() {
        let inv = Inventory::from_roadmap_str("");
        let m = model_with_ops(&[]);
        let report = walk_model(&m, "empty.onnx", &inv).unwrap();
        assert_eq!(report.total_nodes, 0);
        assert_eq!(report.distinct_ops, 0);
        assert_eq!(report.verdict, Verdict::LoadableToday);
    }

    #[test]
    fn single_supported_op_is_loadable_today() {
        let inv = Inventory::from_roadmap_str("");
        let m = model_with_ops(&["MatMul"]);
        let report = walk_model(&m, "m.onnx", &inv).unwrap();
        assert_eq!(report.total_nodes, 1);
        assert_eq!(report.distinct_ops, 1);
        assert_eq!(report.per_op[0].status, OperatorStatus::Implemented);
        assert_eq!(report.verdict, Verdict::LoadableToday);
    }

    #[test]
    fn multi_op_model_tallies_correctly() {
        let inv = Inventory::from_roadmap_str("");
        let m = model_with_ops(&["MatMul", "Add", "MatMul", "Relu", "MatMul"]);
        let report = walk_model(&m, "m.onnx", &inv).unwrap();
        assert_eq!(report.total_nodes, 5);
        assert_eq!(report.distinct_ops, 3);
        // Should be sorted by count desc.
        assert_eq!(report.per_op[0].op_type, "MatMul");
        assert_eq!(report.per_op[0].count, 3);
    }

    #[test]
    fn unknown_op_is_unrecognized_and_marks_verdict() {
        let inv = Inventory::from_roadmap_str("");
        let m = model_with_ops(&["MatMul", "FrobnicateTheGizmo"]);
        let report = walk_model(&m, "m.onnx", &inv).unwrap();
        assert_eq!(report.counts.unrecognized, 1);
        assert_eq!(report.verdict, Verdict::HasUnrecognizedOps);
    }

    #[test]
    fn planned_p2_op_marks_phase2_verdict() {
        // After Phase 1 merged, no ops are still Planned-P1 in develop.
        // The same logic applies to any future phase: a model that uses
        // a Planned-P2 op should produce a LoadableAfterPhase2 verdict.
        let src = "| Bernoulli | Planned-P2 | stochastic masking |";
        let inv = Inventory::from_roadmap_str(src);
        let m = model_with_ops(&["MatMul", "Bernoulli"]);
        let report = walk_model(&m, "llm-ish.onnx", &inv).unwrap();
        assert_eq!(report.counts.planned_p2, 1);
        assert_eq!(report.counts.implemented, 1);
        assert_eq!(report.verdict, Verdict::LoadableAfterPhase2);
    }

    #[test]
    fn resize_cubic_mode_is_flagged() {
        let mut n = node("Resize");
        n.attribute.push(AttributeProto {
            name: "mode".to_string(),
            attr_type: AttributeType::String,
            s: b"cubic".to_vec(),
            ..Default::default()
        });
        let concerns = attribute_concerns_for(&n);
        assert_eq!(concerns.len(), 1);
        assert!(concerns[0].contains("cubic"));
    }

    #[test]
    fn resize_linear_mode_is_not_flagged() {
        let mut n = node("Resize");
        n.attribute.push(AttributeProto {
            name: "mode".to_string(),
            attr_type: AttributeType::String,
            s: b"linear".to_vec(),
            ..Default::default()
        });
        assert!(attribute_concerns_for(&n).is_empty());
    }

    #[test]
    fn empty_model_proto_without_graph_errors() {
        let inv = Inventory::from_roadmap_str("");
        let m = ModelProto::default();
        let err = walk_model(&m, "m.onnx", &inv).unwrap_err();
        assert!(matches!(err, WalkError::EmptyModel));
    }

    #[test]
    fn roialign_max_mode_is_flagged() {
        let mut n = node("RoiAlign");
        n.attribute.push(AttributeProto {
            name: "mode".to_string(),
            attr_type: AttributeType::String,
            s: b"max".to_vec(),
            ..Default::default()
        });
        let c = attribute_concerns_for(&n);
        assert_eq!(c.len(), 1);
    }
}
