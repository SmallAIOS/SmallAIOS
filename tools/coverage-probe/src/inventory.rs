// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Parses `docs/onnx-coverage-roadmap.md` into an operator status map and
//! cross-references entries against the live `OperatorRegistry` from
//! `smallaios-onnx-rt`.
//!
//! The roadmap contains Markdown tables with rows of the form:
//!
//! ```text
//! | Sin | Planned-P1 | Sinusoidal positional embeddings |
//! ```
//!
//! We recognize the following status tokens (case-insensitive, hyphens
//! or underscores accepted):
//!
//! - `Planned-P1`, `Planned-P2`, `Planned-P3`, `Planned-P4`
//! - `Deferred-subsystem` (optionally `Deferred-subsystem:<name>`)
//! - `Skipped-training`, `Skipped-deprecated`, `Skipped-vendor`
//!
//! Any op not named in the roadmap but recognized by the registry is
//! treated as [`OperatorStatus::Implemented`]. Any op neither in the
//! registry nor the roadmap is [`OperatorStatus::Unrecognized`].

use smallaios_onnx_rt::operators::OperatorRegistry;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Roadmap phase identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Phase 1 (transformers baseline: BERT / ViT).
    P1,
    /// Phase 2 (generative LLMs: GPT-2 / Llama).
    P2,
    /// Phase 3 (audio / advanced vision).
    P3,
    /// Phase 4 (long-tail completion).
    P4,
}

impl Phase {
    /// Short display label, e.g. "P1".
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::P1 => "P1",
            Phase::P2 => "P2",
            Phase::P3 => "P3",
            Phase::P4 => "P4",
        }
    }
}

/// Status of an ONNX operator relative to the SmallAIOS roadmap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorStatus {
    /// Present in the live `OperatorRegistry` — loadable today.
    Implemented,
    /// Planned in one of the roadmap phases.
    Planned(Phase),
    /// Deferred pending a subsystem (sequence, optional, control-flow,
    /// string, ml, …).
    DeferredSubsystem(String),
    /// Explicitly skipped: training-only op.
    SkippedTraining,
    /// Explicitly skipped: deprecated and rewritten by converters.
    SkippedDeprecated,
    /// Explicitly skipped: vendor-specific (`com.microsoft`, …).
    SkippedVendor,
    /// Not mentioned in the roadmap and not implemented.
    Unrecognized,
}

impl OperatorStatus {
    /// Human-readable label used in Markdown and CLI output.
    pub fn label(&self) -> String {
        match self {
            OperatorStatus::Implemented => "Implemented".to_string(),
            OperatorStatus::Planned(p) => format!("Planned-{}", p.as_str()),
            OperatorStatus::DeferredSubsystem(s) if s.is_empty() => {
                "Deferred-subsystem".to_string()
            }
            OperatorStatus::DeferredSubsystem(s) => format!("Deferred-subsystem:{s}"),
            OperatorStatus::SkippedTraining => "Skipped-training".to_string(),
            OperatorStatus::SkippedDeprecated => "Skipped-deprecated".to_string(),
            OperatorStatus::SkippedVendor => "Skipped-vendor".to_string(),
            OperatorStatus::Unrecognized => "Unrecognized".to_string(),
        }
    }
}

/// A case where the roadmap claims a status incompatible with the
/// current `OperatorRegistry` — e.g. the doc says `Implemented` but the
/// registry does not recognize the op, or (more commonly) the doc marks
/// an op `Planned-P1` when the registry already supports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocDrift {
    /// Operator name as written in the roadmap.
    pub op: String,
    /// Status listed in the roadmap.
    pub doc_status: OperatorStatus,
    /// Short explanation.
    pub note: String,
}

/// Parsed roadmap plus the set of registry-supported ops.
#[derive(Debug, Clone)]
pub struct Inventory {
    /// Map from op type name to its roadmap-declared status, with
    /// registry overrides applied (registry always wins for
    /// `Implemented`).
    statuses: BTreeMap<String, OperatorStatus>,
    /// Ops detected as both "Implemented in registry" and "Planned in
    /// roadmap" — surfaces doc drift without blocking anything.
    drifts: Vec<DocDrift>,
    /// Names of ops the registry currently recognizes.
    registered: Vec<String>,
}

impl Inventory {
    /// Build an inventory from the roadmap file at `path`.
    pub fn from_roadmap_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let src = fs::read_to_string(path)?;
        Ok(Self::from_roadmap_str(&src))
    }

    /// Build an inventory from raw roadmap Markdown plus the live
    /// operator registry. The registry is consulted via
    /// `OperatorRegistry::new()`.
    pub fn from_roadmap_str(src: &str) -> Self {
        let registered = all_registered_op_names();
        let mut statuses: BTreeMap<String, OperatorStatus> = BTreeMap::new();
        let mut drifts: Vec<DocDrift> = Vec::new();

        // Seed from the registry.
        for name in &registered {
            statuses.insert(name.clone(), OperatorStatus::Implemented);
        }

        // Parse roadmap rows.
        for row in parse_status_rows(src) {
            let (op, status) = row;
            // Registry wins for `Implemented`, but we record the drift.
            if registered.iter().any(|n| n == &op) {
                if !matches!(status, OperatorStatus::Implemented) {
                    drifts.push(DocDrift {
                        op: op.clone(),
                        doc_status: status.clone(),
                        note: format!(
                            "registry already implements `{op}` but roadmap lists `{}`",
                            status.label()
                        ),
                    });
                }
                continue;
            }
            // Otherwise, record the roadmap status. Later duplicate
            // rows overwrite earlier ones (unlikely in practice).
            statuses.insert(op, status);
        }

        Inventory {
            statuses,
            drifts,
            registered,
        }
    }

    /// Look up the status of an op by type name. Returns
    /// [`OperatorStatus::Unrecognized`] when the op is neither
    /// registered nor in the roadmap.
    pub fn status_of(&self, op_type: &str) -> OperatorStatus {
        self.statuses
            .get(op_type)
            .cloned()
            .unwrap_or(OperatorStatus::Unrecognized)
    }

    /// Names of the ops the live registry recognizes.
    pub fn registered_ops(&self) -> &[String] {
        &self.registered
    }

    /// Doc-drift entries detected during parsing.
    pub fn drifts(&self) -> &[DocDrift] {
        &self.drifts
    }

    /// All roadmap-listed ops (including implemented).
    pub fn len(&self) -> usize {
        self.statuses.len()
    }

    /// Whether the inventory is empty (only meaningful for tests).
    pub fn is_empty(&self) -> bool {
        self.statuses.is_empty()
    }
}

/// Collects the canonical names of every op in `OperatorRegistry::new()`.
///
/// Instead of re-deriving the list from `OpKind`, we probe the registry
/// for every ONNX op we know the name of. This keeps the probe working
/// even as the registry grows — it just needs the probing list to stay
/// reasonably comprehensive (expanded when new ops are added).
fn all_registered_op_names() -> Vec<String> {
    let reg = OperatorRegistry::new();
    // Baseline candidate set. When new ops are added to `OpKind` in
    // `smallaios-onnx-rt`, they should be added here as well. This list
    // is intentionally a superset of the current `OpKind` enum.
    const CANDIDATES: &[&str] = &[
        // Arithmetic / elementwise
        "Add",
        "Sub",
        "Mul",
        "Div",
        "MatMul",
        "Mod",
        "Pow",
        "Neg",
        "Abs",
        "Reciprocal",
        "Sqrt",
        "Sign",
        "Sum",
        "Mean",
        "Max",
        "Min",
        "And",
        "Or",
        "Xor",
        "Not",
        "Equal",
        "Greater",
        "Less",
        "GreaterOrEqual",
        "LessOrEqual",
        // Trig / exponential
        "Exp",
        "Log",
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asin",
        "Acos",
        "Atan",
        "Asinh",
        "Acosh",
        "Atanh",
        "Erf",
        // Activations
        "Relu",
        "Sigmoid",
        "Softmax",
        "LogSoftmax",
        "PRelu",
        "LeakyRelu",
        "Elu",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "Selu",
        "Celu",
        "Softplus",
        "Softsign",
        "ThresholdedRelu",
        // Convolution / pooling
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "BatchNormalization",
        "InstanceNormalization",
        "GroupNormalization",
        // Shape manipulation
        "Reshape",
        "Transpose",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
        "Expand",
        "Tile",
        "Identity",
        "Shape",
        "Size",
        // Data movement
        "Concat",
        "Gather",
        "GatherElements",
        "GatherND",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Slice",
        "Pad",
        "Where",
        "Split",
        // Linear algebra / normalization
        "Gemm",
        "LayerNormalization",
        "LpNormalization",
        "MeanVarianceNormalization",
        // Type / reduction
        "Cast",
        "Clip",
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ArgMax",
        "ArgMin",
        "TopK",
        // Random / constants
        "Constant",
        "ConstantOfShape",
        "Range",
        // Quantization
        "DequantizeLinear",
        "QuantizeLinear",
    ];
    CANDIDATES
        .iter()
        .filter(|name| reg.is_supported(name))
        .map(|n| (*n).to_string())
        .collect()
}

/// Extract `(op_name, status)` pairs from the roadmap Markdown source.
///
/// Only rows whose second column matches a known status token are
/// retained. Table headers like `| Op | Status | Rationale |` are
/// filtered out because `Status` is not a status token.
fn parse_status_rows(src: &str) -> Vec<(String, OperatorStatus)> {
    let mut rows = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim())
            .collect();
        if cells.len() < 2 {
            continue;
        }
        let op = cells[0];
        let status_cell = cells[1];
        // Skip header / separator rows.
        if op.is_empty() || op.chars().next().is_some_and(|c| !c.is_ascii_alphabetic()) {
            continue;
        }
        if !is_likely_op_name(op) {
            continue;
        }
        if let Some(status) = parse_status_token(status_cell) {
            rows.push((op.to_string(), status));
        }
    }
    rows
}

/// Heuristic: ONNX op names are CamelCase identifiers of up to a few
/// words (no spaces, no punctuation). This filters out prose like
/// "Phase" or "Op count".
fn is_likely_op_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if !s.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return false;
    }
    if s.chars().any(|c| c.is_whitespace()) {
        return false;
    }
    // Reject known non-op header labels.
    !matches!(s, "Op" | "Bucket" | "Phase" | "Tier" | "Group")
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parse a status cell into [`OperatorStatus`]. Returns `None` for
/// unrecognized tokens so that prose rows (e.g. the header) are
/// silently dropped.
fn parse_status_token(cell: &str) -> Option<OperatorStatus> {
    let normalized = cell.replace('_', "-").to_ascii_lowercase();
    let normalized = normalized.trim();
    match normalized {
        "planned-p1" => Some(OperatorStatus::Planned(Phase::P1)),
        "planned-p2" => Some(OperatorStatus::Planned(Phase::P2)),
        "planned-p3" => Some(OperatorStatus::Planned(Phase::P3)),
        "planned-p4" => Some(OperatorStatus::Planned(Phase::P4)),
        "skipped-training" => Some(OperatorStatus::SkippedTraining),
        "skipped-deprecated" => Some(OperatorStatus::SkippedDeprecated),
        "skipped-vendor" => Some(OperatorStatus::SkippedVendor),
        "implemented" => Some(OperatorStatus::Implemented),
        s if s.starts_with("deferred-subsystem") => {
            let detail = s
                .strip_prefix("deferred-subsystem")
                .and_then(|rest| rest.strip_prefix(':'))
                .unwrap_or("")
                .to_string();
            Some(OperatorStatus::DeferredSubsystem(detail))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_planned_row() {
        let src = "| Sin | Planned-P1 | Sinusoidal positional embeddings |";
        let rows = parse_status_rows(src);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "Sin");
        assert_eq!(rows[0].1, OperatorStatus::Planned(Phase::P1));
    }

    #[test]
    fn parses_multiple_phases() {
        let src = "\
| Sin | Planned-P1 | trig |
| Softplus | Planned-P2 | smooth relu |
| Sinh | Planned-P3 | hyperbolic |
| Tan | Planned-P4 | rare |
";
        let rows = parse_status_rows(src);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[3].1, OperatorStatus::Planned(Phase::P4));
    }

    #[test]
    fn parses_deferred_and_skipped() {
        let src = "\
| SplitToSequence | Deferred-subsystem | Sequence type required |
| Affine | Skipped-deprecated | Replaced by Mul+Add |
| QLinearConv | Skipped-vendor | com.microsoft |
";
        let rows = parse_status_rows(src);
        assert_eq!(rows.len(), 3);
        assert!(matches!(rows[0].1, OperatorStatus::DeferredSubsystem(_)));
        assert_eq!(rows[1].1, OperatorStatus::SkippedDeprecated);
        assert_eq!(rows[2].1, OperatorStatus::SkippedVendor);
    }

    #[test]
    fn header_rows_are_filtered() {
        let src = "\
| Op | Status | Rationale |
|----|--------|-----------|
| Sin | Planned-P1 | trig |
";
        let rows = parse_status_rows(src);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "Sin");
    }

    #[test]
    fn inventory_marks_matmul_as_implemented_via_registry() {
        let inv = Inventory::from_roadmap_str("");
        assert_eq!(inv.status_of("MatMul"), OperatorStatus::Implemented);
        assert_eq!(inv.status_of("Add"), OperatorStatus::Implemented);
    }

    #[test]
    fn inventory_returns_unrecognized_for_unknown_ops() {
        let inv = Inventory::from_roadmap_str("");
        assert_eq!(
            inv.status_of("FrobnicateTheGizmo"),
            OperatorStatus::Unrecognized
        );
    }

    #[test]
    fn inventory_records_doc_drift_when_registry_beats_roadmap() {
        // Roadmap claims Add is Planned-P1, but the registry already
        // implements it. This should be recorded as a drift and the
        // effective status should remain Implemented.
        let src = "| Add | Planned-P1 | bogus |";
        let inv = Inventory::from_roadmap_str(src);
        assert_eq!(inv.status_of("Add"), OperatorStatus::Implemented);
        assert_eq!(inv.drifts().len(), 1);
        assert_eq!(inv.drifts()[0].op, "Add");
    }

    #[test]
    fn inventory_records_planned_op_not_in_registry() {
        let src = "| Gelu | Planned-P1 | transformer activation |";
        let inv = Inventory::from_roadmap_str(src);
        // Gelu is not in the develop registry (29 ops).
        // The walker will see it as Planned(P1).
        assert_eq!(inv.status_of("Gelu"), OperatorStatus::Planned(Phase::P1));
    }

    #[test]
    fn parses_real_roadmap_file() {
        // Test that the actual roadmap file parses without panicking
        // and produces a non-trivial inventory.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/onnx-coverage-roadmap.md"
        );
        let inv = Inventory::from_roadmap_file(path).expect("roadmap file should exist");
        assert!(
            inv.len() > 30,
            "expected roadmap to contribute >30 ops, got {}",
            inv.len()
        );
        // Known Phase 1 planned ops.
        assert_eq!(inv.status_of("Sin"), OperatorStatus::Planned(Phase::P1));
        assert_eq!(inv.status_of("Cos"), OperatorStatus::Planned(Phase::P1));
        // Known deprecated skip.
        assert_eq!(inv.status_of("Affine"), OperatorStatus::SkippedDeprecated);
    }
}
