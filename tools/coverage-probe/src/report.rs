// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Report rendering: Markdown and JSON serialization of a coverage walk.

use crate::inventory::{DocDrift, OperatorStatus};
use std::fmt::Write;

/// Per-op entry in a [`CoverageReport`].
#[derive(Debug, Clone)]
pub struct NodeSummary {
    /// ONNX operator type string (e.g. "MatMul").
    pub op_type: String,
    /// Number of nodes with this op type in the model.
    pub count: usize,
    /// Effective status after registry + roadmap reconciliation.
    pub status: OperatorStatus,
    /// Human-readable per-node attribute warnings (detailed mode).
    pub attribute_concerns: Vec<String>,
}

/// Status-bucket counts, used for summary tables.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusCounts {
    /// Ops already implemented in the live `OperatorRegistry`.
    pub implemented: usize,
    /// Ops planned in Phase 1.
    pub planned_p1: usize,
    /// Ops planned in Phase 2.
    pub planned_p2: usize,
    /// Ops planned in Phase 3.
    pub planned_p3: usize,
    /// Ops planned in Phase 4.
    pub planned_p4: usize,
    /// Ops deferred pending a subsystem.
    pub deferred: usize,
    /// Ops explicitly skipped (training, deprecated, vendor).
    pub skipped: usize,
    /// Ops not recognized at all.
    pub unrecognized: usize,
}

impl StatusCounts {
    /// Sum across all buckets — equals the number of distinct ops.
    pub fn total(&self) -> usize {
        self.implemented
            + self.planned_p1
            + self.planned_p2
            + self.planned_p3
            + self.planned_p4
            + self.deferred
            + self.skipped
            + self.unrecognized
    }
}

/// Aggregate verdict for whether the runtime can execute the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// All ops already implemented — loadable on the current `develop`.
    LoadableToday,
    /// Loadable once Phase 1 PRs land.
    LoadableAfterPhase1,
    /// Loadable once Phase 2 PRs land.
    LoadableAfterPhase2,
    /// Loadable once Phase 3 PRs land.
    LoadableAfterPhase3,
    /// Loadable once Phase 4 PRs land.
    LoadableAfterPhase4,
    /// Model uses at least one op the roadmap does not plan.
    NotLoadable,
    /// Model uses at least one op we do not recognize at all.
    HasUnrecognizedOps,
}

impl Verdict {
    /// Short machine-readable string, used in JSON output and tests.
    pub fn as_tag(&self) -> &'static str {
        match self {
            Verdict::LoadableToday => "loadable_today",
            Verdict::LoadableAfterPhase1 => "loadable_after_phase1",
            Verdict::LoadableAfterPhase2 => "loadable_after_phase2",
            Verdict::LoadableAfterPhase3 => "loadable_after_phase3",
            Verdict::LoadableAfterPhase4 => "loadable_after_phase4",
            Verdict::NotLoadable => "not_loadable",
            Verdict::HasUnrecognizedOps => "has_unrecognized_ops",
        }
    }

    /// Human-readable sentence for Markdown output.
    pub fn explain(&self) -> &'static str {
        match self {
            Verdict::LoadableToday => "Model is loadable on the current `develop` branch.",
            Verdict::LoadableAfterPhase1 => "Model becomes loadable after Phase 1 PRs merge.",
            Verdict::LoadableAfterPhase2 => "Model becomes loadable after Phase 2 PRs merge.",
            Verdict::LoadableAfterPhase3 => "Model becomes loadable after Phase 3 PRs merge.",
            Verdict::LoadableAfterPhase4 => "Model becomes loadable after Phase 4 PRs merge.",
            Verdict::NotLoadable => {
                "Model uses ops the roadmap has deferred or skipped; not loadable."
            }
            Verdict::HasUnrecognizedOps => {
                "Model uses ops neither implemented nor listed on the roadmap."
            }
        }
    }
}

/// The full structural analysis of one model file.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Label for the model, typically the file path.
    pub model_label: String,
    /// Total node count in the graph.
    pub total_nodes: usize,
    /// Distinct op-type count.
    pub distinct_ops: usize,
    /// Per-op breakdown sorted by count desc.
    pub per_op: Vec<NodeSummary>,
    /// Aggregate status counts.
    pub counts: StatusCounts,
    /// Overall verdict.
    pub verdict: Verdict,
    /// Doc-drift entries from the inventory (not filtered by model).
    pub doc_drifts: Vec<DocDrift>,
}

impl CoverageReport {
    /// Render as Markdown.
    pub fn to_markdown(&self, detailed: bool) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# ONNX Coverage Report");
        let _ = writeln!(out);
        let _ = writeln!(out, "**Model:** `{}`", self.model_label);
        let _ = writeln!(out, "**Total nodes:** {}", self.total_nodes);
        let _ = writeln!(out, "**Distinct op kinds:** {}", self.distinct_ops);
        let _ = writeln!(out);
        let _ = writeln!(out, "## Summary");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Status | Count | Percentage |");
        let _ = writeln!(out, "|---|---|---|");
        let total = self.distinct_ops.max(1);
        let pct = |n: usize| -> String { format!("{}%", (n * 100) / total) };
        let c = &self.counts;
        let _ = writeln!(
            out,
            "| Implemented in develop | {} | {} |",
            c.implemented,
            pct(c.implemented)
        );
        let _ = writeln!(
            out,
            "| Planned Phase 1 | {} | {} |",
            c.planned_p1,
            pct(c.planned_p1)
        );
        let _ = writeln!(
            out,
            "| Planned Phase 2 | {} | {} |",
            c.planned_p2,
            pct(c.planned_p2)
        );
        let _ = writeln!(
            out,
            "| Planned Phase 3 | {} | {} |",
            c.planned_p3,
            pct(c.planned_p3)
        );
        let _ = writeln!(
            out,
            "| Planned Phase 4 | {} | {} |",
            c.planned_p4,
            pct(c.planned_p4)
        );
        let _ = writeln!(
            out,
            "| Deferred subsystem | {} | {} |",
            c.deferred,
            pct(c.deferred)
        );
        let _ = writeln!(out, "| Skipped | {} | {} |", c.skipped, pct(c.skipped));
        let _ = writeln!(
            out,
            "| Unrecognized | {} | {} |",
            c.unrecognized,
            pct(c.unrecognized)
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "**Verdict:** {}", self.verdict.explain());
        let _ = writeln!(out);

        let _ = writeln!(out, "## Per-op breakdown");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Op | Count | Status |");
        let _ = writeln!(out, "|---|---|---|");
        for op in &self.per_op {
            let _ = writeln!(
                out,
                "| {} | {} | {} |",
                op.op_type,
                op.count,
                op.status.label()
            );
        }
        let _ = writeln!(out);

        if detailed {
            let all_concerns: Vec<&String> = self
                .per_op
                .iter()
                .flat_map(|op| op.attribute_concerns.iter())
                .collect();
            if !all_concerns.is_empty() {
                let _ = writeln!(out, "## Attribute concerns");
                let _ = writeln!(out);
                for concern in all_concerns {
                    let _ = writeln!(out, "- {concern}");
                }
                let _ = writeln!(out);
            }
        }

        if !self.doc_drifts.is_empty() {
            let _ = writeln!(out, "## Doc drift warnings");
            let _ = writeln!(out);
            for d in &self.doc_drifts {
                let _ = writeln!(out, "- {}", d.note);
            }
            let _ = writeln!(out);
        }

        out
    }

    /// Render as JSON.
    #[cfg(feature = "json")]
    pub fn to_json(&self) -> String {
        use serde_json::json;
        let per_op: Vec<_> = self
            .per_op
            .iter()
            .map(|op| {
                json!({
                    "op": op.op_type,
                    "count": op.count,
                    "status": op.status.label(),
                    "attribute_concerns": op.attribute_concerns,
                })
            })
            .collect();
        let drifts: Vec<_> = self
            .doc_drifts
            .iter()
            .map(|d| {
                json!({
                    "op": d.op,
                    "doc_status": d.doc_status.label(),
                    "note": d.note,
                })
            })
            .collect();
        let v = json!({
            "model": self.model_label,
            "total_nodes": self.total_nodes,
            "distinct_ops": self.distinct_ops,
            "summary": {
                "implemented_in_develop": self.counts.implemented,
                "planned_p1": self.counts.planned_p1,
                "planned_p2": self.counts.planned_p2,
                "planned_p3": self.counts.planned_p3,
                "planned_p4": self.counts.planned_p4,
                "deferred": self.counts.deferred,
                "skipped": self.counts.skipped,
                "unrecognized": self.counts.unrecognized,
            },
            "per_op": per_op,
            "doc_drifts": drifts,
            "verdict": self.verdict.as_tag(),
        });
        serde_json::to_string_pretty(&v).expect("json serialization should not fail")
    }

    /// Render as a short plain-text summary for the default CLI mode.
    pub fn to_text_summary(&self, quiet: bool) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Coverage report: {}", self.model_label);
        let _ = writeln!(out);
        let _ = writeln!(out, "Total nodes:        {}", self.total_nodes);
        let _ = writeln!(out, "Distinct op kinds:  {}", self.distinct_ops);
        let _ = writeln!(out);
        let c = &self.counts;
        let _ = writeln!(out, "Implemented in develop:      {}", c.implemented);
        let _ = writeln!(out, "Planned Phase 1:             {}", c.planned_p1);
        let _ = writeln!(out, "Planned Phase 2:             {}", c.planned_p2);
        let _ = writeln!(out, "Planned Phase 3:             {}", c.planned_p3);
        let _ = writeln!(out, "Planned Phase 4:             {}", c.planned_p4);
        let _ = writeln!(out, "Deferred subsystem:          {}", c.deferred);
        let _ = writeln!(out, "Skipped (training/deprecated/vendor): {}", c.skipped);
        let _ = writeln!(out, "Unrecognized:                {}", c.unrecognized);
        let _ = writeln!(out);
        let _ = writeln!(out, "Verdict: {}", self.verdict.explain());
        let _ = writeln!(out);

        if !quiet {
            let _ = writeln!(out, "Per-op breakdown:");
            for op in &self.per_op {
                let _ = writeln!(
                    out,
                    "  {:<24} {:>6} nodes   {}",
                    op.op_type,
                    op.count,
                    op.status.label()
                );
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{Inventory, Phase};
    use crate::walker::walk_model;
    use smallaios_onnx_rt::onnx_types::{GraphProto, ModelProto, NodeProto};

    fn model_with_ops(ops: &[&str]) -> ModelProto {
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
    fn markdown_contains_verdict_and_breakdown() {
        // Bernoulli is Phase 2 generative, not yet in develop registry.
        let inv = Inventory::from_roadmap_str("| Bernoulli | Planned-P2 | x |");
        let m = model_with_ops(&["MatMul", "Bernoulli"]);
        let report = walk_model(&m, "m.onnx", &inv).unwrap();
        let md = report.to_markdown(false);
        assert!(md.contains("# ONNX Coverage Report"));
        assert!(md.contains("MatMul"));
        assert!(md.contains("Bernoulli"));
        assert!(md.contains("Phase 2"));
    }

    #[cfg(feature = "json")]
    #[test]
    fn json_round_trip_contains_verdict_tag() {
        let inv = Inventory::from_roadmap_str("| Bernoulli | Planned-P2 | x |");
        let m = model_with_ops(&["MatMul", "Bernoulli"]);
        let report = walk_model(&m, "m.onnx", &inv).unwrap();
        let json = report.to_json();
        assert!(json.contains("\"verdict\""));
        assert!(json.contains("loadable_after_phase2"));
    }

    #[test]
    fn status_counts_verdict_precedence() {
        // P3 beats P2; unrecognized beats everything.
        let inv =
            Inventory::from_roadmap_str("| Bernoulli | Planned-P2 | |\n| STFT | Planned-P3 | |");
        let m = model_with_ops(&["MatMul", "Bernoulli", "STFT"]);
        let r = walk_model(&m, "m.onnx", &inv).unwrap();
        assert_eq!(r.verdict, Verdict::LoadableAfterPhase3);
        assert_eq!(r.counts.planned_p2, 1);
        assert_eq!(r.counts.planned_p3, 1);

        let m2 = model_with_ops(&["MatMul", "FooBar"]);
        let r2 = walk_model(&m2, "m.onnx", &inv).unwrap();
        assert_eq!(r2.verdict, Verdict::HasUnrecognizedOps);
    }

    #[test]
    fn planned_phase_parsing_smoke() {
        assert_eq!(Phase::P1.as_str(), "P1");
        assert_eq!(Phase::P4.as_str(), "P4");
    }
}
