// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS ONNX coverage probe.
//!
//! Walks an ONNX [`ModelProto`](smallaios_onnx_rt::onnx_types::ModelProto),
//! tallies each op kind, and cross-references it against two sources of
//! truth:
//!
//! 1. The live [`OperatorRegistry`](smallaios_onnx_rt::operators::OperatorRegistry)
//!    in `smallaios-onnx-rt` (what the runtime *actually* supports today).
//! 2. The roadmap document `docs/onnx-coverage-roadmap.md` (what we plan
//!    to support, and when).
//!
//! The result is a [`CoverageReport`] which the CLI renders as Markdown
//! or JSON. This crate is used to empirically validate that upcoming
//! OpenSpec phases cover the target model families before spending the
//! implementation effort.
//!
//! # Example
//!
//! ```no_run
//! use smallaios_coverage_probe::{Inventory, walk_bytes};
//!
//! let bytes = std::fs::read("model.onnx").unwrap();
//! let inventory = Inventory::from_roadmap_file("docs/onnx-coverage-roadmap.md").unwrap();
//! let report = walk_bytes(&bytes, "model.onnx", &inventory).unwrap();
//! println!("{}", report.to_markdown(false));
//! ```

pub mod inventory;
pub mod report;
pub mod walker;

pub use inventory::{DocDrift, Inventory, OperatorStatus, Phase};
pub use report::{CoverageReport, NodeSummary, StatusCounts, Verdict};
pub use walker::{walk_bytes, walk_model, WalkError};
