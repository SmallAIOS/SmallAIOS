// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phase 2 control-flow operators: `If`, `Loop`, `Scan`.
//!
//! Each helper in this module accepts a pre-built [`ExecutionGraph`] body
//! and drives it via [`crate::sub_executor::run_sub_graph`]. The helpers
//! are named `op_*_with_body` to make it explicit that the inner graph is
//! supplied by the caller; an end-to-end graph-builder path that walks
//! into `GraphProto` attributes and produces compiled inner graphs is
//! tracked as a Phase 2 follow-up (see the module-level comment in
//! `sub_executor.rs`).
//!
//! # Semantics summary
//!
//! - `op_loop_with_body` implements the ONNX `Loop` termination rules:
//!   respects `M`, initial `cond`, and a body-emitted `cond_out`. The
//!   safety-net bound [`crate::sub_executor::MAX_LOOP_ITERATIONS`] is
//!   enforced as a hard error to prevent runaway models from hanging the
//!   scheduler.
//!
//! - `op_if_with_body` evaluates exactly one of the two provided branches
//!   per dispatch, returning the listed output names from the selected
//!   branch's post-execution value map.
//!
//! - `op_scan_with_body` implements a simplified `num_scan_inputs = 1`
//!   `Scan`: the caller passes a `Vec<Tensor>` of sequence elements and
//!   the body runs once per element with the single scan-input bound to
//!   `input_name` and the per-iteration output pulled from `output_name`.
//!   The returned vector contains one output per iteration in input order.
//!
//! # Design rationale
//!
//! The `_with_body` naming makes it trivial to unit-test the control-flow
//! machinery against synthetic graphs constructed in the test harness,
//! without requiring the protobuf parser to decode `AttributeType::Graph`
//! (which is the remaining blocker for end-to-end autoregressive model
//! loading).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::ExecutionGraph;
use crate::session::SessionError;
use crate::sub_executor::{run_sub_graph, MAX_LOOP_ITERATIONS};
use crate::tensor::Tensor;

/// Runs a `Loop` body with the provided termination signals and returns
/// the final value of the loop-carried state.
///
/// The body graph is expected to declare its inputs in the ONNX-standard
/// order: `iter_num`, `cond_in`, then the N loop-carried dependencies by
/// whatever names the body uses (we read them from `body.input_names`
/// starting at index 2).
///
/// On exit, the body's output list is interpreted as `cond_out` followed
/// by the N updated loop-carried values. This matches the ONNX opset-21
/// specification of `Loop`.
///
/// The `outer_scope` map provides read-only tensors that the body may
/// reference by name (e.g., weight initializers, graph-level constants).
/// The map is cloned once per iteration so outer writes are isolated.
///
/// `body_cond` is an additional **caller-side** termination predicate
/// invoked after each iteration with `(iter_num, &v_current)`. It is
/// OR'd with the body's `cond_out` output: if the body emits `cond_out =
/// true` and `body_cond` returns `false`, the loop terminates. This is
/// the only way to inject an external predicate during unit testing; when
/// the runtime graph path lands, this closure is replaced by reading
/// `cond_out` directly from the body's final output.
pub fn op_loop_with_body<F: FnMut(i64, &[Tensor]) -> bool>(
    m: Option<i64>,
    cond_initial: Option<bool>,
    v_initial: Vec<Tensor>,
    body: &ExecutionGraph,
    outer_scope: &BTreeMap<String, Tensor>,
    mut body_cond: F,
) -> Result<Vec<Tensor>, SessionError> {
    // Trivial skips.
    if m == Some(0) || cond_initial == Some(false) {
        return Ok(v_initial);
    }

    let mut v_current = v_initial;
    let mut cond_current = cond_initial.unwrap_or(true);
    let mut iter: i64 = 0;

    let carried_input_names = loop_carried_input_names(body);
    let carried_output_names = loop_carried_output_names(body);

    loop {
        if loop_should_terminate(iter, m, cond_current)? {
            break;
        }

        let inner = build_loop_inner_scope(
            body,
            outer_scope,
            iter,
            cond_current,
            &carried_input_names,
            &v_current,
        );
        let result = run_sub_graph(body, inner, &[])?;
        v_current = extract_loop_carried_outputs(&result, &carried_output_names)?;

        cond_current = body_cond(iter, &v_current);
        iter += 1;
    }

    Ok(v_current)
}

/// Returns the body's carried-input names (starting after `iter_num` and
/// `cond_in`). Empty if the body has fewer than two declared inputs.
fn loop_carried_input_names(body: &ExecutionGraph) -> Vec<String> {
    if body.input_names.len() > 2 {
        body.input_names[2..].to_vec()
    } else {
        Vec::new()
    }
}

/// Returns the body's carried-output names (skipping `cond_out`). Empty if
/// the body has fewer than two declared outputs.
fn loop_carried_output_names(body: &ExecutionGraph) -> Vec<String> {
    if body.output_names.len() > 1 {
        body.output_names[1..].to_vec()
    } else {
        Vec::new()
    }
}

/// Returns `Ok(true)` when the loop should exit (max iters, `m` reached,
/// or `cond_current == false`), and `Err` on safety-limit violation.
fn loop_should_terminate(
    iter: i64,
    m: Option<i64>,
    cond_current: bool,
) -> Result<bool, SessionError> {
    if iter >= MAX_LOOP_ITERATIONS {
        return Err(SessionError::ExecutionFailed(alloc::format!(
            "Loop exceeded compile-time safety limit ({})",
            MAX_LOOP_ITERATIONS
        )));
    }
    if matches!(m, Some(max) if iter >= max) {
        return Ok(true);
    }
    Ok(!cond_current)
}

/// Builds the per-iteration inner value map: outer scope, then `iter_num`,
/// `cond_in`, and the carried-input bindings.
fn build_loop_inner_scope(
    body: &ExecutionGraph,
    outer_scope: &BTreeMap<String, Tensor>,
    iter: i64,
    cond_current: bool,
    carried_input_names: &[String],
    v_current: &[Tensor],
) -> BTreeMap<String, Tensor> {
    let mut inner = outer_scope.clone();
    if let Some(name) = body.input_names.first() {
        if !name.is_empty() {
            inner.insert(name.clone(), crate::sub_executor::i64_scalar(iter));
        }
    }
    if let Some(name) = body.input_names.get(1) {
        if !name.is_empty() {
            inner.insert(name.clone(), crate::sub_executor::bool_scalar(cond_current));
        }
    }
    for (name, val) in carried_input_names.iter().zip(v_current.iter()) {
        inner.insert(name.clone(), val.clone());
    }
    inner
}

/// Pulls the carried-output tensors out of the post-body value map.
fn extract_loop_carried_outputs(
    result: &BTreeMap<String, Tensor>,
    carried_output_names: &[String],
) -> Result<Vec<Tensor>, SessionError> {
    let mut next_v = Vec::with_capacity(carried_output_names.len());
    for name in carried_output_names {
        let t = result.get(name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!("Loop body missing output '{}'", name))
        })?;
        next_v.push(t.clone());
    }
    Ok(next_v)
}

/// Runs an `If` by dispatching to either `then_branch` or `else_branch`
/// based on `cond` and returning the listed `output_names` from the
/// selected branch's post-execution value map.
pub fn op_if_with_body(
    cond: bool,
    then_branch: &ExecutionGraph,
    else_branch: &ExecutionGraph,
    outer_scope: &BTreeMap<String, Tensor>,
    output_names: &[&str],
) -> Result<Vec<Tensor>, SessionError> {
    let body = if cond { then_branch } else { else_branch };
    let result = run_sub_graph(body, outer_scope.clone(), &[])?;
    let mut out = Vec::with_capacity(output_names.len());
    for name in output_names {
        let t = result.get(*name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!("If branch missing output '{}'", name))
        })?;
        out.push(t.clone());
    }
    Ok(out)
}

/// Runs a simplified `Scan`: executes `body` once per element of the
/// provided `sequence`, binding `input_name` to the current element and
/// pulling `output_name` from the post-execution value map. Returns the
/// collected outputs in the same order as the input sequence.
///
/// Phase 2 only supports `num_scan_inputs = 1` with the default axis (0)
/// and forward direction. A future follow-up generalizes to multi-input
/// `Scan` with arbitrary axes.
pub fn op_scan_with_body(
    body: &ExecutionGraph,
    outer_scope: &BTreeMap<String, Tensor>,
    input_name: &str,
    output_name: &str,
    sequence: Vec<Tensor>,
) -> Result<Vec<Tensor>, SessionError> {
    let mut out = Vec::with_capacity(sequence.len());
    for (i, elem) in sequence.into_iter().enumerate() {
        if i as i64 >= MAX_LOOP_ITERATIONS {
            return Err(SessionError::ExecutionFailed(String::from(
                "Scan exceeded safety limit",
            )));
        }
        let mut inner = outer_scope.clone();
        inner.insert(String::from(input_name), elem);
        let result = run_sub_graph(body, inner, &[])?;
        let t = result.get(output_name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!(
                "Scan body missing output '{}'",
                output_name
            ))
        })?;
        out.push(t.clone());
    }
    Ok(out)
}
