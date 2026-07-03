// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Host-CPU [`ExecutionBackend`] implementation.
//!
//! Wraps the existing per-op dispatcher (`dispatch_node_with_domain`)
//! behind the [`ExecutionBackend`] trait surface. The behavior of the
//! wrapped dispatcher is unchanged from the pre-refactor implementation
//! — outputs are byte-identical for any model the runtime previously
//! supported. See `openspec/changes/fpga-accelerator-hal-v1/design.md`
//! Decision 2 ("CPU is a backend, not a special case") for the
//! rationale.

use alloc::format;

use crate::executor;
use crate::operators::OpKind;

use super::{ExecError, ExecutionBackend, OpDescriptor, TensorEnv};

/// Host-CPU execution backend.
///
/// Routes every supported ONNX operator through the runtime's
/// pre-existing CPU dispatch path (NEON/SVE on aarch64, AVX/AVX-512 on
/// x86, scalar fallback elsewhere). When the `cuda` feature is
/// enabled and a `CudaRuntime` is attached to the inference call, the
/// underlying dispatcher will still hand certain ops (MatMul, Gemm,
/// Conv, MatMulInteger) to the GPU as a per-op fast-path; that
/// behavior is preserved verbatim from the pre-refactor implementation
/// to keep this change behavior-identical. A future change will lift
/// the GPU fast-path into its own [`ExecutionBackend`] implementation.
pub struct CpuBackend {
    _private: (),
}

impl CpuBackend {
    /// Constructs a new CPU backend.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn probe(&self) -> Result<(), ExecError> {
        // The host CPU is always present. There is nothing to detect.
        Ok(())
    }

    fn can_run(&self, op: &OpDescriptor) -> bool {
        // The CPU backend supports every operator the runtime knows
        // about, across both the standard ONNX domain and the
        // `com.microsoft` contrib op set. Resolve via the same
        // `(domain, op_type)` lookup the dispatcher uses; if the
        // resolver returns `None` the op is unknown, which by
        // construction means CPU does not support it.
        OpKind::lookup_by_domain_and_name(op.domain, op.op_type).is_some()
    }

    fn estimated_ns(&self, _op: &OpDescriptor) -> u64 {
        // The v1 dispatch policy is strict priority order; cost-based
        // selection is not yet wired in. Backends with no real
        // estimate return the documented sentinel.
        u64::MAX
    }

    fn dispatch(&self, op: &OpDescriptor, env: &mut TensorEnv) -> Result<(), ExecError> {
        // Forward to the existing dispatcher. The CUDA fast-path
        // inside `dispatch_node_with_domain` remains in place for now
        // — Phase 1 of fpga-accelerator-hal-v1 only refactors the CPU
        // path. A future change will refactor the GPU fast-path into
        // its own backend.
        let outputs = executor::dispatch_node_with_domain(
            op.op_type,
            op.domain,
            env.inputs(),
            op.attributes,
            op.output_count,
            #[cfg(feature = "gpu")]
            None,
            #[cfg(feature = "cuda")]
            None,
        )
        .map_err(|e| {
            // Wrap the operator-level error in `Internal` — none of
            // these are recoverable at the trait layer. Callers see
            // the same diagnostic text the pre-refactor code path
            // produced.
            ExecError::Internal(format!(
                "{}: {}",
                if op.node_name.is_empty() {
                    op.op_type
                } else {
                    op.node_name
                },
                e
            ))
        })?;
        env.set_outputs(outputs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec::Vec;

    fn op<'a>(op_type: &'a str) -> OpDescriptor<'a> {
        OpDescriptor {
            op_type,
            domain: "",
            node_name: "",
            attributes: &[],
            output_count: 1,
            inputs: &[],
        }
    }

    #[test]
    fn cpu_backend_name_is_stable() {
        let cpu = CpuBackend::new();
        assert_eq!(cpu.name(), "cpu");
    }

    #[test]
    fn cpu_backend_probe_succeeds() {
        let cpu = CpuBackend::new();
        assert!(cpu.probe().is_ok());
    }

    #[test]
    fn cpu_backend_claims_known_ops() {
        let cpu = CpuBackend::new();
        assert!(cpu.can_run(&op("Relu")));
        assert!(cpu.can_run(&op("MatMul")));
        assert!(cpu.can_run(&op("Conv")));
    }

    #[test]
    fn cpu_backend_rejects_unknown_ops() {
        let cpu = CpuBackend::new();
        assert!(!cpu.can_run(&op("ThisOpDoesNotExist")));
    }

    #[test]
    fn cpu_backend_estimated_ns_returns_sentinel() {
        let cpu = CpuBackend::new();
        assert_eq!(cpu.estimated_ns(&op("Relu")), u64::MAX);
    }

    #[test]
    fn cpu_backend_dispatches_relu_with_byte_identical_output() {
        use crate::byte_io::write_f32;
        use crate::tensor::{DataType, Tensor, TensorShape};
        // Reproduces the existing `test_session_run_after_init_executes_relu`
        // input/output to prove byte-identical CPU output post-refactor.
        let mut data = alloc::vec![0u8; 3 * 4];
        for (i, &val) in [-1.0f32, 0.0, 2.0].iter().enumerate() {
            write_f32(&mut data, i, val);
        }
        let mut tensor = Tensor::new(
            DataType::Float,
            TensorShape::new(alloc::vec![1, 3]),
            String::from("x"),
        );
        tensor.raw_data = data;

        let inputs: Vec<Option<&Tensor>> = alloc::vec![Some(&tensor)];
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut env = TensorEnv::new(&inputs, &mut outputs);

        let cpu = CpuBackend::new();
        let res = cpu.dispatch(&op("Relu"), &mut env);
        assert!(res.is_ok(), "got {:?}", res);
        assert_eq!(outputs.len(), 1);

        let out = &outputs[0];
        let out_floats: Vec<f32> = (0..3)
            .map(|i| {
                f32::from_le_bytes([
                    out.raw_data[i * 4],
                    out.raw_data[i * 4 + 1],
                    out.raw_data[i * 4 + 2],
                    out.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(out_floats, alloc::vec![0.0_f32, 0.0, 2.0]);
    }

    #[test]
    fn cpu_backend_dispatch_unknown_op_returns_internal_error() {
        let cpu = CpuBackend::new();
        let inputs: Vec<Option<&crate::tensor::Tensor>> = Vec::new();
        let mut outputs: Vec<crate::tensor::Tensor> = Vec::new();
        let mut env = TensorEnv::new(&inputs, &mut outputs);
        let res = cpu.dispatch(&op("UnknownOp"), &mut env);
        match res {
            Err(ExecError::Internal(msg)) => {
                assert!(msg.contains("UnknownOp"), "got {}", msg);
            }
            other => panic!("expected Internal error, got {:?}", other),
        }
    }
}
