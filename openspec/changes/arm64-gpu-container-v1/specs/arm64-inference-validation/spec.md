## ADDED Requirements

### Requirement: ARM64 container image builds and runs ONNX inference
The SmallAIOS container image SHALL build for `aarch64-unknown-linux-musl` and execute CPU-based ONNX inference on ARM64 hardware.

#### Scenario: Cross-compile ARM64 container image
- **WHEN** `just build-container-arm` is run on an x86-64 or ARM64 host
- **THEN** the resulting binary SHALL target `aarch64-unknown-linux-musl`
- **AND** the Docker image SHALL be a valid OCI image for `linux/arm64`

#### Scenario: Run CPU inference on ARM64 hardware
- **WHEN** the ARM64 container image is started on a DGX Spark (Grace CPU)
- **THEN** the container SHALL load ONNX models and execute inference using the CPU provider
- **AND** all 29+ implemented operators SHALL produce numerically correct results

#### Scenario: Standard model validation suite
- **WHEN** the ARM64 container runs the validation model suite (ResNet-50, MobileNetV2, SqueezeNet, simple MLP)
- **THEN** each model SHALL load, execute, and produce outputs matching x86-64 reference outputs within 1e-5 relative tolerance
- **AND** any model that fails SHALL be logged with the specific operator that caused the failure

### Requirement: ARM64 operator gap analysis
The validation process SHALL produce a structured report of operator coverage on ARM64.

#### Scenario: Operator gap report
- **WHEN** a model fails to load or execute on ARM64
- **THEN** the system SHALL report the unsupported operator name, opset version, and the model that requires it
- **AND** the report SHALL distinguish between missing operators and operators that produce incorrect results

### Requirement: ARM64 CI integration
The CI pipeline SHALL include an ARM64 container build validation job.

#### Scenario: QEMU-emulated ARM64 build in CI
- **WHEN** a push or PR triggers the CI pipeline
- **THEN** an ARM64 container build job SHALL cross-compile the container image via QEMU emulation
- **AND** the job SHALL verify the image is a valid `linux/arm64` OCI image

#### Scenario: ARM64 basic inference smoke test in CI
- **WHEN** the ARM64 CI job completes the build
- **THEN** it SHALL run a minimal ONNX model (simple MLP) under QEMU emulation
- **AND** the smoke test SHALL verify the model loads and produces a non-error output
