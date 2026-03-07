## MODIFIED Requirements

### Requirement: Boot Security
SmallAIOS SHALL provide software-level boot integrity verification when the `verified-boot` feature is enabled. The kernel SHALL verify its own integrity via an embedded SHA-3-256 hash signed with Ed25519, verify ONNX model signatures at load time using Ed25519/ML-DSA-65/hybrid, and maintain an immutable boot measurement log recording the hash of every loaded component. Hardware-level boot verification (UEFI Secure Boot, ARM TrustZone, RISC-V PMP) is documented as platform-dependent and remains outside SmallAIOS's direct control.

The secure boot chain for bare metal/VM deployments SHALL be:
```
[Platform firmware verification] → Signed kernel image (Ed25519) → Verified ONNX models (Ed25519/ML-DSA-65/hybrid)
```

Container mode deployments SHALL rely on OCI image signing (cosign/notation) for image-level integrity, with SmallAIOS model verification providing an additional layer within the container.

#### Scenario: Verified boot chain on bare metal
- **WHEN** SmallAIOS boots on bare metal with `verified-boot` enabled
- **THEN** the kernel verifies its own integrity before proceeding to model loading, and every ONNX model is verified before execution

#### Scenario: Verified boot in container mode
- **WHEN** SmallAIOS runs in container mode with `verified-boot` enabled
- **THEN** ONNX model signatures are verified at load time, and the boot measurement log records the verification status of all loaded models

#### Scenario: Degraded mode without verified-boot feature
- **WHEN** SmallAIOS is built without the `verified-boot` feature flag
- **THEN** boot proceeds as before with no integrity checks, maintaining backward compatibility
