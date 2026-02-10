# Spec 06: Security Model

## Overview

SmallAIOS uses a **defense-in-depth** security model with four layers:

1. **Language safety**: Rust eliminates memory corruption bugs at compile time.
2. **Capability-based access control**: No ambient authority; every operation
   requires an explicit capability token.
3. **Minimal syscall surface**: ~38 syscalls vs. ~450 in Linux — smaller attack surface.
4. **No unnecessary components**: No shell, no writable filesystem, no dynamic
   linking, no debugging interfaces in production.

## Threat Model

### In Scope

| Threat | Mitigation |
|---|---|
| Malicious ONNX model (crafted protobuf) | Protobuf parser fuzzing, input validation |
| Buffer overflow in tensor operations | Rust memory safety, bounds checking |
| Network-based attack on IPC port | Minimal TCP stack, TLS mutual auth, input validation |
| Container escape via kernel exploit | Minimal syscall surface, no shell to escape to |
| Supply chain attack on dependencies | Zero runtime deps, reproducible builds, SBOM |
| Side-channel attacks (Spectre, etc.) | Retpoline, IBRS, cache partitioning where applicable |
| Denial of service via resource exhaustion | Resource quotas, bounded allocations |

### Out of Scope (accepted risks)

- Physical access to hardware (assume trusted data center)
- Compromised firmware/UEFI (assume verified boot chain)
- NVIDIA GPU hardware vulnerabilities (trust hardware)
- Malicious host kernel in container mode (trust container runtime)

## Capability-Based Security

### Capability Tokens

Every kernel resource is accessed via a **capability** — an unforgeable token
that grants specific permissions on a specific resource.

```rust
pub struct Capability {
    /// Unique capability identifier
    pub id: CapId,
    /// Resource this capability grants access to
    pub resource: ResourceRef,
    /// Permitted operations (bitmask)
    pub permissions: Permissions,
    /// Optional: expiry time (nanoseconds since boot)
    pub expires: Option<u64>,
}

bitflags! {
    pub struct Permissions: u32 {
        const READ    = 0b0001;
        const WRITE   = 0b0010;
        const EXECUTE = 0b0100;
        const GRANT   = 0b1000;  // Can delegate to others
    }
}
```

### Resource Types

| Resource | Capabilities Needed | Example |
|---|---|---|
| Tensor buffer | READ, WRITE | Inference I/O |
| ONNX model | READ, EXECUTE | Load and run model |
| IPC endpoint | READ (subscribe), WRITE (publish) | Messaging |
| GPU device | READ, WRITE, EXECUTE | GPU inference |
| Network socket | READ, WRITE | External IPC |
| System config | READ | Query configuration |
| System control | EXECUTE | Shutdown |

### Capability Lifecycle

```
1. Boot: Root capability set created (all permissions)
2. ONNX runtime receives: model READ/EXECUTE, tensor READ/WRITE, GPU access
3. IPC router receives: network READ/WRITE, endpoint management
4. Inference tasks receive: model EXECUTE, input tensor READ, output tensor WRITE
5. Capabilities cannot be forged — only delegated (with GRANT permission)
6. Expired capabilities are automatically revoked
```

### No Ambient Authority

Unlike POSIX (where root can do anything), SmallAIOS enforces that:
- The ONNX runtime cannot access the network directly
- The IPC router cannot execute models
- No component can access GPU memory without explicit GPU capability
- There is no "root" bypass — even the kernel init code delegates capabilities
  and then drops its root set

## Memory Safety

### Rust Guarantees

- No null pointer dereferences (Option<T>)
- No buffer overflows (bounds-checked indexing)
- No use-after-free (ownership system)
- No data races (Send/Sync traits)
- No uninitialized memory reads (MaybeUninit)

### Unsafe Code Policy

`unsafe` blocks are necessary for:
- Inline assembly (HAL, SIMD kernels)
- Raw pointer manipulation (page tables, MMIO)
- FFI boundaries (GPU command submission)

Rules for `unsafe`:
1. Every `unsafe` block must have a `// SAFETY:` comment explaining the invariant
2. Unsafe code must be wrapped in safe abstractions
3. No unsafe code in business logic (ONNX ops, IPC routing)
4. All unsafe code is reviewed and fuzz-tested

## Network Security

### Minimal Network Stack

SmallAIOS implements only what's needed for IPC:
- TCP (no UDP, no ICMP, no raw sockets)
- TLS 1.3 (optional, for encrypted IPC)
- No DNS (addresses are configured, not resolved)
- No DHCP (IP configured at boot or from container runtime)

### TLS Configuration

- TLS 1.3 only (no downgrade)
- Certificate-based mutual authentication
- Minimal cipher suites: AES-256-GCM, ChaCha20-Poly1305
- No session resumption (stateless)
- Certificates loaded from container image at boot

### Input Validation

All external input is validated before processing:
- IPC messages: Size limits, format validation, schema checking
- ONNX models: Protobuf validation, opset version check, operator allowlist
- Tensor data: Shape validation, size limits, NaN/Inf handling
- TCP: Connection limits, rate limiting, timeout enforcement

## Boot Security

### Secure Boot Chain (bare metal/VM)

```
UEFI Secure Boot → Signed kernel image → Verified ONNX models
```

- Kernel binary is signed with a project-specific key
- ONNX model files can optionally be signed (SHA-256 hash in manifest)
- Container images use standard OCI signing (cosign/notation)

### Container Mode Security

- Kernel runs as unprivileged container (no `--privileged`)
- No Linux capabilities required except `CAP_NET_BIND_SERVICE` (if port < 1024)
- GPU access via NVIDIA Container Toolkit (device passthrough)
- Read-only container filesystem
- No volume mounts required (models embedded in image)

## Resource Limits

| Resource | Default Limit | Configurable |
|---|---|---|
| Heap memory | 256 MB | Yes |
| Tensor pool | 1 GB | Yes |
| GPU memory | Device VRAM | No (uses all) |
| Open file descriptors | 256 | Yes |
| Concurrent inference requests | 64 | Yes |
| TCP connections | 128 | Yes |
| IPC message size | 64 MB | Yes |
| Model file size | 2 GB | Yes |

## Auditing and Logging

- All capability grants/revocations are logged
- All inference requests are logged (key expression, size, latency)
- Failed operations are logged with error details
- Logs are stored in a fixed-size kernel ring buffer
- Logs are published via IPC (`smallaios/v1/logs`) for external collection
- No sensitive data (tensor contents) in logs

## Crate Structure

```
security/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── capability.rs    # Capability type and operations
    ├── registry.rs      # Capability registry (who holds what)
    ├── policy.rs        # Default capability assignment policy
    ├── audit.rs         # Security event logging
    └── crypto.rs        # Minimal crypto (SHA-256, signature verification)
```
