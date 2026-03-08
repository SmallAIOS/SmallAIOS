# Test Coverage Improvement - Design Document

## Context

SmallAIOS achieves 93.84% overall line coverage across 18 crates and 4,143 tests. However, 15 peripheral driver files fall below 70% coverage, with some as low as 34%. For a safety-critical OS targeting DO-178C DAL A compliance, these gaps are unacceptable. The peripheral crate contains MMIO-based hardware drivers that interact with registers at fixed memory addresses -- these cannot be tested on real hardware in CI but CAN be tested with mock register backends.

The bench crate is at 0% coverage (no benchmarks exist). Fuzzing covers crypto but not the ONNX protobuf parser, network packet handling, or USB descriptor parsing. No integration tests validate cross-crate workflows. No CI enforcement prevents coverage regression.

## Goals / Non-Goals

**Goals:**
- Bring all 15 peripheral driver files below 70% to at least 90% line coverage
- Add cross-crate integration tests for key workflows (kernel<->security, net<->onnx-rt, container boot)
- Add fuzz targets for parsers/decoders: ONNX protobuf, TCP packets, USB descriptors
- Add performance benchmarks with regression detection in the bench crate
- Configure Codecov with per-crate targets and path exclusions
- Enforce coverage thresholds in CI to prevent regression

**Non-Goals:**
- 100% MC/DC coverage on all code (only safety-critical paths require MC/DC)
- Hardware-in-the-loop testing (requires physical devices)
- Rewriting drivers to be more testable (tests adapt to existing code)
- Benchmarking on real GPU hardware (GPU crates are stubs)

## Testing Pyramid Strategy

```
                    /  E2E  \           QEMU boot, Docker container
                   /----------\
                  / Integration \       Cross-crate workflows
                 /----------------\
                /    Fuzzing        \   Parser/decoder robustness
               /--------------------\
              /     Benchmarks       \  Performance regression
             /------------------------\
            /       Unit Tests          \ Mock-based driver tests
           /------------------------------\
```

### Layer 1: Unit Tests (Mock-Based for Hardware Drivers)

**Pattern: Mock Register Backend**

All peripheral drivers use MMIO register access through volatile read/write operations. The existing code uses `read_volatile`/`write_volatile` on raw pointers derived from a base address. Tests provide a mock memory region (a `[u8; N]` array or `Vec<u8>`) and pass its address as the base address to the driver constructor.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_mock_regs() -> Vec<u8> {
        vec![0u8; 0x1000]  // 4K register space
    }

    #[test]
    fn test_init_sequence() {
        let mut regs = make_mock_regs();
        let base = regs.as_mut_ptr() as usize;
        let mut driver = Ns16550a::new(base);
        driver.init(&UartConfig::default());
        // Verify expected register values were written
        let lcr = unsafe { *(regs.as_ptr().add(LCR_OFFSET) as *const u8) };
        assert_eq!(lcr & 0x03, 0x03); // 8-bit word length
    }
}
```

This pattern works because:
- Drivers operate on a base address + offset, not on a specific physical address
- `read_volatile`/`write_volatile` work on any valid memory, not just MMIO regions
- No hardware side effects to simulate -- we verify register programming sequences
- Tests run on host (x86-64 Linux) via `cargo test`, no QEMU needed

**Coverage targets by driver family:**

| Family | Files | Current Range | Target |
|--------|-------|---------------|--------|
| UART | ns16550a, sifive, pl011, axi_uart_lite | 34-49% | 90%+ |
| GPIO | riscv_mmio, arm_pl061, axi_gpio | 42-61% | 90%+ |
| I2C | riscv_mmio, arm_mmio, bitbang | 55-69% | 90%+ |
| SPI | riscv_mmio, arm_mmio | 58-68% | 90%+ |
| Camera | tegra_vi, broadcom_unicam, fpga_csi | 53-69% | 90%+ |

Each driver file needs tests covering:
1. Initialization with default and custom configs
2. Data transmit/receive (for UART, SPI, I2C) or pin set/get (for GPIO)
3. Interrupt handling / status register polling
4. Error conditions (FIFO overflow, NACK, timeout)
5. Configuration changes (baud rate, mode, polarity)

### Layer 2: Integration Tests (Cross-Crate Workflows)

Key crate combinations that need integration coverage:

| Test Suite | Crates Involved | What to Validate |
|------------|----------------|------------------|
| `kernel_security` | kernel + security | Capability checks on syscalls, PQC key operations through kernel API |
| `net_onnx` | net + onnx-rt | Inference request received over network, result returned |
| `container_boot` | container + kernel + security | Container startup sequence, health check, metrics export |
| `ipc_security` | ipc + security | Formal-gate label enforcement on pub/sub messages |
| `crypto_pipeline` | security + net | TLS 1.3 handshake with ML-KEM-768, encrypted data transfer |

Integration tests live in each crate's `tests/` directory or in a dedicated `tests/` top-level directory. They use `#[cfg(test)]` and can depend on multiple workspace crates.

### Layer 3: End-to-End Tests

- **QEMU boot test**: Already exists in CI (RISC-V smoke test). Extend to x86-64 and AArch64.
- **Docker container test**: Already validated (594KB image). Add automated test that boots container, hits health endpoint, verifies metrics.
- These remain in CI workflows, not in `cargo test`.

### Layer 4: Fuzzing

**Framework: Inline fuzz modules using `cargo-fuzz` / `libfuzzer`**

Fuzz targets in a top-level `fuzz/` directory (cargo-fuzz convention):

| Target | Input | What to Fuzz |
|--------|-------|-------------|
| `fuzz_onnx_protobuf` | arbitrary bytes | `onnx-rt` protobuf parser -- should never panic on malformed input |
| `fuzz_tcp_packet` | arbitrary bytes | `net` TCP packet parsing -- malformed headers, truncated packets |
| `fuzz_udp_packet` | arbitrary bytes | `net` UDP packet parsing |
| `fuzz_usb_descriptor` | arbitrary bytes | `usb` descriptor parsing -- malformed device/config/interface descriptors |
| `fuzz_ipc_message` | arbitrary bytes | `ipc` message deserialization |
| `fuzz_onnx_tensor` | arbitrary bytes | `onnx-rt` tensor data handling -- shape/stride validation |

Each fuzz target:
- Uses `#![no_main]` with `libfuzzer_sys::fuzz_target!` macro
- Feeds arbitrary bytes to the parser/decoder entry point
- Success = no panic, no undefined behavior
- Runs in CI with a time budget (e.g., 60 seconds per target per CI run)
- Corpus stored in `fuzz/corpus/<target>/` and committed to repo

### Layer 5: Benchmarks

**Framework: Custom `no_std`-compatible benchmark harness in the `bench` crate**

The `bench` crate already exists but has no content. Since SmallAIOS is `#![no_std]`, criterion (which requires `std`) cannot be used directly in the kernel. Instead:

- **Host benchmarks** (run with `cargo bench`): Use criterion for crates that compile on host (onnx-rt, security/crypto, net, ipc). These measure throughput and latency.
- **Kernel benchmarks** (run in QEMU): Use a custom harness that measures TSC/cycle counts. These are not part of CI initially.

| Benchmark | Crate | Metric |
|-----------|-------|--------|
| ONNX operator throughput | onnx-rt | ops/sec for MatMul, Conv, Relu, Sigmoid, Softmax, Gemm |
| Protobuf parse latency | onnx-rt | ns/parse for various model sizes |
| SHA-3-256 throughput | security | bytes/sec |
| AES-256-GCM throughput | security | bytes/sec |
| ML-KEM-768 keygen/encaps/decaps | security | ops/sec |
| ML-DSA-65 sign/verify | security | ops/sec |
| Ed25519 sign/verify | security | ops/sec |
| TCP packet parse | net | packets/sec |
| IPC pub/sub latency | ipc | ns/message |
| Memory allocator | kernel | allocs/sec, fragmentation ratio |

Regression detection:
- Criterion outputs JSON results; CI compares against baseline
- Baseline stored in `bench/baselines/` as JSON
- CI fails if any benchmark regresses more than 10% from baseline
- Baselines updated manually via `make bench-update-baseline`

## Codecov Configuration

`codecov.yml` at repository root:

```yaml
coverage:
  status:
    project:
      default:
        target: 93%          # Current level, ratchet upward
        threshold: 1%        # Allow 1% fluctuation
    patch:
      default:
        target: 90%          # New code must be 90%+ covered

  # Per-crate flags for granular tracking
  flags:
    kernel:
      paths: ["kernel/src/"]
      carryforward: true
    security:
      paths: ["security/src/"]
      carryforward: true
    onnx-rt:
      paths: ["onnx-rt/src/"]
      carryforward: true
    net:
      paths: ["net/src/"]
      carryforward: true
    peripheral:
      paths: ["peripheral/src/"]
      carryforward: true
    container:
      paths: ["container/src/"]
      carryforward: true
    ipc:
      paths: ["ipc/src/"]
      carryforward: true
    bus:
      paths: ["bus/src/"]
      carryforward: true
    usb:
      paths: ["usb/src/"]
      carryforward: true
    sdr:
      paths: ["sdr/src/"]
      carryforward: true

ignore:
  - "arch/**"               # Bare-metal arch crates (not host-testable)
  - "container/src/main.rs" # Binary entry point
  - "bench/**"              # Benchmark code
  - "fuzz/**"               # Fuzz targets
  - "docs/**"               # Documentation

comment:
  layout: "diff, flags, files"
  behavior: default
  require_changes: true
```

## CI Enforcement

Coverage enforcement is added to the existing `.github/workflows/ci.yml`:

1. **Coverage job** (already exists): Runs `cargo-llvm-cov` with lcov output, uploads to Codecov
2. **Codecov status checks**: Configured via `codecov.yml` to report pass/fail on PRs
3. **Project coverage gate**: PR fails if overall coverage drops below target (93%, ratcheted up as coverage improves)
4. **Patch coverage gate**: PR fails if new/changed lines have less than 90% coverage
5. **Per-crate flag tracking**: Each crate's coverage is tracked independently via Codecov flags

The `codecov.yml` file is the single source of truth for thresholds. No custom scripts needed -- Codecov's GitHub integration handles pass/fail reporting on PRs.

## Risks / Trade-offs

- **[Mock register fidelity]** -- Mock register backends don't perfectly simulate hardware behavior (e.g., status bits that auto-clear on read). Mitigated by testing the register programming sequence, not hardware response. Hardware-in-the-loop testing is a separate future effort.
- **[Fuzz time budget in CI]** -- 60 seconds per target is short. Mitigated by accumulating corpus over time and running extended fuzz sessions manually before releases.
- **[Benchmark noise in CI]** -- CI runners have variable performance. Mitigated by using 10% regression threshold (well above typical noise) and allowing manual baseline updates.
- **[Coverage target ratcheting]** -- Setting targets too aggressively may block PRs with legitimate low-coverage code (e.g., error paths that are hard to trigger). Mitigated by allowing `// LCOV_EXCL_LINE` annotations with mandatory justification comments.
- **[Criterion std dependency]** -- Criterion requires `std`, so host benchmarks must compile with `std`. This is fine because `cargo bench` already runs on the host, and all targeted crates support both `std` and `no_std` compilation.
