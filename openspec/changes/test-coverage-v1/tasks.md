## 1. UART Driver Unit Tests

- [x] 1.1 Add unit tests to peripheral/src/uart/ns16550a.rs — mock register region, test init sequence (LCR, IER, FCR programming), TX/RX byte operations, baud rate divisor calculation, error conditions (overrun, parity, framing). Target: 34.52% -> 90%+
- [x] 1.2 Add unit tests to peripheral/src/uart/sifive.rs — mock register region, test init with SiFive-specific register layout, TX/RX operations, watermark configuration, error handling. Target: 40.13% -> 90%+
- [x] 1.3 Add unit tests to peripheral/src/uart/pl011.rs — mock register region, test PL011 init (UARTCR, UARTLCR_H, UARTIBRD/UARTFBRD), FIFO TX/RX, modem control, interrupt enable/clear, error flags. Target: 44.66% -> 90%+
- [x] 1.4 Add unit tests to peripheral/src/uart/axi_uart_lite.rs — mock register region, test Xilinx AXI UART Lite init, TX/RX via status polling, FIFO full/empty detection, reset sequence. Target: 49.14% -> 90%+

## 2. GPIO Driver Unit Tests

- [x] 2.1 Add unit tests to peripheral/src/gpio/riscv_mmio.rs — mock register region, test pin direction set (input/output), pin read/write, interrupt edge/level config, invalid pin error. Target: 42.93% -> 90%+
- [x] 2.2 Add unit tests to peripheral/src/gpio/arm_pl061.rs — mock register region, test PL061 direction register, data register with bit-masking, interrupt sense/edge/event registers, alternate function select. Target: 51.98% -> 90%+
- [x] 2.3 Add unit tests to peripheral/src/gpio/axi_gpio.rs — mock register region, test Xilinx AXI GPIO channel 1/2 data and tri-state registers, interrupt enable/status, dual-channel operation. Target: 61.08% -> 90%+

## 3. I2C Driver Unit Tests

- [x] 3.1 Add unit tests to peripheral/src/i2c/riscv_mmio.rs — mock register region, test write transaction (address + data bytes), read transaction, NACK handling, bus busy detection, clock stretching. Target: 55.71% -> 90%+
- [x] 3.2 Add unit tests to peripheral/src/i2c/arm_mmio.rs — mock register region, test ARM I2C controller init, 7-bit address write/read, repeated start, arbitration loss detection. Target: 68.90% -> 90%+
- [x] 3.3 Add unit tests to peripheral/src/i2c/bitbang.rs — mock GPIO pins, test bitbang I2C protocol sequence (start, stop, byte TX with ACK check, byte RX, clock stretching). Target: 69.17% -> 90%+

## 4. SPI Driver Unit Tests

- [x] 4.1 Add unit tests to peripheral/src/spi/riscv_mmio.rs — mock register region, test RISC-V SPI init (CPOL, CPHA, clock divider), full-duplex transfer, chip select assertion, FIFO status polling. Target: 58.62% -> 90%+
- [x] 4.2 Add unit tests to peripheral/src/spi/arm_mmio.rs — mock register region, test ARM SPI controller init, transfer with TX/RX FIFO, DMA configuration, error conditions (overflow, underflow). Target: 68.28% -> 90%+

## 5. Camera Driver Unit Tests

- [x] 5.1 Add unit tests to peripheral/src/camera/tegra_vi.rs — mock register region, test Tegra VI CSI init (lane config, data format), frame capture start/stop, DMA buffer setup, error flags (CRC, sync loss). Target: 53.11% -> 90%+
- [x] 5.2 Add unit tests to peripheral/src/camera/broadcom_unicam.rs — mock register region, test Unicam CSI init, lane configuration, frame capture control, interrupt handling, buffer management. Target: 55.61% -> 90%+
- [x] 5.3 Add unit tests to peripheral/src/camera/fpga_csi.rs — mock register region, test FPGA CSI receiver init, pixel format configuration, DMA setup, frame sync detection, error reporting. Target: 68.90% -> 90%+

## 6. Integration Tests

- [x] 6.1 Create kernel-security integration test — test capability-gated syscall enforcement: task without required capability gets permission denied, task with capability succeeds. Add to kernel/tests/ or tests/ directory.
- [ ] 6.2 Create net-onnx-rt integration test — test inference request parsing from a network payload, model execution, and result serialization. Test malformed payload handling. Add to onnx-rt/tests/ or tests/ directory.
- [x] 6.3 Create container boot integration test — test container startup sequence, health check response, and metrics export. Verify startup completes within timeout. Add to container/tests/ directory.
- [ ] 6.4 Create IPC-security integration test — test formal-gate label enforcement: matched labels deliver messages, mismatched labels reject. Requires feature flag formal-gate. Add to ipc/tests/ directory.
- [ ] 6.5 Create crypto-network pipeline test — test TLS 1.3 handshake with ML-KEM-768 key exchange through security + net crates, verify encrypted data round-trip. Add to security/tests/ or net/tests/ directory.

## 7. Fuzz Targets

- [x] 7.1 Create fuzz/fuzz_targets/fuzz_onnx_protobuf.rs — fuzz the ONNX protobuf parser with arbitrary bytes, ensure no panics on malformed/truncated input
- [x] 7.2 Create fuzz/fuzz_targets/fuzz_tcp_packet.rs — fuzz TCP packet parsing from the net crate, ensure no panics or out-of-bounds reads
- [x] 7.3 Create fuzz/fuzz_targets/fuzz_udp_packet.rs — fuzz UDP packet parsing from the net crate
- [x] 7.4 Create fuzz/fuzz_targets/fuzz_usb_descriptor.rs — fuzz USB descriptor chain parsing from the usb crate, handle zero-length and oversized descriptors
- [x] 7.5 Create fuzz/fuzz_targets/fuzz_ipc_message.rs — fuzz IPC message deserialization from the ipc crate
- [x] 7.6 Create fuzz/fuzz_targets/fuzz_onnx_tensor.rs — fuzz tensor construction with arbitrary shape/data, ensure shape overflow detection
- [x] 7.7 Create seed corpus files in fuzz/corpus/ for each fuzz target — at least one valid and one near-valid example per target
- [x] 7.8 Add fuzz/Cargo.toml with cargo-fuzz configuration and dependencies on workspace crates

## 8. Benchmarks

- [ ] 8.1 Add criterion dependency and benchmark harness to bench/Cargo.toml — configure criterion with JSON output, add dev-dependencies on onnx-rt, security, net, ipc, kernel
- [ ] 8.2 Create bench/benches/onnx_operators.rs — criterion benchmarks for MatMul (64x64, 256x256, 1024x1024), Conv, Relu, Sigmoid, Softmax, Gemm with representative input sizes
- [ ] 8.3 Create bench/benches/crypto.rs — criterion benchmarks for SHA-3-256 (64B/1KB/64KB/1MB), AES-256-GCM encrypt/decrypt, ML-KEM-768 keygen/encaps/decaps, ML-DSA-65 sign/verify, Ed25519 sign/verify
- [ ] 8.4 Create bench/benches/network.rs — criterion benchmarks for TCP packet parsing, UDP packet parsing, Ethernet frame processing throughput
- [ ] 8.5 Create bench/benches/ipc.rs — criterion benchmarks for IPC pub/sub message throughput (64B/1KB/64KB) and latency measurement
- [ ] 8.6 Create bench/benches/memory.rs — criterion benchmarks for kernel memory allocator: alloc/free throughput (64B/4KB/64KB/1MB), fragmentation under mixed workload
- [ ] 8.7 Create bench/baselines/ directory and baseline JSON format — document baseline update process, add make bench-update-baseline target to Makefile
- [ ] 8.8 Add benchmark regression detection script — compare current results to baselines, fail on >10% regression, produce human-readable report

## 9. CI and Codecov Configuration

- [x] 9.1 Create codecov.yml at repository root — configure project target (93%), patch target (90%), 1% threshold, per-crate flags (kernel, security, onnx-rt, net, peripheral, container, ipc, bus, usb, sdr), path exclusions (arch/**, container/src/main.rs, bench/**, fuzz/**, docs/**), PR comment settings
- [ ] 9.2 Update .github/workflows/ci.yml coverage job — ensure lcov output is uploaded with per-crate flags matching codecov.yml flag definitions
- [ ] 9.3 Add fuzzing CI job to .github/workflows/ci.yml — install cargo-fuzz, run each fuzz target for 60 seconds, fail on any crash/panic, report failing input
- [ ] 9.4 Add benchmark CI job to .github/workflows/ci.yml — run cargo bench, compare against baselines, fail on >10% regression, upload results as artifact
- [x] 9.5 Add Makefile targets — make fuzz (run all fuzz targets locally), make bench (run benchmarks), make bench-update-baseline (update baseline files), make coverage-report (generate local HTML coverage report)
