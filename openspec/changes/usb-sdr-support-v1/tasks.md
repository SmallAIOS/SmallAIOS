## 1. USB Core Protocol Stack

- [x] 1.1 Create `smallaios-usb` crate with feature flags (xhci, gadget, inference-gadget)
- [x] 1.2 Implement USB device descriptor parser (zero-copy from byte buffer)
- [x] 1.3 Implement USB configuration descriptor parser (nested interface/endpoint chain)
- [x] 1.4 Implement USB endpoint descriptor parser (address, direction, transfer type, max packet size)
- [x] 1.5 Implement USB string descriptor parser (UTF-16LE decoding)
- [x] 1.6 Implement USB device enumeration state machine (reset → SET_ADDRESS → GET_DESCRIPTOR → SET_CONFIGURATION)
- [x] 1.7 Implement USB control transfer state machine (SETUP → DATA → STATUS)
- [x] 1.8 Implement USB bulk transfer management (submit, complete, resubmit ring)
- [x] 1.9 Implement USB endpoint state tracking (active, halted, idle) and halt recovery
- [x] 1.10 Implement USB device registry (VID/PID matching, interface class matching, driver binding)
- [x] 1.11 Unit tests for descriptor parsers (100% coverage on valid/invalid/truncated descriptors)
- [x] 1.12 Unit tests for enumeration state machine (success, timeout, stall scenarios)
- [ ] 1.13 Write TLA+ model for USB enumeration state machine

## 2. xHCI Host Controller Driver

- [x] 2.1 Extend PCIe enumeration in arch crates to detect xHCI controllers (class 0x0C/0x03/0x30)
- [x] 2.2 Implement xHCI capability register parsing (HCSPARAMS1/2/3, HCCPARAMS1/2)
- [x] 2.3 Implement xHCI controller reset and initialization (USBCMD.HCRST, wait CNR, configure MaxSlots)
- [x] 2.4 Implement Device Context Base Address Array (DCBAA) allocation and initialization
- [x] 2.5 Implement scratchpad buffer allocation from HCSPARAMS2
- [x] 2.6 Implement Command Ring (enqueue TRBs, ring doorbell, handle Link TRBs)
- [x] 2.7 Implement Event Ring (dequeue events, advance ERDP, handle EHB)
- [x] 2.8 Implement per-endpoint Transfer Rings (Normal TRBs, Link TRBs, cycle bit management)
- [x] 2.9 Implement Enable Slot and Address Device command sequences
- [x] 2.10 Implement Configure Endpoint command for bulk endpoint setup
- [x] 2.11 Implement port status change detection and port reset sequence
- [x] 2.12 Implement MSI-X interrupt configuration for xHCI interrupter
- [x] 2.13 Implement polling fallback when MSI-X is unavailable
- [x] 2.14 Implement device disconnection handling (Disable Slot, resource cleanup)
- [x] 2.15 Unit tests for TRB encoding/decoding (all TRB types: Normal, Setup, Data, Status, Link, Command, Event)
- [x] 2.16 Unit tests for ring management (enqueue, dequeue, wrap-around, cycle bit)
- [ ] 2.17 Integration test: xHCI init → enumerate USB device → bulk transfer (mock or QEMU)
- [ ] 2.18 Write TLA+ model for xHCI transfer ring producer/consumer

## 3. USB Device/Gadget Controller Framework

- [x] 3.1 Define `UsbDeviceController` HAL trait in kernel/src/hal.rs
- [x] 3.2 Implement gadget function registration framework
- [x] 3.3 Implement USB descriptor composition (device, config, interface, endpoint from registered functions)
- [x] 3.4 Implement composite device support with Interface Association Descriptors (IAD)
- [x] 3.5 Implement EP0 control transfer handling for standard device requests (GET_DESCRIPTOR, SET_ADDRESS, SET_CONFIGURATION)
- [x] 3.6 Implement gadget endpoint data transfer API (write IN, read OUT, stall)
- [x] 3.7 Implement USB bus event handling (reset, suspend, resume, speed negotiation)
- [x] 3.8 Implement DWC3 device controller driver (shared by Zynq and Tegra platforms)
- [x] 3.9 Unit tests for descriptor composition (single function, composite, IAD)
- [x] 3.10 Unit tests for EP0 request handling (standard requests, vendor requests, stall)

## 4. USB Inference Gadget

- [x] 4.1 Implement inference gadget function (vendor class 0xFF, bulk IN + bulk OUT endpoints)
- [x] 4.2 Implement inference request parser (request_id, model_name, tensor_size, tensor_data)
- [x] 4.3 Implement inference response formatter (request_id, status, result_size, result_data)
- [x] 4.4 Implement request validation (model name length ≤ 256, tensor size ≤ 256 MiB)
- [x] 4.5 Implement Zenoh bridge: publish inference requests to `usb/inference/{model}`
- [x] 4.6 Implement Zenoh bridge: subscribe to inference results and route to USB response
- [x] 4.7 Implement concurrent request tracking (up to 4 outstanding requests by request_id)
- [x] 4.8 Implement DMA integration for zero-copy tensor transfer (tensors > 4096 bytes)
- [x] 4.9 Unit tests for request/response protocol (valid, malformed, unknown model, concurrent)
- [ ] 4.10 Integration test: host submits inference request over USB → ONNX result returned

## 5. HackRF One SDR Driver

- [ ] 5.1 Create `smallaios-sdr` crate with feature flags (hackrf, pluto, iq-pipeline)
- [ ] 5.2 Implement HackRF device detection (VID 0x1D50 / PID 0x6089, board ID verification)
- [ ] 5.3 Implement vendor control transfer helper (bmRequestType, bRequest, wValue, wIndex, data)
- [ ] 5.4 Implement SET_FREQ command (center frequency configuration)
- [ ] 5.5 Implement SAMPLE_RATE_SET command (sample rate + baseband filter)
- [ ] 5.6 Implement gain commands (SET_LNA_GAIN, SET_VGA_GAIN, SET_TXVGA_GAIN, AMP_ENABLE)
- [ ] 5.7 Implement SET_TRANSCEIVER_MODE command (OFF, RX, TX, SWEEP)
- [ ] 5.8 Implement bulk RX streaming (4 concurrent transfers, 262,144 bytes each, resubmit ring)
- [ ] 5.9 Implement bulk TX streaming (bulk OUT on EP 0x02)
- [ ] 5.10 Implement sweep mode (INIT_SWEEP + SWEEP transceiver mode + header parsing)
- [ ] 5.11 Implement half-duplex enforcement (prevent simultaneous TX+RX)
- [ ] 5.12 Implement device reset and error recovery
- [ ] 5.13 Implement board info queries (BOARD_ID_READ, VERSION_STRING_READ, BOARD_REV_READ)
- [ ] 5.14 Unit tests for all vendor request encoding (verify bmRequestType, bRequest, wValue for each command)
- [ ] 5.15 Unit tests for IQ data format parsing (8-bit signed interleaved I/Q)
- [ ] 5.16 Unit tests for gain range validation (reject out-of-range values)
- [ ] 5.17 Integration test: configure HackRF → start RX → receive IQ samples (mock USB)

## 6. ADALM-PLUTO SDR Driver

- [ ] 6.1 Implement PlutoSDR device detection (VID 0x0456 / PID 0xb673, composite device parsing)
- [ ] 6.2 Implement vendor USB interface identification (locate IIO interface among CDC/MSC/DFU)
- [ ] 6.3 Implement IIOD text protocol command encoder (PRINT, READ, WRITE, OPEN, CLOSE, READBUF, WRITEBUF, TIMEOUT)
- [ ] 6.4 Implement IIOD text protocol response parser (return code + optional data payload)
- [ ] 6.5 Implement IIO context discovery via PRINT command (extract device and channel list)
- [ ] 6.6 Implement AD9363 RX frequency configuration (altvoltage0 frequency attribute)
- [ ] 6.7 Implement AD9363 TX frequency configuration (altvoltage1 frequency attribute)
- [ ] 6.8 Implement AD9363 sample rate configuration (cf-ad9361-lpc sampling_frequency attribute)
- [ ] 6.9 Implement AD9363 gain configuration (gain_control_mode + hardwaregain attributes)
- [ ] 6.10 Implement AD9363 bandwidth configuration (rf_bandwidth attribute)
- [ ] 6.11 Implement frequency range validation (325 MHz to 3.8 GHz for AD9363)
- [ ] 6.12 Implement IIOD buffer streaming (OPEN, READBUF loop, CLOSE)
- [ ] 6.13 Implement WRITEBUF for TX streaming (full-duplex support)
- [ ] 6.14 Implement IIOD TIMEOUT configuration
- [ ] 6.15 Unit tests for IIOD protocol encoder/decoder (all command types, error responses)
- [ ] 6.16 Unit tests for AD9363 attribute formatting (frequency, gain, sample rate value encoding)
- [ ] 6.17 Unit tests for IQ data format parsing (16-bit signed interleaved I/Q)
- [ ] 6.18 Integration test: configure PlutoSDR → open buffer → stream IQ samples (mock USB)

## 7. SDR-to-ONNX Inference Pipeline

- [ ] 7.1 Implement lock-free IQ ring buffer (lossy overwrite mode, configurable depth)
- [ ] 7.2 Implement ring buffer overflow tracking and Zenoh reporting
- [ ] 7.3 Implement Hann window function
- [ ] 7.4 Implement Hamming window function
- [ ] 7.5 Implement rectangular (passthrough) window function
- [ ] 7.6 Implement configurable window overlap and stride
- [ ] 7.7 Implement radix-2 FFT for power-of-2 window sizes (no_std, no alloc)
- [ ] 7.8 Implement magnitude spectrum computation (sqrt(re² + im²))
- [ ] 7.9 Implement power spectral density computation (10*log10, dB floor clamping)
- [ ] 7.10 Implement tensor formatting: 2D real/imaginary [1, N, 2]
- [ ] 7.11 Implement tensor formatting: 1D magnitude [1, N]
- [ ] 7.12 Implement input normalization (running mean/variance, configurable window)
- [ ] 7.13 Implement continuous inference loop (window → preprocess → ONNX → Zenoh publish)
- [ ] 7.14 Implement inference backpressure handling (window skipping, skip counter)
- [ ] 7.15 Implement multi-device pipeline manager (independent pipelines per SDR device)
- [ ] 7.16 Implement pipeline configuration struct and validation
- [ ] 7.17 Unit tests for ring buffer (write, read, overflow, concurrent access)
- [ ] 7.18 Unit tests for window functions (known-good reference values)
- [ ] 7.19 Unit tests for FFT (known-good reference: single tone, DC, Nyquist)
- [ ] 7.20 Unit tests for tensor formatting (shape validation, value correctness)
- [ ] 7.21 Integration test: SDR mock → ring buffer → FFT → ONNX inference → Zenoh publish
- [ ] 7.22 Write TLA+ model for ring buffer overflow protection

## 8. HAL and Syscall Integration

- [x] 8.1 Add `UsbHostController` trait to kernel/src/hal.rs
- [x] 8.2 Add `UsbDeviceController` trait to kernel/src/hal.rs
- [x] 8.3 Add USB-specific error variants to HalError (UsbTransferError, UsbStall, UsbDeviceNotFound, UsbEndpointHalted)
- [ ] 8.4 Implement sys_dev_enumerate for USB device listing
- [ ] 8.5 Implement sys_dev_open / sys_dev_close for USB device handles
- [ ] 8.6 Implement sys_dev_ioctl dispatch to USB driver operations
- [ ] 8.7 Implement sys_dev_dma_alloc integration with USB DMA buffers
- [x] 8.8 Unit tests for HAL trait method signatures and error types
- [ ] 8.9 Unit tests for device syscall USB integration (enumerate, open, ioctl, DMA)
