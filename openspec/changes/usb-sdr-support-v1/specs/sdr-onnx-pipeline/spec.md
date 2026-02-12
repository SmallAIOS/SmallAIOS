# Delta for SDR-to-ONNX Inference Pipeline

## ADDED Requirements

### Requirement: IQ Sample Ring Buffer
The SDR pipeline SHALL provide a lock-free ring buffer for decoupling USB IQ streaming rate from ONNX inference rate.

#### Scenario: Write IQ samples to ring buffer
- WHEN the SDR driver delivers a batch of IQ samples
- THEN the ring buffer MUST accept the samples without blocking the USB streaming callback
- AND MUST overwrite the oldest samples if the buffer is full (lossy mode for real-time inference)

#### Scenario: Read windowed samples from ring buffer
- WHEN the inference task requests a window of N samples
- THEN the ring buffer MUST return the N most recent contiguous samples
- AND MUST advance the read pointer past the consumed samples (or by the stride if overlap is configured)

#### Scenario: Report buffer overflow
- WHEN the ring buffer overwrites unread samples due to the inference task falling behind
- THEN the ring buffer MUST increment an overflow counter
- AND MUST publish the overflow count to Zenoh key expression `sdr/{device}/overflow` for monitoring

#### Scenario: Configure ring buffer depth
- WHEN the pipeline is initialized with a ring buffer depth of 1,048,576 samples
- THEN the ring buffer MUST allocate sufficient DMA-capable memory for the configured depth
- AND MUST support sample sizes of 2 bytes (8-bit I + 8-bit Q from HackRF) and 4 bytes (16-bit I + 16-bit Q from PlutoSDR)

### Requirement: Windowing Functions
The SDR pipeline SHALL apply configurable windowing functions to IQ sample blocks before inference.

#### Scenario: Apply Hann window
- WHEN the pipeline is configured with the Hann window function
- THEN each sample in the window MUST be multiplied by `0.5 * (1 - cos(2π * n / (N-1)))` where n is the sample index and N is the window size

#### Scenario: Apply Hamming window
- WHEN the pipeline is configured with the Hamming window function
- THEN each sample in the window MUST be multiplied by `0.54 - 0.46 * cos(2π * n / (N-1))`

#### Scenario: Apply rectangular window (no windowing)
- WHEN the pipeline is configured with the rectangular window function
- THEN the samples MUST be passed through unchanged (identity multiplication)

#### Scenario: Configure window overlap
- WHEN the pipeline is configured with 50% overlap and a window size of 1024
- THEN consecutive windows MUST overlap by 512 samples
- AND the stride between windows MUST be 512 samples

### Requirement: FFT Preprocessing
The SDR pipeline SHALL optionally compute FFT to convert time-domain IQ samples to frequency-domain features for spectral inference models.

#### Scenario: Compute FFT on IQ window
- WHEN FFT preprocessing is enabled and a windowed IQ block of 1024 complex samples is ready
- THEN the pipeline MUST compute a 1024-point complex FFT
- AND MUST produce 1024 frequency bins as complex values (real + imaginary)

#### Scenario: Compute magnitude spectrum
- WHEN the pipeline is configured for magnitude output
- THEN it MUST compute `sqrt(re^2 + im^2)` for each frequency bin
- AND MUST output a 1024-element real-valued magnitude vector

#### Scenario: Compute power spectral density (dB)
- WHEN the pipeline is configured for PSD output in decibels
- THEN it MUST compute `10 * log10(re^2 + im^2)` for each frequency bin
- AND MUST handle zero-magnitude bins by clamping to a configurable floor (default: -120 dB)

#### Scenario: Bypass FFT for time-domain models
- WHEN FFT preprocessing is disabled
- THEN the pipeline MUST pass windowed IQ samples directly to tensor formatting
- AND MUST format them as interleaved real/imaginary pairs

### Requirement: Tensor Formatting for ONNX Input
The SDR pipeline SHALL convert processed IQ data into ONNX-compatible input tensors.

#### Scenario: Format as 2D real/imaginary tensor
- WHEN the ONNX model expects input shape [1, N, 2] (batch, samples, channels)
- THEN the pipeline MUST format the IQ data as a float32 tensor with I values in channel 0 and Q values in channel 1

#### Scenario: Format as 1D magnitude tensor
- WHEN the ONNX model expects input shape [1, N] (batch, frequency bins)
- THEN the pipeline MUST provide the magnitude spectrum as a float32 tensor

#### Scenario: Normalize input tensor
- WHEN normalization is configured (e.g., zero-mean, unit-variance)
- THEN the pipeline MUST compute running mean and variance over a configurable window
- AND MUST normalize each sample as `(x - mean) / std`

### Requirement: Continuous Streaming Inference
The SDR pipeline SHALL continuously feed windowed IQ data into the ONNX runtime and publish results to Zenoh.

#### Scenario: Run inference on each window
- WHEN the pipeline is running and a new window of IQ data is available
- THEN the pipeline MUST format the window as an ONNX input tensor
- AND MUST submit the tensor to the ONNX runtime for inference
- AND MUST publish the inference output to Zenoh key expression `sdr/{device}/{model}`

#### Scenario: Handle inference backpressure
- WHEN inference takes longer than the window arrival rate
- THEN the pipeline MUST skip windows to keep up with real-time data
- AND MUST increment a skip counter published to Zenoh key expression `sdr/{device}/skipped`
- AND MUST NOT queue unbounded inference requests

#### Scenario: Publish classification result
- WHEN the ONNX model outputs a classification vector (e.g., signal type probabilities)
- THEN the pipeline MUST publish the result with metadata including timestamp, center frequency, sample rate, and window index to Zenoh key expression `sdr/{device}/{model}`

### Requirement: Multi-Device Pipeline Support
The SDR pipeline SHALL support running independent pipelines for multiple SDR devices simultaneously.

#### Scenario: Run HackRF and PlutoSDR pipelines concurrently
- WHEN both a HackRF One and an ADALM-PLUTO are connected and configured
- THEN the pipeline manager MUST create independent ring buffers and inference tasks for each device
- AND each device's results MUST be published to distinct Zenoh key expressions (`sdr/hackrf/...` and `sdr/pluto/...`)

### Requirement: Pipeline Configuration
The SDR pipeline SHALL accept a configuration structure specifying all pipeline parameters.

#### Scenario: Configure pipeline from structured config
- WHEN the application provides a pipeline configuration with device, model path, sample rate, center frequency, window size, window function, FFT enabled, and ring buffer depth
- THEN the pipeline MUST validate all parameters
- AND MUST configure the SDR device, allocate the ring buffer, and start the inference loop
- AND MUST return an error if any parameter is invalid or the SDR device is not available
