## ADDED Requirements

### Requirement: Metal Device Initialization
The Metal backend SHALL initialize an Apple GPU device via Metal framework FFI in container mode.

#### Scenario: Create Metal device on Apple Silicon
- **WHEN** the Metal backend initializes on a macOS system with Apple Silicon (M1/M2/M3/M4)
- **THEN** it MUST create an `MTLDevice` instance for the default system GPU
- **AND** MUST report device name, unified memory size, and GPU family via `device_info()`

#### Scenario: Fail gracefully on non-macOS platforms
- **WHEN** the Metal backend is compiled on a non-macOS target
- **THEN** initialization MUST return an error indicating Metal is unavailable
- **AND** the system MUST fall back to CPU execution

### Requirement: Metal Buffer Management
The Metal backend SHALL manage GPU memory buffers using Metal's buffer allocation API.

#### Scenario: Allocate a device buffer
- **WHEN** `alloc(size)` is called
- **THEN** the backend MUST create an `MTLBuffer` with the requested size
- **AND** on Apple Silicon MUST use shared storage mode (unified memory)

#### Scenario: Host-to-device transfer on unified memory
- **WHEN** `copy_host_to_device` is called on Apple Silicon (unified memory)
- **THEN** the backend MUST copy data into the shared `MTLBuffer` contents pointer
- **AND** the transfer MUST NOT require DMA (direct memory copy)

### Requirement: Metal Compute Kernel Execution
The Metal backend SHALL compile and execute Metal Shading Language (MSL) kernels for supported ONNX operators.

#### Scenario: Load an MSL compute kernel
- **WHEN** `load_kernel(name, msl_source)` is called with valid MSL source code
- **THEN** the backend MUST compile the MSL into a `MTLComputePipelineState`
- **AND** MUST cache the pipeline state for reuse across inference calls

#### Scenario: Launch a compute kernel
- **WHEN** `launch(kernel, grid, block, args)` is called
- **THEN** the backend MUST create a command buffer and compute command encoder
- **AND** MUST dispatch the kernel with the specified threadgroup configuration
- **AND** MUST commit the command buffer for execution

#### Scenario: Synchronize after kernel execution
- **WHEN** `synchronize()` is called
- **THEN** the backend MUST wait until all committed command buffers have completed
- **AND** output buffers MUST contain the computed results

### Requirement: Metal Kernel Implementations for Core Operators
The Metal backend SHALL provide MSL kernel implementations for compute-intensive ONNX operators.

#### Scenario: MatMul/Gemm via SIMD group matrix operations
- **WHEN** a MatMul or Gemm operator is dispatched to the Metal backend
- **THEN** the kernel MUST use Apple's SIMD group matrix multiply intrinsics where available
- **AND** MUST fall back to tiled matrix multiplication for unsupported matrix sizes
- **AND** MUST produce results within f32 epsilon of the CPU reference implementation

#### Scenario: Convolution kernel
- **WHEN** a Conv operator is dispatched to the Metal backend
- **THEN** the kernel MUST implement convolution (im2col + MatMul or direct)
- **AND** MUST support padding, strides, and dilation attributes

#### Scenario: Element-wise operation kernels
- **WHEN** Add, Mul, Sub, Div, Relu, Sigmoid, or Tanh operators are dispatched
- **THEN** the kernel MUST compute the element-wise operation in parallel across GPU threads
- **AND** MUST support broadcasting for binary operators

#### Scenario: Softmax reduction kernel
- **WHEN** a Softmax operator is dispatched to the Metal backend
- **THEN** the kernel MUST compute softmax using parallel reduction for the max and sum passes
- **AND** MUST handle numerical stability (subtract max before exp)
