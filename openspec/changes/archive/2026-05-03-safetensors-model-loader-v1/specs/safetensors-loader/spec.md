## ADDED Requirements

### Requirement: Parse safetensors binary format
The runtime SHALL parse `.safetensors` files containing the JSON header and tensor data sections according to the safetensors format specification.

#### Scenario: Parse single-file safetensors model
- **WHEN** a `.safetensors` file is opened
- **THEN** the runtime SHALL read the 8-byte little-endian u64 header length
- **AND** parse the JSON header containing tensor metadata (dtype, shape, data_offsets)
- **AND** make tensor data accessible by name without copying the entire file into memory

#### Scenario: Tensor lookup by name
- **WHEN** the runtime requests a tensor by its name (e.g. `model.layers.0.self_attn.q_proj.weight`)
- **THEN** the loader SHALL return a `TensorView` containing the dtype, shape, and a slice into the mmap'd file data
- **AND** the lookup SHALL be O(log n) where n is the number of tensors in the file

### Requirement: Sharded safetensors model support
The runtime SHALL load multi-file safetensors models that are split across shards via an index file.

#### Scenario: Load sharded model via index file
- **WHEN** a model directory contains `model.safetensors.index.json` plus multiple `model-NNNNN-of-MMMMM.safetensors` shards
- **THEN** the runtime SHALL parse the index to map each tensor name to its shard file
- **AND** open all shard files (eagerly mmap) at load time
- **AND** present a unified tensor namespace across all shards

#### Scenario: Sharded tensor lookup
- **WHEN** a tensor name is requested from a sharded model
- **THEN** the loader SHALL resolve which shard contains it via the index
- **AND** return a `TensorView` into that shard's mmap region

### Requirement: Memory-efficient weight loading
The safetensors loader SHALL avoid copying tensor data into Rust-owned memory before transfer to GPU.

#### Scenario: Direct mmap-to-GPU transfer
- **WHEN** loading weights for a GPU-resident model
- **THEN** the loader SHALL transfer each tensor from its mmap region directly to a `DeviceBuffer` via `cudaMemcpy`
- **AND** SHALL NOT allocate intermediate host-side `Tensor` storage for the tensor data
- **AND** SHALL release mmap regions after all tensors have been transferred to GPU

### Requirement: Supported safetensors dtypes
The loader SHALL recognize the dtype strings used in safetensors headers and map them to SmallAIOS `DataType` values.

#### Scenario: BF16 tensor recognition
- **WHEN** a tensor header has `"dtype": "BF16"`
- **THEN** the loader SHALL set the tensor's `DataType` to `BFloat16`
- **AND** treat the data as raw 2-byte BF16 values

#### Scenario: F16 tensor recognition
- **WHEN** a tensor header has `"dtype": "F16"`
- **THEN** the loader SHALL set the tensor's `DataType` to `Float16`

#### Scenario: F32 tensor recognition
- **WHEN** a tensor header has `"dtype": "F32"`
- **THEN** the loader SHALL set the tensor's `DataType` to `Float`

#### Scenario: Unsupported dtype error
- **WHEN** a tensor header has a dtype not yet supported (e.g. `F64`, `BOOL`, `U64`)
- **THEN** the loader SHALL return an error identifying the unsupported dtype
- **AND** SHALL NOT crash or silently coerce
