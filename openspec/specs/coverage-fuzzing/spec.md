# coverage-fuzzing Specification

## Purpose
TBD - created by archiving change test-coverage-v1. Update Purpose after archive.
## Requirements
### Requirement: ONNX Protobuf Fuzz Target
A fuzz target SHALL exercise the ONNX protobuf parser with arbitrary byte sequences to ensure it never panics or causes undefined behavior.

#### Scenario: Malformed protobuf input
- GIVEN the ONNX protobuf parser entry point
- WHEN arbitrary bytes of length 0 to 65536 are fed as input
- THEN the parser MUST either return a valid parse result or a well-formed error
- AND the parser MUST NOT panic, abort, or trigger undefined behavior

#### Scenario: Truncated model data
- GIVEN a valid ONNX model serialized as protobuf
- WHEN the serialized data is truncated at every possible byte offset
- THEN the parser MUST return a parse error for each truncation
- AND no truncation MUST cause a panic

### Requirement: TCP Packet Fuzz Target
A fuzz target SHALL exercise the TCP packet parser with arbitrary byte sequences.

#### Scenario: Malformed TCP packets
- GIVEN the TCP packet parsing function from the net crate
- WHEN arbitrary bytes are provided as a raw TCP packet
- THEN the parser MUST either return a valid TCP header or an error
- AND the parser MUST NOT panic on any input
- AND the parser MUST NOT read beyond the provided input buffer

#### Scenario: Malformed UDP packets
- GIVEN the UDP packet parsing function from the net crate
- WHEN arbitrary bytes are provided as a raw UDP packet
- THEN the parser MUST either return a valid UDP header or an error
- AND the parser MUST NOT panic on any input

### Requirement: USB Descriptor Fuzz Target
A fuzz target SHALL exercise the USB descriptor parser with arbitrary byte sequences.

#### Scenario: Malformed USB descriptors
- GIVEN the USB descriptor parsing function from the usb crate
- WHEN arbitrary bytes are provided as a USB descriptor chain
- THEN the parser MUST either return valid descriptors or an error
- AND the parser MUST NOT panic on any input
- AND the parser MUST handle zero-length descriptors, oversized length fields, and unknown descriptor types

### Requirement: IPC Message Fuzz Target
A fuzz target SHALL exercise the IPC message deserialization path with arbitrary byte sequences.

#### Scenario: Malformed IPC messages
- GIVEN the IPC message deserialization function from the ipc crate
- WHEN arbitrary bytes are provided as a serialized message
- THEN the deserializer MUST either return a valid message or an error
- AND the deserializer MUST NOT panic on any input

### Requirement: ONNX Tensor Data Fuzz Target
A fuzz target SHALL exercise the ONNX tensor creation and validation paths with arbitrary data.

#### Scenario: Malformed tensor shapes and data
- GIVEN the tensor construction function from the onnx-rt crate
- WHEN arbitrary bytes are interpreted as shape dimensions and tensor data
- THEN the constructor MUST validate shape/data size consistency
- AND the constructor MUST NOT panic on mismatched shape and data lengths
- AND overflow in shape dimension multiplication MUST be detected and returned as an error

### Requirement: Fuzz Corpus Management
All fuzz targets SHALL maintain a seed corpus in the repository.

#### Scenario: Seed corpus for each target
- GIVEN a fuzz target directory in fuzz/corpus/<target>/
- THEN the directory MUST contain at least one valid seed input
- AND the seed inputs SHOULD include both valid and near-valid examples
- AND the corpus MUST be committed to the repository

### Requirement: Fuzz CI Integration
Fuzz targets SHALL run in CI with a bounded time budget.

#### Scenario: CI fuzz execution
- GIVEN the CI pipeline runs on a PR or push
- WHEN the fuzzing job executes
- THEN each fuzz target MUST run for at least 30 seconds and at most 60 seconds
- AND any panic or crash MUST cause the CI job to fail
- AND the failing input MUST be reported in the CI output

