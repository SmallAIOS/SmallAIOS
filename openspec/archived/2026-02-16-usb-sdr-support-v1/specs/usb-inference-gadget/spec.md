# Delta for USB Inference Gadget

## ADDED Requirements

### Requirement: USB Inference Device Presentation
The USB inference gadget SHALL present SmallAIOS as a vendor-class USB device (class 0xFF) with bulk IN and bulk OUT endpoints for inference request/response.

#### Scenario: Host enumerates inference gadget
- WHEN a host PC connects to SmallAIOS via USB
- THEN the gadget MUST present a device descriptor with SmallAIOS vendor/product identification
- AND MUST expose one interface with class 0xFF (vendor-specific), one bulk OUT endpoint (requests), and one bulk IN endpoint (responses)

#### Scenario: Host reads product string
- WHEN the host requests the product string descriptor
- THEN the gadget MUST return "SmallAIOS Inference Engine" as a UTF-16LE string

### Requirement: Inference Request Protocol
The USB inference gadget SHALL accept inference requests on the bulk OUT endpoint using a binary framing protocol.

#### Scenario: Submit a valid inference request
- WHEN the host sends a request with format [4-byte request_id][2-byte model_name_len][model_name UTF-8][4-byte tensor_size][tensor_data]
- THEN the gadget MUST parse the request, locate the named ONNX model, and submit the input tensor for inference
- AND MUST queue the request for processing with the associated request_id

#### Scenario: Reject request with unknown model
- WHEN the host submits a request referencing a model name that is not loaded
- THEN the gadget MUST respond with the request_id, status code 0x0002 (MODEL_NOT_FOUND), and zero-length result

#### Scenario: Reject malformed request
- WHEN the host sends a request with model_name_len exceeding 256 bytes or tensor_size exceeding 256 MiB
- THEN the gadget MUST respond with the request_id, status code 0x0001 (INVALID_REQUEST), and zero-length result

### Requirement: Inference Response Protocol
The USB inference gadget SHALL return inference results on the bulk IN endpoint using a binary framing protocol.

#### Scenario: Return successful inference result
- WHEN ONNX inference completes for a queued request
- THEN the gadget MUST send a response with format [4-byte request_id][2-byte status (0x0000 = OK)][4-byte result_size][result tensor data]
- AND the result tensor MUST match the model's output tensor format

#### Scenario: Return inference error
- WHEN ONNX inference fails (runtime error, OOM, timeout)
- THEN the gadget MUST send a response with the request_id, status code 0x0003 (INFERENCE_ERROR), and zero-length result

#### Scenario: Handle concurrent requests
- WHEN the host submits multiple inference requests before receiving responses
- THEN the gadget MUST process them in order and return responses with matching request_ids
- AND MUST support at least 4 outstanding requests

### Requirement: Zenoh Bridge for USB Inference
The USB inference gadget SHALL bridge inference requests to the Zenoh IPC layer, making USB-submitted requests indistinguishable from network-submitted requests.

#### Scenario: Publish USB inference request to Zenoh
- WHEN the gadget receives an inference request for model "mobilenet_v2"
- THEN the gadget MUST publish the input tensor to Zenoh key expression `usb/inference/mobilenet_v2`
- AND the existing ONNX runtime subscriber MUST process the request identically to TCP/QUIC-submitted requests

#### Scenario: Receive inference result from Zenoh
- WHEN the ONNX runtime publishes an inference result on the corresponding Zenoh key expression
- THEN the gadget MUST receive the result and format it as a USB response for the host

### Requirement: DMA Integration for Tensor Data
The USB inference gadget SHALL use DMA for transferring tensor data between USB endpoints and inference memory to minimize CPU overhead.

#### Scenario: Zero-copy DMA for large input tensors
- WHEN the host submits an input tensor larger than 4096 bytes
- THEN the gadget SHOULD use DMA to transfer the tensor data directly from the USB endpoint buffer to inference-accessible memory
- AND MUST NOT require the CPU to copy the tensor data byte-by-byte
