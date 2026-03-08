# Delta for Kubernetes Integration

## ADDED Requirements

### Requirement: Virtual Kubelet Provider
The system SHALL include a Virtual Kubelet provider implemented in Go that runs on a Linux control node and presents SmallAIOS instances as Kubernetes nodes.

#### Scenario: Register as a Kubernetes node
- WHEN the Virtual Kubelet provider starts on a Linux control node
- THEN it MUST register itself as a node in the Kubernetes API server
- AND the node MUST appear with a SmallAIOS-specific taint to prevent unrelated pod scheduling
- AND the node status MUST report as Ready when at least one SmallAIOS instance is reachable

#### Scenario: Provider manages multiple SmallAIOS instances
- WHEN the Virtual Kubelet provider is configured with multiple SmallAIOS instance endpoints
- THEN it MUST register each instance as a distinct virtual node in the Kubernetes cluster
- AND each virtual node MUST independently report its own resources and health status

### Requirement: SmallAIOS Management API over Zenoh TCP
The SmallAIOS instance SHALL expose a management API over Zenoh TCP transport for the Virtual Kubelet provider to control model deployment and query status.

#### Scenario: Deploy a model
- WHEN the provider sends a deploy request with a model URL and configuration
- THEN the SmallAIOS instance MUST download the ONNX model, load it into the inference runtime, and report deployment status
- AND if the model URL is unreachable or the model is invalid, the instance MUST return a descriptive error

#### Scenario: Undeploy a model
- WHEN the provider sends an undeploy request for a loaded model
- THEN the SmallAIOS instance MUST unload the model, release associated memory, and confirm removal
- AND the instance MUST reject undeploy requests for models that are not currently loaded

#### Scenario: Query instance status
- WHEN the provider sends a status request
- THEN the SmallAIOS instance MUST return its current state including loaded models, uptime, and resource utilization
- AND the response MUST be delivered within 500 ms under normal operating conditions

#### Scenario: Update model configuration
- WHEN the provider sends a config request with updated parameters (e.g., batch size, execution provider)
- THEN the SmallAIOS instance MUST apply the configuration to the specified model
- AND if the configuration is invalid, the instance MUST reject it and retain the previous configuration

### Requirement: Node Resource Reporting
The Virtual Kubelet provider SHALL report accurate resource capacity and utilization for each SmallAIOS instance to the Kubernetes API server.

#### Scenario: Report CPU and memory resources
- WHEN the Kubernetes scheduler queries the virtual node's allocatable resources
- THEN the provider MUST report the number of CPU cores and total memory available on the SmallAIOS instance
- AND the reported values MUST reflect the actual hardware capacity

#### Scenario: Report GPU resources
- WHEN a SmallAIOS instance has one or more NVIDIA GPUs
- THEN the provider MUST report the GPU count as an extended resource (nvidia.com/gpu)
- AND the GPU memory MUST be included in the node status annotations

#### Scenario: Report loaded models
- WHEN the Kubernetes API server queries node status
- THEN the provider MUST include the list of currently loaded ONNX models in the node annotations
- AND each model entry MUST include the model name, size, and current state (loading, ready, error)

### Requirement: Pod Spec Translation
The Virtual Kubelet provider SHALL translate Kubernetes pod specifications into SmallAIOS model deployment operations.

#### Scenario: Translate container image to ONNX model URL
- WHEN a pod is scheduled on a SmallAIOS virtual node with a container image reference
- THEN the provider MUST interpret the container image field as an ONNX model URL or OCI artifact reference
- AND the provider MUST pass the resolved URL to the SmallAIOS deploy API

#### Scenario: Translate environment variables to configuration
- WHEN a pod spec includes environment variables
- THEN the provider MUST map them to SmallAIOS model configuration parameters
- AND recognized variables (e.g., BATCH_SIZE, EXECUTION_PROVIDER) MUST be applied to the model configuration
- AND unrecognized variables MUST be passed through as opaque key-value metadata

#### Scenario: Translate resource requests to scheduling constraints
- WHEN a pod spec includes resource requests or limits (CPU, memory, nvidia.com/gpu)
- THEN the provider MUST verify the SmallAIOS instance has sufficient resources before deploying
- AND if resources are insufficient, the provider MUST reject the pod with a clear reason

### Requirement: Pod Lifecycle Management
The Virtual Kubelet provider SHALL manage pod lifecycle transitions corresponding to SmallAIOS model deployment states.

#### Scenario: Pod transitions to Running
- WHEN a model deployment succeeds on the SmallAIOS instance
- THEN the provider MUST update the pod status to Running with a ready condition
- AND the pod's start time MUST reflect when the model became ready for inference

#### Scenario: Pod transitions to Failed
- WHEN a model deployment fails (download error, OOM, invalid model)
- THEN the provider MUST update the pod status to Failed with a descriptive reason and message
- AND the provider MUST NOT automatically retry unless the pod's restart policy requires it

#### Scenario: Pod transitions to Succeeded
- WHEN a model is explicitly undeployed via pod deletion
- THEN the provider MUST undeploy the model on the SmallAIOS instance
- AND the provider MUST update the pod phase to Succeeded and remove the pod from the node

#### Scenario: Pod in Pending state
- WHEN a pod is first scheduled on the SmallAIOS virtual node
- THEN the provider MUST set the pod status to Pending
- AND the pod MUST remain Pending until the SmallAIOS instance confirms model loading has begun

### Requirement: K3s Edge Deployment Support
The Virtual Kubelet provider SHALL support K3s lightweight Kubernetes distributions for edge deployments on resource-constrained hardware.

#### Scenario: Provider joins K3s cluster
- WHEN the Virtual Kubelet provider is deployed alongside a K3s server or agent
- THEN it MUST register with the K3s API server using the same kubeconfig mechanism as standard K3s agents
- AND the provider MUST function correctly with K3s on Jetson Orin Nano and Raspberry Pi platforms

#### Scenario: Edge resource constraints
- WHEN running on an edge K3s cluster with limited bandwidth
- THEN the provider MUST minimize API server communication to essential status updates and pod events
- AND model download URLs MUST support resumable transfers to handle intermittent connectivity

### Requirement: K8s Datacenter Deployment Support
The Virtual Kubelet provider SHALL support standard Kubernetes distributions for datacenter deployments.

#### Scenario: Provider joins K8s cluster
- WHEN the Virtual Kubelet provider is deployed in a standard Kubernetes cluster
- THEN it MUST register with the Kubernetes API server using standard kubeconfig or in-cluster service account credentials
- AND the provider MUST function correctly with SmallAIOS instances running on DGX Spark and Intel Xeon platforms

#### Scenario: Multi-node datacenter deployment
- WHEN multiple SmallAIOS instances run across datacenter hardware
- THEN the provider MUST present each as a schedulable node
- AND the Kubernetes scheduler MUST be able to place pods based on GPU availability and resource capacity

### Requirement: Health Probe Integration
The Virtual Kubelet provider SHALL integrate SmallAIOS health status with Kubernetes liveness probes.

#### Scenario: Liveness probe via /health endpoint
- WHEN Kubernetes performs a liveness check on a pod running on a SmallAIOS virtual node
- THEN the provider MUST query the SmallAIOS instance's existing /health endpoint
- AND the provider MUST translate the health response to a Kubernetes probe result (success or failure)

#### Scenario: Instance unreachable
- WHEN the provider cannot reach the SmallAIOS instance's /health endpoint
- THEN the provider MUST mark the node condition as NotReady
- AND all pods on that node MUST have their ready condition set to false

### Requirement: Metrics Integration
The Virtual Kubelet provider SHALL expose SmallAIOS metrics through the existing Prometheus /metrics endpoint for Kubernetes monitoring integration.

#### Scenario: Metrics endpoint discovery
- WHEN a Prometheus instance scrapes metrics from the SmallAIOS virtual node
- THEN the provider MUST proxy or redirect to the SmallAIOS instance's /metrics endpoint
- AND the metrics MUST include inference latency, throughput, model load status, and resource utilization

#### Scenario: Metrics availability during model transitions
- WHEN a model is being loaded or unloaded
- THEN the /metrics endpoint MUST remain available and responsive
- AND transitional states MUST be reflected in the metric labels (e.g., model_status="loading")

### Requirement: GPU Resource Advertisement
The Virtual Kubelet provider SHALL advertise NVIDIA GPU resources using the standard Kubernetes extended resource mechanism.

#### Scenario: Advertise nvidia.com/gpu extended resource
- WHEN a SmallAIOS instance has NVIDIA GPUs available
- THEN the provider MUST report nvidia.com/gpu with the count of available GPUs in the node's allocatable resources
- AND pods requesting nvidia.com/gpu resources MUST be schedulable on the virtual node

#### Scenario: GPU resource accounting
- WHEN a pod is deployed that requests nvidia.com/gpu resources
- THEN the provider MUST decrement the available GPU count in the node status
- AND when the pod is removed, the provider MUST restore the GPU count

### Requirement: Certification Boundary
The Virtual Kubelet provider SHALL operate outside the safety-critical certification boundary of SmallAIOS.

#### Scenario: Provider failure does not affect SmallAIOS operation
- WHEN the Virtual Kubelet provider crashes or becomes unavailable
- THEN all running SmallAIOS instances MUST continue operating and serving inference requests
- AND the SmallAIOS instances MUST NOT depend on the provider for runtime correctness

#### Scenario: Provider excluded from safety certification
- WHEN safety certification artifacts are produced (DO-178C, ISO 26262)
- THEN the Virtual Kubelet provider code MUST NOT be included in the certification scope
- AND the certification boundary MUST be documented as ending at the SmallAIOS management API interface
