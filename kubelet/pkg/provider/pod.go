// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// PodPhase represents the SmallAIOS-internal workload phase.
type PodPhase string

const (
	// PodPhasePending means the workload has been accepted but is not yet running.
	PodPhasePending PodPhase = "Pending"
	// PodPhaseLoading means the ONNX model is being downloaded/loaded.
	PodPhaseLoading PodPhase = "Loading"
	// PodPhaseRunning means the workload is actively serving inference.
	PodPhaseRunning PodPhase = "Running"
	// PodPhaseSucceeded means batch inference completed successfully.
	PodPhaseSucceeded PodPhase = "Succeeded"
	// PodPhaseFailed means the workload terminated with an error.
	PodPhaseFailed PodPhase = "Failed"
	// PodPhaseTerminating means the workload is being shut down.
	PodPhaseTerminating PodPhase = "Terminating"
)

// SmallAIOSDeploySpec holds the translated workload specification sent
// to SmallAIOS via the Zenoh management API.
type SmallAIOSDeploySpec struct {
	// WorkloadID maps to the K8s pod UID.
	WorkloadID string
	// ModelURL is the ONNX model URL extracted from the container image.
	ModelURL string
	// Name is the K8s pod name.
	Name string
	// Namespace is the K8s namespace.
	Namespace string
	// Env holds configuration key-value pairs.
	Env map[string]string
	// CPUMillis is the requested CPU in millicores.
	CPUMillis int64
	// MemoryBytes is the requested memory in bytes.
	MemoryBytes int64
	// GPURequired indicates whether the workload needs a GPU.
	GPURequired bool
}

// PodState tracks the lifecycle of a SmallAIOS workload mapped from a K8s pod.
type PodState struct {
	// Pod is the original Kubernetes pod spec.
	Pod *corev1.Pod
	// Phase is the current SmallAIOS workload phase.
	Phase PodPhase
	// Spec is the translated SmallAIOS deploy specification.
	Spec SmallAIOSDeploySpec
	// Message is a human-readable status message.
	Message string
	// StartTime is when the workload was created.
	StartTime time.Time
	// InferenceCount is the number of inferences served.
	InferenceCount uint64
}

// TranslatePodSpec converts a Kubernetes PodSpec into a SmallAIOS deploy
// specification. The translation follows these rules:
//
//   - Container image -> ONNX model URL (the image field contains the model URL)
//   - Container env vars -> SmallAIOS configuration
//   - Resource requests/limits -> SmallAIOS resource allocation
//   - GPU resource requests -> SmallAIOS GPU flag
func TranslatePodSpec(pod *corev1.Pod) SmallAIOSDeploySpec {
	spec := SmallAIOSDeploySpec{
		WorkloadID: string(pod.UID),
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Env:        make(map[string]string),
	}

	// Use the first container's image as the ONNX model URL.
	// In SmallAIOS, the "container image" is actually the ONNX model location.
	if len(pod.Spec.Containers) > 0 {
		container := pod.Spec.Containers[0]
		spec.ModelURL = translateImageToModelURL(container.Image)

		// Translate environment variables.
		for _, env := range container.Env {
			spec.Env[env.Name] = env.Value
		}

		// Translate resource requests.
		if cpu, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
			spec.CPUMillis = cpu.MilliValue()
		}
		if mem, ok := container.Resources.Requests[corev1.ResourceMemory]; ok {
			spec.MemoryBytes = mem.Value()
		}

		// Check for GPU resource request.
		gpuRes := corev1.ResourceName("nvidia.com/gpu")
		if gpu, ok := container.Resources.Limits[gpuRes]; ok {
			spec.GPURequired = gpu.Value() > 0
		}
	}

	return spec
}

// translateImageToModelURL converts a container image reference to an ONNX
// model URL. If the image looks like a URL (starts with http:// or https://),
// it is used directly. Otherwise it is treated as an OCI reference that
// should be fetched from a model registry.
func translateImageToModelURL(image string) string {
	if strings.HasPrefix(image, "http://") || strings.HasPrefix(image, "https://") {
		return image
	}
	// For OCI-style references, construct a model registry URL.
	// e.g. "models.example.com/resnet50:v1" -> "oci://models.example.com/resnet50:v1"
	return fmt.Sprintf("oci://%s", image)
}

// KubernetesPhase maps the SmallAIOS workload phase to a K8s PodPhase.
func (ps *PodState) KubernetesPhase() corev1.PodPhase {
	switch ps.Phase {
	case PodPhasePending, PodPhaseLoading:
		return corev1.PodPending
	case PodPhaseRunning, PodPhaseTerminating:
		return corev1.PodRunning
	case PodPhaseSucceeded:
		return corev1.PodSucceeded
	case PodPhaseFailed:
		return corev1.PodFailed
	default:
		return corev1.PodUnknown
	}
}

// KubernetesPodStatus builds a K8s PodStatus from the current PodState.
func (ps *PodState) KubernetesPodStatus() *corev1.PodStatus {
	now := metav1.Now()
	phase := ps.KubernetesPhase()

	containerState := corev1.ContainerState{}
	switch ps.Phase {
	case PodPhasePending, PodPhaseLoading:
		containerState.Waiting = &corev1.ContainerStateWaiting{
			Reason:  string(ps.Phase),
			Message: ps.Message,
		}
	case PodPhaseRunning, PodPhaseTerminating:
		startTime := metav1.NewTime(ps.StartTime)
		containerState.Running = &corev1.ContainerStateRunning{
			StartedAt: startTime,
		}
	case PodPhaseSucceeded:
		containerState.Terminated = &corev1.ContainerStateTerminated{
			ExitCode:   0,
			Reason:     "Completed",
			Message:    ps.Message,
			FinishedAt: now,
		}
	case PodPhaseFailed:
		containerState.Terminated = &corev1.ContainerStateTerminated{
			ExitCode:   1,
			Reason:     "Error",
			Message:    ps.Message,
			FinishedAt: now,
		}
	}

	containerStatuses := make([]corev1.ContainerStatus, 0)
	if len(ps.Pod.Spec.Containers) > 0 {
		containerStatuses = append(containerStatuses, corev1.ContainerStatus{
			Name:  ps.Pod.Spec.Containers[0].Name,
			Image: ps.Pod.Spec.Containers[0].Image,
			Ready: ps.Phase == PodPhaseRunning,
			State: containerState,
		})
	}

	return &corev1.PodStatus{
		Phase:             phase,
		Message:           ps.Message,
		ContainerStatuses: containerStatuses,
	}
}
