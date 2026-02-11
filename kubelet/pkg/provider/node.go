// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NodeAdvertiser manages the Kubernetes node resource advertisement for
// a SmallAIOS virtual node. It reports CPU, memory, and GPU capacity
// as Kubernetes extended resources.
type NodeAdvertiser struct {
	config Config
}

// NewNodeAdvertiser creates a new node advertiser with the given config.
func NewNodeAdvertiser(cfg Config) *NodeAdvertiser {
	return &NodeAdvertiser{config: cfg}
}

// ConfigureNode sets the node's capacity, allocatable resources, conditions,
// and SmallAIOS-specific labels and taints.
func (n *NodeAdvertiser) ConfigureNode(node *corev1.Node) {
	// Set labels to identify this as a SmallAIOS virtual node.
	if node.Labels == nil {
		node.Labels = make(map[string]string)
	}
	node.Labels["type"] = "virtual-kubelet"
	node.Labels["kubernetes.io/role"] = "agent"
	node.Labels["smallaios.io/os"] = "smallaios"
	node.Labels["smallaios.io/arch"] = "amd64"

	// Add a taint so only SmallAIOS-targeted pods schedule here.
	node.Spec.Taints = []corev1.Taint{
		{
			Key:    "virtual-kubelet.io/provider",
			Value:  "smallaios",
			Effect: corev1.TaintEffectNoSchedule,
		},
	}

	// Advertise capacity.
	capacity := corev1.ResourceList{
		corev1.ResourceCPU:    *resource.NewMilliQuantity(n.config.CPUMillis, resource.DecimalSI),
		corev1.ResourceMemory: *resource.NewQuantity(n.config.MemoryBytes, resource.BinarySI),
		corev1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
	}
	if n.config.GPUCount > 0 {
		capacity[corev1.ResourceName("nvidia.com/gpu")] = *resource.NewQuantity(n.config.GPUCount, resource.DecimalSI)
	}

	node.Status.Capacity = capacity
	node.Status.Allocatable = capacity.DeepCopy()

	// Set node conditions.
	now := metav1.Now()
	node.Status.Conditions = []corev1.NodeCondition{
		{
			Type:               corev1.NodeReady,
			Status:             corev1.ConditionTrue,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSReady",
			Message:            "SmallAIOS inference engine operational",
		},
		{
			Type:               corev1.NodeMemoryPressure,
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSNoMemoryPressure",
			Message:            "Memory within bounds",
		},
		{
			Type:               corev1.NodeDiskPressure,
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSNoDiskPressure",
			Message:            "SmallAIOS unikernel has no disk",
		},
		{
			Type:               corev1.NodeNetworkUnavailable,
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSNetworkReady",
			Message:            "Network stack operational",
		},
	}

	// Node info.
	node.Status.NodeInfo = corev1.NodeSystemInfo{
		OperatingSystem: "smallaios",
		Architecture:    "amd64",
		KubeletVersion:  "v0.1.0-smallaios",
	}
}

// Capacity returns the node's resource capacity.
func (n *NodeAdvertiser) Capacity() corev1.ResourceList {
	resources := corev1.ResourceList{
		corev1.ResourceCPU:    *resource.NewMilliQuantity(n.config.CPUMillis, resource.DecimalSI),
		corev1.ResourceMemory: *resource.NewQuantity(n.config.MemoryBytes, resource.BinarySI),
		corev1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
	}
	if n.config.GPUCount > 0 {
		resources[corev1.ResourceName("nvidia.com/gpu")] = *resource.NewQuantity(n.config.GPUCount, resource.DecimalSI)
	}
	return resources
}
