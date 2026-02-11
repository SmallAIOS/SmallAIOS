// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Package provider implements the Virtual Kubelet PodLifecycleHandler for SmallAIOS.
//
// It translates Kubernetes PodSpecs into SmallAIOS ONNX inference sessions,
// communicating with SmallAIOS instances through the Zenoh management API.
package provider

import (
	"context"
	"fmt"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Config holds the configuration for the SmallAIOS provider.
type Config struct {
	// NodeName is the Kubernetes node name for this virtual kubelet.
	NodeName string
	// ZenohAddr is the Zenoh router address (e.g. "tcp/127.0.0.1:7447").
	ZenohAddr string
	// CPUMillis is the advertised CPU capacity in millicores.
	CPUMillis int64
	// MemoryBytes is the advertised memory capacity in bytes.
	MemoryBytes int64
	// GPUCount is the number of NVIDIA GPUs to advertise.
	GPUCount int64
	// MetricsAddr is the address for Prometheus metrics passthrough.
	MetricsAddr string
	// HealthAddr is the address for health probe proxy.
	HealthAddr string
}

// Provider implements the virtual-kubelet PodLifecycleHandler interface,
// managing SmallAIOS ONNX inference workloads as Kubernetes pods.
type Provider struct {
	mu       sync.RWMutex
	config   Config
	zenoh    *ZenohClient
	pods     map[string]*PodState
	node     *NodeAdvertiser
	health   *HealthProber
	metrics  *MetricsProxy
}

// New creates a new SmallAIOS provider with the given configuration.
func New(cfg Config) (*Provider, error) {
	if cfg.NodeName == "" {
		return nil, fmt.Errorf("node name must not be empty")
	}
	if cfg.ZenohAddr == "" {
		return nil, fmt.Errorf("zenoh address must not be empty")
	}

	zenoh := NewZenohClient(cfg.ZenohAddr)
	node := NewNodeAdvertiser(cfg)
	health := NewHealthProber(zenoh)
	metrics := NewMetricsProxy(zenoh)

	return &Provider{
		config:  cfg,
		zenoh:   zenoh,
		pods:    make(map[string]*PodState),
		node:    node,
		health:  health,
		metrics: metrics,
	}, nil
}

// podKey returns a unique key for a pod (namespace/name).
func podKey(namespace, name string) string {
	return namespace + "/" + name
}

// CreatePod accepts a Pod definition and translates it into a SmallAIOS
// inference session via the Zenoh management API.
func (p *Provider) CreatePod(ctx context.Context, pod *corev1.Pod) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := podKey(pod.Namespace, pod.Name)
	if _, exists := p.pods[key]; exists {
		return fmt.Errorf("pod %s already exists", key)
	}

	spec := TranslatePodSpec(pod)

	if err := p.zenoh.Deploy(ctx, spec); err != nil {
		return fmt.Errorf("deploy failed for %s: %w", key, err)
	}

	p.pods[key] = &PodState{
		Pod:   pod.DeepCopy(),
		Phase: PodPhasePending,
		Spec:  spec,
	}

	return nil
}

// UpdatePod updates an existing pod (re-deploys with new spec).
func (p *Provider) UpdatePod(ctx context.Context, pod *corev1.Pod) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := podKey(pod.Namespace, pod.Name)
	state, exists := p.pods[key]
	if !exists {
		return fmt.Errorf("pod %s not found", key)
	}

	spec := TranslatePodSpec(pod)
	state.Pod = pod.DeepCopy()
	state.Spec = spec

	return nil
}

// DeletePod stops and removes a SmallAIOS inference session.
func (p *Provider) DeletePod(ctx context.Context, pod *corev1.Pod) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := podKey(pod.Namespace, pod.Name)
	state, exists := p.pods[key]
	if !exists {
		return fmt.Errorf("pod %s not found", key)
	}

	if err := p.zenoh.Undeploy(ctx, state.Spec.WorkloadID, false); err != nil {
		return fmt.Errorf("undeploy failed for %s: %w", key, err)
	}

	delete(p.pods, key)
	return nil
}

// GetPod returns a pod by name and namespace.
func (p *Provider) GetPod(_ context.Context, namespace, name string) (*corev1.Pod, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	key := podKey(namespace, name)
	state, exists := p.pods[key]
	if !exists {
		return nil, fmt.Errorf("pod %s not found", key)
	}

	return state.Pod.DeepCopy(), nil
}

// GetPodStatus returns the status of a pod.
func (p *Provider) GetPodStatus(_ context.Context, namespace, name string) (*corev1.PodStatus, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	key := podKey(namespace, name)
	state, exists := p.pods[key]
	if !exists {
		return nil, fmt.Errorf("pod %s not found", key)
	}

	return state.KubernetesPodStatus(), nil
}

// GetPods returns all pods managed by this provider.
func (p *Provider) GetPods(_ context.Context) ([]*corev1.Pod, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pods := make([]*corev1.Pod, 0, len(p.pods))
	for _, state := range p.pods {
		pods = append(pods, state.Pod.DeepCopy())
	}
	return pods, nil
}

// ConfigureNode sets the node's capacity, conditions, and labels.
func (p *Provider) ConfigureNode(_ context.Context, node *corev1.Node) {
	p.node.ConfigureNode(node)
}

// NodeCapacity returns the resource capacity for the virtual node.
func (p *Provider) NodeCapacity() corev1.ResourceList {
	resources := corev1.ResourceList{
		corev1.ResourceCPU:    *resource.NewMilliQuantity(p.config.CPUMillis, resource.DecimalSI),
		corev1.ResourceMemory: *resource.NewQuantity(p.config.MemoryBytes, resource.BinarySI),
		corev1.ResourcePods:   *resource.NewQuantity(32, resource.DecimalSI),
	}
	if p.config.GPUCount > 0 {
		resources[corev1.ResourceName("nvidia.com/gpu")] = *resource.NewQuantity(p.config.GPUCount, resource.DecimalSI)
	}
	return resources
}

// NodeConditions returns the node conditions for the virtual node.
func (p *Provider) NodeConditions(_ context.Context) []corev1.NodeCondition {
	now := metav1.Now()
	return []corev1.NodeCondition{
		{
			Type:               corev1.NodeReady,
			Status:             corev1.ConditionTrue,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSReady",
			Message:            "SmallAIOS inference engine is operational",
		},
		{
			Type:               corev1.NodeMemoryPressure,
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSNoMemoryPressure",
			Message:            "SmallAIOS memory is within bounds",
		},
		{
			Type:               corev1.NodeDiskPressure,
			Status:             corev1.ConditionFalse,
			LastHeartbeatTime:  now,
			LastTransitionTime: now,
			Reason:             "SmallAIOSNoDiskPressure",
			Message:            "SmallAIOS is a unikernel with no disk",
		},
	}
}

// NodeAddresses returns the addresses advertised for this virtual node.
func (p *Provider) NodeAddresses(_ context.Context) []corev1.NodeAddress {
	return []corev1.NodeAddress{
		{
			Type:    corev1.NodeInternalIP,
			Address: "127.0.0.1",
		},
	}
}
