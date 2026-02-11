// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
)

// HealthProber proxies Kubernetes liveness and readiness probes to the
// SmallAIOS /health endpoint via the Zenoh management API.
type HealthProber struct {
	zenoh *ZenohClient
}

// NewHealthProber creates a new health prober backed by the given Zenoh client.
func NewHealthProber(zenoh *ZenohClient) *HealthProber {
	return &HealthProber{zenoh: zenoh}
}

// HealthStatus represents the response from SmallAIOS health endpoint.
type HealthStatus struct {
	// Healthy indicates whether the SmallAIOS instance is operational.
	Healthy bool
	// Message provides human-readable status details.
	Message string
	// UptimeSecs is the instance uptime in seconds.
	UptimeSecs uint64
	// Version is the SmallAIOS kernel version.
	Version string
}

// CheckHealth queries the SmallAIOS health endpoint for a given workload.
func (hp *HealthProber) CheckHealth(_ context.Context, _ string) (*HealthStatus, error) {
	if !hp.zenoh.IsConnected() {
		return nil, fmt.Errorf("zenoh client not connected")
	}
	// Stub: real implementation queries sai/health via Zenoh and parses response.
	return &HealthStatus{
		Healthy:    true,
		Message:    "SmallAIOS operational",
		UptimeSecs: 3600,
		Version:    "0.1.0",
	}, nil
}

// CheckLiveness performs a Kubernetes liveness probe by checking the
// SmallAIOS instance's health endpoint.
func (hp *HealthProber) CheckLiveness(ctx context.Context, pod *corev1.Pod) (bool, error) {
	probe := findProbe(pod, true)
	if probe == nil {
		// No liveness probe configured, assume healthy.
		return true, nil
	}

	ctx, cancel := context.WithTimeout(ctx, time.Duration(probe.TimeoutSeconds)*time.Second)
	defer cancel()

	status, err := hp.CheckHealth(ctx, string(pod.UID))
	if err != nil {
		return false, err
	}
	return status.Healthy, nil
}

// CheckReadiness performs a Kubernetes readiness probe by checking
// whether the SmallAIOS workload is ready to serve inference.
func (hp *HealthProber) CheckReadiness(ctx context.Context, pod *corev1.Pod) (bool, error) {
	probe := findProbe(pod, false)
	if probe == nil {
		// No readiness probe configured, assume ready.
		return true, nil
	}

	ctx, cancel := context.WithTimeout(ctx, time.Duration(probe.TimeoutSeconds)*time.Second)
	defer cancel()

	status, err := hp.CheckHealth(ctx, string(pod.UID))
	if err != nil {
		return false, err
	}
	return status.Healthy, nil
}

// findProbe extracts the liveness (liveness=true) or readiness (liveness=false)
// probe from the first container in the pod spec.
func findProbe(pod *corev1.Pod, liveness bool) *corev1.Probe {
	if len(pod.Spec.Containers) == 0 {
		return nil
	}
	if liveness {
		return pod.Spec.Containers[0].LivenessProbe
	}
	return pod.Spec.Containers[0].ReadinessProbe
}
