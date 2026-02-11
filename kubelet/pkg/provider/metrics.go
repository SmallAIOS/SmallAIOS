// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"context"
	"fmt"
	"strings"
)

// MetricsProxy proxies Prometheus /metrics requests from Kubernetes
// monitoring to SmallAIOS instances via the Zenoh management API.
type MetricsProxy struct {
	zenoh *ZenohClient
}

// NewMetricsProxy creates a new metrics proxy backed by the given Zenoh client.
func NewMetricsProxy(zenoh *ZenohClient) *MetricsProxy {
	return &MetricsProxy{zenoh: zenoh}
}

// MetricEntry represents a single Prometheus metric.
type MetricEntry struct {
	// Name is the metric name (e.g. "smallaios_inference_count").
	Name string
	// Labels are the Prometheus labels.
	Labels map[string]string
	// Value is the metric value.
	Value float64
	// Type is "counter", "gauge", or "histogram".
	Type string
}

// FetchMetrics retrieves Prometheus metrics from a SmallAIOS instance
// for the given workload.
func (mp *MetricsProxy) FetchMetrics(_ context.Context, workloadID string) ([]MetricEntry, error) {
	if !mp.zenoh.IsConnected() {
		return nil, fmt.Errorf("zenoh client not connected")
	}

	// Stub: real implementation queries sai/metrics via Zenoh,
	// parses the response, and returns structured metrics.
	return []MetricEntry{
		{
			Name:   "smallaios_inference_total",
			Labels: map[string]string{"workload_id": workloadID},
			Value:  0,
			Type:   "counter",
		},
		{
			Name:   "smallaios_inference_latency_seconds",
			Labels: map[string]string{"workload_id": workloadID},
			Value:  0.001,
			Type:   "gauge",
		},
		{
			Name:   "smallaios_memory_used_bytes",
			Labels: map[string]string{"workload_id": workloadID},
			Value:  0,
			Type:   "gauge",
		},
	}, nil
}

// FormatPrometheus formats metrics into Prometheus text exposition format.
func FormatPrometheus(metrics []MetricEntry) string {
	var sb strings.Builder
	for _, m := range metrics {
		// TYPE line
		sb.WriteString(fmt.Sprintf("# TYPE %s %s\n", m.Name, m.Type))
		// Metric line
		sb.WriteString(m.Name)
		if len(m.Labels) > 0 {
			sb.WriteString("{")
			first := true
			for k, v := range m.Labels {
				if !first {
					sb.WriteString(",")
				}
				sb.WriteString(fmt.Sprintf(`%s="%s"`, k, v))
				first = false
			}
			sb.WriteString("}")
		}
		sb.WriteString(fmt.Sprintf(" %g\n", m.Value))
	}
	return sb.String()
}
