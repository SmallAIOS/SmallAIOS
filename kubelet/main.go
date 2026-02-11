// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

// SmallAIOS Virtual Kubelet Provider
//
// Registers SmallAIOS instances as Kubernetes virtual nodes, translating
// K8s PodSpecs into SmallAIOS ONNX inference sessions via the Zenoh
// management API.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/SmallAIOS/smallaios-kubelet/pkg/provider"
)

var (
	nodeName     = flag.String("nodename", "smallaios-node", "Kubernetes node name for this virtual kubelet")
	zenohAddr    = flag.String("zenoh-addr", "tcp/127.0.0.1:7447", "Zenoh router address for SmallAIOS management API")
	cpuMillis    = flag.Int64("cpu", 4000, "Advertised CPU capacity in millicores")
	memoryBytes  = flag.Int64("memory", 8589934592, "Advertised memory capacity in bytes")
	gpuCount     = flag.Int64("gpu", 0, "Advertised NVIDIA GPU count")
	metricsAddr  = flag.String("metrics-addr", ":10255", "Address for Prometheus metrics endpoint")
	healthAddr   = flag.String("health-addr", ":10256", "Address for health probe endpoint")
	version      = "0.1.0"
)

func main() {
	flag.Parse()

	cfg := provider.Config{
		NodeName:    *nodeName,
		ZenohAddr:   *zenohAddr,
		CPUMillis:   *cpuMillis,
		MemoryBytes: *memoryBytes,
		GPUCount:    *gpuCount,
		MetricsAddr: *metricsAddr,
		HealthAddr:  *healthAddr,
	}

	fmt.Fprintf(os.Stdout, "SmallAIOS Virtual Kubelet v%s\n", version)
	fmt.Fprintf(os.Stdout, "Node: %s, Zenoh: %s\n", cfg.NodeName, cfg.ZenohAddr)
	fmt.Fprintf(os.Stdout, "Resources: CPU=%dm, Memory=%d, GPU=%d\n",
		cfg.CPUMillis, cfg.MemoryBytes, cfg.GPUCount)

	p, err := provider.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create provider: %v\n", err)
		os.Exit(1)
	}

	_ = p
	// In production: register with virtual-kubelet node controller
	// node.NewNodeController(p, ...).Run(ctx)
	fmt.Fprintln(os.Stdout, "Provider initialized. Ready for virtual-kubelet node controller.")
}
