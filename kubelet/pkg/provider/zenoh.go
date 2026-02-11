// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"context"
	"encoding/binary"
	"fmt"
)

// Zenoh management API key expressions (must match ipc/src/mgmt.rs).
const (
	MgmtDeploy    = "sai/mgmt/deploy"
	MgmtUndeploy  = "sai/mgmt/undeploy"
	MgmtStatus    = "sai/mgmt/status"
	MgmtConfig    = "sai/mgmt/config"
	MgmtResources = "sai/mgmt/resources"
)

// ZenohClient provides a Go interface to the SmallAIOS management API
// over the Zenoh protocol. It encodes requests in the same wire format
// as the Rust-side DeployRequest/UndeployRequest in ipc/src/mgmt.rs.
type ZenohClient struct {
	// addr is the Zenoh router address (e.g. "tcp/127.0.0.1:7447").
	addr string
	// connected indicates whether the client has an active session.
	connected bool
}

// NewZenohClient creates a new Zenoh client targeting the given address.
func NewZenohClient(addr string) *ZenohClient {
	return &ZenohClient{
		addr:      addr,
		connected: false,
	}
}

// Connect establishes a Zenoh session. In a real implementation this
// would open a TCP connection to the Zenoh router and perform the
// session handshake.
func (z *ZenohClient) Connect(_ context.Context) error {
	if z.connected {
		return nil
	}
	// Stub: real implementation would use zenoh-go client library
	z.connected = true
	return nil
}

// Close terminates the Zenoh session.
func (z *ZenohClient) Close() error {
	z.connected = false
	return nil
}

// IsConnected returns whether the client has an active session.
func (z *ZenohClient) IsConnected() bool {
	return z.connected
}

// Addr returns the Zenoh router address.
func (z *ZenohClient) Addr() string {
	return z.addr
}

// Deploy sends a deploy request to the SmallAIOS management API.
// The wire format matches the Rust DeployRequest::encode() in ipc/src/mgmt.rs:
//
//	workload_id\0model_url\0name\0namespace\0cpu_millis(4B LE)\0memory_bytes(8B LE)\0gpu(1B)\0env_count(2B LE)..
func (z *ZenohClient) Deploy(_ context.Context, spec SmallAIOSDeploySpec) error {
	if !z.connected {
		return fmt.Errorf("zenoh client not connected")
	}
	_ = EncodeDeployRequest(spec)
	// Stub: real implementation would publish to MgmtDeploy key expression
	// and wait for DeployResponse.
	return nil
}

// Undeploy sends an undeploy request to the SmallAIOS management API.
func (z *ZenohClient) Undeploy(_ context.Context, workloadID string, force bool) error {
	if !z.connected {
		return fmt.Errorf("zenoh client not connected")
	}
	_ = EncodeUndeployRequest(workloadID, force)
	// Stub: real implementation would publish to MgmtUndeploy key expression.
	return nil
}

// QueryStatus queries workload status from the SmallAIOS management API.
func (z *ZenohClient) QueryStatus(_ context.Context, workloadID string) (string, error) {
	if !z.connected {
		return "", fmt.Errorf("zenoh client not connected")
	}
	// Stub: real implementation would query MgmtStatus and return JSON response.
	return fmt.Sprintf(`[{"id":"%s","phase":"Running","inferences":0}]`, workloadID), nil
}

// QueryResources queries node resources from the SmallAIOS management API.
func (z *ZenohClient) QueryResources(_ context.Context) (string, error) {
	if !z.connected {
		return "", fmt.Errorf("zenoh client not connected")
	}
	// Stub: real implementation would query MgmtResources.
	return `{"cpu":{"capacity":4000,"allocatable":4000},"memory":{"capacity":8589934592,"allocatable":8589934592}}`, nil
}

// EncodeDeployRequest encodes a SmallAIOSDeploySpec into the wire format
// compatible with the Rust DeployRequest::decode().
func EncodeDeployRequest(spec SmallAIOSDeploySpec) []byte {
	buf := make([]byte, 0, 256)

	buf = appendString(buf, spec.WorkloadID)
	buf = appendString(buf, spec.ModelURL)
	buf = appendString(buf, spec.Name)
	buf = appendString(buf, spec.Namespace)

	cpuBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(cpuBytes, uint32(spec.CPUMillis))
	buf = append(buf, cpuBytes...)

	memBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(memBytes, uint64(spec.MemoryBytes))
	buf = append(buf, memBytes...)

	if spec.GPURequired {
		buf = append(buf, 1)
	} else {
		buf = append(buf, 0)
	}

	envCount := make([]byte, 2)
	binary.LittleEndian.PutUint16(envCount, uint16(len(spec.Env)))
	buf = append(buf, envCount...)
	for k, v := range spec.Env {
		buf = appendString(buf, k)
		buf = appendString(buf, v)
	}

	return buf
}

// EncodeUndeployRequest encodes an undeploy request into wire format.
func EncodeUndeployRequest(workloadID string, force bool) []byte {
	buf := appendString(nil, workloadID)
	if force {
		buf = append(buf, 1)
	} else {
		buf = append(buf, 0)
	}
	return buf
}

// appendString appends a length-prefixed string (2-byte LE length + UTF-8 bytes).
func appendString(buf []byte, s string) []byte {
	lenBytes := make([]byte, 2)
	binary.LittleEndian.PutUint16(lenBytes, uint16(len(s)))
	buf = append(buf, lenBytes...)
	buf = append(buf, []byte(s)...)
	return buf
}
