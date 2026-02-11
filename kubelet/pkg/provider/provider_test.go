// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"context"
	"encoding/binary"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// --- Config & Provider creation ---

func testConfig() Config {
	return Config{
		NodeName:    "test-node",
		ZenohAddr:   "tcp/127.0.0.1:7447",
		CPUMillis:   4000,
		MemoryBytes: 8589934592,
		GPUCount:    1,
		MetricsAddr: ":10255",
		HealthAddr:  ":10256",
	}
}

func TestNewProvider(t *testing.T) {
	p, err := New(testConfig())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if p == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestNewProviderEmptyNodeName(t *testing.T) {
	cfg := testConfig()
	cfg.NodeName = ""
	_, err := New(cfg)
	if err == nil {
		t.Fatal("expected error for empty node name")
	}
}

func TestNewProviderEmptyZenohAddr(t *testing.T) {
	cfg := testConfig()
	cfg.ZenohAddr = ""
	_, err := New(cfg)
	if err == nil {
		t.Fatal("expected error for empty zenoh address")
	}
}

// --- Pod spec translation ---

func testPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "resnet50-inference",
			Namespace: "default",
			UID:       types.UID("pod-uid-abc123"),
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "inference",
					Image: "https://models.example.com/resnet50.onnx",
					Env: []corev1.EnvVar{
						{Name: "BATCH_SIZE", Value: "4"},
						{Name: "PRECISION", Value: "fp16"},
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    *resource.NewMilliQuantity(1000, resource.DecimalSI),
							corev1.ResourceMemory: *resource.NewQuantity(536870912, resource.BinarySI), // 512 MiB
						},
						Limits: corev1.ResourceList{
							corev1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
}

func TestTranslatePodSpec(t *testing.T) {
	pod := testPod()
	spec := TranslatePodSpec(pod)

	if spec.WorkloadID != "pod-uid-abc123" {
		t.Errorf("expected workload ID 'pod-uid-abc123', got %q", spec.WorkloadID)
	}
	if spec.ModelURL != "https://models.example.com/resnet50.onnx" {
		t.Errorf("expected model URL from image, got %q", spec.ModelURL)
	}
	if spec.Name != "resnet50-inference" {
		t.Errorf("expected name 'resnet50-inference', got %q", spec.Name)
	}
	if spec.Namespace != "default" {
		t.Errorf("expected namespace 'default', got %q", spec.Namespace)
	}
	if spec.CPUMillis != 1000 {
		t.Errorf("expected 1000 millicores, got %d", spec.CPUMillis)
	}
	if spec.MemoryBytes != 536870912 {
		t.Errorf("expected 536870912 bytes, got %d", spec.MemoryBytes)
	}
	if !spec.GPURequired {
		t.Error("expected GPU required")
	}
	if spec.Env["BATCH_SIZE"] != "4" {
		t.Errorf("expected BATCH_SIZE=4, got %q", spec.Env["BATCH_SIZE"])
	}
	if spec.Env["PRECISION"] != "fp16" {
		t.Errorf("expected PRECISION=fp16, got %q", spec.Env["PRECISION"])
	}
}

func TestTranslateImageToModelURL_HTTP(t *testing.T) {
	url := translateImageToModelURL("https://models.example.com/bert.onnx")
	if url != "https://models.example.com/bert.onnx" {
		t.Errorf("expected passthrough URL, got %q", url)
	}
}

func TestTranslateImageToModelURL_OCI(t *testing.T) {
	url := translateImageToModelURL("models.example.com/resnet50:v1")
	if url != "oci://models.example.com/resnet50:v1" {
		t.Errorf("expected OCI URL, got %q", url)
	}
}

func TestTranslatePodSpec_NoContainers(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "empty", Namespace: "default", UID: "uid-1"},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{}},
	}
	spec := TranslatePodSpec(pod)
	if spec.ModelURL != "" {
		t.Errorf("expected empty model URL for no containers, got %q", spec.ModelURL)
	}
}

// --- Pod lifecycle state machine ---

func TestPodPhaseMapping(t *testing.T) {
	tests := []struct {
		phase    PodPhase
		expected corev1.PodPhase
	}{
		{PodPhasePending, corev1.PodPending},
		{PodPhaseLoading, corev1.PodPending},
		{PodPhaseRunning, corev1.PodRunning},
		{PodPhaseTerminating, corev1.PodRunning},
		{PodPhaseSucceeded, corev1.PodSucceeded},
		{PodPhaseFailed, corev1.PodFailed},
	}
	for _, tc := range tests {
		ps := &PodState{Phase: tc.phase, Pod: testPod()}
		if got := ps.KubernetesPhase(); got != tc.expected {
			t.Errorf("phase %s: expected %v, got %v", tc.phase, tc.expected, got)
		}
	}
}

func TestKubernetesPodStatus_Pending(t *testing.T) {
	ps := &PodState{Phase: PodPhasePending, Pod: testPod(), Message: "waiting"}
	status := ps.KubernetesPodStatus()
	if status.Phase != corev1.PodPending {
		t.Errorf("expected Pending, got %v", status.Phase)
	}
	if len(status.ContainerStatuses) != 1 {
		t.Fatalf("expected 1 container status, got %d", len(status.ContainerStatuses))
	}
	if status.ContainerStatuses[0].Ready {
		t.Error("expected not ready for pending pod")
	}
	if status.ContainerStatuses[0].State.Waiting == nil {
		t.Error("expected waiting state for pending pod")
	}
}

func TestKubernetesPodStatus_Running(t *testing.T) {
	ps := &PodState{Phase: PodPhaseRunning, Pod: testPod(), Message: "serving"}
	status := ps.KubernetesPodStatus()
	if status.Phase != corev1.PodRunning {
		t.Errorf("expected Running, got %v", status.Phase)
	}
	if !status.ContainerStatuses[0].Ready {
		t.Error("expected ready for running pod")
	}
	if status.ContainerStatuses[0].State.Running == nil {
		t.Error("expected running state")
	}
}

func TestKubernetesPodStatus_Failed(t *testing.T) {
	ps := &PodState{Phase: PodPhaseFailed, Pod: testPod(), Message: "OOM"}
	status := ps.KubernetesPodStatus()
	if status.Phase != corev1.PodFailed {
		t.Errorf("expected Failed, got %v", status.Phase)
	}
	if status.ContainerStatuses[0].State.Terminated == nil {
		t.Error("expected terminated state")
	}
	if status.ContainerStatuses[0].State.Terminated.ExitCode != 1 {
		t.Error("expected exit code 1 for failed pod")
	}
}

// --- Node resource advertisement ---

func TestNodeCapacity(t *testing.T) {
	cfg := testConfig()
	p, _ := New(cfg)
	cap := p.NodeCapacity()

	cpu := cap[corev1.ResourceCPU]
	if cpu.MilliValue() != 4000 {
		t.Errorf("expected 4000m CPU, got %d", cpu.MilliValue())
	}
	mem := cap[corev1.ResourceMemory]
	if mem.Value() != 8589934592 {
		t.Errorf("expected 8Gi memory, got %d", mem.Value())
	}
	gpu := cap[corev1.ResourceName("nvidia.com/gpu")]
	if gpu.Value() != 1 {
		t.Errorf("expected 1 GPU, got %d", gpu.Value())
	}
	pods := cap[corev1.ResourcePods]
	if pods.Value() != 32 {
		t.Errorf("expected 32 pods, got %d", pods.Value())
	}
}

func TestNodeCapacityNoGPU(t *testing.T) {
	cfg := testConfig()
	cfg.GPUCount = 0
	p, _ := New(cfg)
	cap := p.NodeCapacity()
	if _, ok := cap[corev1.ResourceName("nvidia.com/gpu")]; ok {
		t.Error("expected no GPU resource when GPUCount=0")
	}
}

func TestNodeConditions(t *testing.T) {
	p, _ := New(testConfig())
	conds := p.NodeConditions(context.Background())
	if len(conds) != 3 {
		t.Fatalf("expected 3 conditions, got %d", len(conds))
	}
	// First condition should be NodeReady = True
	if conds[0].Type != corev1.NodeReady || conds[0].Status != corev1.ConditionTrue {
		t.Errorf("expected NodeReady=True, got %v=%v", conds[0].Type, conds[0].Status)
	}
}

func TestConfigureNode(t *testing.T) {
	adv := NewNodeAdvertiser(testConfig())
	node := &corev1.Node{}
	adv.ConfigureNode(node)

	if node.Labels["smallaios.io/os"] != "smallaios" {
		t.Error("expected smallaios.io/os label")
	}
	if len(node.Spec.Taints) != 1 {
		t.Fatalf("expected 1 taint, got %d", len(node.Spec.Taints))
	}
	if node.Spec.Taints[0].Key != "virtual-kubelet.io/provider" {
		t.Error("expected virtual-kubelet.io/provider taint")
	}
	if node.Status.NodeInfo.OperatingSystem != "smallaios" {
		t.Error("expected smallaios OS in node info")
	}
}

// --- Zenoh client ---

func TestZenohClientConnect(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	if z.IsConnected() {
		t.Error("expected not connected initially")
	}
	if err := z.Connect(context.Background()); err != nil {
		t.Fatalf("connect error: %v", err)
	}
	if !z.IsConnected() {
		t.Error("expected connected after Connect()")
	}
	if z.Addr() != "tcp/127.0.0.1:7447" {
		t.Errorf("unexpected addr: %s", z.Addr())
	}
}

func TestZenohClientDeployNotConnected(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	err := z.Deploy(context.Background(), SmallAIOSDeploySpec{})
	if err == nil {
		t.Error("expected error when not connected")
	}
}

func TestZenohClientDeployConnected(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	err := z.Deploy(context.Background(), SmallAIOSDeploySpec{
		WorkloadID: "wl-1",
		ModelURL:   "https://example.com/model.onnx",
		Name:       "test",
		Namespace:  "default",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestZenohClientUndeployNotConnected(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	err := z.Undeploy(context.Background(), "wl-1", false)
	if err == nil {
		t.Error("expected error when not connected")
	}
}

func TestZenohClientQueryStatus(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	resp, err := z.QueryStatus(context.Background(), "wl-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == "" {
		t.Error("expected non-empty status response")
	}
}

func TestZenohClientClose(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	_ = z.Close()
	if z.IsConnected() {
		t.Error("expected disconnected after Close()")
	}
}

// --- Wire format encoding ---

func TestEncodeDeployRequest(t *testing.T) {
	spec := SmallAIOSDeploySpec{
		WorkloadID:  "wl-123",
		ModelURL:    "https://example.com/model.onnx",
		Name:        "test-model",
		Namespace:   "default",
		Env:         map[string]string{"KEY": "VAL"},
		CPUMillis:   1000,
		MemoryBytes: 536870912,
		GPURequired: true,
	}
	data := EncodeDeployRequest(spec)
	if len(data) == 0 {
		t.Fatal("expected non-empty encoded data")
	}

	// Verify workload ID is at the start (2-byte length prefix + string).
	wlLen := binary.LittleEndian.Uint16(data[0:2])
	if int(wlLen) != len("wl-123") {
		t.Errorf("expected workload ID length %d, got %d", len("wl-123"), wlLen)
	}
	wlID := string(data[2 : 2+wlLen])
	if wlID != "wl-123" {
		t.Errorf("expected 'wl-123', got %q", wlID)
	}
}

func TestEncodeUndeployRequest(t *testing.T) {
	data := EncodeUndeployRequest("wl-abc", true)
	wlLen := binary.LittleEndian.Uint16(data[0:2])
	wlID := string(data[2 : 2+wlLen])
	if wlID != "wl-abc" {
		t.Errorf("expected 'wl-abc', got %q", wlID)
	}
	force := data[2+wlLen]
	if force != 1 {
		t.Errorf("expected force=1, got %d", force)
	}
}

func TestEncodeUndeployRequestNoForce(t *testing.T) {
	data := EncodeUndeployRequest("wl-1", false)
	wlLen := binary.LittleEndian.Uint16(data[0:2])
	force := data[2+wlLen]
	if force != 0 {
		t.Errorf("expected force=0, got %d", force)
	}
}

// --- Health probes ---

func TestHealthCheck(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	hp := NewHealthProber(z)

	status, err := hp.CheckHealth(context.Background(), "wl-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !status.Healthy {
		t.Error("expected healthy status")
	}
}

func TestHealthCheckNotConnected(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	hp := NewHealthProber(z)

	_, err := hp.CheckHealth(context.Background(), "wl-1")
	if err == nil {
		t.Error("expected error when zenoh not connected")
	}
}

func TestCheckLivenessNoProbe(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	hp := NewHealthProber(z)

	pod := testPod()
	live, err := hp.CheckLiveness(context.Background(), pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !live {
		t.Error("expected liveness=true when no probe configured")
	}
}

func TestCheckReadinessNoProbe(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	hp := NewHealthProber(z)

	pod := testPod()
	ready, err := hp.CheckReadiness(context.Background(), pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ready {
		t.Error("expected readiness=true when no probe configured")
	}
}

// --- Metrics ---

func TestFetchMetrics(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	_ = z.Connect(context.Background())
	mp := NewMetricsProxy(z)

	metrics, err := mp.FetchMetrics(context.Background(), "wl-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(metrics) != 3 {
		t.Fatalf("expected 3 metrics, got %d", len(metrics))
	}
	if metrics[0].Name != "smallaios_inference_total" {
		t.Errorf("expected first metric 'smallaios_inference_total', got %q", metrics[0].Name)
	}
}

func TestFetchMetricsNotConnected(t *testing.T) {
	z := NewZenohClient("tcp/127.0.0.1:7447")
	mp := NewMetricsProxy(z)
	_, err := mp.FetchMetrics(context.Background(), "wl-1")
	if err == nil {
		t.Error("expected error when not connected")
	}
}

func TestFormatPrometheus(t *testing.T) {
	metrics := []MetricEntry{
		{Name: "test_counter", Labels: map[string]string{"id": "1"}, Value: 42, Type: "counter"},
	}
	output := FormatPrometheus(metrics)
	if output == "" {
		t.Error("expected non-empty prometheus output")
	}
	if !containsStr(output, "# TYPE test_counter counter") {
		t.Error("expected TYPE line in output")
	}
	if !containsStr(output, "test_counter{") {
		t.Error("expected metric line with labels")
	}
	if !containsStr(output, "42") {
		t.Error("expected value 42 in output")
	}
}

// --- CRUD operations ---

func TestCreateAndGetPod(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	ctx := context.Background()
	pod := testPod()

	if err := p.CreatePod(ctx, pod); err != nil {
		t.Fatalf("create pod error: %v", err)
	}

	got, err := p.GetPod(ctx, "default", "resnet50-inference")
	if err != nil {
		t.Fatalf("get pod error: %v", err)
	}
	if got.Name != "resnet50-inference" {
		t.Errorf("expected name 'resnet50-inference', got %q", got.Name)
	}
}

func TestCreatePodDuplicate(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	ctx := context.Background()
	pod := testPod()

	_ = p.CreatePod(ctx, pod)
	err := p.CreatePod(ctx, pod)
	if err == nil {
		t.Error("expected error for duplicate pod")
	}
}

func TestDeletePod(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	ctx := context.Background()
	pod := testPod()

	_ = p.CreatePod(ctx, pod)
	if err := p.DeletePod(ctx, pod); err != nil {
		t.Fatalf("delete pod error: %v", err)
	}

	_, err := p.GetPod(ctx, "default", "resnet50-inference")
	if err == nil {
		t.Error("expected not found after delete")
	}
}

func TestDeletePodNotFound(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	pod := testPod()
	err := p.DeletePod(context.Background(), pod)
	if err == nil {
		t.Error("expected error for non-existent pod")
	}
}

func TestGetPods(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	ctx := context.Background()

	pod1 := testPod()
	_ = p.CreatePod(ctx, pod1)

	pod2 := testPod()
	pod2.Name = "bert-inference"
	pod2.UID = "uid-2"
	_ = p.CreatePod(ctx, pod2)

	pods, err := p.GetPods(ctx)
	if err != nil {
		t.Fatalf("get pods error: %v", err)
	}
	if len(pods) != 2 {
		t.Errorf("expected 2 pods, got %d", len(pods))
	}
}

func TestGetPodStatus(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	ctx := context.Background()
	pod := testPod()
	_ = p.CreatePod(ctx, pod)

	status, err := p.GetPodStatus(ctx, "default", "resnet50-inference")
	if err != nil {
		t.Fatalf("get pod status error: %v", err)
	}
	if status.Phase != corev1.PodPending {
		t.Errorf("expected Pending, got %v", status.Phase)
	}
}

func TestUpdatePod(t *testing.T) {
	p, _ := New(testConfig())
	_ = p.zenoh.Connect(context.Background())
	ctx := context.Background()
	pod := testPod()
	_ = p.CreatePod(ctx, pod)

	pod.Spec.Containers[0].Image = "https://models.example.com/resnet50-v2.onnx"
	if err := p.UpdatePod(ctx, pod); err != nil {
		t.Fatalf("update pod error: %v", err)
	}
}

func TestUpdatePodNotFound(t *testing.T) {
	p, _ := New(testConfig())
	err := p.UpdatePod(context.Background(), testPod())
	if err == nil {
		t.Error("expected error for non-existent pod")
	}
}

// --- Helpers ---

func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstring(s, substr))
}

func containsSubstring(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
