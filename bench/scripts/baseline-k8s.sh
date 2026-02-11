#!/bin/bash
# Kubernetes/K3s baseline: deploy ONNX Runtime pod and measure.
# SPDX-License-Identifier: Apache-2.0
#
# Prerequisites:
#   - kubectl configured with a running cluster (K8s or K3s)
#   - Models accessible via PVC or node-local path
#
# Usage: ./baseline-k8s.sh [config-file]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/../configs/default.env}"
MANIFESTS_DIR="${SCRIPT_DIR}/../configs/k8s"
RESULTS_DIR="${SCRIPT_DIR}/../results/k8s"

# Load config
if [ -f "$CONFIG" ]; then
    # shellcheck source=/dev/null
    source "$CONFIG"
fi

NAMESPACE="${BENCH_K8S_NAMESPACE:-smallaios-bench}"
ITERATIONS="${BENCH_ITERATIONS:-1000}"

mkdir -p "$RESULTS_DIR"

echo "=== Kubernetes Baseline ==="
echo "Namespace: ${NAMESPACE}"
echo "Cluster: $(kubectl config current-context 2>/dev/null || echo 'unknown')"
echo ""

# Create namespace if needed
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Deploy the benchmark pod
echo "Deploying benchmark pod..."
kubectl apply -n "$NAMESPACE" -f "${MANIFESTS_DIR}/benchmark-pod.yaml"

# Wait for pod to be ready
echo "Waiting for pod to be ready..."
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod/onnxrt-bench --timeout=120s

# Measure cold start: time from pod creation to ready
CREATED=$(kubectl get pod -n "$NAMESPACE" onnxrt-bench -o jsonpath='{.metadata.creationTimestamp}')
READY=$(kubectl get pod -n "$NAMESPACE" onnxrt-bench -o jsonpath='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}')
echo "Pod created:  ${CREATED}"
echo "Pod ready:    ${READY}"

# Run benchmarks inside the pod
echo ""
echo "Running MobileNetV2 benchmark..."
kubectl exec -n "$NAMESPACE" onnxrt-bench -- \
    onnxruntime_perf_test \
        -m /models/mobilenetv2-7.onnx \
        -r "$ITERATIONS" \
        -w 10 \
        -o /tmp/mobilenetv2-results.json 2>&1 | tail -5

# Copy results back
kubectl cp "$NAMESPACE/onnxrt-bench:/tmp/mobilenetv2-results.json" \
    "${RESULTS_DIR}/mobilenetv2-k8s.json" 2>/dev/null || true

# Cleanup
echo ""
echo "Cleaning up..."
kubectl delete -n "$NAMESPACE" -f "${MANIFESTS_DIR}/benchmark-pod.yaml" --ignore-not-found

echo "=== Results saved to ${RESULTS_DIR}/ ==="
