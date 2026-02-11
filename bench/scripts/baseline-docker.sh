#!/bin/bash
# Docker baseline: run ONNX Runtime in a container.
# SPDX-License-Identifier: Apache-2.0
#
# Prerequisites:
#   - Docker installed and running
#   - Models downloaded (run download-models.sh first)
#
# Usage: ./baseline-docker.sh [config-file]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/../configs/default.env}"
MODEL_DIR="${SCRIPT_DIR}/../models"
RESULTS_DIR="${SCRIPT_DIR}/../results/docker"

# Load config
if [ -f "$CONFIG" ]; then
    # shellcheck source=/dev/null
    source "$CONFIG"
fi

WARMUP="${BENCH_WARMUP:-10}"
ITERATIONS="${BENCH_ITERATIONS:-1000}"
DOCKER_IMAGE="${ONNXRT_DOCKER_IMAGE:-mcr.microsoft.com/onnxruntime/server:latest}"

mkdir -p "$RESULTS_DIR"

echo "=== Docker Baseline ==="
echo "Image: ${DOCKER_IMAGE}"
echo "Warmup: ${WARMUP}, Iterations: ${ITERATIONS}"
echo ""

# Build a lightweight benchmark container if not using the official image
BENCH_DOCKERFILE="${SCRIPT_DIR}/Dockerfile.bench"
if [ -f "$BENCH_DOCKERFILE" ]; then
    echo "Building benchmark container..."
    docker build -t smallaios-bench:latest -f "$BENCH_DOCKERFILE" "$SCRIPT_DIR/.."
    DOCKER_IMAGE="smallaios-bench:latest"
fi

run_docker_benchmark() {
    local model_name="$1"
    local model_file="$2"
    local batch_sizes=("${@:3}")

    if [ ! -f "$model_file" ]; then
        echo "SKIP: $model_name (model not found)"
        return
    fi

    for bs in "${batch_sizes[@]}"; do
        local out="${RESULTS_DIR}/${model_name}-b${bs}.json"
        echo "--- ${model_name} batch=${bs} (Docker) ---"

        # Measure cold start time (container creation to first output)
        local start_ns
        start_ns=$(date +%s%N)

        docker run --rm \
            -v "${MODEL_DIR}:/models:ro" \
            -v "${RESULTS_DIR}:/results" \
            "$DOCKER_IMAGE" \
            onnxruntime_perf_test \
                -m "/models/$(basename "$model_file")" \
                -r "$ITERATIONS" \
                -w "$WARMUP" \
                -b "$bs" \
                -o "/results/${model_name}-b${bs}.json" 2>&1 | tail -5

        local end_ns
        end_ns=$(date +%s%N)
        local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
        echo "  Container total time: ${elapsed_ms} ms"
        echo ""
    done
}

# MobileNetV2
run_docker_benchmark "mobilenetv2" "${MODEL_DIR}/mobilenetv2-7.onnx" 1 4 16 64

# DistilBERT
run_docker_benchmark "distilbert" "${MODEL_DIR}/distilbert-base-uncased.onnx" 1 4 16

# Whisper-tiny
run_docker_benchmark "whisper-tiny" "${MODEL_DIR}/whisper-tiny-encoder.onnx" 1 4

echo "=== Results saved to ${RESULTS_DIR}/ ==="
