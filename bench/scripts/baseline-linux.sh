#!/bin/bash
# Linux bare metal baseline: run ONNX Runtime C++ on host OS.
# SPDX-License-Identifier: Apache-2.0
#
# Prerequisites:
#   - ONNX Runtime C++ installed (libonnxruntime.so in LD_LIBRARY_PATH)
#   - onnxruntime_perf_test binary available
#   - Models downloaded (run download-models.sh first)
#
# Usage: ./baseline-linux.sh [config-file]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/../configs/default.env}"
MODEL_DIR="${SCRIPT_DIR}/../models"
RESULTS_DIR="${SCRIPT_DIR}/../results/linux-bare-metal"

# Load config
if [ -f "$CONFIG" ]; then
    # shellcheck source=/dev/null
    source "$CONFIG"
fi

WARMUP="${BENCH_WARMUP:-10}"
ITERATIONS="${BENCH_ITERATIONS:-1000}"
PERF_TEST="${ONNXRT_PERF_TEST:-onnxruntime_perf_test}"

mkdir -p "$RESULTS_DIR"

echo "=== Linux Bare Metal Baseline ==="
echo "Platform: $(uname -r) on $(uname -m)"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
echo "Warmup: ${WARMUP}, Iterations: ${ITERATIONS}"
echo ""

run_benchmark() {
    local model_name="$1"
    local model_file="$2"
    local batch_sizes=("${@:3}")

    if [ ! -f "$model_file" ]; then
        echo "SKIP: $model_name (model not found: $model_file)"
        return
    fi

    for bs in "${batch_sizes[@]}"; do
        local out="${RESULTS_DIR}/${model_name}-b${bs}.json"
        echo "--- ${model_name} batch=${bs} ---"
        "$PERF_TEST" \
            -m "$model_file" \
            -r "$ITERATIONS" \
            -w "$WARMUP" \
            -b "$bs" \
            -o "$out" 2>&1 | tail -5
        echo ""
    done
}

# MobileNetV2 — small vision model
run_benchmark "mobilenetv2" "${MODEL_DIR}/mobilenetv2-7.onnx" 1 4 16 64

# DistilBERT — text model
run_benchmark "distilbert" "${MODEL_DIR}/distilbert-base-uncased.onnx" 1 4 16

# Whisper-tiny — audio/signal model
run_benchmark "whisper-tiny" "${MODEL_DIR}/whisper-tiny-encoder.onnx" 1 4

echo "=== Results saved to ${RESULTS_DIR}/ ==="
