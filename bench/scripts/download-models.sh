#!/bin/bash
# Download benchmark ONNX models from the ONNX Model Zoo.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
mkdir -p "$MODEL_DIR"

# MobileNetV2 — vision (ImageNet classification, ~14 MB)
MOBILENET_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
MOBILENET_FILE="${MODEL_DIR}/mobilenetv2-7.onnx"
if [ ! -f "$MOBILENET_FILE" ]; then
    echo "Downloading MobileNetV2..."
    curl -sSL -o "$MOBILENET_FILE" "$MOBILENET_URL"
    echo "  -> $(stat --format='%s' "$MOBILENET_FILE" 2>/dev/null || stat -f'%z' "$MOBILENET_FILE") bytes"
else
    echo "MobileNetV2 already downloaded."
fi

# DistilBERT — text (NLI, ~256 MB)
DISTILBERT_URL="https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/distilbert/model/distilbert-base-uncased.onnx"
DISTILBERT_FILE="${MODEL_DIR}/distilbert-base-uncased.onnx"
if [ ! -f "$DISTILBERT_FILE" ]; then
    echo "Downloading DistilBERT..."
    curl -sSL -o "$DISTILBERT_FILE" "$DISTILBERT_URL"
    echo "  -> $(stat --format='%s' "$DISTILBERT_FILE" 2>/dev/null || stat -f'%z' "$DISTILBERT_FILE") bytes"
else
    echo "DistilBERT already downloaded."
fi

# Whisper-tiny encoder — audio/signal (~151 MB)
WHISPER_URL="https://github.com/onnx/models/raw/main/validated/speech/speech_recognition/whisper/model/whisper-tiny-encoder.onnx"
WHISPER_FILE="${MODEL_DIR}/whisper-tiny-encoder.onnx"
if [ ! -f "$WHISPER_FILE" ]; then
    echo "Downloading Whisper-tiny encoder..."
    curl -sSL -o "$WHISPER_FILE" "$WHISPER_URL"
    echo "  -> $(stat --format='%s' "$WHISPER_FILE" 2>/dev/null || stat -f'%z' "$WHISPER_FILE") bytes"
else
    echo "Whisper-tiny encoder already downloaded."
fi

echo ""
echo "All models downloaded to ${MODEL_DIR}/"
ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || true
