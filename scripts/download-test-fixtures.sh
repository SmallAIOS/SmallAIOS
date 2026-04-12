#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Downloads public ONNX model graph files for integration testing.
# These are graph-structure files (typically <100 MB) from public HuggingFace mirrors.
# Fixture files are gitignored and NOT committed to the repository.
#
# Usage:
#   ./scripts/download-test-fixtures.sh
#
# After downloading, run the fixture tests:
#   cargo test -p smallaios-onnx-rt --test test_model_fixtures -- --ignored

set -euo pipefail

DEST="tests/fixtures/onnx-models"
mkdir -p "$DEST"

download() {
    local name="$1"
    local url="$2"
    local dest_file="$DEST/$name"

    if [ -f "$dest_file" ]; then
        echo "SKIP: $name already exists ($(du -h "$dest_file" | cut -f1))"
        return 0
    fi

    echo "Downloading $name ..."
    if curl -L --fail-with-body --progress-bar -o "$dest_file" "$url"; then
        echo "  OK: $(du -h "$dest_file" | cut -f1)"
    else
        echo "  SKIP: $name (download failed)"
        rm -f "$dest_file"
    fi
}

# BERT-base (Xenova mirror, public)
download "bert-base-uncased.onnx" \
    "https://huggingface.co/Xenova/bert-base-uncased/resolve/main/onnx/model.onnx"

# ViT-base (Xenova mirror, public)
download "vit-base-patch16-224.onnx" \
    "https://huggingface.co/Xenova/vit-base-patch16-224/resolve/main/onnx/model.onnx"

# DistilGPT-2 (Xenova mirror, public)
download "distilgpt2.onnx" \
    "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/model.onnx"

# Llama-3.2-1B-Instruct (onnx-community, graph-only file)
download "llama-3.2-1b.onnx" \
    "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct/resolve/main/onnx/model.onnx"

# Gemma 3 1B (onnx-community, public)
download "gemma-3-1b-it.onnx" \
    "https://huggingface.co/onnx-community/gemma-3-1b-it/resolve/main/onnx/model.onnx"

# DeepSeek-R1-Distill-Qwen-1.5B (onnx-community, public)
download "deepseek-r1-distill-qwen-1.5b.onnx" \
    "https://huggingface.co/onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/resolve/main/onnx/model.onnx"

# MobileNetV2 (onnx/models GitHub mirror, public)
download "mobilenet_v2.onnx" \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"

echo ""
echo "Done. Downloaded fixtures:"
ls -lh "$DEST/"
