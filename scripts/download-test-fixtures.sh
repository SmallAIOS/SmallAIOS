#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Downloads public ONNX model graph files for integration testing.
# These are graph-structure files (typically <100 MB) from public HuggingFace mirrors.
# Fixture files are gitignored and NOT committed to the repository.
#
# Supports HuggingFace authentication for gated models (e.g. Gemma 2):
#   - Set $HF_TOKEN env var, OR
#   - Run `hf auth login` (stores token at ~/.cache/huggingface/token)
#
# Usage:
#   ./scripts/download-test-fixtures.sh
#
# After downloading, run the fixture tests:
#   cargo test -p smallaios-onnx-rt --test test_model_fixtures -- --ignored

set -euo pipefail

DEST="tests/fixtures/onnx-models"
mkdir -p "$DEST"

# ---------------------------------------------------------------------------
# Resolve HuggingFace token for authenticated downloads
# ---------------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")}"
AUTH_ARGS=()
if [ -n "$HF_TOKEN" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer $HF_TOKEN")
    echo "Using HuggingFace token for authenticated downloads"
else
    echo "No HF token found; gated models will be skipped"
    echo "  To authenticate: export HF_TOKEN=<token> or run 'hf auth login'"
fi

download() {
    local name="$1"
    local url="$2"
    local dest_file="$DEST/$name"

    if [ -f "$dest_file" ]; then
        echo "SKIP: $name already exists ($(du -h "$dest_file" | cut -f1))"
        return 0
    fi

    echo "Downloading $name ..."
    if curl -L --fail-with-body --progress-bar "${AUTH_ARGS[@]}" -o "$dest_file" "$url"; then
        echo "  OK: $(du -h "$dest_file" | cut -f1)"
    else
        local exit_code=$?
        echo "  SKIP: $name (download failed, curl exit=$exit_code)"
        rm -f "$dest_file"
    fi
}

# ---------------------------------------------------------------------------
# Public models (no token required, but token is passed anyway — harmless)
# ---------------------------------------------------------------------------

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

# DeepSeek-R1-Distill-Qwen-1.5B (onnx-community, public)
download "deepseek-r1-distill-qwen-1.5b.onnx" \
    "https://huggingface.co/onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/resolve/main/onnx/model.onnx"

# MobileNetV2 (onnx/models GitHub mirror, public)
download "mobilenet_v2.onnx" \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"

# ResNet-50 (onnx/models GitHub mirror, public)
download "resnet50.onnx" \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"

# SqueezeNet 1.1 (onnx/models GitHub mirror, public)
download "squeezenet.onnx" \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"

# ---------------------------------------------------------------------------
# Gemma 3 (onnx-community public ONNX exports)
# ---------------------------------------------------------------------------

# Gemma 3 1B IT (onnx-community, public graph-only)
# Note: repo name has -ONNX suffix
download "gemma-3-1b-it.onnx" \
    "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/onnx/model.onnx"

# Gemma 3 270M IT (onnx-community, public graph-only, smaller variant)
download "gemma-3-270m-it.onnx" \
    "https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX/resolve/main/onnx/model.onnx"

# ---------------------------------------------------------------------------
# Gated models (require HF token with accepted license)
# ---------------------------------------------------------------------------

# Gemma 2 2B IT — onnx-community/gemma-2-2b-it does NOT exist (404).
# Only onnx-community/gemma-2-2b-jpn-it exists (gated, Gemma license).
# Uncomment and adjust if the repo becomes available:
# download "gemma-2-2b-it.onnx" \
#     "https://huggingface.co/onnx-community/gemma-2-2b-it/resolve/main/onnx/model.onnx"

echo ""
echo "Downloaded fixtures:"
ls -lh "$DEST/"
