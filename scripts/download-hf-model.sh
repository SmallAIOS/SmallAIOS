#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Download a HuggingFace safetensors model into
# tests/fixtures/safetensors-models/<model-id>/ for local testing of
# the safetensors Session path (change/safetensors-model-loader-v1).
#
# Usage:
#   scripts/download-hf-model.sh <hf-model-id> [--revision <rev>]
#
# Example:
#   scripts/download-hf-model.sh google/gemma-4-270m
#   HF_TOKEN=hf_xxx scripts/download-hf-model.sh google/gemma-4-31b-it
#
# Gated models (Gemma, Llama) require a HuggingFace access token with
# license acceptance. Set HF_TOKEN in the environment.
#
# Prefers `huggingface-cli` when available; falls back to `curl`.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <hf-model-id> [--revision <rev>]" >&2
    exit 2
fi

MODEL_ID="$1"
shift
REVISION="main"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --revision)
            REVISION="$2"
            shift 2
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAFE_NAME="${MODEL_ID//\//__}"
DEST="$REPO_ROOT/tests/fixtures/safetensors-models/$SAFE_NAME"
mkdir -p "$DEST"

echo "Downloading $MODEL_ID @ $REVISION -> $DEST"

if command -v huggingface-cli >/dev/null 2>&1; then
    HF_ARGS=(download "$MODEL_ID" --revision "$REVISION" --local-dir "$DEST" --local-dir-use-symlinks False)
    if [[ -n "${HF_TOKEN:-}" ]]; then
        HF_ARGS+=(--token "$HF_TOKEN")
    fi
    # Only pull the files the safetensors loader actually needs.
    HF_ARGS+=(
        "config.json"
        "tokenizer.json"
        "tokenizer_config.json"
        "model.safetensors"
        "model.safetensors.index.json"
        "*.safetensors"
    )
    huggingface-cli "${HF_ARGS[@]}" || {
        echo "huggingface-cli failed — retrying without file filter" >&2
        huggingface-cli download "$MODEL_ID" --revision "$REVISION" --local-dir "$DEST" \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
    }
else
    echo "huggingface-cli not found; falling back to curl" >&2
    BASE="https://huggingface.co/$MODEL_ID/resolve/$REVISION"
    AUTH=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        AUTH=(-H "Authorization: Bearer $HF_TOKEN")
    fi
    for f in config.json tokenizer.json tokenizer_config.json model.safetensors model.safetensors.index.json; do
        echo "  fetching $f"
        curl -sSL -o "$DEST/$f" "${AUTH[@]}" "$BASE/$f" || {
            echo "  (missing $f — ok for some models)" >&2
            rm -f "$DEST/$f"
        }
    done
    # If we got an index, pull each shard listed in `weight_map`.
    if [[ -f "$DEST/model.safetensors.index.json" ]]; then
        python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    idx = json.load(f)
for shard in sorted(set(idx.get("weight_map", {}).values())):
    print(shard)
' "$DEST/model.safetensors.index.json" | while read -r shard; do
            echo "  fetching shard $shard"
            curl -sSL -o "$DEST/$shard" "${AUTH[@]}" "$BASE/$shard"
        done
    fi
fi

echo "Done. Model layout at $DEST:"
ls -lh "$DEST"
