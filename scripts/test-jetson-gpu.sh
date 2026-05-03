#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Jetson Orin GPU smoke test.
#
# Pass `slim` as $1 to test Dockerfile.jetson.slim (jetson-slim profile)
# instead of the default fat-base Dockerfile.jetson (jetson profile).
#
# Builds smallaios:jetson{,-slim} via docker compose, boots it on the
# local Jetson with the NVIDIA Container Runtime, downloads SqueezeNet
# if needed, and validates:
#   1. /healthz returns 200 within the readiness window
#   2. /readyz returns 200 (model loaded)
#   3. Container logs include "compute 8.7" — proves the integrated
#      Tegra Orin GPU was probed by cudaGetDeviceProperties (NOT a
#      CPU fallback)
#   4. POST /v1/inference against squeezenet returns 200
#
# Always tears the service down on exit (success or failure).
#
# Exit codes:
#   0   all checks passed
#   10  docker compose build failed
#   20  health/ready did not become green within the timeout
#   30  GPU init line missing (silent CPU fallback or wrong device)
#   40  inference request did not return 200, or response shape is wrong
#   50  prerequisite check failed (missing docker / curl / nvidia runtime / compose plugin / python3)
#   51  unknown variant argument (only "slim", "fat", "full", or empty are accepted)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

READY_TIMEOUT=120
case "${1:-}" in
    slim)
        COMPOSE_SERVICE=smallaios-jetson-slim
        COMPOSE_PROFILE=jetson-slim
        VARIANT_LABEL="slim"
        ;;
    ""|fat|full)
        COMPOSE_SERVICE=smallaios-jetson
        COMPOSE_PROFILE=jetson
        VARIANT_LABEL="full"
        ;;
    *)
        echo "Unknown variant: $1 (expected 'slim' or '' / 'fat' / 'full')" >&2
        exit 51
        ;;
esac

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
info()   { yellow "[jetson-gpu] $*"; }
ok()     { green  "[jetson-gpu] $*"; }
fail()   { red    "[jetson-gpu] $*"; }

TMP_FILES_TO_REMOVE=()

cleanup() {
    info "Tearing down container ..."
    docker compose --profile "$COMPOSE_PROFILE" down --remove-orphans >/dev/null 2>&1 || true
    for f in "${TMP_FILES_TO_REMOVE[@]}"; do
        rm -f "$f" 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- Prereqs --------------------------------------------------------
command -v docker >/dev/null 2>&1 || { fail "docker not installed"; exit 50; }
command -v curl   >/dev/null 2>&1 || { fail "curl not installed";   exit 50; }
command -v python3 >/dev/null 2>&1 || { fail "python3 not installed (needed to build the f32 inference payload)"; exit 50; }
docker compose version >/dev/null 2>&1 \
    || { fail "docker compose plugin not installed"; exit 50; }
docker info 2>/dev/null | grep -q "Runtimes:.*nvidia" \
    || { fail "NVIDIA Container Runtime not configured (install nvidia-container-toolkit)"; exit 50; }

# --- Fetch the test model if needed --------------------------------
mkdir -p models
if [ ! -f models/squeezenet.onnx ]; then
    info "Downloading SqueezeNet ONNX (~5 MB) ..."
    curl -L --fail-with-body --progress-bar \
        -o models/squeezenet.onnx \
        https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
fi

# --- Build ----------------------------------------------------------
info "Building smallaios:jetson [${VARIANT_LABEL}] (this can take 5-15 min cold; L4T base ~3 GB) ..."
if ! docker compose --profile "$COMPOSE_PROFILE" build "$COMPOSE_SERVICE"; then
    fail "compose build failed"
    exit 10
fi
ok "Image built."

# --- Boot -----------------------------------------------------------
info "Starting service ..."
docker compose --profile "$COMPOSE_PROFILE" up -d "$COMPOSE_SERVICE"

# --- Wait for ready -------------------------------------------------
info "Waiting up to ${READY_TIMEOUT}s for /healthz + /readyz ..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
    if curl -sf http://localhost:8080/healthz >/dev/null 2>&1 \
        && curl -sf http://localhost:8080/readyz >/dev/null 2>&1; then
        ok "Healthy + ready."
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        fail "service did not become ready within ${READY_TIMEOUT}s"
        docker compose --profile "$COMPOSE_PROFILE" logs --tail=50 "$COMPOSE_SERVICE" || true
        exit 20
    fi
    sleep 2
done

# --- GPU init log check --------------------------------------------
# Poll up to 10s — the GPU init line is logged before the readiness
# probe flips, but `docker compose logs` is async and may briefly lag
# the in-container stdout flush, especially on the slim variant whose
# ldconfig + cuDNN dlopen are heavier on first launch. Use --no-color
# so the grep is not fooled by ANSI escape sequences.
info "Checking container logs for 'compute 8.7' ..."
gpu_log_ok=false
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if docker compose --profile "$COMPOSE_PROFILE" logs --no-color "$COMPOSE_SERVICE" 2>&1 \
        | grep -F "compute 8.7" >/dev/null; then
        gpu_log_ok=true
        break
    fi
    sleep 1
done
if $gpu_log_ok; then
    ok "GPU initialised at compute 8.7 (Tegra Orin)."
else
    fail "Expected 'compute 8.7' in logs — silent CPU fallback or wrong device"
    docker compose --profile "$COMPOSE_PROFILE" logs --no-color --tail=80 "$COMPOSE_SERVICE" || true
    exit 30
fi

# --- Inference round-trip ------------------------------------------
info "Generating a [1, 3, 224, 224] f32 input for SqueezeNet ..."
# Build a real, deterministic input payload so /v1/inference receives
# a well-formed tensor and the smoke test exercises the GPU-side
# parse-decode-dispatch path end-to-end. Use python3 (universally
# present on JetPack hosts) instead of jq, which doesn't handle a
# 150528-element float array efficiently.
REQ_FILE="$(mktemp -t smallaios-jetson-req.XXXXXX.json)"
RESP_FILE="$(mktemp -t smallaios-jetson-resp.XXXXXX.txt)"
# Make sure the temp files are cleaned up alongside container teardown.
TMP_FILES_TO_REMOVE+=("$REQ_FILE" "$RESP_FILE")
python3 - <<'PY' > "$REQ_FILE"
import json
shape = [1, 3, 224, 224]
n = shape[0] * shape[1] * shape[2] * shape[3]
data = [(i % 256) / 255.0 for i in range(n)]
print(json.dumps({
    "model": "squeezenet",
    "inputs": {"data": {"shape": shape, "dtype": "float32", "data": data}}
}))
PY

info "POST /v1/inference (squeezenet, real f32 input, expect 200) ..."
HTTP_CODE=$(curl -s -o "$RESP_FILE" -w "%{http_code}" \
    -X POST http://localhost:8080/v1/inference \
    -H "Content-Type: application/json" \
    --data-binary @"$REQ_FILE" || true)

if [ "$HTTP_CODE" = "200" ]; then
    # Sanity-check the response shape: must contain a 1000-element
    # output tensor (SqueezeNet ImageNet logits) under the model's
    # output name. Don't assert exact values — those depend on
    # numerical determinism we don't guarantee across CUDA versions.
    if python3 -c "
import json, sys
with open('$RESP_FILE') as f: d = json.load(f)
out = next(iter(d['outputs'].values()))
assert out['dtype'] == 'float32', f'wrong dtype: {out[\"dtype\"]}'
assert len(out['data']) == 1000, f'wrong output size: {len(out[\"data\"])}'
" 2>/dev/null; then
        ok "/v1/inference returned 200 with valid [1, 1000] f32 logits."
    else
        fail "/v1/inference returned 200 but response shape is wrong"
        head -c 400 "$RESP_FILE" 2>/dev/null
        exit 40
    fi
else
    fail "/v1/inference returned ${HTTP_CODE} (expected 200)"
    head -c 400 "$RESP_FILE" 2>/dev/null
    exit 40
fi

ok "All Jetson GPU smoke checks passed."
exit 0
