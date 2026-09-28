#!/usr/bin/env bash
set -euo pipefail

serve_args=(
  "$VLM_MODEL"
  --host 0.0.0.0
  --port "$VLM_PORT"
  --served-model-name "$VLM_SERVED_MODEL_NAME"
  --dtype "$VLM_DTYPE"
  --max-model-len "$VLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLM_GPU_MEMORY_UTILIZATION"
  --enable-auto-tool-choice
  --tool-call-parser hermes
  --enforce-eager
)

if [[ "${VLM_ENABLE_MULTIMODAL:-true}" == "true" ]]; then
  serve_args+=(--limit-mm-per-prompt "{\"image\":${VLM_MAX_IMAGES_PER_PROMPT},\"video\":0}")
fi

exec vllm serve "${serve_args[@]}"
