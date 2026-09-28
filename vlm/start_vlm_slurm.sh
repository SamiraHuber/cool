#!/bin/bash
#SBATCH --job-name=vlm_server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=vlm_server_%j.out
# TODO: adjust to your cluster, e.g.:
#SBATCH --partition=gpu
##SBATCH --account=your_account
##SBATCH --qos=your_qos

# Start the Qwen3-VL vLLM OpenAI-compatible server on a SLURM compute node.
#
# Usage:
#   1. sbatch vlm/start_vlm_slurm.sh
#   2. Find the node:  squeue -u $USER -n vlm_server -o "%N"
#      (or read the first lines of vlm_server_<jobid>.out)
#   3. From your local machine, tunnel through the login node:
#      ssh -N -L 8002:<node>:8000 <login-node>
#   4. The simulator's localhost:8002 candidate now reaches the remote VLM.
#      For the web UI, set VLM_API_URL=http://localhost:8002/v1 in
#      docker-compose.yaml (or tunnel straight into the web container's
#      network by pointing it at the remote IP directly if reachable).
#
# Notes:
#   - SLURM kills the job at --time; resubmit when it expires.
#   - The model downloads into HF_HOME on first run; keep it on a
#     persistent/scratch path, not /tmp.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)"
echo "job:  $SLURM_JOB_ID"

# --- config from vlm/vlm.env ------------------------------------------------
set -a
source vlm/vlm.env
set +a

# Persistent Hugging Face cache (adjust to a scratch/project dir if needed).
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

# If your cluster needs modules or a different env, do it here, e.g.:
# module load cuda/12.4
# source .venv-vllm/bin/activate

VLLM=.venv-vllm/bin/vllm
if [[ ! -x "$VLLM" ]]; then
  echo "ERROR: $VLLM not found. Create the venv first:" >&2
  echo "  uv venv .venv-vllm && .venv-vllm/bin/pip install vllm" >&2
  exit 1
fi

args=(
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
  args+=(--limit-mm-per-prompt "{\"image\":${VLM_MAX_IMAGES_PER_PROMPT},\"video\":0}")
fi

exec "$VLLM" serve "${args[@]}"
