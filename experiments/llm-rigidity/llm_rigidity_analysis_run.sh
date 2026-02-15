#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="llm_rigidity_analysis.py"
LOG_FILE="llm-rigidity/results/analysis.log"
mkdir -p "llm-rigidity/results"

# Use GPU 1 specifically
GPU_ID=1

# Verify GPU 1 is available
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU check unavailable." | tee "$LOG_FILE"
  exit 1
fi

# Check GPU 1 status
GPU_STATUS=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")
if [[ -z "${GPU_STATUS}" ]]; then
  echo "ERROR: GPU ${GPU_ID} not found or not accessible" | tee "$LOG_FILE"
  exit 1
fi

# Extract memory info
FREE_MEM=$(echo "${GPU_STATUS}" | awk -F',' '{
  gsub(/ /,"");
  mem_total = $2 + 0;
  mem_used = $1 + 0;
  print mem_total - mem_used
}')
GPU_UTIL=$(echo "${GPU_STATUS}" | awk -F',' '{gsub(/ /,""); print $3}')

echo "Using GPU ${GPU_ID}: ${FREE_MEM} MB free, ${GPU_UTIL}% utilization" | tee "$LOG_FILE"

# Warn if GPU is busy
if [[ ${GPU_UTIL} -gt 50 ]] || [[ ${FREE_MEM} -lt 4096 ]]; then
  echo "WARNING: GPU ${GPU_ID} may not have enough resources (${FREE_MEM} MB free, ${GPU_UTIL}% util)" | tee -a "$LOG_FILE"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

source sid/bin/activate

python "$SCRIPT_NAME" 2>&1 | tee -a "$LOG_FILE"
