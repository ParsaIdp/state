#!/bin/bash
#SBATCH --account=cell_reason
#SBATCH --job-name=embed_zscore_cells
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=cell_reason_gpu_priority
#SBATCH --array=0-63%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

MODEL_FOLDER="${MODEL_FOLDER:-/large_storage/goodarzilab/cytoprism/models/state/SE-600M}"
INPUT_PARQUET="${INPUT_PARQUET:-/large_storage/goodarzilab/bioreason_cell/scBaseCount/scBaseCount_per_cell_topN_zscored.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/large_storage/goodarzilab/bioreason_cell/embeddings}"
HF_CONFIG="${HF_CONFIG:-reasoning-scBaseCount-stringent-pathway-zscores}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT}}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_NAME="${OUTPUT_NAME:-embeddings.cell_bf16.pt}"
SHARD_DIR_NAME="${SHARD_DIR_NAME:-cell_shards_bf16}"
MAX_FILES="${MAX_FILES:-}"
MAX_ROWS="${MAX_ROWS:-}"

mkdir -p logs

args=(
  scripts/embed_zscore_cell_mean_pool.py
  --hf-config "${HF_CONFIG}"
  --input-parquet "${INPUT_PARQUET}"
  --model-folder "${MODEL_FOLDER}"
  --output-root "${OUTPUT_ROOT}"
  --output-name "${OUTPUT_NAME}"
  --shard-dir-name "${SHARD_DIR_NAME}"
  --array-index "${SLURM_ARRAY_TASK_ID}"
  --array-count "${SHARD_COUNT}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
)

if [[ -n "${MAX_FILES}" ]]; then
  args+=(--max-files "${MAX_FILES}")
fi

if [[ -n "${MAX_ROWS}" ]]; then
  args+=(--max-rows "${MAX_ROWS}")
fi

echo "[$(date --iso-8601=seconds)] Embedding zscore cell shard ${SLURM_ARRAY_TASK_ID}/${SHARD_COUNT}" >&2
uv run python "${args[@]}"
echo "[$(date --iso-8601=seconds)] Shard complete" >&2
