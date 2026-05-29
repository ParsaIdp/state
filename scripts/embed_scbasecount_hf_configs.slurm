#!/bin/bash
#SBATCH --account=cell_reason
#SBATCH --job-name=embed_scbasecount
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=cell_reason_gpu_priority
#SBATCH --array=0-47%24
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

MODEL_FOLDER="${MODEL_FOLDER:-/large_storage/goodarzilab/cytoprism/models/state/SE-600M}"
INPUT_DIR="${INPUT_DIR:-/large_storage/goodarzilab/bioreason_cell/scBaseCount/sample_h5ads_by_celltype_tissue}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/large_storage/goodarzilab/bioreason_cell/embeddings}"
HF_DATASET="${HF_DATASET:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/large_storage/goodarzilab/bioreason_cell/cache_dir}"
MANIFEST="${MANIFEST:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_INPUTS="${MAX_INPUTS:-}"

CONFIGS=(
  reasoning-scBaseCount-stringent-pathway-zscores
  reasoning-scBaseCount-within-tissue-contrastive
  reasoning-scBaseCount-cross-tissue-contrastive
)

if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a CONFIGS <<< "${CONFIGS_OVERRIDE}"
fi

SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT}}"

declare -A PARQUETS=(
  [reasoning-scBaseCount-stringent-pathway-zscores]="/large_storage/goodarzilab/bioreason_cell/scBaseCount/scBaseCount_per_cell_topN_zscored.parquet"
  [reasoning-scBaseCount-within-tissue-contrastive]="/large_storage/goodarzilab/bioreason_cell/scBaseCount/scBaseCount_within_tissue_contrastive_hf.parquet"
  [reasoning-scBaseCount-cross-tissue-contrastive]="/large_storage/goodarzilab/bioreason_cell/scBaseCount/scBaseCount_cross_tissue_contrastive_hf.parquet"
)

declare -A PATH_COLUMNS=(
  [reasoning-scBaseCount-stringent-pathway-zscores]="cell_file"
  [reasoning-scBaseCount-within-tissue-contrastive]="cell_file_A cell_file_B"
  [reasoning-scBaseCount-cross-tissue-contrastive]="cell_file_A cell_file_B"
)

mkdir -p logs

for hf_config in "${CONFIGS[@]}"; do
  echo "[$(date --iso-8601=seconds)] Embedding ${hf_config} shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" >&2

  args=(
    scripts/embed_h5ad_mean_pool.py
    --hf-config "${hf_config}"
    --model-folder "${MODEL_FOLDER}"
    --output-root "${OUTPUT_ROOT}"
    --array-index "${SLURM_ARRAY_TASK_ID}"
    --array-count "${SHARD_COUNT}"
  )

  if [[ -n "${MANIFEST}" ]]; then
    args+=(--manifest "${MANIFEST}")
  elif [[ -n "${HF_DATASET}" ]]; then
    args+=(--hf-dataset "${HF_DATASET}" --hf-cache-dir "${HF_CACHE_DIR}")
  elif [[ -f "${PARQUETS[${hf_config}]}" ]]; then
    # These local parquet files are the materialized Hugging Face config rows.
    # They carry the h5ad file paths in config-specific columns.
    read -r -a cols <<< "${PATH_COLUMNS[${hf_config}]}"
    args+=(--input-parquet "${PARQUETS[${hf_config}]}" --path-columns "${cols[@]}")
  else
    args+=(--input-dir "${INPUT_DIR}")
  fi

  if [[ -n "${BATCH_SIZE}" ]]; then
    args+=(--batch-size "${BATCH_SIZE}")
  fi

  if [[ -n "${MAX_INPUTS}" ]]; then
    args+=(--max-inputs "${MAX_INPUTS}")
  fi

  uv run python "${args[@]}"
done

echo "[$(date --iso-8601=seconds)] Shard complete" >&2
