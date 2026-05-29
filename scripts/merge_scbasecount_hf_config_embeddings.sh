#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/large_storage/goodarzilab/bioreason_cell/embeddings}"

CONFIGS=(
  reasoning-scBaseCount-stringent-pathway-zscores
  reasoning-scBaseCount-within-tissue-contrastive
  reasoning-scBaseCount-cross-tissue-contrastive
)

for hf_config in "${CONFIGS[@]}"; do
  echo "[$(date --iso-8601=seconds)] Merging ${hf_config}" >&2
  uv run python scripts/embed_h5ad_mean_pool.py \
    --hf-config "${hf_config}" \
    --output-root "${OUTPUT_ROOT}" \
    --merge \
    --overwrite
done
