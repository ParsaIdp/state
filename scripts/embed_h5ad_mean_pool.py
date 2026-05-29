#!/usr/bin/env python
"""Embed h5ad files with State and save one mean-pooled torch index."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_INPUT_DIR = Path("/large_storage/goodarzilab/bioreason_cell/scBaseCount/sample_h5ads_by_celltype_tissue")
DEFAULT_OUTPUT_ROOT = Path("/large_storage/goodarzilab/bioreason_cell/embeddings")
DEFAULT_MODEL_FOLDER = Path("/large_storage/goodarzilab/cytoprism/models/state/SE-600M")
H5AD_COLUMNS = (
    "h5ad",
    "h5ad_path",
    "adata",
    "adata_path",
    "file",
    "file_path",
    "path",
)

log = logging.getLogger("embed_h5ad_mean_pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-config", required=True, help="Hugging Face config name; also used as output folder name.")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to newest *.ckpt in model folder.")
    parser.add_argument("--config", type=Path, default=None, help="Optional State config override.")
    parser.add_argument("--protein-embeddings", type=Path, default=None, help="Optional protein_embeddings.pt override.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing input .h5ad files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Text/JSON manifest of h5ad paths.")
    parser.add_argument("--input-parquet", type=Path, default=None, help="Parquet file containing h5ad path columns.")
    parser.add_argument(
        "--path-columns",
        nargs="+",
        default=None,
        help="Column names in --input-parquet that contain h5ad paths. Defaults to common path column names.",
    )
    parser.add_argument("--hf-dataset", default=None, help="Optional HF dataset name to resolve local h5ad paths from.")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("/large_storage/goodarzilab/bioreason_cell/cache_dir"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", default="embeddings.pt")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding forward batch size.")
    parser.add_argument("--max-inputs", type=int, default=None, help="Limit resolved inputs for smoke tests.")
    parser.add_argument("--array-index", type=int, default=None, help="0-based shard index. Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--array-count", type=int, default=None, help="Number of shards. Defaults to SLURM_ARRAY_TASK_COUNT.")
    parser.add_argument("--merge", action="store_true", help="Merge existing shards instead of embedding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing shard/final output.")
    return parser.parse_args()


def resolve_checkpoint(model_folder: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint

    ckpts = sorted(model_folder.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No *.ckpt files found under {model_folder}")
    return ckpts[-1]


def load_protein_embeddings(model_folder: Path, path: Path | None) -> Any:
    if path is not None:
        return torch.load(path, map_location="cpu", weights_only=False)

    candidates = [model_folder / "protein_embeddings.pt", *sorted(model_folder.glob("protein_embeddings*.pt"))]
    for candidate in candidates:
        if candidate.exists():
            log.info("Using protein embeddings: %s", candidate)
            return torch.load(candidate, map_location="cpu", weights_only=False)
    return None


def load_manifest(path: Path) -> list[Path]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for key in ("paths", "h5ad_paths", "files"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"JSON manifest must contain a list of paths: {path}")
        return [Path(str(item)) for item in payload]

    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def paths_from_hf_dataset(dataset_name: str, config: str, cache_dir: Path) -> list[Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` or use --input-dir/--manifest instead of --hf-dataset.") from exc

    dataset = load_dataset(dataset_name, config, cache_dir=str(cache_dir))
    paths: list[Path] = []
    splits = dataset.values() if hasattr(dataset, "values") else [dataset]
    for split in splits:
        columns = getattr(split, "column_names", [])
        for column in H5AD_COLUMNS:
            if column not in columns:
                continue
            for value in split[column]:
                if value:
                    paths.append(Path(str(value)))
            break
    return sorted(dict.fromkeys(paths))


def paths_from_parquet(path: Path, columns: list[str] | None) -> list[Path]:
    import pandas as pd

    df = pd.read_parquet(path)
    if columns is None:
        columns = [column for column in (*H5AD_COLUMNS, "cell_file", "cell_file_A", "cell_file_B") if column in df]
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(f"Missing path columns in {path}: {missing}")

    paths: list[Path] = []
    for column in columns:
        paths.extend(Path(str(value)) for value in df[column].dropna().unique() if str(value).endswith(".h5ad"))
    return sorted(dict.fromkeys(paths))


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.manifest is not None:
        paths = load_manifest(args.manifest)
    elif args.input_parquet is not None:
        paths = paths_from_parquet(args.input_parquet, args.path_columns)
    elif args.hf_dataset is not None:
        paths = paths_from_hf_dataset(args.hf_dataset, args.hf_config, args.hf_cache_dir)
    else:
        paths = sorted(args.input_dir.glob("*.h5ad"))

    paths = [path for path in paths if path.suffix == ".h5ad"]
    if args.max_inputs is not None:
        paths = paths[: args.max_inputs]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} h5ad inputs do not exist; first missing path: {missing[0]}")
    if not paths:
        raise FileNotFoundError("No .h5ad inputs resolved.")
    return paths


def shard_paths(paths: list[Path], index: int | None, count: int | None) -> tuple[int, int, list[Path]]:
    if index is None:
        slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        index = int(slurm_index) if slurm_index is not None else 0
    if count is None:
        slurm_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        count = int(slurm_count) if slurm_count is not None else 1

    if index < 0 or count < 1 or index >= count:
        raise ValueError(f"Invalid shard index/count: {index}/{count}")
    return index, count, paths[index::count]


def output_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.hf_config


def embed_shard(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from state.emb.inference import Inference

    all_paths = resolve_inputs(args)
    index, count, paths = shard_paths(all_paths, args.array_index, args.array_count)
    out_dir = output_dir(args)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{index:05d}_of_{count:05d}.pt"

    if shard_path.exists() and not args.overwrite:
        log.info("Shard already exists, skipping: %s", shard_path)
        return

    checkpoint = resolve_checkpoint(args.model_folder, args.checkpoint)
    conf = OmegaConf.load(args.config) if args.config is not None else None
    protein_embeddings = load_protein_embeddings(args.model_folder, args.protein_embeddings)

    log.info("Loading checkpoint: %s", checkpoint)
    inferer = Inference(cfg=conf, protein_embeds=protein_embeddings)
    inferer.load_model(str(checkpoint))

    records: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    for path in paths:
        log.info("Embedding %s", path)
        embeddings = inferer.encode_adata(
            input_adata_path=str(path),
            output_adata_path=None,
            emb_key="X_state",
            batch_size=args.batch_size,
        )
        vector = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).mean(dim=0)
        records.append({"key": path.stem, "path": str(path), "n_cells": int(embeddings.shape[0])})
        vectors.append(vector.cpu())

    payload = {
        "hf_config": args.hf_config,
        "model_folder": str(args.model_folder),
        "checkpoint": str(checkpoint),
        "shard_index": index,
        "shard_count": count,
        "records": records,
        "keys": [record["key"] for record in records],
        "paths": [record["path"] for record in records],
        "n_cells": torch.tensor([record["n_cells"] for record in records], dtype=torch.int32),
        "embeddings": torch.stack(vectors) if vectors else torch.empty((0, 0), dtype=torch.float32),
    }
    torch.save(payload, shard_path)
    log.info("Wrote %s with %d vectors", shard_path, len(vectors))


def merge_shards(args: argparse.Namespace) -> None:
    out_dir = output_dir(args)
    output_path = out_dir / args.output_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    shard_paths = sorted((out_dir / "shards").glob("shard_*_of_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {out_dir / 'shards'}")

    records: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        records.extend(shard["records"])
        embeddings.append(shard["embeddings"])

    order = sorted(range(len(records)), key=lambda i: records[i]["path"])
    merged_embeddings = torch.cat(embeddings, dim=0)[order].contiguous()
    merged_records = [records[i] for i in order]
    key_to_index = {record["key"]: i for i, record in enumerate(merged_records)}

    payload = {
        "hf_config": args.hf_config,
        "records": merged_records,
        "keys": [record["key"] for record in merged_records],
        "paths": [record["path"] for record in merged_records],
        "key_to_index": key_to_index,
        "n_cells": torch.tensor([record["n_cells"] for record in merged_records], dtype=torch.int32),
        "embeddings": merged_embeddings,
    }
    torch.save(payload, output_path)
    log.info("Wrote %s with shape %s", output_path, tuple(merged_embeddings.shape))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    if args.merge:
        merge_shards(args)
    else:
        embed_shard(args)


if __name__ == "__main__":
    main()
