#!/usr/bin/env python
"""Embed zscore rows at (cell_file, cell_id) granularity."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_PARQUET = Path("/large_storage/goodarzilab/bioreason_cell/scBaseCount/scBaseCount_per_cell_topN_zscored.parquet")
DEFAULT_OUTPUT_ROOT = Path("/large_storage/goodarzilab/bioreason_cell/embeddings")
DEFAULT_MODEL_FOLDER = Path("/large_storage/goodarzilab/cytoprism/models/state/SE-600M")
DEFAULT_HF_CONFIG = "reasoning-scBaseCount-stringent-pathway-zscores"

log = logging.getLogger("embed_zscore_cell_mean_pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG, help="Output folder name.")
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--path-column", default="cell_file")
    parser.add_argument("--cell-id-column", default="cell_id")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--protein-embeddings", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", default="embeddings.pt")
    parser.add_argument("--shard-dir-name", default="cell_shards")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-files", type=int, default=None, help="Limit unique cell files for smoke tests.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit parquet rows before deduping for smoke tests.")
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--array-count", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def load_pairs(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input_parquet, columns=[args.path_column, args.cell_id_column])
    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()

    df = df.dropna(subset=[args.path_column, args.cell_id_column]).copy()
    df[args.path_column] = df[args.path_column].astype(str)
    df[args.cell_id_column] = df[args.cell_id_column].astype(str)
    df["source_row"] = df.index.astype(np.int64)
    df = df.drop_duplicates([args.path_column, args.cell_id_column], keep="first").reset_index(drop=True)

    missing = [path for path in df[args.path_column].unique() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} h5ad inputs do not exist; first missing path: {missing[0]}")
    return df


def shard_files(files: list[str], index: int | None, count: int | None) -> tuple[int, int, list[str]]:
    if index is None:
        slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        index = int(slurm_index) if slurm_index is not None else 0
    if count is None:
        slurm_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        count = int(slurm_count) if slurm_count is not None else 1

    if index < 0 or count < 1 or index >= count:
        raise ValueError(f"Invalid shard index/count: {index}/{count}")
    return index, count, files[index::count]


def output_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.hf_config


def tensor_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def encode_adata_object(inferer: Any, adata: anndata.AnnData, dataset_name: str, batch_size: int | None) -> np.ndarray:
    from omegaconf import OmegaConf

    from state.emb.data import create_dataloader
    from state.emb.utils import get_precision_config

    adata = inferer._convert_to_csr(adata)
    gene_column = inferer._auto_detect_gene_column(adata)

    dataloader_cfg = inferer._vci_conf
    if batch_size is not None:
        dataloader_cfg = OmegaConf.create(OmegaConf.to_container(inferer._vci_conf, resolve=True))
        if not hasattr(dataloader_cfg, "model"):
            dataloader_cfg["model"] = {}
        dataloader_cfg.model.batch_size = int(batch_size)
        log.info("Using override batch size: %s", batch_size)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    precision = get_precision_config(device_type=device_type)
    dataloader = create_dataloader(
        dataloader_cfg,
        adata=adata,
        adata_name=dataset_name,
        shape_dict={dataset_name: adata.shape},
        data_dir=None,
        shuffle=False,
        protein_embeds=inferer.protein_embeds,
        precision=precision,
        gene_column=gene_column,
    )

    all_embeddings = []
    all_ds_embeddings = []
    for embeddings, ds_embeddings in inferer.encode(dataloader):
        all_embeddings.append(embeddings)
        if ds_embeddings is not None:
            all_ds_embeddings.append(ds_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    if all_ds_embeddings:
        ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(np.float32)
        embeddings = np.concatenate([embeddings, ds_embeddings], axis=-1)
    return embeddings


def records_for_file(
    path: str,
    file_rows: pd.DataFrame,
    embeddings: np.ndarray,
    spans: list[tuple[str, int, int, int]],
    cell_id_column: str,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    records: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    rows_by_cell = {str(cell_id): int(source_row) for cell_id, source_row in file_rows[[cell_id_column, "source_row"]].itertuples(index=False, name=None)}
    stem = Path(path).stem

    for cell_id, start, end, n_matches in spans:
        vector = torch.from_numpy(embeddings[start:end].mean(axis=0).astype(np.float32))
        key = f"{stem}::{cell_id}"
        records.append(
            {
                "key": key,
                "cell_file": path,
                "cell_id": cell_id,
                "path": path,
                "n_cells": int(n_matches),
                "source_row": rows_by_cell[cell_id],
            }
        )
        vectors.append(vector)

    return records, vectors


def embed_shard(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from state.emb.inference import Inference

    df = load_pairs(args)
    all_files = sorted(df[args.path_column].unique())
    if args.max_files is not None:
        all_files = all_files[: args.max_files]
        df = df[df[args.path_column].isin(all_files)].copy()

    index, count, files = shard_files(all_files, args.array_index, args.array_count)
    out_dir = output_dir(args)
    shard_dir = out_dir / args.shard_dir_name
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

    for path in files:
        file_rows = df[df[args.path_column] == path].copy()
        requested_ids = file_rows[args.cell_id_column].tolist()
        requested_set = set(requested_ids)

        log.info("Embedding %s (%d requested cell IDs)", path, len(requested_ids))
        adata = anndata.read_h5ad(path, backed="r")

        positions_by_id: dict[str, list[int]] = defaultdict(list)
        for pos, cell_id in enumerate(map(str, adata.obs_names)):
            if cell_id in requested_set:
                positions_by_id[cell_id].append(pos)

        missing = [cell_id for cell_id in requested_ids if cell_id not in positions_by_id]
        if missing:
            raise KeyError(f"{len(missing)} requested cell IDs missing from {path}; first missing: {missing[0]}")

        selected_positions: list[int] = []
        spans: list[tuple[str, int, int, int]] = []
        for cell_id in requested_ids:
            positions = positions_by_id[cell_id]
            start = len(selected_positions)
            selected_positions.extend(positions)
            end = len(selected_positions)
            spans.append((cell_id, start, end, len(positions)))

        sub_adata = adata[selected_positions, :].to_memory()
        if getattr(adata, "isbacked", False):
            adata.file.close()
        embeddings = encode_adata_object(inferer, sub_adata, dataset_name=Path(path).stem, batch_size=args.batch_size)
        file_records, file_vectors = records_for_file(
            path=path,
            file_rows=file_rows,
            embeddings=embeddings,
            spans=spans,
            cell_id_column=args.cell_id_column,
        )
        records.extend(file_records)
        vectors.extend(file_vectors)

    out_dtype = tensor_dtype(args.dtype)
    payload = {
        "hf_config": args.hf_config,
        "granularity": "cell_file_cell_id",
        "model_folder": str(args.model_folder),
        "checkpoint": str(checkpoint),
        "shard_index": index,
        "shard_count": count,
        "records": records,
        "keys": [record["key"] for record in records],
        "cell_files": [record["cell_file"] for record in records],
        "cell_ids": [record["cell_id"] for record in records],
        "paths": [record["path"] for record in records],
        "n_cells": torch.tensor([record["n_cells"] for record in records], dtype=torch.int16),
        "source_rows": torch.tensor([record["source_row"] for record in records], dtype=torch.int64),
        "embeddings": torch.stack(vectors).to(dtype=out_dtype) if vectors else torch.empty((0, 0), dtype=out_dtype),
    }
    torch.save(payload, shard_path)
    log.info("Wrote %s with %d vectors", shard_path, len(vectors))


def merge_shards(args: argparse.Namespace) -> None:
    out_dir = output_dir(args)
    output_path = out_dir / args.output_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    shard_paths = sorted((out_dir / args.shard_dir_name).glob("shard_*_of_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {out_dir / args.shard_dir_name}")

    records: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        records.extend(shard["records"])
        embeddings.append(shard["embeddings"])

    merged_embeddings = torch.cat(embeddings, dim=0)
    order = sorted(range(len(records)), key=lambda i: (int(records[i]["source_row"]), records[i]["cell_file"], records[i]["cell_id"]))
    merged_embeddings = merged_embeddings[order].contiguous()
    merged_records = [records[i] for i in order]
    key_to_index = {record["key"]: i for i, record in enumerate(merged_records)}
    pair_to_index = {f"{record['cell_file']}\t{record['cell_id']}": i for i, record in enumerate(merged_records)}

    payload = {
        "hf_config": args.hf_config,
        "granularity": "cell_file_cell_id",
        "records": merged_records,
        "keys": [record["key"] for record in merged_records],
        "cell_files": [record["cell_file"] for record in merged_records],
        "cell_ids": [record["cell_id"] for record in merged_records],
        "paths": [record["path"] for record in merged_records],
        "key_to_index": key_to_index,
        "pair_to_index": pair_to_index,
        "n_cells": torch.tensor([record["n_cells"] for record in merged_records], dtype=torch.int16),
        "source_rows": torch.tensor([record["source_row"] for record in merged_records], dtype=torch.int64),
        "embeddings": merged_embeddings,
    }
    torch.save(payload, output_path)
    log.info("Wrote merged output %s with %d vectors", output_path, len(merged_records))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    if args.merge:
        merge_shards(args)
    else:
        embed_shard(args)


if __name__ == "__main__":
    main()
