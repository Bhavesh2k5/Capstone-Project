import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import DATASETS, build_dataset
from data.data_utils import collate_fn
from models.encoder_factory import get_encoder
from utils.common import clear_gpu_cache, ensure_dir, get_logger, load_config, set_seed

logger = get_logger("extract_embeddings")

CHUNK_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description="Batch-encode dataset clips into embedding files")
    p.add_argument("--model", required=True, choices=["vljepa", "clip"])
    p.add_argument("--dataset", required=True, help="ucf_crime | ucf_arg | rlvs | tinyvirat | all")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--split", default=None, help="dataset split override (ucf_crime mode / tinyvirat split)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap number of clips (smoke tests)")
    p.add_argument("--variant", default=None, help="CLIP variant override: primary | secondary")
    return p.parse_args()


def output_paths(config, model_name, dataset_name):
    out_dir = ensure_dir(config["retrieval"]["embedding_save_dir"])
    base = out_dir / f"{model_name}_{dataset_name}_embeddings"
    return base.with_suffix(".npy"), base.with_suffix(".json")


def load_existing(npy_path, json_path):
    if not npy_path.is_file() or not json_path.is_file():
        return [], []
    try:
        embeddings = np.load(npy_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        records = data.get("records", [])
        if len(records) == len(embeddings):
            return [row for row in embeddings], records
    except Exception as exc:
        logger.warning("could not reuse existing embeddings (%s); starting fresh", exc)
    return [], []


def save_all(npy_path, json_path, embedding_rows, records, model_name, dataset_name, dim):
    matrix = np.vstack(embedding_rows).astype(np.float32) if embedding_rows else np.zeros((0, dim), np.float32)
    np.save(npy_path, matrix)
    json_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "dataset": dataset_name,
                "dim": int(dim),
                "count": len(records),
                "records": records,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def encode_dataset(model_name, dataset_name, config, args):
    vcfg = config["models"][model_name]
    batch_size = args.batch_size or int(vcfg.get("batch_size", 8))
    workers = args.num_workers
    if workers is None:
        workers = 0 if sys.platform.startswith("win") else min(4, (torch.get_num_threads() or 1))

    npy_path, json_path = output_paths(config, model_name, dataset_name)
    embedding_rows, records = load_existing(npy_path, json_path)
    processed = {r["path"] for r in records}

    dataset = build_dataset(dataset_name, config, split=args.split)
    if args.limit:
        dataset.samples = dataset.samples[: args.limit]
    if not len(dataset):
        logger.warning("[%s] %s: empty dataset - nothing to encode", model_name, dataset_name)
        return None

    order = [i for i in range(len(dataset)) if dataset.samples[i]["path"] not in processed]
    if not order:
        logger.info("[%s] %s: all %d clips already encoded - skipping", model_name, dataset_name, len(records))
        return npy_path, json_path
    logger.info(
        "[%s] %s: %d clips total (%d done, %d remaining)",
        model_name, dataset_name, len(dataset), len(records), len(order),
    )

    encoder = get_encoder(model_name, config, variant=args.variant)
    dim = encoder.get_embedding_dim()

    def encode_chunk(chunk_indices, bs):
        subset = Subset(dataset, chunk_indices)
        loader = DataLoader(subset, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate_fn)
        done = 0
        try:
            for batch in loader:
                feats = encoder.encode_video(batch["video"]).numpy().astype(np.float32)
                for row, path, label, meta in zip(feats, batch["path"], batch["label"], batch["metadata"]):
                    embedding_rows.append(row)
                    records.append({"path": path, "label": label, "dataset": dataset_name, "metadata": meta})
                done += len(feats)
                save_all(npy_path, json_path, embedding_rows, records, model_name, dataset_name, dim)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            clear_gpu_cache()
            new_bs = max(1, bs // 2)
            logger.warning("CUDA OOM - halving batch size to %d and retrying", new_bs)
            return done, new_bs
        return done, bs

    current_batch = batch_size
    pos = 0
    while pos < len(order):
        chunk = order[pos : pos + CHUNK_SIZE]
        offset = 0
        while offset < len(chunk):
            remaining_chunk = chunk[offset:]
            done, current_batch = encode_chunk(remaining_chunk, current_batch)
            if done == 0:
                break
            offset += done
        pos += len(chunk)
        clear_gpu_cache()

    save_all(npy_path, json_path, embedding_rows, records, model_name, dataset_name, dim)
    logger.info("[%s] %s: saved %d embeddings -> %s", model_name, dataset_name, len(records), npy_path)
    return npy_path, json_path


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"].get("seed", 42))
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset '{name}'; expected one of {DATASETS} or 'all'")
        encode_dataset(args.model, name, config, args)


if __name__ == "__main__":
    main()
