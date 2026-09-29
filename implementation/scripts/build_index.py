import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from retrieval.embedding_indexer import EmbeddingIndexer
from utils.common import ensure_dir, get_logger, load_config

logger = get_logger("build_index")


def parse_args():
    p = argparse.ArgumentParser(description="Build a FAISS index from saved embeddings")
    p.add_argument("--model", required=True, help="model name used during extraction (e.g. clip)")
    p.add_argument("--dataset", required=True, help="dataset name used during extraction (e.g. ucf_crime)")
    p.add_argument("--config", default=None)
    p.add_argument("--embeddings-dir", default=None)
    p.add_argument("--index-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path(config["retrieval"]["embedding_save_dir"])
    idx_dir = ensure_dir(args.index_dir or config["retrieval"]["index_save_dir"])

    npy_path = emb_dir / f"{args.model}_{args.dataset}_embeddings.npy"
    json_path = emb_dir / f"{args.model}_{args.dataset}_embeddings.json"
    if not npy_path.is_file() or not json_path.is_file():
        raise SystemExit(f"embeddings not found: {npy_path} / {json_path}. Run extract_embeddings.py first.")

    embeddings = np.load(npy_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if len(records) != len(embeddings):
        raise SystemExit(f"embedding/metadata mismatch: {len(embeddings)} vectors vs {len(records)} records")

    indexer = EmbeddingIndexer(
        embedding_dim=embeddings.shape[1],
        index_type=config["retrieval"].get("index_type", "FlatIP"),
        auto_ivf_threshold=config["retrieval"].get("auto_ivf_threshold", 20000),
    )
    indexer.add(embeddings, metadata=records)
    out_path = idx_dir / f"{args.model}_{args.dataset}.faiss"
    indexer.save(out_path)
    logger.info("indexed %d vectors (dim=%d) -> %s", len(indexer), embeddings.shape[1], out_path)


if __name__ == "__main__":
    main()
