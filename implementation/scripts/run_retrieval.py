import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoder_factory import get_encoder
from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retriever import CCTVRetriever
from utils.common import ensure_dir, get_logger, load_config

logger = get_logger("run_retrieval")


def parse_args():
    p = argparse.ArgumentParser(description="Text query -> top-k CCTV clip retrieval")
    p.add_argument("--model", required=True, choices=["vljepa", "clip"])
    p.add_argument("--query", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--index", default=None, help="path to a .faiss index file")
    p.add_argument("--dataset", default=None, help="dataset name to resolve <model>_<dataset>.faiss")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--variant", default=None, help="CLIP variant override")
    p.add_argument("--save", default=None, help="optional output JSON path")
    return p.parse_args()


def resolve_index_path(config, args):
    if args.index:
        return Path(args.index)
    if not args.dataset:
        raise SystemExit("provide either --index or --dataset")
    idx_dir = Path(config["retrieval"]["index_save_dir"])
    return idx_dir / f"{args.model}_{args.dataset}.faiss"


def main():
    args = parse_args()
    config = load_config(args.config)
    index_path = resolve_index_path(config, args)
    if not index_path.is_file():
        raise SystemExit(f"index not found: {index_path}. Run build_index.py first.")

    encoder = get_encoder(args.model, config, variant=args.variant)
    indexer = EmbeddingIndexer.load(index_path)
    retriever = CCTVRetriever(encoder, indexer)

    k = args.k or int(config["retrieval"].get("top_k", 5))
    results = retriever.retrieve(args.query, k=k)

    logger.info("top-%d results for: %s", k, args.query)
    for r in results:
        logger.info(
            "%2d. [%.4f] %-8s %-40s %s",
            r["rank"], r["score"], r.get("dataset", "?"), str(r.get("label", "?"))[:40], r.get("path", "?"),
        )

    if args.save:
        out = ensure_dir(Path(args.save).parent) / Path(args.save).name
        out.write_text(json.dumps({"query": args.query, "model": args.model, "results": results}, indent=2), encoding="utf-8")
        logger.info("saved results -> %s", out)


if __name__ == "__main__":
    main()
