import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.faithfulness_metrics import grounding_score
from models.encoder_factory import get_encoder
from reporting.evidence_utils import evidence_text
from reporting.llm_reporter import IncidentReporter
from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retriever import CCTVRetriever
from utils.common import ensure_dir, get_logger, load_config

logger = get_logger("generate_report")


def parse_args():
    p = argparse.ArgumentParser(description="Retrieve evidence and generate an LLM incident report")
    p.add_argument("--model", required=True, choices=["vljepa", "clip"])
    p.add_argument("--query", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--index", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--ungrounded", action="store_true", help="H3 control: query only, no evidence")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def resolve_index_path(config, args):
    if args.index:
        return Path(args.index)
    if not args.dataset:
        raise SystemExit("provide either --index or --dataset")
    return Path(config["retrieval"]["index_save_dir"]) / f"{args.model}_{args.dataset}.faiss"


def main():
    args = parse_args()
    config = load_config(args.config)
    index_path = resolve_index_path(config, args)
    if not index_path.is_file():
        raise SystemExit(f"index not found: {index_path}. Run build_index.py first.")

    encoder = get_encoder(args.model, config)
    retriever = CCTVRetriever(encoder, EmbeddingIndexer.load(index_path))
    k = args.k or int(config["retrieval"].get("top_k", 5))
    evidence = retriever.retrieve(args.query, k=k)

    reporter = IncidentReporter(config)
    start = time.perf_counter()
    report = reporter.generate_report(evidence, args.query, grounded=not args.ungrounded)
    elapsed = time.perf_counter() - start

    ev_text = evidence_text(evidence)
    score, _ = grounding_score(report, ev_text)
    payload = {
        "query": args.query,
        "retrieval_model": args.model,
        "llm": config["llm"]["model"],
        "grounded": not args.ungrounded,
        "report": report,
        "evidence": evidence,
        "grounding_score": score,
        "hallucination_rate": 1.0 - score,
        "generation_seconds": elapsed,
    }

    out_dir = ensure_dir(args.output_dir or (Path(config["experiment"].get("output_dir", "results")) / "reports"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.model}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("report generated in %.1fs (grounding %.2f) -> %s", elapsed, score, out_path)
    logger.info("\n%s", report)


if __name__ == "__main__":
    main()
