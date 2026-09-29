import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data import build_dataset
from evaluation import build_queries
from evaluation.efficiency_metrics import (
    measure_encoding_latency,
    measure_memory_usage,
    measure_throughput,
)
from evaluation.faithfulness_metrics import compute_faithfulness_report
from evaluation.retrieval_metrics import compute_all_metrics
from models.encoder_factory import get_encoder
from reporting.evidence_utils import evidence_text
from reporting.llm_reporter import IncidentReporter
from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retriever import CCTVRetriever
from utils.common import ensure_dir, get_logger, load_config, set_seed

logger = get_logger("run_evaluation")


def parse_args():
    p = argparse.ArgumentParser(description="Run retrieval / efficiency / faithfulness evaluation")
    p.add_argument("--model", required=True, choices=["vljepa", "clip"])
    p.add_argument("--dataset", required=True, help="ucf_crime | ucf_arg | rlvs | tinyvirat")
    p.add_argument("--config", default=None)
    p.add_argument("--metrics", default="retrieval,efficiency",
                   help="comma list of: retrieval,efficiency,faithfulness")
    p.add_argument("--variant", default=None, help="CLIP variant override")
    return p.parse_args()


def resolve_index(config, model_name, dataset_name):
    path = Path(config["retrieval"]["index_save_dir"]) / f"{model_name}_{dataset_name}.faiss"
    if not path.is_file():
        raise SystemExit(f"index not found: {path}. Run extract_embeddings.py + build_index.py first.")
    return path


def retrieval_eval(config, dataset_name, retriever):
    ecfg = config["evaluation"]
    k_values = list(ecfg.get("recall_k_values", [1, 5, 10]))
    k_max = max(k_values)
    queries = build_queries(dataset_name, config)
    results_labels, ground_truth = [], []
    for item in queries:
        results = retriever.retrieve(item["query"], k=k_max)
        results_labels.append([r.get("label", "") for r in results])
        ground_truth.append(item["relevant"])
    return compute_all_metrics(results_labels, ground_truth, k_values=k_values)


def efficiency_eval(config, encoder, dataset_name):
    ecfg = config["evaluation"]
    dataset = build_dataset(dataset_name, config)
    if not len(dataset):
        return {}
    cap = int(ecfg.get("max_eval_clips", 0)) or len(dataset)
    n = min(len(dataset), cap, 8)
    samples = [dataset[i]["video"] for i in range(n)]
    latency = measure_encoding_latency(encoder, samples, n_runs=int(ecfg.get("latency_runs", 50)))
    memory = measure_memory_usage(encoder, samples[0])
    throughput = measure_throughput(
        encoder, samples, batch_size=min(4, n), n_batches=int(ecfg.get("throughput_batches", 4))
    )
    return {
        "latency_mean_ms": round(latency["mean_ms"], 2),
        "latency_std_ms": round(latency["std_ms"], 2),
        "peak_gpu_mb": round(memory["peak_gpu_mb"], 1) if memory["peak_gpu_mb"] is not None else None,
        "clips_per_second": round(throughput["clips_per_second"], 2),
    }


def faithfulness_eval(config, retriever, dataset_name):
    reporter = IncidentReporter(config)
    queries = build_queries(dataset_name, config)
    reports, evidence_texts, modes = [], [], []
    for item in queries[: min(4, len(queries))]:
        results = retriever.retrieve(item["query"], k=int(config["retrieval"].get("top_k", 5)))
        ev_text = evidence_text(results)
        reports.append(reporter.generate_report(results, item["query"], grounded=True))
        evidence_texts.append(ev_text)
        modes.append("grounded")
        reports.append(reporter.generate_report(results, item["query"], grounded=False))
        evidence_texts.append("no evidence provided")
        modes.append("ungrounded")
    df = compute_faithfulness_report(
        reports,
        evidence_texts,
        use_bertscore=config["evaluation"].get("faithfulness_metric") in ("bertscore", "both"),
    )
    df.insert(1, "mode", modes)
    return df


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"].get("seed", 42))
    requested = {m.strip() for m in args.metrics.split(",") if m.strip()}

    encoder = get_encoder(args.model, config, variant=args.variant)
    retriever = CCTVRetriever(encoder, EmbeddingIndexer.load(resolve_index(config, args.model, args.dataset)))

    out_dir = ensure_dir(Path(config["experiment"].get("output_dir", "results")) / "metrics")
    row = {"model": args.model, "dataset": args.dataset}

    if "retrieval" in requested:
        row.update(retrieval_eval(config, args.dataset, retriever))
    if "efficiency" in requested:
        row.update(efficiency_eval(config, encoder, args.dataset))

    df = pd.DataFrame([row])
    out_csv = out_dir / f"{args.model}_{args.dataset}_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info("metrics -> %s", out_csv)
    logger.info("\n%s", df.to_string(index=False))

    if "faithfulness" in requested:
        fdf = faithfulness_eval(config, retriever, args.dataset)
        fcsv = out_dir / f"{args.model}_{args.dataset}_faithfulness.csv"
        fdf.to_csv(fcsv, index=False)
        logger.info("faithfulness -> %s", fcsv)
        logger.info("\n%s", fdf.to_string(index=False))


if __name__ == "__main__":
    main()
