import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation import build_queries
from evaluation.efficiency_metrics import measure_encoding_latency
from evaluation.retrieval_metrics import compute_all_metrics
from models.encoder_factory import get_encoder
from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retriever import CCTVRetriever
from utils.common import clear_gpu_cache, ensure_dir, get_logger, load_config, set_seed

logger = get_logger("compare_models")


def parse_args():
    p = argparse.ArgumentParser(description="Head-to-head VL-JEPA vs CLIP comparison")
    p.add_argument("--models", default="vljepa,clip", help="comma list of models to compare")
    p.add_argument("--datasets", default="ucf_crime,rlvs", help="comma list of datasets")
    p.add_argument("--config", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--latency-runs", type=int, default=20)
    return p.parse_args()


def resolve_index(config, model_name, dataset_name):
    return Path(config["retrieval"]["index_save_dir"]) / f"{model_name}_{dataset_name}.faiss"


def eval_model_on_dataset(config, model_name, dataset_name, args):
    index_path = resolve_index(config, model_name, dataset_name)
    if not index_path.is_file():
        logger.warning("missing index %s - skipping", index_path)
        return None
    encoder = get_encoder(model_name, config, variant=args.variant)
    retriever = CCTVRetriever(encoder, EmbeddingIndexer.load(index_path))

    k_values = list(config["evaluation"].get("recall_k_values", [1, 5, 10]))
    queries = build_queries(dataset_name, config)
    results_labels, ground_truth = [], []
    for item in queries:
        results = retriever.retrieve(item["query"], k=max(k_values))
        results_labels.append([r.get("label", "") for r in results])
        ground_truth.append(item["relevant"])
    metrics = compute_all_metrics(results_labels, ground_truth, k_values=k_values)
    metrics["model"] = model_name
    metrics["dataset"] = dataset_name

    dataset = build_dataset(dataset_name, config)
    if len(dataset):
        samples = [dataset[i]["video"] for i in range(min(4, len(dataset)))]
        latency = measure_encoding_latency(encoder, samples, n_runs=args.latency_runs)
        metrics["latency_mean_ms"] = round(latency["mean_ms"], 2)
    clear_gpu_cache()
    return metrics


def plot_metric(df, metric, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{m}\n{d}" for m, d in zip(df["model"], df["dataset"])]
    values = df[metric].astype(float)
    ax.bar(labels, values, color=["#4c72b0" if m == "vljepa" else "#dd8452" for m in df["model"]])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"].get("seed", 42))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rows = []
    for model_name in models:
        for dataset_name in datasets:
            logger.info("evaluating %s on %s ...", model_name, dataset_name)
            metrics = eval_model_on_dataset(config, model_name, dataset_name, args)
            if metrics:
                rows.append(metrics)

    if not rows:
        raise SystemExit("no evaluations completed - check that indices exist")

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(Path(config["experiment"].get("output_dir", "results")) / "metrics")
    plots_dir = ensure_dir(Path(config["experiment"].get("output_dir", "results")) / "plots")
    table_path = out_dir / "comparison_table.csv"
    df.to_csv(table_path, index=False)
    logger.info("comparison table -> %s\n%s", table_path, df.to_string(index=False))

    for metric, title, name in (
        ("recall@1", "Recall@1 by model and dataset", "recall_at1_comparison.png"),
        ("recall@5", "Recall@5 by model and dataset", "recall_at5_comparison.png"),
        ("recall@10", "Recall@10 by model and dataset", "recall_at10_comparison.png"),
        ("latency_mean_ms", "Encoding latency (ms)", "latency_comparison.png"),
    ):
        if metric in df.columns:
            plot_metric(df, metric, title, plots_dir / name)
    logger.info("plots -> %s", plots_dir)


if __name__ == "__main__":
    main()
