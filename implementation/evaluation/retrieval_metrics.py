import math


def _as_label_set(labels):
    if isinstance(labels, str):
        return {labels}
    return {str(lbl) for lbl in labels}


def recall_at_k(retrieved_labels, relevant_labels, k: int) -> float:
    relevant = _as_label_set(relevant_labels)
    if not relevant:
        return 0.0
    hit = any(str(lbl) in relevant for lbl in list(retrieved_labels)[:k])
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved_labels, relevant_labels) -> float:
    relevant = _as_label_set(relevant_labels)
    for rank, lbl in enumerate(list(retrieved_labels), 1):
        if str(lbl) in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_labels, relevant_labels, k: int) -> float:
    relevant = _as_label_set(relevant_labels)
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, lbl in enumerate(list(retrieved_labels)[:k], 1):
        if str(lbl) in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def average_precision(retrieved_labels, relevant_labels) -> float:
    relevant = _as_label_set(relevant_labels)
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, lbl in enumerate(list(retrieved_labels), 1):
        if str(lbl) in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


def mean_reciprocal_rank(results_labels, relevant_labels_list) -> float:
    values = [reciprocal_rank(r, g) for r, g in zip(results_labels, relevant_labels_list)]
    return sum(values) / len(values) if values else 0.0


def mean_average_precision(results_labels, relevant_labels_list) -> float:
    values = [average_precision(r, g) for r, g in zip(results_labels, relevant_labels_list)]
    return sum(values) / len(values) if values else 0.0


def compute_all_metrics(results, ground_truth, k_values=(1, 5, 10)) -> dict:
    if len(results) != len(ground_truth):
        raise ValueError("results and ground_truth must have the same length")
    n = max(len(results), 1)
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = sum(recall_at_k(r, g, k) for r, g in zip(results, ground_truth)) / n
    for k in k_values:
        metrics[f"ndcg@{k}"] = sum(ndcg_at_k(r, g, k) for r, g in zip(results, ground_truth)) / n
    metrics["mrr"] = mean_reciprocal_rank(results, ground_truth)
    metrics["map"] = mean_average_precision(results, ground_truth)
    metrics["num_queries"] = len(results)
    return metrics
