from collections import defaultdict

import numpy as np
import pandas as pd

from retrieval.retrieval_utils import l2_normalize_np


def _recall(query_emb, index_emb, query_labels, index_labels, k: int) -> float:
    sims = index_emb @ query_emb.T
    order = np.argsort(-sims, axis=0)[:k]
    hits = 0
    total = 0
    for j in range(query_emb.shape[0]):
        top = [index_labels[i] for i in order[:, j]]
        total += 1
        if any(str(lbl) == str(query_labels[j]) for lbl in top):
            hits += 1
    return hits / max(total, 1)


def cross_viewpoint_matrix(embeddings_by_view, labels_by_view, k: int = 5, max_per_class: int = 16):
    views = sorted(embeddings_by_view)
    norm = {
        v: l2_normalize_np(np.asarray(embeddings_by_view[v], dtype=np.float32)) for v in views
    }
    matrix = pd.DataFrame(index=views, columns=views, dtype=float)
    for qv in views:
        per_class = defaultdict(list)
        for i, lbl in enumerate(labels_by_view[qv]):
            per_class[str(lbl)].append(i)
        picked = []
        for lbl in sorted(per_class):
            picked += per_class[lbl][:max_per_class]
        if not picked:
            continue
        q_emb = norm[qv][picked]
        q_lbl = [labels_by_view[qv][i] for i in picked]
        for iv in views:
            matrix.loc[qv, iv] = _recall(q_emb, norm[iv], q_lbl, labels_by_view[iv], k)
    return matrix


def plot_heatmap(matrix, path, title="Cross-viewpoint Recall@K"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if not np.isnan(data[r, c]):
                ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel("Index viewpoint")
    ax.set_ylabel("Query viewpoint")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
