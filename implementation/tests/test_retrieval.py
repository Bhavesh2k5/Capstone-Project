import numpy as np
import pytest
import torch

from evaluation.retrieval_metrics import (
    average_precision,
    compute_all_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.embedding_indexer import EmbeddingIndexer
from retrieval.retrieval_utils import l2_normalize_np, l2_normalize_tensor
from retrieval.retriever import CCTVRetriever


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, "b", 2) == 1.0
    assert recall_at_k(retrieved, "b", 1) == 0.0
    assert recall_at_k(retrieved, ["z"], 4) == 0.0
    assert recall_at_k(retrieved, ["d"], 10) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "target"], "target") == pytest.approx(1 / 3)
    assert reciprocal_rank(["target"], "target") == 1.0
    assert reciprocal_rank(["a", "b"], "target") == 0.0
    assert mean_reciprocal_rank([["target"], ["a"]], ["target", "target"]) == pytest.approx(0.75)


def test_ndcg_at_k():
    assert ndcg_at_k(["target", "a", "b"], "target", 3) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "target", "b"], "target", 3) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(["a", "b", "c"], "target", 3) == 0.0


def test_average_precision():
    assert average_precision(["t1", "x", "t2"], ["t1", "t2"]) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision(["x", "y"], "t1") == 0.0


def test_compute_all_metrics_shapes():
    results = [["a", "b"], ["c", "a"]]
    truth = [["a"], ["c"]]
    metrics = compute_all_metrics(results, truth, k_values=(1, 2))
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["num_queries"] == 2


def test_l2_normalize():
    t = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(l2_normalize_tensor(t), torch.tensor([[0.6, 0.8]]))
    n = l2_normalize_np([[3.0, 4.0]])
    assert np.allclose(n, [[0.6, 0.8]])


class FakeEncoder:
    def get_embedding_dim(self):
        return 4

    def encode_text(self, texts):
        table = {"cat": [1.0, 0.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0, 0.0]}
        return torch.tensor([table[t] for t in texts], dtype=torch.float32)


def test_indexer_roundtrip_and_retriever(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"path": "cat.mp4", "label": "cat", "dataset": "toy"},
        {"path": "dog.mp4", "label": "dog", "dataset": "toy"},
        {"path": "bird.mp4", "label": "bird", "dataset": "toy"},
    ]
    indexer = EmbeddingIndexer(embedding_dim=4)
    indexer.add(embeddings, metadata=metadata)
    assert len(indexer) == 3

    index_path = tmp_path / "toy.faiss"
    indexer.save(index_path)
    loaded = EmbeddingIndexer.load(index_path)
    assert len(loaded) == 3
    assert loaded.metadata[0]["label"] == "cat"

    hits = loaded.search([[1.0, 0.0, 0.0, 0.0]], k=2)[0]
    assert hits[0][1]["path"] == "cat.mp4"

    retriever = CCTVRetriever(FakeEncoder(), loaded)
    results = retriever.retrieve("cat", k=2)
    assert results[0]["path"] == "cat.mp4"
    assert results[0]["rank"] == 1

    with pytest.raises(ValueError):
        EmbeddingIndexer(embedding_dim=8).add(embeddings, metadata)
