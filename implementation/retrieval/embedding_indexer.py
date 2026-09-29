import json
from pathlib import Path

import faiss
import numpy as np

from retrieval.retrieval_utils import as_float32


class EmbeddingIndexer:
    def __init__(self, embedding_dim: int, index_type: str = "FlatIP", auto_ivf_threshold: int = 20000):
        self.dim = int(embedding_dim)
        self.index_type = index_type
        self.auto_ivf_threshold = int(auto_ivf_threshold)
        self.index = None
        self.metadata = []

    def _create(self, n=None):
        if self.index_type in ("IVF", "auto"):
            if self.index_type == "IVF" or (n and n > self.auto_ivf_threshold):
                nlist = max(1, min(int(np.sqrt(max(n, 1))), 4096))
                quantizer = faiss.IndexFlatIP(self.dim)
                return faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexFlatIP(self.dim)

    def add(self, embeddings, metadata=None):
        emb = as_float32(embeddings)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"embedding shape {emb.shape} incompatible with dim {self.dim}")
        if self.index is None:
            self.index = self._create(n=len(emb))
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            if len(emb) < self.index.nlist * 5:
                self.index = faiss.IndexFlatIP(self.dim)
            else:
                self.index.train(emb)
        self.index.add(emb)
        if metadata:
            self.metadata.extend(metadata)
        return self

    def save(self, path):
        if self.index is None:
            raise RuntimeError("nothing to save: index is empty")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        sidecar = path.with_name(path.stem + ".meta.json")
        sidecar.write_text(
            json.dumps({"dim": self.dim, "ntotal": int(self.index.ntotal), "metadata": self.metadata}),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        index = faiss.read_index(str(path))
        obj = cls(int(index.d))
        obj.index = index
        sidecar = path.with_name(path.stem + ".meta.json")
        if sidecar.is_file():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            obj.metadata = data.get("metadata", [])
        return obj

    def search(self, query, k: int = 5):
        if self.index is None:
            raise RuntimeError("index is empty")
        q = as_float32(query)
        if q.ndim == 1:
            q = q[None, :]
        k = min(int(k), max(int(self.index.ntotal), 1))
        scores, ids = self.index.search(q, k)
        results = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            for s, i in zip(row_scores, row_ids):
                if i == -1:
                    continue
                meta = self.metadata[i] if 0 <= i < len(self.metadata) else {"faiss_id": int(i)}
                hits.append((float(s), meta))
            results.append(hits)
        return results

    def __len__(self):
        return 0 if self.index is None else int(self.index.ntotal)
