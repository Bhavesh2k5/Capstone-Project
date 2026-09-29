class CCTVRetriever:
    def __init__(self, encoder, indexer):
        if encoder.get_embedding_dim() != indexer.dim:
            raise ValueError(
                f"encoder dim {encoder.get_embedding_dim()} does not match index dim {indexer.dim}"
            )
        self.encoder = encoder
        self.indexer = indexer

    def retrieve(self, text_query: str, k: int = 5):
        q = self.encoder.encode_text([text_query]).numpy()
        hits = self.indexer.search(q, k)[0]
        results = []
        for i, (score, meta) in enumerate(hits):
            item = {"rank": i + 1, "score": score}
            item.update(dict(meta))
            results.append(item)
        return results

    def batch_retrieve(self, queries, k: int = 5):
        return [self.retrieve(q, k) for q in queries]
