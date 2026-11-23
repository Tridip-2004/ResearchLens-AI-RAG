import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.store import get_vector_store


def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


class HybridRetriever:

    def __init__(self):

        self.store = get_vector_store()

        self.reranker = CrossEncoder(
            settings.reranker_model
        )

    def semantic_search(self, query, limit=15):

        return self.store.search(
            query=query,
            limit=limit
        )

    def keyword_search(self, query, limit=15):

        documents = self.store.get_all_documents()

        if not documents:
            return []

        tokenized_corpus = [
            tokenize(document["text"])
            for document in documents
        ]

        if not any(tokenized_corpus):
            return []

        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:limit]:

            if scores[index] <= 0:
                continue

            document = documents[index].copy()

            document["score"] = float(scores[index])

            results.append(document)

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results,
        keyword_results,
        k=60
    ):

        combined = {}

        for results in [
            semantic_results,
            keyword_results
        ]:

            for rank, document in enumerate(
                results,
                start=1
            ):

                document_id = document["id"]

                if document_id not in combined:

                    combined[document_id] = {
                        **document,
                        "rrf_score": 0.0
                    }

                combined[document_id]["rrf_score"] += (
                    1 / (k + rank)
                )

        ranked = sorted(
            combined.values(),
            key=lambda document: document["rrf_score"],
            reverse=True
        )

        return ranked

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.reranker.predict(pairs)

        for document, score in zip(
            documents,
            scores
        ):

            document["rerank_score"] = float(score)

        ranked = sorted(
            documents,
            key=lambda document: document["rerank_score"],
            reverse=True
        )

        return ranked[:top_k]

    def retrieve(self, query, top_k=None):

        if top_k is None:
            top_k = settings.top_k

        semantic_results = self.semantic_search(
            query,
            limit=settings.retrieval_limit
        )

        keyword_results = self.keyword_search(
            query,
            limit=settings.retrieval_limit
        )

        merged_results = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        final_results = self.rerank(
            query,
            merged_results,
            top_k=top_k
        )

        return final_results


_retriever = None


def get_retriever():

    global _retriever

    if _retriever is None:
        _retriever = HybridRetriever()

    return _retriever