import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.config import settings


class VectorStore:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            settings.embedding_model
        )

        database_path = (
            Path(settings.data_dir) / "qdrant"
        )

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = QdrantClient(
            path=str(database_path)
        )

        self.collection_name = settings.collection_name

        self.vector_size = (
            self.embedding_model
            .get_sentence_embedding_dimension()
        )

        self.create_collection()

    def create_collection(self):

        if self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )

    def add_documents(self, documents):

        if not documents:
            return 0

        texts = [
            document["text"]
            for document in documents
        ]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        batch_size = 100

        for start in range(0, len(documents), batch_size):

            batch_documents = documents[
                start:start + batch_size
            ]

            batch_embeddings = embeddings[
                start:start + batch_size
            ]

            points = []

            for document, embedding in zip(
                batch_documents,
                batch_embeddings
            ):

                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "text": document["text"],
                        "metadata": document["metadata"]
                    }
                )

                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

        return len(documents)

    def search(self, query, limit=10):

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append({
                "id": str(point.id),
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}),
                "score": float(point.score)
            })

        return results

    def get_all_documents(self):

        documents = []

        offset = None

        while True:

            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in points:

                payload = point.payload or {}

                documents.append({
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {})
                })

            if offset is None:
                break

        return documents

    def delete_document(self, filename):

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(
                                value=filename
                            )
                        )
                    ]
                )
            ),
            wait=True
        )

    def document_count(self):

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )

        return result.count


_store = None


def get_vector_store():

    global _store

    if _store is None:
        _store = VectorStore()

    return _store