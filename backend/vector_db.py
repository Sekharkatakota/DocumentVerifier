import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


class QdrantStorage:

    def __init__(self):

        # Read configuration
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        self.collection = os.getenv(
            "QDRANT_COLLECTION",
            "docs"
        )

        self.dim = 3072

        if not self.url:
            raise ValueError("QDRANT_URL is missing")

        # Connect to Qdrant Cloud
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30
        )

        # Create collection if it doesn't exist
        self.create_collection()

    def create_collection(self):

        if not self.client.collection_exists(self.collection):

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    def upsert(self, ids, vectors, payloads):

        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                "IDs, vectors and payloads must have equal lengths"
            )

        batch_size = 100

        for i in range(0, len(ids), batch_size):

            points = [
                PointStruct(
                    id=ids[j],
                    vector=vectors[j],
                    payload=payloads[j]
                )
                for j in range(
                    i,
                    min(i + batch_size, len(ids))
                )
            ]

            self.client.upsert(
                collection_name=self.collection,
                points=points
            )

    def search(self, query_vector, top_k=5):

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points

        contexts = []
        sources = set()

        for result in results:

            payload = result.payload or {}

            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }