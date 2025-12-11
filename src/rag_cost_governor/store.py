from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    doc_id: str
    page: int
    text: str


@dataclass
class Hit:
    doc_id: str
    page: int
    text: str
    score: float

    @property
    def cite(self) -> str:
        return f"[{self.doc_id}:{self.page}]"


class VectorStore:
    def __init__(self, url: str, collection: str, dim: int) -> None:
        if url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url)
        self._collection = collection
        self._ensure(dim)

    def _ensure(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        points = [
            qm.PointStruct(
                id=str(uuid4()),
                vector=v,
                payload={"doc_id": c.doc_id, "page": c.page, "text": c.text},
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def search(self, vector: list[float], top_k: int = 5) -> list[Hit]:
        results = self._client.query_points(
            collection_name=self._collection, query=vector, limit=top_k, with_payload=True
        ).points
        return [
            Hit(
                doc_id=p.payload["doc_id"],
                page=int(p.payload["page"]),
                text=p.payload["text"],
                score=float(p.score),
            )
            for p in results
        ]

    def count(self) -> int:
        return self._client.count(self._collection).count
