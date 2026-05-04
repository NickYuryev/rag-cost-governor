from __future__ import annotations

import pytest

from rag_cost_governor.embeddings import LocalEmbedder
from rag_cost_governor.llm_client import InstrumentedLLM, MockBackend
from rag_cost_governor.nodes import NodeDeps
from rag_cost_governor.store import Chunk, VectorStore


@pytest.fixture(scope="session")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture
def store(embedder: LocalEmbedder) -> VectorStore:
    s = VectorStore(":memory:", "test", embedder.dim)
    chunks = [
        Chunk(
            "msft-10k-2024",
            12,
            "Microsoft Cloud revenue was $137.4 billion in fiscal 2024, an increase of 23%.",
        ),
        Chunk(
            "msft-10k-2024",
            13,
            "Server products and cloud services revenue increased $19.5 billion or 22%.",
        ),
        Chunk(
            "googl-10k-2024",
            42,
            "Google Services revenues were $305.6 billion in 2024, an increase of 13%.",
        ),
        Chunk(
            "googl-10k-2024",
            43,
            "Google Cloud revenues were $43.2 billion in 2024, an increase of 31%.",
        ),
    ]
    s.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return s


@pytest.fixture
def mock_llm() -> InstrumentedLLM:
    canned = {
        "You are decomposing an analyst question into focused sub-queries for a": (
            "microsoft cloud revenue fy2024\nserver products and cloud services growth"
        ),
        "Answer the question using only the context below. Every factual": (
            "Microsoft Cloud revenue was $137.4 billion in fiscal 2024, an increase of 23%. "
            "[msft-10k-2024:12]"
        ),
    }
    return InstrumentedLLM(MockBackend(canned=canned))


@pytest.fixture
def deps(mock_llm: InstrumentedLLM, embedder: LocalEmbedder, store: VectorStore) -> NodeDeps:
    return NodeDeps(llm=mock_llm, embedder=embedder, store=store, top_k=4, max_subqueries=3)
