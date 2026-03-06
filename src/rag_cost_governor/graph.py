from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from .budget import Budget
from .config import Settings
from .embeddings import LocalEmbedder
from .llm_client import InstrumentedLLM
from .nodes import GraphState, NodeDeps, drafter, planner, retriever, verifier
from .store import VectorStore

log = logging.getLogger(__name__)


def _after_planner(state: GraphState) -> str:
    return "end" if state.get("budget_exhausted") else "retrieve"


def _after_drafter(state: GraphState) -> str:
    return "end" if state.get("budget_exhausted") else "verify"


def _after_verifier(state: GraphState) -> str:
    """One retry if the verifier dropped anything, then stop."""
    if state.get("unsupported") and state.get("retries", 0) < 1:
        state["retries"] = state.get("retries", 0) + 1
        state["unsupported"] = []  # cleared for the retry pass; dropped list is in answer
        return "retrieve"
    return "end"


def build(deps: NodeDeps) -> Any:
    g = StateGraph(GraphState)
    g.add_node("plan", lambda s: planner(s, deps))
    g.add_node("retrieve", lambda s: retriever(s, deps))
    g.add_node("draft", lambda s: drafter(s, deps))
    g.add_node("verify", lambda s: verifier(s, deps))

    g.set_entry_point("plan")
    g.add_conditional_edges("plan", _after_planner, {"retrieve": "retrieve", "end": END})
    g.add_edge("retrieve", "draft")
    g.add_conditional_edges("draft", _after_drafter, {"verify": "verify", "end": END})
    g.add_conditional_edges("verify", _after_verifier, {"retrieve": "retrieve", "end": END})
    return g.compile()


def build_default(settings: Settings, llm: InstrumentedLLM) -> tuple[Any, NodeDeps]:
    embedder = LocalEmbedder(model_name=settings.embed_model)
    store = VectorStore(settings.qdrant_url, settings.qdrant_collection, embedder.dim)
    deps = NodeDeps(
        llm=llm,
        embedder=embedder,
        store=store,
        top_k=settings.top_k,
        max_subqueries=settings.max_subqueries,
        verifier_tolerance_pct=settings.verifier_tolerance_pct,
    )
    return build(deps), deps


def ask(graph: Any, question: str, budget_usd: float) -> GraphState:
    initial: GraphState = {
        "question": question,
        "budget": Budget(ceiling_usd=budget_usd),
        "retries": 0,
        "unsupported": [],
    }
    return graph.invoke(initial)
