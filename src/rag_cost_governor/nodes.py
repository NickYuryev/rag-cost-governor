from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TypedDict

from .budget import Budget
from .embeddings import LocalEmbedder
from .llm_client import InstrumentedLLM, worst_case_cost
from .store import Hit, VectorStore

log = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    question: str
    subqueries: list[str]
    hits: list[Hit]
    draft: str
    answer: str
    unsupported: list[str]
    retries: int
    budget: Budget
    budget_exhausted: bool


@dataclass
class NodeDeps:
    llm: InstrumentedLLM
    embedder: LocalEmbedder
    store: VectorStore
    top_k: int = 5
    max_subqueries: int = 4
    verifier_tolerance_pct: float = 1.0
    max_retries: int = 1
    # Max tokens budgeted per call (used as the reserve worst-case).
    plan_max_tokens: int = 160
    draft_max_tokens: int = 500
    _reserves: list[tuple[str, float, float]] = field(default_factory=list)

    def call(self, node: str, prompt: str, state: GraphState, max_tokens: int) -> str | None:
        """Reserve, call, settle. Returns None if the budget cannot cover the call."""
        budget: Budget = state["budget"]
        prompt_tokens = len(prompt.split())
        reserve = worst_case_cost(self.llm.model, prompt_tokens, max_tokens)
        if not budget.try_reserve(reserve):
            state["budget_exhausted"] = True
            return None
        resp = self.llm.call(node, prompt, max_tokens=max_tokens)
        budget.settle(reserve, resp.cost_usd)
        return resp.text


# --- planner ------------------------------------------------------------

PLANNER_PROMPT = """You are decomposing an analyst question into focused sub-queries for a
retrieval system over SEC 10-K filings. Return at most {k} short sub-queries, one per
line, no numbering, no commentary. If the question is already atomic, return it as one line.

Question: {q}
"""


def planner(state: GraphState, deps: NodeDeps) -> GraphState:
    prompt = PLANNER_PROMPT.format(k=deps.max_subqueries, q=state["question"])
    text = deps.call("planner", prompt, state, deps.plan_max_tokens)
    if text is None:
        return {**state, "subqueries": [state["question"]]}
    lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
    subs = [ln for ln in lines if 4 <= len(ln) <= 200][: deps.max_subqueries]
    if not subs:
        subs = [state["question"]]
    return {**state, "subqueries": subs}


# --- retriever ----------------------------------------------------------

def retriever(state: GraphState, deps: NodeDeps) -> GraphState:
    seen: set[tuple[str, int]] = set()
    hits: list[Hit] = []
    for q in state.get("subqueries", [state["question"]]):
        vec = deps.embedder.embed_one(q)
        for h in deps.store.search(vec, top_k=deps.top_k):
            key = (h.doc_id, h.page)
            if key in seen:
                continue
            seen.add(key)
            hits.append(h)
    hits.sort(key=lambda h: h.score, reverse=True)
    return {**state, "hits": hits[: deps.top_k * 2]}


