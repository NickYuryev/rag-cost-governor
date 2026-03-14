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


# --- drafter ------------------------------------------------------------

DRAFTER_PROMPT = """Answer the question using only the context below. Every factual
sentence must end with a citation of the form [doc_id:page], copied from the context.
Do not invent citations. If the context does not contain the answer, say so plainly.

Question: {q}

Context:
{ctx}

Answer:
"""


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.doc_id}:{h.page}] {h.text}" for h in hits)


def drafter(state: GraphState, deps: NodeDeps) -> GraphState:
    hits = state.get("hits", [])
    if not hits:
        return {**state, "draft": "no relevant passages were retrieved."}
    prompt = DRAFTER_PROMPT.format(q=state["question"], ctx=_format_context(hits))
    text = deps.call("drafter", prompt, state, deps.draft_max_tokens)
    if text is None:
        return state
    return {**state, "draft": text.strip()}


# --- verifier -----------------------------------------------------------

# Match percentages, currency, counts, and bare numbers with units. The pattern is
# deliberately loose; the verifier's job is to flag claims to check, not to parse them.
_CLAIM_RE = re.compile(
    r"""
    (?:\$|US\$|USD\s)?           # optional currency sign
    \d+(?:,\d{3})*               # one or more digits, with optional thousands commas
    (?:\.\d+)?                   # optional decimal
    (?:\s*(?:%|percent|bps|bn|billion|million|m|k|thousand|USD))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
_CITE_RE = re.compile(r"\[([a-z0-9_\-]+):(\d+)\]")


def _normalise(num: str) -> float | None:
    s = num.replace(",", "").replace(" ", "").replace("$", "").upper()
    mult = 1.0
    if s.endswith("%"):
        s = s[:-1]
    elif s.endswith("BPS"):
        s = s[:-3]
        mult = 0.01
    elif s.endswith(("BN", "BILLION")):
        s = s.removesuffix("BILLION").removesuffix("BN")
        mult = 1e9
    elif s.endswith(("M", "MILLION")):
        s = s.removesuffix("MILLION").removesuffix("M")
        mult = 1e6
    elif s.endswith(("K", "THOUSAND")):
        s = s.removesuffix("THOUSAND").removesuffix("K")
        mult = 1e3
    try:
        return float(s) * mult
    except ValueError:
        return None


def _supported(sentence: str, hits_by_cite: dict[str, str], tol_pct: float) -> bool:
    citations = _CITE_RE.findall(sentence)
    if not citations:
        return True  # not a quantitative claim worth gating on
    corpus = " ".join(hits_by_cite.get(f"{d}:{p}", "") for d, p in citations)
    if not corpus:
        return False
    # Strip the citation markers before extracting numbers, so the page number in
    # a citation does not count as a claim.
    body = _CITE_RE.sub(" ", sentence)
    numbers = [m.group(0) for m in _CLAIM_RE.finditer(body) if any(ch.isdigit() for ch in m.group(0))]
    if not numbers:
        return True
    corpus_numbers = [m.group(0) for m in _CLAIM_RE.finditer(corpus) if any(ch.isdigit() for ch in m.group(0))]
    corpus_values = [v for v in (_normalise(n) for n in corpus_numbers) if v is not None]
    for n in numbers:
        v = _normalise(n)
        if v is None:
            continue
        if not any(_close(v, c, tol_pct) for c in corpus_values):
            return False
    return True


def _close(a: float, b: float, tol_pct: float) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) * 100.0 <= tol_pct


def verifier(state: GraphState, deps: NodeDeps) -> GraphState:
    draft = state.get("draft", "")
    hits_by_cite = {f"{h.doc_id}:{h.page}": h.text for h in state.get("hits", [])}
    raw = re.split(r"(?<=[\.\?!])\s+", draft)
    # A trailing citation after a sentence boundary ("... 2024. [doc:page]") belongs
    # to the sentence it follows; merge it back so the citation gates the right claim.
    sentences: list[str] = []
    for part in raw:
        if _CITE_RE.fullmatch(part.strip()) and sentences:
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)
    kept: list[str] = []
    unsupported: list[str] = list(state.get("unsupported", []))
    for s in sentences:
        if not s.strip():
            continue
        try:
            ok = _supported(s, hits_by_cite, deps.verifier_tolerance_pct)
        except Exception as e:  # verifier is fail-open: a parser miss should not kill the query
            log.warning("verifier regex miss: %s (%s)", e, s[:80])
            ok = True
        if ok:
            kept.append(s)
        else:
            unsupported.append(s)
    answer = " ".join(kept).strip() or draft
    if unsupported:
        answer += "\n\n[dropped unsupported claims: " + " | ".join(unsupported) + "]"
    return {**state, "answer": answer, "unsupported": unsupported}
