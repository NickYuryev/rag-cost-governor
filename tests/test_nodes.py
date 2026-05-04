from rag_cost_governor.budget import Budget
from rag_cost_governor.nodes import drafter, planner, retriever, verifier
from rag_cost_governor.store import Hit


def _state(**kw):
    s = {"budget": Budget(ceiling_usd=0.05), "retries": 0, "unsupported": []}
    s.update(kw)
    return s


def test_planner_splits_into_subqueries(deps):
    out = planner(_state(question="any"), deps)
    assert len(out["subqueries"]) >= 1
    assert all(isinstance(q, str) and q for q in out["subqueries"])


def test_retriever_dedupes_by_doc_and_page(deps):
    s = _state(question="microsoft cloud", subqueries=["microsoft cloud", "microsoft cloud"])
    out = retriever(s, deps)
    keys = [(h.doc_id, h.page) for h in out["hits"]]
    assert len(keys) == len(set(keys))


def test_drafter_falls_back_when_no_hits(deps):
    out = drafter(_state(question="q", hits=[]), deps)
    assert "no relevant" in out["draft"].lower()


def test_verifier_keeps_supported_claim(deps):
    hit = Hit(
        "msft-10k-2024",
        12,
        "Microsoft Cloud revenue was $137.4 billion in fiscal 2024.",
        0.9,
    )
    draft = "Microsoft Cloud revenue was $137.4 billion in fiscal 2024. [msft-10k-2024:12]"
    out = verifier(_state(hits=[hit], draft=draft), deps)
    assert out["unsupported"] == []
    assert "137.4" in out["answer"]


def test_verifier_drops_unsupported_number(deps):
    hit = Hit(
        "msft-10k-2024",
        12,
        "Microsoft Cloud revenue was $137.4 billion in fiscal 2024.",
        0.9,
    )
    draft = "Microsoft Cloud revenue was $999.9 billion in fiscal 2024. [msft-10k-2024:12]"
    out = verifier(_state(hits=[hit], draft=draft), deps)
    assert out["unsupported"], "fabricated 999.9 should be dropped"
    assert "dropped unsupported" in out["answer"]
