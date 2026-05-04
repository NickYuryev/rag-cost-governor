from rag_cost_governor.budget import Budget
from rag_cost_governor.graph import ask, build


def test_end_to_end_mock(deps):
    graph = build(deps)
    out = graph.invoke(
        {
            "question": "what was microsoft cloud revenue in fy2024?",
            "budget": Budget(ceiling_usd=0.05),
            "retries": 0,
            "unsupported": [],
        }
    )
    assert out["hits"], "retriever should find at least one chunk"
    assert "137.4" in out["answer"]
    assert out["budget"].settled_usd > 0


def test_budget_exhaustion_short_circuits(deps):
    graph = build(deps)
    # ceiling so small no reserve can be made
    out = ask(graph, "anything", budget_usd=1e-9)
    assert out.get("budget_exhausted") is True
    assert out["budget"].settled_usd == 0
