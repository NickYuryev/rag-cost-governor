# rag-cost-governor

A cost-governed multi-agent RAG system over SEC 10-K filings. Equity and credit
analysts ask natural-language questions ("what was Microsoft Cloud revenue in
fiscal 2024", "list the top risk factors in Item 1A"); the system plans
sub-queries, retrieves cited passages from Qdrant, drafts an answer with inline
`[doc:page]` citations, verifies every numeric claim against the retrieved
chunks and enforces a per-query USD ceiling using the reserve-then-settle
pattern from the gateway in
[budget-governor](https://github.com/NickYuryev/budget-governor).

Naive LLM calls over a 400-page 10-K blow the token budget. Vanilla RAG
hallucinates numbers without a verification pass. This repo stitches four
small LangGraph nodes together, puts a hard dollar cap in front of them, and
surfaces per-node cost and latency on a Prometheus dashboard so you can see
where the money is going before the invoice lands.

## How it fits together

```
   question
      |
      v
   +------+   +----------+   +-------+   +--------+
   | plan |-->| retrieve |-->| draft |-->| verify |---> answer
   +------+   +----------+   +-------+   +--------+
     | every llm call reserves worst-case cost first,
     | settles actual on return. a reserve that would
     | blow the ceiling short-circuits the graph.
     v
   Budget(settled + outstanding <= ceiling_usd)
```

1. **Planner** (LLM) decomposes the question into up to four sub-queries.
2. **Retriever** (local embeddings, Qdrant top-k) collects cited passages per
   sub-query, deduped by `(doc_id, page)`.
3. **Drafter** (LLM) writes the answer pinned to the retrieved context, with
   `[doc:page]` citations copied from it.
4. **Verifier** (deterministic, no LLM) re-parses every numeric claim against
   the text of its cited chunk. A claim that fails verification is dropped;
   if anything was dropped, the graph gets one retrieve-draft retry before
   stopping.

A per-query `Budget(ceiling_usd=…)` wraps every LLM call with
`try_reserve(worst_case)` and `settle(actual)`. The invariant is
`settled + outstanding <= ceiling_usd`; a reserve that would blow the cap is
refused and the conditional edges turn that into an early stop with a
`budget_exhausted` flag. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full
state graph and metrics table.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai]"

export OPENAI_API_KEY=...
export RCG_LLM_BACKEND=openai
export RCG_LLM_MODEL=gpt-4o-mini
export RCG_QDRANT_URL=data/qdrant
export RCG_EDGAR_USER_AGENT="rag-cost-governor you@example.com"

python scripts/ingest.py --tickers MSFT GOOGL --year 2024
python -m rag_cost_governor.cli "What was Microsoft Cloud revenue in fiscal 2024?"
```

Anthropic works the same way: `pip install -e ".[dev,anthropic]"`,
`ANTHROPIC_API_KEY=…`, `RCG_LLM_BACKEND=anthropic`,
`RCG_LLM_MODEL=claude-haiku-5-5`.

Full stack with Prometheus and Grafana:

```bash
docker compose up -d qdrant prometheus grafana
export RCG_QDRANT_URL=http://localhost:6333
python scripts/ingest.py --tickers MSFT GOOGL --year 2024
python -m rag_cost_governor.cli "..."
# grafana at http://localhost:3000, dashboard "rag-cost-governor"
```

CI runs ruff + mypy + pytest with a stub backend, so pull requests don't
need provider credentials.

## Measured numbers

Ten questions in `eval/questions.jsonl` over the 2024 10-K filings of
Microsoft and Alphabet (275 chunks after `bge-small-en-v1.5` embedding),
`gpt-4o-mini`. Raw CSVs in `eval/results/`.

### Baseline, $0.05 ceiling

| metric                             | value         |
| ---------------------------------- | ------------- |
| questions                          | 10            |
| citation_hit (expected doc hit)    | 10 / 10       |
| number_hit (expected figure in answer) | 8 / 10    |
| budget_exhausted                   | 0 / 10        |
| mean cost per query                | $0.000299     |
| mean wall latency per query        | 2.275 s       |

Retrieval hits the right filing on every question. The two `number_hit`
misses are on q06 (gross margin not stated as a single figure in the 10-K)
and q10 (verifier rejects a paraphrased percentage; retry does not recover
it either). Both are dropped from the answer and surfaced in
`unsupported`.

### Budget pinch, $1e-5 ceiling

| metric            | value   |
| ----------------- | ------- |
| budget_exhausted  | 10 / 10 |
| settled cost      | $0.00   |
| citation_hit      | 0 / 10  |

Every query short-circuits before the first LLM call: the governor doing
its job. `rcg_budget_exhausted_total` increments, `settled_usd` stays at
zero, no provider request is ever made.

### /metrics snapshot

After three baseline queries plus one budget-pinch run (full dump in
`eval/results/metrics_dump.txt`):

```
rcg_budget_reserved_usd_total       0.001566
rcg_budget_settled_usd_total        0.000693
rcg_budget_exhausted_total          1.0
rcg_tokens_in_total{node="planner",model="gpt-4o-mini"}  120
rcg_tokens_in_total{node="drafter",model="gpt-4o-mini"}  2412
```

Reserved is roughly 2.3x settled because every call reserves its worst-case
output tokens; most calls settle well below that, which is the point.

## Repo layout

```
src/rag_cost_governor/
  config.py              env-driven Settings
  llm_client.py          openai / anthropic backends + prometheus metrics
  budget.py              reserve-then-settle usd ceiling
  embeddings.py          fastembed local embeddings
  store.py               qdrant wrapper (http / embedded / in-memory)
  nodes.py               planner, retriever, drafter, verifier
  graph.py               langgraph wiring with budget edges
  cli.py                 entrypoint + prometheus exporter

scripts/
  ingest.py              fetch 10-Ks from EDGAR, chunk, embed, upsert
  run_eval.py            offline eval, writes eval/results/<name>.csv

tests/                   pytest, 12 tests
eval/                    questions.jsonl + results/
grafana/                 dashboard.json + provisioning
docker-compose.yml       qdrant + prometheus + grafana
```

## What is in and what is not

In scope: a working end-to-end RAG agent, a hard dollar ceiling that fires
on real metrics, a deterministic offline eval that catches retrieval and
verifier regressions, dashboards that are importable rather than
screenshots.

Out of scope on purpose, documented in `ARCHITECTURE.md`: judge-model eval,
cross-encoder reranking, cross-request budget sharing (the gateway in
`budget-governor` already does that), prompt-injection scrubbing on
EDGAR-authored text.

## License

MIT. See [LICENSE](LICENSE).
