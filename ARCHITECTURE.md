# Architecture

## What this is

A cost-governed RAG agent over SEC 10-K filings. The system plans sub-queries,
retrieves cited passages from a Qdrant vector store, drafts an answer with
inline `[doc:page]` citations, verifies every quantitative claim against the
cited chunks, and enforces a per-query USD ceiling using the reserve-then-settle
pattern the gateway in
[budget-governor](https://github.com/NickYuryev/budget-governor) applies at
the request layer.

## State graph

```
         +---------+
         |  plan   |---+budget_exhausted---> END
         +---------+
              |
              v
         +---------+
         | retrieve|
         +---------+
              |
              v
         +---------+
         | draft   |---+budget_exhausted---> END
         +---------+
              |
              v
         +---------+     +unsupported & retries<1+
         | verify  |-----------------------------> retrieve
         +---------+
              |
              v
             END
```

* **plan** — one LLM call. Decomposes the question into ≤ `max_subqueries`
  short sub-queries. If the planner cannot reserve the budget for its call,
  the sub-query list falls back to the raw question and the graph proceeds;
  the budget flag short-circuits only when the reserve fails.
* **retrieve** — no LLM. Embeds each sub-query locally (fastembed,
  BAAI/bge-small-en-v1.5, 384-dim) and takes top-k from Qdrant. Hits are
  deduped by `(doc_id, page)` and sorted by score; the retriever returns up
  to `2 * top_k`.
* **draft** — one LLM call. The prompt pins the drafter to the retrieved
  context and requires `[doc:page]` citations copied from it.
* **verify** — no LLM. Splits the draft into sentences, re-attaches trailing
  citations to the sentence they anchor, extracts numeric claims with a loose
  regex, and checks each against the text of its cited chunk within
  `verifier_tolerance_pct`. A claim that fails verification is removed from
  the final answer and surfaced in `unsupported`. If anything was unsupported
  and the retrier has not yet fired, the edge routes back to `retrieve` for
  one more pass; a second miss stops instead of looping.

## Budget invariants

A per-query `Budget(ceiling_usd=…)` tracks three numbers:

* `settled_usd` — real cost of the LLM calls that have returned,
* `outstanding_usd` — sum of worst-case reservations for calls in flight,
* `ceiling_usd` — the hard cap.

The invariant is:

    settled_usd + outstanding_usd <= ceiling_usd

enforced by `Budget.try_reserve(worst_case)`. The reason to reserve the
worst case before the call rather than reconcile after: a slow or runaway
response could blow through the cap before we learned about it. Reserving
first means settling can only shrink `outstanding_usd`.

Fail-closed on the budget: a reserve that would blow the cap is refused
and the node returns `None`; the conditional edges turn that into an early
END and the `budget_exhausted` flag goes to the client. Fail-open on the
verifier: an unexpected regex failure is logged and the sentence is kept,
not dropped. A bad analyst answer is a worse outcome than a dropped claim,
but killing the whole query on a parser quirk is worse still.

## Metrics

Every LLM call goes through the `InstrumentedLLM` wrapper, which emits:

| name                                | kind      | labels              |
| ----------------------------------- | --------- | ------------------- |
| `rcg_tokens_in_total`               | counter   | node, model         |
| `rcg_tokens_out_total`              | counter   | node, model         |
| `rcg_cost_usd_total`                | counter   | node, model         |
| `rcg_request_duration_seconds`      | histogram | node, model         |
| `rcg_request_failures_total`        | counter   | node, model, reason |
| `rcg_budget_reserved_usd_total`     | counter   | –                   |
| `rcg_budget_settled_usd_total`      | counter   | –                   |
| `rcg_budget_exhausted_total`        | counter   | –                   |

The `node` label is what makes the dashboards useful: the drafter is the
single most expensive node, and watching cost and token flow by node is how
you decide to raise `top_k`, cap `draft_max_tokens`, or swap the model.

## What is not here

* No request-level concurrency control. One question at a time per process;
  the gateway in `budget-governor` handles the cross-request case with
  Redis + Lua.
* No judge-model eval. `scripts/run_eval.py` grades on citation hit and
  numeric substring hit. Ragas would need a judge LLM and a reference-answer
  corpus to maintain.
* No chunk reranking. Dense retrieval with bge-small is the first pass; a
  cross-encoder reranker is a reasonable next step.
* No prompt-injection scrubbing on 10-K text. EDGAR filings are trusted
  authored text and the drafter prompt is pinned to cited context. For
  user-uploaded filings, add a strip pass on ingest.
