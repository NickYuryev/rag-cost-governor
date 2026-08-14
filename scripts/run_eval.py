"""Offline eval. Runs every question in eval/questions.jsonl and writes
one row per question to eval/results/<name>.csv.

Measured metrics (no judge model needed):
  * citation_hit: did any retrieved hit come from the expected doc?
  * number_hit:   does the answer contain every expected number as a substring?
  * cost_usd, latency_s, retries, budget_exhausted

Ragas is intentionally not required; it would need a judge llm and a reference
answer corpus which we do not want to maintain for a weekend project. The csv is
plenty for a baseline table.

Usage:
    python scripts/run_eval.py --name baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_cost_governor.config import load
from rag_cost_governor.graph import ask, build_default
from rag_cost_governor.llm_client import build_from_env

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


def contains_number(answer: str, number: str) -> bool:
    # exact substring match, no normalisation; a stricter check would do rounding
    # tolerance but we want the eval to tell us when the drafter dropped a figure.
    return number in answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output file name (without .csv)")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()

    settings = load()
    llm = build_from_env()
    graph, _ = build_default(settings, llm)
    budget = args.budget if args.budget is not None else settings.budget_usd

    rows: list[dict] = []
    with open(args.questions) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            start = time.perf_counter()
            result = ask(graph, q["question"], budget_usd=budget)
            elapsed = time.perf_counter() - start
            answer = result.get("answer") or result.get("draft") or ""
            cited_docs = {h.doc_id for h in result.get("hits", [])}
            citation_hit = q["expected_doc"] in cited_docs
            number_hit = all(contains_number(answer, n) for n in q["expected_numbers"]) if q["expected_numbers"] else True
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "citation_hit": citation_hit,
                "number_hit": number_hit,
                "retries": result.get("retries", 0),
                "budget_exhausted": result.get("budget_exhausted", False),
                "cost_usd": round(result["budget"].settled_usd, 6),
                "latency_s": round(elapsed, 3),
                "n_subqueries": len(result.get("subqueries", [])),
                "n_hits": len(result.get("hits", [])),
                "n_unsupported": len(result.get("unsupported", [])),
                "model": llm.model,
            })

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    cit = sum(r["citation_hit"] for r in rows)
    num = sum(r["number_hit"] for r in rows)
    cost_avg = sum(r["cost_usd"] for r in rows) / n
    lat_avg = sum(r["latency_s"] for r in rows) / n
    exh = sum(r["budget_exhausted"] for r in rows)
    print(f"wrote {out}")
    print(f"n={n}  citation_hit={cit}/{n}  number_hit={num}/{n}  budget_exhausted={exh}/{n}")
    print(f"mean cost ${cost_avg:.6f}  mean latency {lat_avg:.3f}s  model={llm.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
