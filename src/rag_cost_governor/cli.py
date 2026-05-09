from __future__ import annotations

import argparse
import json
import logging
import sys

from prometheus_client import generate_latest, start_http_server

from .config import load
from .graph import ask, build_default
from .llm_client import build_from_env

log = logging.getLogger("rag_cost_governor")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-cost-governor")
    ap.add_argument("question", nargs="?", help="analyst question to answer")
    ap.add_argument("--budget", type=float, default=None, help="override RCG_BUDGET_USD")
    ap.add_argument("--metrics-dump", action="store_true", help="print /metrics text and exit")
    ap.add_argument("--json", action="store_true", help="emit result as json instead of prose")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load()

    if settings.metrics_port and not args.metrics_dump:
        try:
            start_http_server(settings.metrics_port)
            log.info("prometheus exporter on :%d", settings.metrics_port)
        except OSError as e:
            log.warning("could not start metrics exporter: %s", e)

    try:
        llm = build_from_env()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    graph, _ = build_default(settings, llm)

    if not args.question:
        print("error: a question is required (use --help)", file=sys.stderr)
        return 2

    budget = args.budget if args.budget is not None else settings.budget_usd
    result = ask(graph, args.question, budget_usd=budget)
    answer = result.get("answer") or result.get("draft") or ""
    b = result["budget"]

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": answer,
                    "unsupported": result.get("unsupported", []),
                    "subqueries": result.get("subqueries", []),
                    "budget_exhausted": result.get("budget_exhausted", False),
                    "settled_usd": round(b.settled_usd, 6),
                    "remaining_usd": round(b.remaining(), 6),
                    "retries": result.get("retries", 0),
                    "hits": [
                        {"doc": h.doc_id, "page": h.page, "score": round(h.score, 4)}
                        for h in result.get("hits", [])
                    ],
                },
                indent=2,
            )
        )
    else:
        print(answer)
        print()
        print(
            f"-- cost ${b.settled_usd:.4f} / ${b.ceiling_usd:.4f}  "
            f"retries={result.get('retries', 0)}  "
            f"exhausted={result.get('budget_exhausted', False)}"
        )

    if args.metrics_dump:
        print()
        print(generate_latest().decode())

    return 0


if __name__ == "__main__":
    sys.exit(main())
