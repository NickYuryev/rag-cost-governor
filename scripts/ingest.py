"""Fetch SEC filings, chunk, embed and upsert into qdrant.

Usage:
    python scripts/ingest.py --tickers MSFT GOOGL --year 2024
    python scripts/ingest.py --tickers ANTHROPIC --form S-1 --year 2026

--form matches by prefix, so "S-1" also picks up S-1/A amendments and "10-K"
also picks up 10-K/A. the EDGAR api is public but rate-limited and insists on
an identifying User-Agent, read from RCG_EDGAR_USER_AGENT. the raw filing
goes to data/filings/<doc_id>.html so re-runs are cache-friendly.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_cost_governor.config import load
from rag_cost_governor.embeddings import LocalEmbedder
from rag_cost_governor.store import Chunk, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

TICKER_CIK = {
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "NVDA": "0001045810",
    "ANTHROPIC": "0002130357",
}

PAGE_CHARS = 2800  # ~700 tokens per chunk


def fetch_latest_filing_url(cik: str, year: int, form_prefix: str, ua: str) -> str:
    """URL of the first filing whose form starts with `form_prefix` (so S-1 also catches S-1/A)."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if not form.startswith(form_prefix):
            continue
        if not recent["filingDate"][i].startswith(str(year)):
            continue
        acc = recent["accessionNumber"][i].replace("-", "")
        doc = recent["primaryDocument"][i]
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    raise RuntimeError(f"no {form_prefix} found for cik={cik} year={year}")


def strip_html(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = PAGE_CHARS) -> list[str]:
    out = []
    for i in range(0, len(text), size):
        piece = text[i : i + size].strip()
        if len(piece) > 200:
            out.append(piece)
    return out


def ingest_one(ticker: str, year: int, form_prefix: str, ua: str, cache: Path) -> list[Chunk]:
    cik = TICKER_CIK[ticker]
    slug = form_prefix.lower().replace("-", "").replace("/", "")
    doc_id = f"{ticker.lower()}-{slug}-{year}"
    cache_path = cache / f"{doc_id}.html"
    if cache_path.exists():
        log.info("cache hit %s", cache_path)
        html = cache_path.read_text()
    else:
        url = fetch_latest_filing_url(cik, year, form_prefix, ua)
        log.info("fetching %s", url)
        r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
        r.raise_for_status()
        html = r.text
        cache_path.write_text(html)
        time.sleep(0.2)  # EDGAR limits to ~10 req/s per client.
    text = strip_html(html)
    pieces = chunk_text(text)
    log.info("%s chunked into %d pieces (%d chars)", doc_id, len(pieces), len(text))
    return [Chunk(doc_id=doc_id, page=i + 1, text=p) for i, p in enumerate(pieces)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["MSFT", "GOOGL"])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--form", default="10-K", help="form prefix: 10-K, S-1, 20-F etc.")
    ap.add_argument("--cache", default="data/filings")
    args = ap.parse_args()

    cfg = load()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    embedder = LocalEmbedder(model_name=cfg.embed_model)
    store = VectorStore(cfg.qdrant_url, cfg.qdrant_collection, embedder.dim)

    for ticker in args.tickers:
        chunks = ingest_one(ticker, args.year, args.form, cfg.edgar_user_agent, cache)
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        vecs = list(tqdm(embedder.embed_iter(texts), total=len(texts), desc=f"embed {ticker}"))
        n = store.upsert(chunks, vecs)
        log.info("upserted %d chunks for %s", n, ticker)
    log.info("collection now holds %d points", store.count())


if __name__ == "__main__":
    main()
