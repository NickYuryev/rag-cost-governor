from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Published list prices as of 2025-10. Kept here so cost accounting stays in-process
# and does not depend on a provider billing roundtrip.
PRICES_USD_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-haiku-5-5": (0.0008, 0.004),
    "claude-sonnet-5-5": (0.003, 0.015),
    "mock": (0.0001, 0.0004),
}

TOKENS_IN = Counter(
    "rcg_tokens_in_total", "Prompt tokens sent to the LLM", ["node", "model"]
)
TOKENS_OUT = Counter(
    "rcg_tokens_out_total", "Completion tokens received from the LLM", ["node", "model"]
)
COST_USD = Counter(
    "rcg_cost_usd_total", "Settled cost in USD", ["node", "model"]
)
REQ_SECONDS = Histogram(
    "rcg_request_duration_seconds",
    "LLM request wall time",
    ["node", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
REQ_FAIL = Counter(
    "rcg_request_failures_total", "LLM request failures", ["node", "model", "reason"]
)


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


class LLMBackend(Protocol):
    model: str

    def complete(self, prompt: str, max_tokens: int) -> LLMResponse: ...


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    rate_in, rate_out = PRICES_USD_PER_1K.get(model, PRICES_USD_PER_1K["mock"])
    return (tokens_in / 1000.0) * rate_in + (tokens_out / 1000.0) * rate_out


def worst_case_cost(model: str, prompt_tokens: int, max_tokens: int) -> float:
    return price(model, prompt_tokens, max_tokens)


class MockBackend:
    """Deterministic backend for tests and offline eval.

    Returns one of a few canned responses by looking at the first prompt line, and
    counts tokens as whitespace-separated words. Numbers are chosen to exercise
    the verifier.
    """

    model = "mock"

    def __init__(self, canned: dict[str, str] | None = None) -> None:
        self._canned = canned or {}

    def complete(self, prompt: str, max_tokens: int) -> LLMResponse:
        key = prompt.strip().splitlines()[0][:80] if prompt.strip() else ""
        text = self._canned.get(key, "mock answer")
        t_in = len(prompt.split())
        t_out = min(max_tokens, len(text.split()))
        return LLMResponse(text, t_in, t_out, price(self.model, t_in, t_out))


class OpenAIBackend:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI  # imported lazily so the dep stays optional

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str, max_tokens: int) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        t_in = usage.prompt_tokens if usage else len(prompt.split())
        t_out = usage.completion_tokens if usage else len(text.split())
        return LLMResponse(text, t_in, t_out, price(self.model, t_in, t_out))


class AnthropicBackend:
    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, max_tokens: int) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        t_in = resp.usage.input_tokens
        t_out = resp.usage.output_tokens
        return LLMResponse(text, t_in, t_out, price(self.model, t_in, t_out))


class InstrumentedLLM:
    """Wraps a backend with prometheus metrics and a per-call node label.

    `call` is the only thing nodes should touch. The budget check is done by the
    graph edge, not here, so this stays provider-shaped.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    @property
    def model(self) -> str:
        return self._backend.model

    def call(self, node: str, prompt: str, max_tokens: int = 400) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = self._backend.complete(prompt, max_tokens)
        except Exception as e:
            REQ_FAIL.labels(node=node, model=self.model, reason=type(e).__name__).inc()
            raise
        finally:
            REQ_SECONDS.labels(node=node, model=self.model).observe(
                time.perf_counter() - start
            )
        TOKENS_IN.labels(node=node, model=self.model).inc(resp.tokens_in)
        TOKENS_OUT.labels(node=node, model=self.model).inc(resp.tokens_out)
        COST_USD.labels(node=node, model=self.model).inc(resp.cost_usd)
        return resp


def build_from_env() -> InstrumentedLLM:
    from .config import load

    s = load()
    backend_name = s.llm_backend.lower()
    if backend_name == "openai":
        key = s.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("RCG_LLM_BACKEND=openai but OPENAI_API_KEY is empty")
        return InstrumentedLLM(OpenAIBackend(s.llm_model, key))
    if backend_name == "anthropic":
        key = s.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("RCG_LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is empty")
        return InstrumentedLLM(AnthropicBackend(s.llm_model, key))
    if backend_name == "mock":
        return InstrumentedLLM(MockBackend())
    raise ValueError(f"unknown backend: {backend_name}")
