from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

PRICES_USD_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


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
    rate_in, rate_out = PRICES_USD_PER_1K[model]
    return (tokens_in / 1000.0) * rate_in + (tokens_out / 1000.0) * rate_out


class OpenAIBackend:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

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
        t_in = usage.input_tokens if usage else len(prompt.split())
        t_out = usage.output_tokens if usage else len(text.split())
        return LLMResponse(text, t_in, t_out, price(self.model, t_in, t_out))
