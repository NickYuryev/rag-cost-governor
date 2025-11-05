from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RCG_", extra="ignore")

    llm_backend: str = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    budget_usd: float = 0.05

    qdrant_url: str = ":memory:"
    qdrant_collection: str = "filings"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    metrics_port: int = 9090
    edgar_user_agent: str = "rag-cost-governor contact@example.com"

    top_k: int = 5
    max_subqueries: int = 4
    verifier_tolerance_pct: float = 1.0


def load() -> Settings:
    import os
    # pydantic-settings only honours the configured prefix, so OPENAI_API_KEY /
    # ANTHROPIC_API_KEY (which the ecosystem agrees on) have to be picked up by hand.
    s = Settings()
    if not s.openai_api_key:
        s.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not s.anthropic_api_key:
        s.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return s
