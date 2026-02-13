from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    anthropic_model: str
    data_dir: Path
    store_file: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_tokens: int
    temperature: float
    min_relevance_score: float
    min_citations: int
    pubmed_enabled: bool
    pubmed_email: str
    pubmed_tool: str
    pubmed_max_results: int
    pubmed_timeout_seconds: float
    high_grade_min_contraindication_citations: int
    high_grade_min_summary_citations: int
    gold_examples_file: str
    gold_examples_top_k: int
    local_model_path: str
    local_max_new_tokens: int
    local_temperature: float

    @property
    def store_path(self) -> Path:
        return self.data_dir / self.store_file

    @property
    def gold_examples_path(self) -> Path:
        return self.data_dir / self.gold_examples_file


def get_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("TARVIA_DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest").strip()

    pubmed_enabled_raw = os.getenv("TARVIA_PUBMED_ENABLED", "true").strip().lower()
    pubmed_enabled = pubmed_enabled_raw in {"1", "true", "yes", "on"}

    return Settings(
        anthropic_api_key=api_key,
        anthropic_model=model,
        data_dir=data_dir,
        store_file=os.getenv("TARVIA_STORE_FILE", "vector_store.pkl"),
        chunk_size=int(os.getenv("TARVIA_CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("TARVIA_CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TARVIA_TOP_K", "5")),
        max_tokens=int(os.getenv("TARVIA_MAX_TOKENS", "1200")),
        temperature=float(os.getenv("TARVIA_TEMPERATURE", "0.1")),
        min_relevance_score=float(os.getenv("TARVIA_MIN_RELEVANCE_SCORE", "0.08")),
        min_citations=int(os.getenv("TARVIA_MIN_CITATIONS", "1")),
        pubmed_enabled=pubmed_enabled,
        pubmed_email=os.getenv("TARVIA_PUBMED_EMAIL", "").strip(),
        pubmed_tool=os.getenv("TARVIA_PUBMED_TOOL", "tarvia_ai").strip(),
        pubmed_max_results=int(os.getenv("TARVIA_PUBMED_MAX_RESULTS", "3")),
        pubmed_timeout_seconds=float(os.getenv("TARVIA_PUBMED_TIMEOUT_SECONDS", "8")),
        high_grade_min_contraindication_citations=int(
            os.getenv("TARVIA_HIGH_GRADE_MIN_CONTRAINDICATION_CITATIONS", "2")
        ),
        high_grade_min_summary_citations=int(
            os.getenv("TARVIA_HIGH_GRADE_MIN_SUMMARY_CITATIONS", "2")
        ),
        gold_examples_file=os.getenv("TARVIA_GOLD_EXAMPLES_FILE", "gold_examples.jsonl"),
        gold_examples_top_k=int(os.getenv("TARVIA_GOLD_EXAMPLES_TOP_K", "3")),
        local_model_path=os.getenv("TARVIA_LOCAL_MODEL_PATH", "./artifacts/lora_qwen_3b/adapter"),
        local_max_new_tokens=int(os.getenv("TARVIA_LOCAL_MAX_NEW_TOKENS", "1200")),
        local_temperature=float(os.getenv("TARVIA_LOCAL_TEMPERATURE", "0.1")),
    )
