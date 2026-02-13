from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, description="Absolute or relative file paths to ingest.")


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int


class TrainPubMedRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=25)
    max_results_per_query: int | None = Field(default=None, ge=1, le=20)


class TrainPubMedResponse(BaseModel):
    records_fetched: int
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=20)


class BatchAskRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    source: str
    excerpt: str


class TrialCriteria(BaseModel):
    inclusion: list[str]
    exclusion: list[str]


class EvidenceGrade(str, Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    INSUFFICIENT = "Insufficient"


class OncologyAnswerSchema(BaseModel):
    summary: str
    summary_citation_ids: list[str]
    trial_criteria: TrialCriteria
    evidence_grade: EvidenceGrade
    contraindications: list[str]
    contraindication_citations: dict[str, list[str]]
    reasoning_trace: str


class AskResponse(BaseModel):
    schema: OncologyAnswerSchema
    limitations: str
    insufficient_evidence: bool
    citations: list[Citation]
    retrieved_chunks: int


class BatchAskItem(BaseModel):
    question: str
    result: AskResponse


class BatchAskResponse(BaseModel):
    results: list[BatchAskItem]


class AskABResponse(BaseModel):
    anthropic: AskResponse
    local: AskResponse


class GoldExampleInput(BaseModel):
    question: str
    answer: str
    evidence_grade: EvidenceGrade
    contraindications: list[str] = Field(default_factory=list)
    notes: str = ""


class GoldExamplesUpsertRequest(BaseModel):
    examples: list[GoldExampleInput] = Field(min_length=1, max_length=500)


class GoldExamplesUpsertResponse(BaseModel):
    examples_added: int
    total_examples: int


class GoldExamplesStatsResponse(BaseModel):
    total_examples: int
