from __future__ import annotations

from pathlib import Path
import re

from fastapi import FastAPI, HTTPException

from tarvia_ai.anthropic_client import TarviaAnthropicClient
from tarvia_ai.api_models import (
    AskABResponse,
    AskRequest,
    AskResponse,
    BatchAskItem,
    BatchAskRequest,
    BatchAskResponse,
    Citation,
    EvidenceGrade,
    GoldExamplesStatsResponse,
    GoldExamplesUpsertRequest,
    GoldExamplesUpsertResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    OncologyAnswerSchema,
    TrainPubMedRequest,
    TrainPubMedResponse,
    TrialCriteria,
)
from tarvia_ai.config import Settings, get_settings
from tarvia_ai.document_loader import LoadedDocument, load_documents
from tarvia_ai.gold_store import GoldExample, GoldExampleStore
from tarvia_ai.local_client import TarviaLocalClient
from tarvia_ai.pubmed import fetch_pubmed_abstracts, search_pubmed
from tarvia_ai.vector_store import LocalVectorStore, StoredChunk


settings: Settings = get_settings()
store = LocalVectorStore(store_path=settings.store_path)
gold_store = GoldExampleStore(path=settings.gold_examples_path)
app = FastAPI(title="TARVIA AI RAG API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", chunks_indexed=len(store.chunks))


@app.get("/gold_examples/stats", response_model=GoldExamplesStatsResponse)
def gold_examples_stats() -> GoldExamplesStatsResponse:
    return GoldExamplesStatsResponse(total_examples=gold_store.count())


@app.post("/gold_examples", response_model=GoldExamplesUpsertResponse)
def upsert_gold_examples(payload: GoldExamplesUpsertRequest) -> GoldExamplesUpsertResponse:
    items = [
        GoldExample(
            question=item.question.strip(),
            answer=item.answer.strip(),
            evidence_grade=item.evidence_grade.value,
            contraindications=[x.strip() for x in item.contraindications if x.strip()],
            notes=item.notes.strip(),
        )
        for item in payload.examples
        if item.question.strip() and item.answer.strip()
    ]
    added = gold_store.add_examples(items)
    return GoldExamplesUpsertResponse(examples_added=added, total_examples=gold_store.count())


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    if not payload.paths:
        raise HTTPException(status_code=400, detail="No paths provided.")

    paths = [Path(p).expanduser().resolve() for p in payload.paths]
    documents = load_documents(paths)
    if not documents:
        raise HTTPException(status_code=400, detail="No supported non-empty documents found.")

    chunks_added = store.add_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return IngestResponse(documents_indexed=len(documents), chunks_indexed=chunks_added)


@app.post("/train_pubmed", response_model=TrainPubMedResponse)
def train_pubmed(payload: TrainPubMedRequest) -> TrainPubMedResponse:
    max_results = payload.max_results_per_query or settings.pubmed_max_results
    all_records = []
    for query in payload.queries:
        pmids = search_pubmed(
            query=query,
            max_results=max_results,
            email=settings.pubmed_email,
            tool=settings.pubmed_tool,
            timeout_seconds=settings.pubmed_timeout_seconds,
        )
        records = fetch_pubmed_abstracts(
            pmids=pmids,
            email=settings.pubmed_email,
            tool=settings.pubmed_tool,
            timeout_seconds=settings.pubmed_timeout_seconds,
        )
        all_records.extend(records)

    if not all_records:
        return TrainPubMedResponse(records_fetched=0, chunks_indexed=0)

    documents: list[LoadedDocument] = []
    for record in all_records:
        documents.append(
            LoadedDocument(
                doc_id=f"pubmed_{record.pmid}",
                source=record.source,
                text=record.abstract,
            )
        )

    chunks_added = store.add_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return TrainPubMedResponse(records_fetched=len(all_records), chunks_indexed=chunks_added)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    return _answer_question(payload.question, payload.top_k, provider="anthropic")


@app.post("/ask_local", response_model=AskResponse)
def ask_local(payload: AskRequest) -> AskResponse:
    return _answer_question(payload.question, payload.top_k, provider="local")


@app.post("/ask_ab", response_model=AskABResponse)
def ask_ab(payload: AskRequest) -> AskABResponse:
    anthropic = _answer_question(payload.question, payload.top_k, provider="anthropic")
    local = _answer_question(payload.question, payload.top_k, provider="local")
    return AskABResponse(anthropic=anthropic, local=local)


@app.post("/batch_ask", response_model=BatchAskResponse)
def batch_ask(payload: BatchAskRequest) -> BatchAskResponse:
    results: list[BatchAskItem] = []
    for question in payload.questions:
        result = _answer_question(question, payload.top_k, provider="anthropic")
        results.append(BatchAskItem(question=question, result=result))
    return BatchAskResponse(results=results)


def _answer_question(question: str, top_k_override: int | None, provider: str) -> AskResponse:
    if not store.chunks:
        raise HTTPException(status_code=400, detail="No documents indexed. Call /ingest first.")

    top_k = top_k_override or settings.top_k
    retrieved = store.search(question, top_k=top_k)
    retrieved = _augment_with_pubmed(question, retrieved)
    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant context found for this query.")

    top_score = max(score for _chunk, score in retrieved)
    if top_score < settings.min_relevance_score:
        return _insufficient_evidence_response(
            citations=[_citation_from_chunk(retrieved[0][0])],
            retrieved_chunks=len(retrieved),
            limitations=(
                f"Top retrieval score {top_score:.4f} below threshold "
                f"{settings.min_relevance_score:.4f}."
            ),
        )

    gold_examples = gold_store.search(question=question, top_k=settings.gold_examples_top_k)
    result = _provider_answer(provider, question, retrieved, gold_examples)

    citations: list[Citation] = []
    chosen_ids = set(result.get("citation_ids", []))
    for chunk, _score in retrieved:
        if chunk.chunk_id in chosen_ids:
            citations.append(
                _citation_from_chunk(chunk)
            )

    if not citations:
        citations.append(_citation_from_chunk(retrieved[0][0]))

    model_says_insufficient = bool(result.get("insufficient_evidence", False))
    low_citation_support = len(citations) < settings.min_citations
    evidence_grade = _normalize_evidence_grade(str(result.get("evidence_grade", "Insufficient")).strip())
    contraindications = _normalize_string_list(result.get("contraindications", []))
    summary_citation_ids = _normalize_citation_ids(result.get("summary_citation_ids", []))
    valid_citation_ids = {citation.chunk_id for citation in citations}
    summary_citation_ids = [cid for cid in summary_citation_ids if cid in valid_citation_ids]

    contraindications_required = _question_requires_contraindications(question)
    missing_contraindications = contraindications_required and not contraindications
    unsupported_contraindications = bool(contraindications) and not _contraindications_supported_by_citations(
        contraindications,
        citations,
    )
    contraindication_citations = _normalize_contraindication_citations(
        result.get("contraindication_citations", {}),
        contraindications,
    )
    invalid_contraindication_links = not _validate_contraindication_links(
        contraindications=contraindications,
        contraindication_citations=contraindication_citations,
        citations=citations,
        evidence_grade=evidence_grade,
        high_grade_min_citations=settings.high_grade_min_contraindication_citations,
    )
    invalid_high_grade_summary_support = not _validate_high_grade_summary_support(
        evidence_grade=evidence_grade,
        summary_citation_ids=summary_citation_ids,
        min_summary_citations=settings.high_grade_min_summary_citations,
    )

    if (
        model_says_insufficient
        or low_citation_support
        or evidence_grade == EvidenceGrade.INSUFFICIENT
        or missing_contraindications
        or unsupported_contraindications
        or invalid_contraindication_links
        or invalid_high_grade_summary_support
    ):
        base_limitations = str(result.get("limitations", "")).strip()
        extra_reason = []
        if model_says_insufficient:
            extra_reason.append("Model flagged insufficient evidence.")
        if low_citation_support:
            extra_reason.append(
                f"Only {len(citations)} citation(s), below minimum {settings.min_citations}."
            )
        if evidence_grade == EvidenceGrade.INSUFFICIENT:
            extra_reason.append("Evidence grade is Insufficient.")
        if missing_contraindications:
            extra_reason.append("Question requires contraindications but none were extracted.")
        if unsupported_contraindications:
            extra_reason.append("Contraindications are not supported by cited evidence text.")
        if invalid_contraindication_links:
            extra_reason.append(
                "Each contraindication must map to at least one valid citation id with supporting evidence."
            )
            if evidence_grade == EvidenceGrade.HIGH:
                extra_reason.append(
                    "For High evidence grade, each contraindication needs multiple distinct citations."
                )
        if invalid_high_grade_summary_support:
            extra_reason.append(
                "For High evidence grade, summary claims require multiple distinct summary citation IDs."
            )
        limitations = " ".join([base_limitations] + extra_reason).strip()
        return _insufficient_evidence_response(
            citations=citations,
            retrieved_chunks=len(retrieved),
            limitations=limitations,
        )

    trial_criteria = result.get("trial_criteria", {})
    if not isinstance(trial_criteria, dict):
        trial_criteria = {"inclusion": [], "exclusion": []}
    inclusion = trial_criteria.get("inclusion", [])
    exclusion = trial_criteria.get("exclusion", [])
    if not isinstance(inclusion, list):
        inclusion = [str(inclusion)]
    if not isinstance(exclusion, list):
        exclusion = [str(exclusion)]

    schema = OncologyAnswerSchema(
        summary=str(result.get("summary", "")).strip(),
        summary_citation_ids=summary_citation_ids,
        trial_criteria=TrialCriteria(
            inclusion=[str(item) for item in inclusion],
            exclusion=[str(item) for item in exclusion],
        ),
        evidence_grade=evidence_grade,
        contraindications=contraindications,
        contraindication_citations=contraindication_citations,
        reasoning_trace=str(result.get("reasoning_trace", "")).strip(),
    )

    return AskResponse(schema=schema, limitations=str(result.get("limitations", "")).strip(), insufficient_evidence=False, citations=citations, retrieved_chunks=len(retrieved))


def _provider_answer(
    provider: str,
    question: str,
    retrieved: list[tuple[StoredChunk, float]],
    gold_examples: list[GoldExample],
) -> dict[str, object]:
    if provider == "local":
        client = TarviaLocalClient(
            model_path=settings.local_model_path,
            max_new_tokens=settings.local_max_new_tokens,
            temperature=settings.local_temperature,
        )
        return client.answer_question(question, retrieved, gold_examples=gold_examples)

    client = TarviaAnthropicClient(settings=settings)
    return client.answer_question(question, retrieved, gold_examples=gold_examples)


def _citation_from_chunk(chunk: StoredChunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        excerpt=(chunk.text[:300] + "...") if len(chunk.text) > 300 else chunk.text,
    )


def _insufficient_evidence_response(
    citations: list[Citation],
    retrieved_chunks: int,
    limitations: str,
) -> AskResponse:
    schema = OncologyAnswerSchema(
        summary="Insufficient evidence in the indexed context to answer confidently.",
        summary_citation_ids=[],
        trial_criteria=TrialCriteria(inclusion=[], exclusion=[]),
        evidence_grade=EvidenceGrade.INSUFFICIENT,
        contraindications=[],
        contraindication_citations={},
        reasoning_trace="Guardrail triggered due to weak retrieval support or model uncertainty.",
    )
    return AskResponse(
        schema=schema,
        limitations=limitations or "Insufficient evidence.",
        insufficient_evidence=True,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
    )


def _augment_with_pubmed(
    question: str,
    retrieved: list[tuple[StoredChunk, float]],
) -> list[tuple[StoredChunk, float]]:
    if not settings.pubmed_enabled:
        return retrieved

    try:
        pmids = search_pubmed(
            query=question,
            max_results=settings.pubmed_max_results,
            email=settings.pubmed_email,
            tool=settings.pubmed_tool,
            timeout_seconds=settings.pubmed_timeout_seconds,
        )
        records = fetch_pubmed_abstracts(
            pmids=pmids,
            email=settings.pubmed_email,
            tool=settings.pubmed_tool,
            timeout_seconds=settings.pubmed_timeout_seconds,
        )
    except Exception:
        return retrieved

    if not records:
        return retrieved

    existing_ids = {chunk.chunk_id for chunk, _score in retrieved}
    top_score = max((score for _chunk, score in retrieved), default=0.2)
    pubmed_results: list[tuple[StoredChunk, float]] = []
    for idx, record in enumerate(records):
        chunk_id = f"pubmed::{record.pmid}"
        if chunk_id in existing_ids:
            continue
        pubmed_chunk = StoredChunk(
            chunk_id=chunk_id,
            doc_id=f"pubmed_{record.pmid}",
            source=record.source,
            text=record.abstract,
        )
        score = max(top_score - (idx * 0.01), 0.05)
        pubmed_results.append((pubmed_chunk, score))

    return retrieved + pubmed_results


def _normalize_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_citation_ids(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_evidence_grade(value: str) -> EvidenceGrade:
    normalized = value.strip().lower()
    mapping = {
        "high": EvidenceGrade.HIGH,
        "moderate": EvidenceGrade.MODERATE,
        "low": EvidenceGrade.LOW,
        "insufficient": EvidenceGrade.INSUFFICIENT,
    }
    return mapping.get(normalized, EvidenceGrade.INSUFFICIENT)


def _question_requires_contraindications(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in ["contraindication", "contraindications", "avoid", "not recommended"])


def _contraindications_supported_by_citations(
    contraindications: list[str],
    citations: list[Citation],
) -> bool:
    if not contraindications:
        return False
    combined = " ".join(citation.excerpt.lower() for citation in citations)
    if not combined.strip():
        return False

    for contraindication in contraindications:
        phrase = contraindication.lower().strip()
        if not phrase:
            continue
        if phrase in combined:
            return True
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", phrase) if len(tok) >= 4]
        if tokens and sum(1 for tok in tokens if tok in combined) >= min(2, len(tokens)):
            return True
    return False


def _normalize_contraindication_citations(
    value: object,
    contraindications: list[str],
) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {item: [] for item in contraindications}

    normalized: dict[str, list[str]] = {}
    keys_by_lower = {item.lower(): item for item in contraindications}
    for raw_key, raw_ids in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        canonical_key = keys_by_lower.get(key.lower(), key)
        if isinstance(raw_ids, list):
            ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        elif raw_ids is None:
            ids = []
        else:
            text_id = str(raw_ids).strip()
            ids = [text_id] if text_id else []
        normalized[canonical_key] = ids

    for item in contraindications:
        normalized.setdefault(item, [])
    return normalized


def _validate_contraindication_links(
    contraindications: list[str],
    contraindication_citations: dict[str, list[str]],
    citations: list[Citation],
    evidence_grade: EvidenceGrade,
    high_grade_min_citations: int,
) -> bool:
    if not contraindications:
        return True

    valid_ids = {citation.chunk_id for citation in citations}
    citation_by_id = {citation.chunk_id: citation.excerpt.lower() for citation in citations}
    for contraindication in contraindications:
        citation_ids = contraindication_citations.get(contraindication, [])
        if not citation_ids:
            return False
        unique_ids = list(dict.fromkeys(citation_ids))
        if evidence_grade == EvidenceGrade.HIGH and len(unique_ids) < high_grade_min_citations:
            return False
        for citation_id in unique_ids:
            if citation_id not in valid_ids:
                return False
        evidence_text = " ".join(citation_by_id[citation_id] for citation_id in unique_ids)
        phrase = contraindication.lower().strip()
        if phrase and phrase not in evidence_text:
            tokens = [tok for tok in re.split(r"[^a-z0-9]+", phrase) if len(tok) >= 4]
            if not tokens or sum(1 for tok in tokens if tok in evidence_text) < min(2, len(tokens)):
                return False
    return True


def _validate_high_grade_summary_support(
    evidence_grade: EvidenceGrade,
    summary_citation_ids: list[str],
    min_summary_citations: int,
) -> bool:
    if evidence_grade != EvidenceGrade.HIGH:
        return True
    unique_ids = list(dict.fromkeys(summary_citation_ids))
    return len(unique_ids) >= min_summary_citations
