from tarvia_ai.api_models import Citation, EvidenceGrade
from tarvia_ai.service import (
    _contraindications_supported_by_citations,
    _normalize_contraindication_citations,
    _normalize_evidence_grade,
    _normalize_citation_ids,
    _question_requires_contraindications,
    _validate_high_grade_summary_support,
    _validate_contraindication_links,
)


def test_evidence_grade_normalization() -> None:
    assert _normalize_evidence_grade("High") == EvidenceGrade.HIGH
    assert _normalize_evidence_grade("moderate") == EvidenceGrade.MODERATE
    assert _normalize_evidence_grade("LOW") == EvidenceGrade.LOW
    assert _normalize_evidence_grade("unknown") == EvidenceGrade.INSUFFICIENT


def test_contraindication_support_detection() -> None:
    citations = [
        Citation(
            chunk_id="c1",
            source="doc",
            excerpt="Contraindications include severe hepatic impairment and active bleeding.",
        )
    ]
    assert _contraindications_supported_by_citations(["severe hepatic impairment"], citations)
    assert not _contraindications_supported_by_citations(["pregnancy"], citations)


def test_question_requires_contraindications() -> None:
    assert _question_requires_contraindications("What are the contraindications for this therapy?")
    assert not _question_requires_contraindications("What are inclusion criteria?")


def test_contraindication_link_validation() -> None:
    citations = [
        Citation(
            chunk_id="doc1::chunk_0",
            source="doc",
            excerpt="Contraindications include severe hepatic impairment and active bleeding.",
        )
    ]
    contraindications = ["severe hepatic impairment"]
    mapping = {"severe hepatic impairment": ["doc1::chunk_0"]}
    assert _validate_contraindication_links(
        contraindications, mapping, citations, EvidenceGrade.MODERATE, 2
    )

    bad_mapping = {"severe hepatic impairment": ["unknown_chunk"]}
    assert not _validate_contraindication_links(
        contraindications, bad_mapping, citations, EvidenceGrade.MODERATE, 2
    )


def test_high_grade_requires_two_citations_per_contraindication() -> None:
    citations = [
        Citation(
            chunk_id="doc1::chunk_0",
            source="doc",
            excerpt="Contraindications include severe hepatic impairment and active bleeding.",
        ),
        Citation(
            chunk_id="doc2::chunk_0",
            source="doc",
            excerpt="Avoid treatment in severe hepatic impairment due to toxicity concerns.",
        ),
    ]
    contraindications = ["severe hepatic impairment"]

    one_link = {"severe hepatic impairment": ["doc1::chunk_0"]}
    assert not _validate_contraindication_links(
        contraindications, one_link, citations, EvidenceGrade.HIGH, 2
    )

    two_links = {"severe hepatic impairment": ["doc1::chunk_0", "doc2::chunk_0"]}
    assert _validate_contraindication_links(
        contraindications, two_links, citations, EvidenceGrade.HIGH, 2
    )


def test_normalize_contraindication_citations() -> None:
    contraindications = ["Severe hepatic impairment"]
    raw = {"severe hepatic impairment": "doc1::chunk_0"}
    normalized = _normalize_contraindication_citations(raw, contraindications)
    assert normalized["Severe hepatic impairment"] == ["doc1::chunk_0"]


def test_high_grade_summary_support() -> None:
    assert not _validate_high_grade_summary_support(EvidenceGrade.HIGH, ["c1"], 2)
    assert _validate_high_grade_summary_support(EvidenceGrade.HIGH, ["c1", "c2"], 2)
    assert _validate_high_grade_summary_support(EvidenceGrade.MODERATE, ["c1"], 2)


def test_normalize_citation_ids() -> None:
    assert _normalize_citation_ids(["a", " ", "b"]) == ["a", "b"]
    assert _normalize_citation_ids("a") == ["a"]
