from pathlib import Path

from tarvia_ai.gold_store import GoldExample, GoldExampleStore


def test_gold_example_store_add_and_search(tmp_path: Path) -> None:
    store = GoldExampleStore(path=tmp_path / "gold.jsonl")
    added = store.add_examples(
        [
            GoldExample(
                question="What are contraindications for checkpoint inhibitors?",
                answer="Active autoimmune flare is a contraindication.",
                evidence_grade="Moderate",
                contraindications=["active autoimmune flare"],
                notes="Expert gold answer",
            ),
            GoldExample(
                question="What is first-line approach in relapse setting?",
                answer="Depends on biomarker status and prior exposure.",
                evidence_grade="Low",
                contraindications=[],
                notes="Expert note",
            ),
        ]
    )
    assert added == 2
    assert store.count() == 2

    results = store.search("checkpoint inhibitor contraindication", top_k=1)
    assert len(results) == 1
    assert "contraindications" in results[0].question.lower()
