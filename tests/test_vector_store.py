from pathlib import Path

from tarvia_ai.document_loader import LoadedDocument
from tarvia_ai.vector_store import LocalVectorStore


def test_vector_store_add_and_search(tmp_path: Path) -> None:
    store_path = tmp_path / "store.pkl"
    store = LocalVectorStore(store_path=store_path)

    docs = [
        LoadedDocument(
            doc_id="doc1",
            source="/tmp/doc1.txt",
            text="Immunotherapy response criteria and oncology trial endpoints.",
        ),
        LoadedDocument(
            doc_id="doc2",
            source="/tmp/doc2.txt",
            text="Cardiology note unrelated to cancer treatment.",
        ),
    ]
    count = store.add_documents(documents=docs, chunk_size=200, chunk_overlap=20)
    assert count == 2

    results = store.search("oncology trial endpoints", top_k=1)
    assert len(results) == 1
    assert "doc1" in results[0][0].chunk_id
