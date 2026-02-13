from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tarvia_ai.chunking import chunk_text
from tarvia_ai.document_loader import LoadedDocument


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str


class LocalVectorStore:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.chunks: List[StoredChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._load()

    def add_documents(
        self,
        documents: List[LoadedDocument],
        chunk_size: int,
        chunk_overlap: int,
    ) -> int:
        chunk_count = 0
        for doc in documents:
            chunks = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for idx, chunk in enumerate(chunks):
                self.chunks.append(
                    StoredChunk(
                        chunk_id=f"{doc.doc_id}::chunk_{idx}",
                        doc_id=doc.doc_id,
                        source=doc.source,
                        text=chunk,
                    )
                )
                chunk_count += 1
        if chunk_count > 0:
            self._refit_index()
            self.save()
        return chunk_count

    def search(self, query: str, top_k: int) -> List[Tuple[StoredChunk, float]]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []

        query_matrix = self.vectorizer.transform([query])
        scores = cosine_similarity(query_matrix, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: List[Tuple[StoredChunk, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append((self.chunks[int(idx)], score))
        return results

    def save(self) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("wb") as f:
            pickle.dump(payload, f)

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        with self.store_path.open("rb") as f:
            payload = pickle.load(f)
        self.chunks = [StoredChunk(**item) for item in payload.get("chunks", [])]
        self.vectorizer = payload.get("vectorizer")
        self.matrix = payload.get("matrix")
        if self.chunks and (self.vectorizer is None or self.matrix is None):
            self._refit_index()

    def _refit_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50_000,
            ngram_range=(1, 2),
        )
        corpus = [chunk.text for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)
