from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class GoldExample:
    question: str
    answer: str
    evidence_grade: str
    contraindications: list[str]
    notes: str


class GoldExampleStore:
    def __init__(self, path: Path):
        self.path = path
        self.examples: List[GoldExample] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._load()

    def add_examples(self, items: List[GoldExample]) -> int:
        if not items:
            return 0
        self.examples.extend(items)
        self._persist_append(items)
        self._refit()
        return len(items)

    def search(self, question: str, top_k: int) -> List[GoldExample]:
        if not self.examples or self.vectorizer is None or self.matrix is None:
            return []
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [self.examples[int(idx)] for idx in indices if float(scores[int(idx)]) > 0]

    def count(self) -> int:
        return len(self.examples)

    def _load(self) -> None:
        if not self.path.exists():
            return
        items: List[GoldExample] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items.append(
                    GoldExample(
                        question=str(data.get("question", "")).strip(),
                        answer=str(data.get("answer", "")).strip(),
                        evidence_grade=str(data.get("evidence_grade", "Insufficient")).strip(),
                        contraindications=[str(x).strip() for x in data.get("contraindications", [])],
                        notes=str(data.get("notes", "")).strip(),
                    )
                )
        self.examples = [x for x in items if x.question and x.answer]
        self._refit()

    def _persist_append(self, items: List[GoldExample]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(asdict(item), ensure_ascii=True) + "\n")

    def _refit(self) -> None:
        if not self.examples:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        corpus = [item.question for item in self.examples]
        self.matrix = self.vectorizer.fit_transform(corpus)
