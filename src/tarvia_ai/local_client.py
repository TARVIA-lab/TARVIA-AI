from __future__ import annotations

import json
from typing import Dict, List, Tuple

from tarvia_ai.gold_store import GoldExample
from tarvia_ai.vector_store import StoredChunk


class TarviaLocalClient:
    def __init__(self, model_path: str, max_new_tokens: int = 1200, temperature: float = 0.1):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None

    def answer_question(
        self,
        question: str,
        retrieved: List[Tuple[StoredChunk, float]],
        gold_examples: List[GoldExample] | None = None,
    ) -> Dict[str, object]:
        pipe = self._ensure_pipeline()
        prompt = self._build_prompt(question, retrieved, gold_examples or [])
        generated = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            return_full_text=False,
        )
        text = str(generated[0].get("generated_text", "")).strip()
        parsed = self._safe_parse(text)

        valid_ids = {chunk.chunk_id for chunk, _score in retrieved}
        citation_ids = [cid for cid in parsed.get("citation_ids", []) if cid in valid_ids]
        if not citation_ids and retrieved:
            citation_ids = [retrieved[0][0].chunk_id]

        return {
            "summary": str(parsed.get("summary", "")).strip(),
            "summary_citation_ids": parsed.get("summary_citation_ids", []),
            "trial_criteria": parsed.get("trial_criteria", {"inclusion": [], "exclusion": []}),
            "evidence_grade": str(parsed.get("evidence_grade", "Insufficient")).strip() or "Insufficient",
            "contraindications": parsed.get("contraindications", []),
            "contraindication_citations": parsed.get("contraindication_citations", {}),
            "reasoning_trace": str(parsed.get("reasoning_trace", "")).strip(),
            "citation_ids": citation_ids,
            "limitations": str(parsed.get("limitations", "")).strip(),
            "insufficient_evidence": bool(parsed.get("insufficient_evidence", False)),
        }

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install local model dependencies with: pip install -e '.[finetune]'") from exc

        self._pipe = pipeline("text-generation", model=self.model_path)
        return self._pipe

    @staticmethod
    def _build_prompt(
        question: str,
        retrieved: List[Tuple[StoredChunk, float]],
        gold_examples: List[GoldExample],
    ) -> str:
        context_lines: List[str] = []
        for chunk, score in retrieved:
            context_lines.append(
                f"[{chunk.chunk_id}] source={chunk.source} score={score:.4f}\n{chunk.text}\n"
            )

        example_lines: List[str] = []
        for idx, item in enumerate(gold_examples, start=1):
            example_lines.append(
                f"[Example {idx}] question={item.question}\n"
                f"gold_answer={item.answer}\n"
                f"gold_evidence_grade={item.evidence_grade}\n"
                f"gold_contraindications={item.contraindications}\n"
                f"gold_notes={item.notes}\n"
            )

        examples = "\n".join(example_lines) if example_lines else "None"
        context = "\n".join(context_lines)
        return (
            "You are TARVIA AI, an oncology-focused evaluation assistant. "
            "Use only provided context and return strict JSON with keys: "
            "summary, summary_citation_ids, trial_criteria, evidence_grade, contraindications, "
            "contraindication_citations, reasoning_trace, citation_ids, limitations, insufficient_evidence.\n\n"
            f"Question:\n{question}\n\n"
            f"Gold Standard Examples:\n{examples}\n\n"
            f"Context:\n{context}\n"
        )

    @staticmethod
    def _safe_parse(raw_text: str) -> Dict[str, object]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {
            "summary": raw_text,
            "summary_citation_ids": [],
            "trial_criteria": {"inclusion": [], "exclusion": []},
            "evidence_grade": "Insufficient",
            "contraindications": [],
            "contraindication_citations": {},
            "reasoning_trace": "",
            "citation_ids": [],
            "limitations": "Local model did not return strict JSON.",
            "insufficient_evidence": True,
        }
