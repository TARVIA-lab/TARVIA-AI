from __future__ import annotations

import json
from typing import Dict, List, Tuple

from anthropic import Anthropic

from tarvia_ai.config import Settings
from tarvia_ai.gold_store import GoldExample
from tarvia_ai.vector_store import StoredChunk


class TarviaAnthropicClient:
    def __init__(self, settings: Settings):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing.")
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def answer_question(
        self,
        question: str,
        retrieved: List[Tuple[StoredChunk, float]],
        gold_examples: List[GoldExample] | None = None,
    ) -> Dict[str, object]:
        context = self._build_context(retrieved)
        examples_text = self._build_examples(gold_examples or [])
        response = self.client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            system=(
                "You are TARVIA AI, an oncology-focused evaluation assistant. "
                "Use only the provided context and do not fabricate details."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Answer the oncology question using only context. "
                        "Return strict JSON with keys:\n"
                        "summary (string)\n"
                        "summary_citation_ids (array of citation chunk ids supporting summary)\n"
                        "trial_criteria (object with inclusion array and exclusion array)\n"
                        "evidence_grade (string, must be exactly one of: High, Moderate, Low, Insufficient)\n"
                        "contraindications (array of strings)\n"
                        "contraindication_citations (object: key is contraindication string, value is array of citation chunk ids)\n"
                        "reasoning_trace (string, concise)\n"
                        "citation_ids (array of chunk ids)\n"
                        "limitations (string)\n"
                        "insufficient_evidence (boolean)\n\n"
                        f"Question:\n{question}\n\n"
                        f"Gold Standard Examples (supervised):\n{examples_text}\n\n"
                        f"Context:\n{context}"
                    ),
                }
            ],
        )
        text = self._extract_text(response)
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

    @staticmethod
    def _extract_text(response: object) -> str:
        blocks = getattr(response, "content", [])
        parts: List[str] = []
        for block in blocks:
            block_text = getattr(block, "text", "")
            if block_text:
                parts.append(block_text)
        return "\n".join(parts).strip()

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
            "limitations": "Model did not return strict JSON.",
            "insufficient_evidence": True,
        }

    @staticmethod
    def _build_context(retrieved: List[Tuple[StoredChunk, float]]) -> str:
        lines: List[str] = []
        for chunk, score in retrieved:
            lines.append(
                f"[{chunk.chunk_id}] source={chunk.source} score={score:.4f}\n"
                f"{chunk.text}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_examples(gold_examples: List[GoldExample]) -> str:
        if not gold_examples:
            return "None"
        lines: List[str] = []
        for idx, item in enumerate(gold_examples, start=1):
            lines.append(
                f"[Example {idx}] question={item.question}\n"
                f"gold_answer={item.answer}\n"
                f"gold_evidence_grade={item.evidence_grade}\n"
                f"gold_contraindications={item.contraindications}\n"
                f"gold_notes={item.notes}\n"
            )
        return "\n".join(lines)
