from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are TARVIA AI, an oncology-focused clinical evaluation assistant. "
    "Return structured, evidence-grounded answers and do not hallucinate."
)


def to_training_text(question: str, answer: str, evidence_grade: str, contraindications: list[str], notes: str) -> str:
    output = {
        "summary": answer,
        "summary_citation_ids": [],
        "trial_criteria": {"inclusion": [], "exclusion": []},
        "evidence_grade": evidence_grade,
        "contraindications": contraindications,
        "contraindication_citations": {},
        "reasoning_trace": notes,
    }
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nQuestion:\n{question}\n\nReturn strict JSON.\n"
        f"<|assistant|>\n{json.dumps(output, ensure_ascii=True)}"
    )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL dataset from TARVIA gold examples.")
    parser.add_argument("--gold-file", required=True, help="Path to gold_examples.jsonl")
    parser.add_argument("--out", required=True, help="Output JSONL path with {'text': ...} rows")
    args = parser.parse_args()

    gold_path = Path(args.gold_file).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with gold_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            evidence_grade = str(item.get("evidence_grade", "Insufficient")).strip()
            contraindications = [str(x).strip() for x in item.get("contraindications", []) if str(x).strip()]
            notes = str(item.get("notes", "")).strip()
            if not question or not answer:
                continue

            text = to_training_text(
                question=question,
                answer=answer,
                evidence_grade=evidence_grade,
                contraindications=contraindications,
                notes=notes,
            )
            dst.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
            count += 1

    print(f"Wrote {count} training rows to {out_path}")


if __name__ == "__main__":
    main()
