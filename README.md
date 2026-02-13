# TARVIA AI (Option 1: Anthropic RAG)

Production-oriented starter for TARVIA AI using:
- FastAPI API service
- Local document ingestion and chunking
- Local TF-IDF vector store for retrieval
- Anthropic for grounded answer generation

## 1) Setup

```bash
cd /Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set your Anthropic key in `.env`:

```bash
ANTHROPIC_API_KEY=your_real_key
```

## 2) Ingest Documents

Use API:

```bash
uvicorn tarvia_ai.main:app --reload --app-dir src
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths":["/absolute/path/to/file1.pdf","/absolute/path/to/file2.docx"]}'
```

Or use CLI:

```bash
tarvia-ingest /absolute/path/to/file1.pdf /absolute/path/to/file2.docx
```

## 3) Ask Questions

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize inclusion criteria for trial eligibility."}'
```

Returns:
- `schema.summary`
- `schema.summary_citation_ids`
- `schema.trial_criteria.inclusion`
- `schema.trial_criteria.exclusion`
- `schema.evidence_grade`
- `schema.contraindications`
- `schema.contraindication_citations` (one-to-many mapping)
- `schema.reasoning_trace`
- `limitations`
- `insufficient_evidence`
- `citations` (chunk ids + source + excerpt)
- `retrieved_chunks`

`evidence_grade` is strict and only returns one of:
- `High`
- `Moderate`
- `Low`
- `Insufficient`

Clinical guardrails now enforce:
- contraindication questions require contraindication output
- contraindications must be supported by cited text
- each contraindication must map to at least one valid citation ID
- if `evidence_grade=High`, each contraindication must map to at least 2 distinct citations
- if `evidence_grade=High`, summary claims must map to at least 2 distinct summary citation IDs
- unsupported/weak evidence is downgraded to `insufficient_evidence=true`

Batch evaluate many questions:

```bash
curl -X POST http://127.0.0.1:8000/batch_ask \
  -H "Content-Type: application/json" \
  -d '{"questions":["What are inclusion criteria?","What contraindications are listed?"],"top_k":5}'
```

Ask with local LoRA model:

```bash
curl -X POST http://127.0.0.1:8000/ask_local \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize contraindications for checkpoint inhibitor use."}'
```

A/B compare Anthropic vs local LoRA:

```bash
curl -X POST http://127.0.0.1:8000/ask_ab \
  -H "Content-Type: application/json" \
  -d '{"question":"What are contraindications in relapse treatment?"}'
```

Add your gold-standard supervised examples:

```bash
curl -X POST http://127.0.0.1:8000/gold_examples \
  -H "Content-Type: application/json" \
  -d '{
    "examples":[
      {
        "question":"When should this regimen be avoided?",
        "answer":"Avoid in severe hepatic impairment and active bleeding risk.",
        "evidence_grade":"High",
        "contraindications":["severe hepatic impairment","active bleeding"],
        "notes":"Expert gold standard from oncology review."
      }
    ]
  }'
```

Train the retriever from PubMed:

```bash
curl -X POST http://127.0.0.1:8000/train_pubmed \
  -H "Content-Type: application/json" \
  -d '{"queries":["metastatic breast cancer relapse biomarkers","immune checkpoint contraindications"],"max_results_per_query":3}'
```

## 4) Endpoints

- `GET /health`
- `GET /gold_examples/stats`
- `POST /gold_examples`
- `POST /ingest`
- `POST /train_pubmed`
- `POST /ask`
- `POST /ask_local`
- `POST /ask_ab`
- `POST /batch_ask`

## Notes

- Supported files: `.pdf`, `.docx`, `.txt`, `.md`
- Vector store persists to `TARVIA_DATA_DIR/TARVIA_STORE_FILE`
- Optional PubMed augmentation is enabled by default via NCBI E-utilities.
  - Configure `TARVIA_PUBMED_EMAIL` (recommended by NCBI).
  - Disable with `TARVIA_PUBMED_ENABLED=false`.
- Gold standards are persisted in `TARVIA_DATA_DIR/TARVIA_GOLD_EXAMPLES_FILE` and used as supervised few-shot guidance at inference.
- Retrieval quality can be upgraded later to dense embeddings + reranker without changing API contracts.

## Optional Weight-Level Fine-Tuning (LoRA)

If you want true weight updates on an open-source oncology model, use:
- `/Users/omarlujanoolazaba/Desktop/Luhano,Inc./tarvia_project/training/README.md`

That workflow includes:
- gold-standard to SFT dataset conversion
- LoRA adapter training
- optional merged checkpoint export
