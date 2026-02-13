from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List
from urllib.parse import urlencode
from urllib.request import urlopen


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass(frozen=True)
class PubMedRecord:
    pmid: str
    source: str
    abstract: str


def search_pubmed(
    query: str,
    max_results: int,
    email: str,
    tool: str,
    timeout_seconds: float,
) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "relevance",
        "email": email,
        "tool": tool,
    }
    url = f"{PUBMED_BASE}/esearch.fcgi?{urlencode(params)}"

    with urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))

    return payload.get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_abstracts(
    pmids: List[str],
    email: str,
    tool: str,
    timeout_seconds: float,
) -> List[PubMedRecord]:
    if not pmids:
        return []

    records: List[PubMedRecord] = []
    for pmid in pmids:
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
            "email": email,
            "tool": tool,
        }
        url = f"{PUBMED_BASE}/efetch.fcgi?{urlencode(params)}"
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                text = response.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not text:
            continue
        records.append(
            PubMedRecord(
                pmid=pmid,
                source=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                abstract=text,
            )
        )
    return records
