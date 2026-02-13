from __future__ import annotations

import argparse
from pathlib import Path

from tarvia_ai.config import get_settings
from tarvia_ai.document_loader import load_documents
from tarvia_ai.vector_store import LocalVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into TARVIA local vector store.")
    parser.add_argument("paths", nargs="+", help="Files to ingest (.pdf, .docx, .txt, .md).")
    args = parser.parse_args()

    settings = get_settings()
    store = LocalVectorStore(store_path=settings.store_path)

    paths = [Path(p).expanduser().resolve() for p in args.paths]
    documents = load_documents(paths)
    if not documents:
        print("No supported non-empty documents were found.")
        return

    chunks_added = store.add_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"Indexed documents: {len(documents)}")
    print(f"Indexed chunks: {chunks_added}")
    print(f"Store: {settings.store_path}")


if __name__ == "__main__":
    main()
