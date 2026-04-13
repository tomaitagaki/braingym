"""
Embed the BrainBiz paper corpus into ChromaDB.

Chunks each paper's abstract + metadata into a vector store
for retrieval during chat.

Usage: python3 brainbiz/embed_corpus.py
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

CORPUS_PATH = Path(__file__).parent.parent / "cache/brainbiz_corpus/rag_ready_v2.json"
DB_DIR = Path(__file__).parent / "vectorstore"


def main():
    papers = json.loads(CORPUS_PATH.read_text())
    print(f"Loaded {len(papers)} papers")

    # Use ChromaDB's default embedding (all-MiniLM-L6-v2, runs locally)
    ef = DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete existing collection if re-running
    try:
        client.delete_collection("papers")
    except Exception:
        pass

    collection = client.create_collection(
        name="papers",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs = []
    metadatas = []
    ids = []

    for i, paper in enumerate(papers):
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        year = paper.get("publication_year", "")
        journal = paper.get("source_name", "")
        citations = paper.get("cited_by_count", 0)
        doi = paper.get("doi", "")
        keywords = ", ".join(paper.get("keywords", [])[:10])
        oa_url = paper.get("oa_url", "")

        # Build the document text for embedding
        doc = f"Title: {title}\n"
        if year:
            doc += f"Year: {year}\n"
        if journal:
            doc += f"Journal: {journal}\n"
        if keywords:
            doc += f"Keywords: {keywords}\n"
        doc += f"\nAbstract: {abstract}"

        docs.append(doc)
        metadatas.append({
            "title": str(title or ""),
            "year": int(year) if year else 0,
            "journal": str(journal or ""),
            "citations": int(citations),
            "doi": str(doi or ""),
            "oa_url": str(oa_url or ""),
            "keywords": str(keywords or ""),
        })
        ids.append(f"paper_{i}")

    # ChromaDB has a batch limit of ~5000
    batch_size = 500
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        print(f"  Embedding batch {start}-{end}...")
        collection.add(
            documents=docs[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"\nDone. {collection.count()} documents in vectorstore at {DB_DIR}")
    print(f"DB size: {sum(f.stat().st_size for f in DB_DIR.rglob('*') if f.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
