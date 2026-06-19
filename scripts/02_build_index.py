"""
scripts/02_build_index.py
Reads data/contexts.jsonl, embeds all chunks with gte-multilingual-base,
stores them in ChromaDB, and saves a BM25 index to disk.

Run: python scripts/02_build_index.py
Expected time: ~15-30 min on CPU for 21k chunks
"""

import json
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    PROCESSED_CONTEXTS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    BM25_INDEX_PATH,
)


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_chroma_index(chunks: list[dict], model: SentenceTransformer):
    print(f"[ChromaDB] Initialising persistent store at ./{CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Drop existing collection to allow clean rebuilds
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("  (Deleted existing collection for fresh build)")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]
    metadatas = [{"title": c["title"], "source_split": c["source_split"]} for c in chunks]

    # Embed in batches
    print(f"[ChromaDB] Embedding {len(texts):,} chunks with {EMBEDDING_MODEL} on {EMBEDDING_DEVICE} …")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        embs = model.encode(batch, device=EMBEDDING_DEVICE, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())

    # Upsert in batches of 5000 (ChromaDB limit)
    UPSERT_BATCH = 5000
    print(f"[ChromaDB] Upserting {len(all_embeddings):,} vectors …")
    for i in tqdm(range(0, len(chunks), UPSERT_BATCH), desc="Upserting"):
        collection.upsert(
            ids=ids[i : i + UPSERT_BATCH],
            embeddings=all_embeddings[i : i + UPSERT_BATCH],
            documents=texts[i : i + UPSERT_BATCH],
            metadatas=metadatas[i : i + UPSERT_BATCH],
        )

    print(f"[ChromaDB] Done ✓  ({collection.count():,} vectors stored)")
    return collection


def build_bm25_index(chunks: list[dict]):
    print(f"[BM25] Building index over {len(chunks):,} chunks …")
    from underthesea import word_tokenize
    
    # Segment Vietnamese text properly into compound tokens
    tokenised = [word_tokenize(c["text"].lower(), format="text").split() for c in chunks]
    bm25 = BM25Okapi(tokenised)

    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [{"title": c["title"]} for c in chunks],
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[BM25] Done ✓  saved to {BM25_INDEX_PATH}")


def main():
    if not os.path.exists(PROCESSED_CONTEXTS_PATH):
        print(f"ERROR: {PROCESSED_CONTEXTS_PATH} not found. Run 01_prepare_data.py first.")
        sys.exit(1)

    chunks = load_chunks(PROCESSED_CONTEXTS_PATH)
    print(f"Loaded {len(chunks):,} chunks")

    # scripts/02_build_index.py (Line 111)
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE, trust_remote_code=True)

    build_chroma_index(chunks, model)
    build_bm25_index(chunks)

    print("\n✓ All indexes built successfully.")


if __name__ == "__main__":
    main()
