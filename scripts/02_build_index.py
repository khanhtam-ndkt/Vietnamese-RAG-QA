import json
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
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

def build_chroma_index(chunks: list[dict]):
    print(f"[ChromaDB] Initialising persistent store at ./{CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

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

    # --- RAW TRANSFORMERS IMPLEMENTATION (KAGGLE FIX) ---
    print(f"\nLoading tokenizer and model: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(EMBEDDING_DEVICE)
    model.eval()

    def mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    print(f"[ChromaDB] Embedding {len(texts):,} chunks...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(EMBEDDING_DEVICE)
        
        # Bypass corrupted position_ids
        seq_len = encoded["input_ids"].shape[1]
        encoded["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=EMBEDDING_DEVICE).unsqueeze(0).expand(len(batch), -1)
        
        with torch.no_grad():
            out = model(**encoded, unpad_inputs=False)  
            
        # Sanitize NaNs
        hidden_states = torch.nan_to_num(out.last_hidden_state, nan=0.0)
        emb = mean_pool(hidden_states, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.extend(emb.cpu().float().tolist())

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
    
    tokenised = [word_tokenize(c["text"].lower(), format="text").split() for c in chunks]
    from rank_bm25 import BM25Okapi
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

    build_chroma_index(chunks)
    build_bm25_index(chunks)
    print("\n✓ All indexes built successfully.")

if __name__ == "__main__":
    main()