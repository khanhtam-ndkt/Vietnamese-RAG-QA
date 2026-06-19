"""
retrieval/retriever.py
Hybrid retriever: Dense (ChromaDB) + Sparse (BM25) → RRF fusion → Cross-Encoder rerank
"""

import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Optional
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from underthesea import word_tokenize  # Added for Vietnamese word segmentation

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    BM25_INDEX_PATH,
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
    HYBRID_TOP_K,
    RERANKER_MODEL,
    RERANKER_DEVICE,
    RERANK_TOP_K,
)


class HybridRetriever:
    def __init__(self):
        print("[Retriever] Loading embedding model …")
        # Added trust_remote_code=True for GTE model support
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE, trust_remote_code=True)

        print("[Retriever] Connecting to ChromaDB …")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_collection(CHROMA_COLLECTION)

        print("[Retriever] Loading BM25 index …")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_payload = pickle.load(f)
        self.bm25 = bm25_payload["bm25"]
        self.bm25_chunk_ids = bm25_payload["chunk_ids"]
        self.bm25_texts = bm25_payload["texts"]
        self.bm25_metadatas = bm25_payload["metadatas"]

        print("[Retriever] Loading Cross-Encoder reranker …")
        self.reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)

        print("[Retriever] Ready ✓")

    # ── Dense retrieval ───────────────────────────────────────────────────────
    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
        # GTE-multilingual requires a specific instruction prefix for asymmetric queries
        query_prefix = "Represent this sentence for searching relevant passages: "
        prefixed_query = query_prefix + query
        
        query_emb = self.embedder.encode([prefixed_query], device=EMBEDDING_DEVICE)[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append({
                "chunk_id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "dense_rank": i + 1,
            })
        return hits

    # ── Sparse retrieval ──────────────────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        # Use underthesea to join Vietnamese compound words with underscores
        tokenised_query = word_tokenize(query.lower(), format="text").split()
        scores = self.bm25.get_scores(tokenised_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for rank, idx in enumerate(top_indices):
            hits.append({
                "chunk_id": self.bm25_chunk_ids[idx],
                "text": self.bm25_texts[idx],
                "metadata": self.bm25_metadatas[idx],
                "bm25_rank": rank + 1,
            })
        return hits

    # ── RRF fusion ────────────────────────────────────────────────────────────
    def _rrf_fusion(
        self,
        dense_hits: list[dict],
        bm25_hits: list[dict],
        k: int = RRF_K,
        top_k: int = HYBRID_TOP_K,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        doc_store: dict[str, dict] = {}

        for hit in dense_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + hit["dense_rank"])
            doc_store[cid] = hit

        for hit in bm25_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + hit["bm25_rank"])
            doc_store.setdefault(cid, hit)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (cid, rrf_score) in enumerate(ranked):
            entry = dict(doc_store[cid])
            entry["rrf_score"] = rrf_score
            entry["hybrid_rank"] = rank + 1
            results.append(entry)
        return results

    # ── Cross-Encoder reranking ───────────────────────────────────────────────
    def _rerank(self, query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        for rank, doc in enumerate(reranked):
            doc["final_rank"] = rank + 1
        return reranked

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[dict]:
        dense_hits = self._dense_search(query)
        bm25_hits  = self._bm25_search(query)
        fused      = self._rrf_fusion(dense_hits, bm25_hits)
        reranked   = self._rerank(query, fused)
        return reranked