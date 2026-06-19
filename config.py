"""
Central config — edit here, everything else reads from here.
"""

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "taidng/UIT-ViQuAD2.0"          # HuggingFace dataset id
DATA_DIR = "data"
PROCESSED_CONTEXTS_PATH = "data/contexts.jsonl"   # deduped context chunks

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 80         # Target word-count per chunk
CHUNK_OVERLAP = 30      # Target word overlap between chunks
MIN_CHUNK_WORDS = 20    # Minimum words required to keep a chunk (filters noise)

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
EMBEDDING_DEVICE = "cpu"        # 4GB VRAM — keep embeddings on CPU
EMBEDDING_BATCH_SIZE = 64

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION = "virag_chunks"

# ── BM25 ─────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = "data/bm25_index.pkl"

# ── Hybrid Retrieval ─────────────────────────────────────────────────────────
DENSE_TOP_K = 50        # candidates from ChromaDB dense search
BM25_TOP_K = 50         # candidates from BM25
RRF_K = 60              # RRF constant (standard = 60)
HYBRID_TOP_K = 50       # final candidates passed to reranker

# ── Cross-Encoder Reranker ────────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Changed from English ms-marco
RERANKER_DEVICE = "cuda"
RERANK_TOP_K = 10

# ── Generation (Ollama) ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"     # ~2GB, fits 4GB VRAM; swap to llama3.2:3b if preferred
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

# ── Flask API ─────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_SPLIT = "validation"       # use validation set of ViQuAD2.0
EVAL_METRICS = ["recall_5", "ndcg_5"]
