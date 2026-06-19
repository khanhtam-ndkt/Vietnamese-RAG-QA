# Vietnamese RAG Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) system, focusing on efficient indexing, retrieval, and robust evaluation metrics to ensure high-quality output.

**Stack:** Python · ChromaDB · BM25 · RRF · Cross-Encoder · Ollama · Flask · Streamlit  
**Dataset:** UIT-ViQuAD 2.0  
**Targets:** Recall@10 ≥ 0.9, nDCG@10 ≥ 0.9

---
## Evaluation Results
The system was evaluated on a dataset of **n=2653** samples.

| Metric | Score | Target | Status |
| :--- | :--- | :--- | :--- |
| **Recall@10** | 0.9175 | ≥ 0.9 | ✅ Passed |
| **nDCG@10** | 0.8632 | ≥ 0.9 | ⚠️ Needs Improvement |

---

## Folder Structure

```
vietnamese-rag/
├── config.py                  # All settings in one place
├── requirements.txt
├── data/                      # Downloaded + chunked data (auto-created)
├── chroma_db/                 # ChromaDB persistent store (auto-created)
├── scripts/
│   ├── 01_prepare_data.py     # Download & chunk ViQuAD2.0
│   ├── 02_build_index.py      # Build ChromaDB + BM25 index
│   └── 03_evaluate.py         # Eval Recall@10 / nDCG@10
├── retrieval/
│   └── retriever.py           # HybridRetriever class
├── generation/
│   └── generator.py           # Ollama generator
├── api/
│   └── app.py                 # Flask REST API
└── ui/
    └── app.py                 # Streamlit UI
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama (https://ollama.com/download)
# Then pull the LLM model:
ollama pull qwen2.5:3b
```

---

## Run Order

### Step 1 — Prepare data
```bash
python scripts/01_prepare_data.py
```
Downloads ViQuAD2.0 from HuggingFace, chunks contexts into ~21k chunks.

### Step 2 — Build indexes
```bash
python scripts/02_build_index.py
```
- Embeds all chunks with `gte-multilingual-base` (CPU, ~20-30 min)
- Stores in ChromaDB
- Builds and saves BM25 index

### Step 3 — Start Ollama
```bash
ollama serve          # in a separate terminal
```

### Step 4 — Start Flask API
```bash
python api/app.py
```
API runs at `http://localhost:5000`

### Step 5 — Start Streamlit UI
```bash
streamlit run ui/app.py
```
Opens at `http://localhost:8501`

### Step 6 — Evaluate (optional)
```bash
# With API running:
python scripts/03_evaluate.py
```
> **Note:** the full evaluation (n=2653) was run on Kaggle rather than locally — see [Execution Notes](#execution-notes) below.

---

## API Usage

```bash
# Ask a question
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Thủ đô của Việt Nam là gì?"}'

# Retrieval only (for eval)
curl -X POST http://localhost:5000/retrieve \
     -H "Content-Type: application/json" \
     -d '{"question": "Thủ đô của Việt Nam là gì?"}'

# Health check
curl http://localhost:5000/health
```

---

## Execution Notes

All core components — indexing, the Flask API, and the Streamlit UI — run locally without issue (CPU-only, ~2.5GB VRAM for the LLM via Ollama).

The **full metric evaluation** (`03_evaluate.py`), however, was run on **Kaggle** (see `notebook\viet-rag-03-eval.ipynb`) rather than locally. Evaluating ~2,653 samples requires sending that many sequential requests through the retrieval (and generation) pipeline, which made it prohibitively slow on local hardware. Kaggle's resources were used instead to complete the full-scale evaluation in a practical timeframe.

| Component | Runs locally? | Notes |
|---|---|---|
| `01_prepare_data.py` | ✅ Yes | CPU only |
| `02_build_index.py` | ✅ Yes | CPU, ~20-30 min embedding |
| `api/app.py` (Flask) | ✅ Yes | CPU + ~2.5GB VRAM for Ollama LLM |
| `ui/app.py` (Streamlit) | ✅ Yes | — |
| `03_evaluate.py` (full n=2653) | ⚠️ Ran on Kaggle | Too slow locally at this scale |

### Component VRAM Usage

| Component | Runs on | VRAM used |
|---|---|---|
| `gte-multilingual-base` (embedding) | CPU | 0 |
| BM25 | CPU | 0 |
| `bge-reranker-base` (reranker) | CPU | 0 |
| `qwen2.5:3b` (Ollama LLM) | GPU | ~2.5GB |

---

## Config Tuning

Edit `config.py` to adjust:
- `DENSE_TOP_K`, `BM25_TOP_K` — candidates before RRF
- `RRF_K` — RRF constant (60 is standard)
- `RERANK_TOP_K` — final passages sent to LLM
- `OLLAMA_MODEL` — swap LLM if needed

## Future Improvements
- **nDCG Optimization**: Investigate re-ranking strategies or alternative embedding models to close the gap between the current score (0.8632) and the target (0.9).
- **Failure Analysis**: Analyze the 219 logged failures to identify patterns and potential edge cases in the retrieval pipeline.