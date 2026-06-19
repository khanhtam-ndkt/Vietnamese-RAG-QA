"""
api/app.py
Flask REST API exposing the RAG pipeline.

Endpoints:
  POST /ask        { "question": "..." }  → { "answer": "...", "passages": [...] }
  GET  /health     → { "status": "ok" }
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify

from retrieval.retriever import HybridRetriever
from generation.generator import Generator
from config import API_HOST, API_PORT

app = Flask(__name__)

# Lazy-loaded singletons
_retriever: HybridRetriever = None
_generator: Generator = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required"}), 400

    try:
        retriever = get_retriever()
        generator = get_generator()

        passages  = retriever.retrieve(question)
        result    = generator.generate(question, passages)

        return jsonify({
            "question": question,
            "answer":   result["answer"],
            "model":    result["model"],
            "passages": [
                {
                    "rank":  p["final_rank"],
                    "text":  p["text"],
                    "score": p.get("rerank_score", 0),
                    "title": p.get("metadata", {}).get("title", ""),
                }
                for p in passages
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrieve", methods=["POST"])
def retrieve_only():
    """Retrieval-only endpoint (useful for eval without generation)."""
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required"}), 400

    retriever = get_retriever()
    passages = retriever.retrieve(question)
    return jsonify({
        "question": question,
        "passages": [
            {"rank": p["final_rank"], "text": p["text"], "chunk_id": p["chunk_id"]}
            for p in passages
        ],
    })


if __name__ == "__main__":
    # Pre-load models on startup
    get_retriever()
    get_generator()
    app.run(host=API_HOST, port=API_PORT, debug=False)
