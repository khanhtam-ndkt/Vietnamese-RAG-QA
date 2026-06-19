"""
generation/generator.py
Sends retrieved passages + question to Ollama local LLM for answer generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ollama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_NEW_TOKENS, TEMPERATURE


SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi bằng tiếng Việt. Hãy trả lời câu hỏi dựa trên các đoạn văn được cung cấp. 
Chỉ sử dụng thông tin từ các đoạn văn. Nếu không tìm thấy câu trả lời, hãy nói "Tôi không tìm thấy thông tin liên quan."
Trả lời ngắn gọn, chính xác."""


def build_prompt(question: str, passages: list[dict]) -> str:
    context_parts = []
    for i, p in enumerate(passages, 1):
        context_parts.append(f"[{i}] {p['text']}")
    context = "\n\n".join(context_parts)
    return f"Đoạn văn tham khảo:\n{context}\n\nCâu hỏi: {question}\n\nCâu trả lời:"


class Generator:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = OLLAMA_MODEL
        print(f"[Generator] Using Ollama model: {self.model}")
        # Warm-up ping
        try:
            self.client.list()
            print("[Generator] Ollama connection OK ✓")
        except Exception as e:
            print(f"[Generator] WARNING: Could not connect to Ollama at {OLLAMA_BASE_URL}: {e}")
            print("  → Make sure Ollama is running: `ollama serve`")
            print(f"  → Pull the model: `ollama pull {self.model}`")

    def generate(self, question: str, passages: list[dict]) -> dict:
        prompt = build_prompt(question, passages)
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_NEW_TOKENS,
            },
        )
        answer = response["message"]["content"].strip()
        return {
            "answer": answer,
            "model": self.model,
            "passages_used": len(passages),
        }
