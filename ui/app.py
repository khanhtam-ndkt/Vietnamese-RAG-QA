"""
ui/app.py
Streamlit frontend for the Vietnamese RAG QA system.

Run: streamlit run ui/app.py
(Requires Flask API to be running at localhost:5000)
"""

import requests
import streamlit as st

API_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Hệ thống Hỏi Đáp Tiếng Việt",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Vietnamese RAG Question Answering")
st.caption("UIT-ViQuAD 2.0 | gte-multilingual-base + BM25 + RRF + Cross-Encoder + Ollama")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Mode", ["Full RAG (answer + passages)", "Retrieve only"])
    st.markdown("---")
    st.markdown("**Stack:**")
    st.markdown("- Dense: `gte-multilingual-base`")
    st.markdown("- Sparse: BM25 (rank_bm25)")
    st.markdown("- Fusion: RRF (k=60)")
    st.markdown("- Reranker: `ms-marco-MiniLM-L-6`")
    st.markdown("- LLM: Ollama `qwen2.5:3b`")

# ── Health check ──────────────────────────────────────────────────────────────
try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    if r.status_code == 200:
        st.success("✅ API connected")
    else:
        st.error("❌ API returned non-200")
except Exception:
    st.error(f"❌ Cannot reach API at {API_URL}. Run `python api/app.py` first.")

# ── Main interface ────────────────────────────────────────────────────────────
question = st.text_input(
    "Nhập câu hỏi tiếng Việt:",
    placeholder="Ví dụ: Thủ đô của Việt Nam là gì?",
)

col1, col2 = st.columns([1, 4])
with col1:
    submitted = st.button("🔍 Tìm kiếm", type="primary")

if submitted and question.strip():
    endpoint = "/ask" if "Full" in mode else "/retrieve"
    with st.spinner("Đang xử lý …"):
        try:
            resp = requests.post(
                f"{API_URL}{endpoint}",
                json={"question": question},
                timeout=120,
            )
            data = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    # Answer
    if "answer" in data:
        st.markdown("### 💬 Answer")
        st.info(data["answer"])
        st.caption(f"Model: `{data.get('model', '-')}`")

    # Passages
    st.markdown("### 📄 Retrieved Passages")
    passages = data.get("passages", [])
    for p in passages:
        rank  = p.get("rank", "?")
        text  = p.get("text", "")
        score = p.get("score", None)
        title = p.get("title", "")
        label = f"**#{rank}** — {title}" if title else f"**#{rank}**"
        if score is not None:
            label += f"  *(rerank score: {score:.4f})*"
        with st.expander(label):
            st.write(text)

elif submitted:
    st.warning("Vui lòng nhập câu hỏi.")
