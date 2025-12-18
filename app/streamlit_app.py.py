import sys
from pathlib import Path
import streamlit as st

# -----------------------------
# Path fix (Cloud-safe)
# -----------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.retrieval import rag_answer, retrieve

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Nutrition RAG",
    page_icon="🥗",
    layout="wide"
)

st.title("🥗 Nutrition RAG Assistant")
st.caption("Production-ready RAG system using Pinecone + Voyage + OpenRouter")

# -----------------------------
# Sidebar
# -----------------------------
top_k = st.sidebar.slider("Top-K Chunks", 1, 10, 4)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max Tokens", 128, 2048, 512)

# -----------------------------
# Input
# -----------------------------
query = st.text_input("Ask a question")
if st.button("Ask") and query.strip():

    with st.spinner("Retrieving context..."):
        contexts = retrieve(query, top_k)

    for i, c in enumerate(contexts, 1):
        with st.expander(f"Chunk {i} | Score {c['score']:.3f}"):
            st.write(c["text"])

    with st.spinner("Generating answer..."):
        answer = rag_answer(
            query,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens
        )

    st.success(answer)
