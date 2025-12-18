import numpy as np
from vectorstore.pinecone_client import get_index
from llm.openrouter import generate_answer
from utils.text import format_prompt
from voyageai import Client
import streamlit as st

voyage = Client(api_key=st.secrets["VOYAGE_API_KEY"])
index = get_index()

def embed_query(text: str):
    emb = voyage.embed(texts=[text], model=st.secrets["VOYAGE_MODEL"])
    return np.array(emb.embeddings[0], dtype=np.float32).tolist()

def retrieve(query: str, top_k: int):
    q_emb = embed_query(query)
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    return [
        {
            "text": m.metadata.get("sentence_chunk", ""),
            "score": m.score
        }
        for m in res.matches
    ]

def rag_answer(query, top_k, max_tokens, temperature):
    ctx = retrieve(query, top_k)
    prompt = format_prompt(query, ctx)
    return generate_answer(prompt, max_tokens, temperature)
