import os
import numpy as np
from dotenv import load_dotenv
from voyageai import Client
import pinecone

from utils import prompt_formatter
from llm.llm_openrouter import generate_llm_answer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

if not PINECONE_API_KEY or not VOYAGE_API_KEY:
    raise ValueError("Missing Pinecone/Voyage API keys in .env")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(PINECONE_INDEX_NAME)

# Initialize Voyage
voyage = Client(api_key=VOYAGE_API_KEY)

def embed_query(query: str):
    response = voyage.embed(texts=[query], model=VOYAGE_MODEL)
    embedding = np.array(response.embeddings[0], dtype=np.float32)
    return embedding.tolist()

def retrieve(query: str, top_k: int = 5):
    q_emb = embed_query(query)
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    contexts = [
        {
            "text": match.metadata.get("sentence_chunk", ""),
            "page": match.metadata.get("page_number", "unknown"),
            "score": match.score
        }
        for match in results.matches
    ]
    return contexts

def build_rag_prompt(query: str, top_k: int = 5):
    contexts = retrieve(query, top_k=top_k)
    prompt = prompt_formatter(query, contexts)
    return prompt, contexts

def rag_answer(query: str, top_k: int = 5):
    prompt, contexts = build_rag_prompt(query, top_k)
    answer = generate_llm_answer(prompt)
    return answer
