import os
import requests
from dotenv import load_dotenv

load_dotenv()

# API key + model
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nex-agi/deepseek-v3.1-nex-n1:free")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "X-API-KEY": OPENROUTER_API_KEY,     # DeepSeek models need BOTH
    "HTTP-Referer": "http://localhost",  
    "X-Title": "Nutrition-RAG",
    "Content-Type": "application/json"
}


def generate_llm_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.1):
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "Answer ONLY using the provided context. If unknown, say 'I don’t know'."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=40)
        print("Status code:", response.status_code)
        print("Raw response:", response.text)

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("\n ERROR calling OpenRouter:", e)
        return "API error."


# ---------------------------------------------------------
# Alias expected by retrieval.py
# ---------------------------------------------------------

def generate_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.1):
    return generate_llm_answer(prompt, max_tokens, temperature)
