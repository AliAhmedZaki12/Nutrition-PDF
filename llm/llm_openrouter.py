import requests
import streamlit as st

URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
    "HTTP-Referer": "https://streamlit.io",
    "Content-Type": "application/json",
}

def generate_answer(prompt, max_tokens, temperature):
    payload = {
        "model": st.secrets["OPENROUTER_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = requests.post(URL, headers=HEADERS, json=payload, timeout=40)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

