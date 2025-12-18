import spacy

nlp = spacy.load("en_core_web_sm")

def format_prompt(query, contexts):
    context = "\n\n".join(c["text"] for c in contexts)
    return f"""
Answer ONLY from the context below.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()
