# utils.py
#(text splitting, chunking, prompt)
import re
from typing import List, Dict
from spacy.lang.en import English


# 1. Basic text cleanup

def text_formatter(text: str) -> str:
    """
    Cleans raw text extracted from PDFs.
    Mirrors the simple preprocessing from the notebook.
    """
    cleaned = text.replace("\n", " ").strip()
    return cleaned



# 2. spaCy sentence splitter

# Initialize spaCy sentencizer
nlp = English()
nlp.add_pipe("sentencizer")

def split_sentences_spacy(text: str) -> List[str]:
    """
    Splits a block of text into sentences using spaCy sentencizer.
    Returns a list of clean sentence strings.
    """
    doc = nlp(text)
    return [str(s).strip() for s in doc.sents]



# 3. Split list into chunks

def split_list(input_list: list, slice_size: int) -> list:
    """
    Splits a list into sublists of size slice_size.
    e.g. list of 17 sentences -> [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]



# 4. Create chunks with statistics

def create_sentence_chunks(sentences: List[str], page_number: int, chunk_size: int = 10) -> List[Dict]:
    """
    Converts a list of sentences into paragraph-like chunks.
    Also computes token/char/word stats like the notebook.

    Returns a list of chunk dictionaries:
    {
        "page_number": int,
        "sentence_chunk": str,
        "chunk_char_count": int,
        "chunk_word_count": int,
        "chunk_token_count": float
    }
    """
    chunks = []
    sentence_groups = split_list(sentences, chunk_size)

    for group in sentence_groups:
        joined = "".join(group)
        joined = joined.replace("  ", " ").strip()

        # Fix patterns like ".A" → ". A"
        joined = re.sub(r'\.([A-Z])', r'. \1', joined)

        chunk_dict = {
            "page_number": page_number,
            "sentence_chunk": joined,
            "chunk_char_count": len(joined),
            "chunk_word_count": len(joined.split(" ")),
            "chunk_token_count": len(joined) / 4  # approx tokens
        }
        chunks.append(chunk_dict)

    return chunks


# 5. Filter tiny chunks

def filter_chunks(chunks: List[Dict], min_token_length: int = 30) -> List[Dict]:
    """
    Removes small irrelevant chunks such as headers/footers.
    """
    return [c for c in chunks if c["chunk_token_count"] > min_token_length]


# 6. Retrieval → Prompt formatter


def prompt_formatter(query: str, context_items: list) -> str:
    """
    Formats retrieved chunks into a RAG prompt that encourages
    detailed, well-structured answers while remaining grounded.
    """

    context_block = ""
    for item in context_items:
        source = item.get("source", item.get("page", "-"))
        text = item.get("text", item.get("sentence_chunk", ""))
        context_block += f"Source (Page {source}):\n{text}\n\n"

    prompt = f"""
You are a domain-aware assistant answering STRICTLY based on the provided context.

Instructions:
- Provide a **detailed and well-structured explanation**
- Use paragraphs or bullet points where appropriate
- Explain concepts clearly and thoroughly
- Do NOT introduce information not present in the context
- If the answer is not found in the context, say: "I don’t know."

CONTEXT:
{context_block}

QUESTION:
{query}

DETAILED ANSWER:
""".strip()

    return prompt
