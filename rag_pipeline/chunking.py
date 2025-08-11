# rag_pipeline/chunking.py
import re
from typing import List

def word_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text into overlapping segments (by words)."""
    words = word_tokenize(text)
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(words):
        chunk_words = words[start:start+chunk_size]
        if not chunk_words:
            break
        # Join preserving basic punctuation
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        start += step
    return chunks
