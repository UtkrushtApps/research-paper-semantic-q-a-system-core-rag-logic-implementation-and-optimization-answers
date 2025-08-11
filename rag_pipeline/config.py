# rag_pipeline/config.py

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = "research_papers"
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 100  # words
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 5
SEARCH_METRIC = "cosine"  # or "dot_product"

# Optional: for extensibility
SUPPORTED_FILETYPES = ['.txt']

# Prompt template for answer generation
GENERATION_PROMPT = (
    "You are a helpful research assistant. Using the provided context (cited with title, author, year), "
    "answer the question concisely, citing sources inline. Context follows:\n\n{context}\n\nQuestion: {query}\nAnswer:"
)
