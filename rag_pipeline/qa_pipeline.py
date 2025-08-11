# rag_pipeline/qa_pipeline.py
from .config import (
    CHROMA_COLLECTION_NAME, CHROMA_HOST, CHROMA_PORT,
    EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K
)
from .vector_store import get_chroma_client, get_or_create_collection, query_top_k
from .embedding import embed_texts
from .generation import SimpleGenerator, fallback_generate
from .chunking import chunk_text
from .document_ingestion import load_documents_from
import numpy as np

class ResearchRAGQA:
    def __init__(self, collection_name=CHROMA_COLLECTION_NAME):
        self.collection_name = collection_name
        self.chroma_client = get_chroma_client()
        self.collection = get_or_create_collection(self.chroma_client, collection_name)
        try:
            self.generator = SimpleGenerator()
        except ImportError:
            self.generator = None

    def ingest_corpus(self, doc_dir):
        docs = load_documents_from(doc_dir)
        all_chunks = []
        chunk_metadatas = []
        for doc in docs:
            source_mdata = doc["metadata"].copy()
            chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk in enumerate(chunks):
                m = source_mdata.copy()
                m['chunk_index'] = idx
                all_chunks.append(chunk)
                chunk_metadatas.append(m)
        print(f"Total chunks: {len(all_chunks)}")
        # Embedding
        chunk_embeds = embed_texts(all_chunks, EMBEDDING_MODEL_NAME)
        # Store
        # Chroma requires batch size <= 2048; chunk
        batch_size = 512
        for i in range(0, len(all_chunks), batch_size):
            b_chunks = all_chunks[i:i+batch_size]
            b_embeds = chunk_embeds[i:i+batch_size]
            b_metas = chunk_metadatas[i:i+batch_size]
            self.collection.add(
                embeddings=b_embeds.tolist(),
                documents=b_chunks,
                metadatas=b_metas,
                ids=[f"doc-{i}" for i in range(i, i+len(b_chunks))]
            )
        print("Ingestion complete.")

    def ask(self, query, top_k=RETRIEVAL_TOP_K, show_chunks=False):
        q_embed = embed_texts([query], EMBEDDING_MODEL_NAME)[0]
        top_chunks = query_top_k(self.collection, q_embed.tolist(), top_k)
        # Mitigation: Only use distinct papers, group by citation
        context_blocks = []
        cited = set()
        for hit in top_chunks:
            m = hit['metadata']
            citation = f"[{m.get('title')}, {m.get('author')}, {m.get('year')}]"
            # Avoid repeat citations for conciseness
            if citation not in cited:
                context_blocks.append(f"{hit['document']} {citation}")
                cited.add(citation)
            if len(context_blocks) >= top_k:
                break
        context = '\n\n'.join(context_blocks)
        if show_chunks:
            print("\n---- Retrieved Context Chunks ----")
            for i, ctx in enumerate(context_blocks):
                print(f"Context {i+1}:", ctx[:300], "...\n")
        if self.generator:
            answer = self.generator.generate(context, query)
        else:
            answer = fallback_generate(context, query)
        return answer
