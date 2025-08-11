# rag_pipeline/vector_store.py
from chromadb import Client
import chromadb.config
from chromadb.utils import embedding_functions
from .config import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION_NAME, SEARCH_METRIC
)

def get_chroma_client():
    client = Client(
        chromadb.config.Settings(
            chroma_api_impl="rest",
            chroma_server_host=CHROMA_HOST,
            chroma_server_http_port=CHROMA_PORT,
        )
    )
    return client

def get_or_create_collection(client, collection_name=CHROMA_COLLECTION_NAME):
    try:
        collection = client.get_collection(
            name=collection_name
        )
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": SEARCH_METRIC},
        )
    return collection

def add_embeddings(collection, embeddings, documents, metadata_list):
    ids = [f"doc-{i}" for i in range(len(embeddings))]
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadata_list,
        ids=ids
    )

def query_top_k(collection, query_embedding, k):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for i in range(len(results["ids"][0])):
        hit = {
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }
        hits.append(hit)
    return hits
