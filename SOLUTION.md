# Solution Steps

1. 1. Define configuration parameters in 'config.py': Chroma connection details, chunk size/overlap, embedding model name, retrieval parameters, supported filetypes, and prompt template.

2. 2. Implement document loading and metadata extraction in 'document_ingestion.py'. Parse filenames (e.g., 'Title__Author__Year.txt') to extract metadata, and read document text.

3. 3. Implement chunking logic in 'chunking.py' to split text into overlapping word-based chunks, ensuring each chunk maintains context and supports overlap for recall.

4. 4. Implement semantic embedding in 'embedding.py' using SentenceTransformers. Efficiently encode batches of text to obtain dense vectors.

5. 5. Implement Chroma vector store abstraction in 'vector_store.py': set up the client connection, (get or create) collection with similarity metric, batch document+embedding ingestion, and top-k query functionality including metadata/document return.

6. 6. Implement answer generation in 'generation.py'. Attempt to use a powerful open LLM (via HuggingFace transformers), with a fallback stub if not available. Prompt includes contextual citations.

7. 7. Orchestrate ingestion and question answering in 'qa_pipeline.py': for ingestion, chunk all documents and embed+store chunks with metadata; for RAG-QA, embed query, retrieve top-k relevant chunks, build a cited context, and generate or fallback an answer.

8. 8. Build a simple CLI in 'cli.py' to enable document ingestion and answer queries interactively, exposing arguments for document dir, ingestion mode, and question input. Show cited context if requested.

9. 9. Optionally, for evaluating and optimizing retrieval: group context chunks by citation (to reduce context dilution), adjust chunking parameters, tune retrieval k, and template prompt for hallucination mitigation.

10. 10. Test ingestion and Q&A via the CLI using sample research question(s), validating that answers cite relevant sources and context comes directly from stored papers.

