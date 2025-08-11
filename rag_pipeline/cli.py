# rag_pipeline/cli.py
import argparse
from .qa_pipeline import ResearchRAGQA
import sys

def main():
    parser = argparse.ArgumentParser(description="Research Paper Semantic Q&A System")
    parser.add_argument('--doc_dir', type=str, help='Directory with academic documents to ingest')
    parser.add_argument('--ingest', action='store_true', help='Ingest all documents in --doc_dir')
    parser.add_argument('--query', type=str, help='Pose a question to the system')
    parser.add_argument('--top_k', type=int, default=None, help='Number of top context chunks')
    parser.add_argument('--show_chunks', action='store_true', help='Show retrieved context chunks')

    args = parser.parse_args()
    rag = ResearchRAGQA()

    if args.ingest:
        if not args.doc_dir:
            print("Specify --doc_dir for ingestion.")
            sys.exit(1)
        print(f"Ingesting documents from {args.doc_dir} ...")
        rag.ingest_corpus(args.doc_dir)

    if args.query:
        k = args.top_k if args.top_k is not None else None
        answer = rag.ask(args.query, top_k=k, show_chunks=args.show_chunks)
        print("\n=== Answer ===\n")
        print(answer)

if __name__ == "__main__":
    main()
