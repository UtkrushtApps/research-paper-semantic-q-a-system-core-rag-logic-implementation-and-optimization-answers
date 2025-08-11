# rag_pipeline/document_ingestion.py
import os
import glob
import json
from typing import List, Dict
from .config import SUPPORTED_FILETYPES

def parse_metadata_from_filename(filename: str) -> Dict:
    """
    Expect filenames like: 'Title__Author__Year.txt'
    Return dict with title, author, year
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("__")
    if len(parts) != 3:
        return {"title": name, "author": "", "year": ""}
    title, author, year = parts
    return {"title": title.replace('_', ' '), "author": author.replace('_', ' '), "year": year}


def load_documents_from(dir_path: str) -> List[Dict]:
    documents = []
    for ext in SUPPORTED_FILETYPES:
        for filepath in glob.glob(os.path.join(dir_path, f"*{ext}")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            mdata = parse_metadata_from_filename(filepath)
            documents.append({
                "text": text,
                "metadata": mdata,
                "filepath": filepath
            })
    return documents
