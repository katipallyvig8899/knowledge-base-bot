import os
import argparse
import glob
import uuid
from typing import List, Dict
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def discover_files(source_dir: str) -> List[str]:
    patterns = ["**/*.pdf", "**/*.txt", "**/*.md"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(source_dir, p), recursive=True))
    return files

def load_docs(paths: List[str]):
    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(p)
            docs.extend(loader.load())
        elif ext in [".txt", ".md"]:
            loader = TextLoader(p, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def tag_metadata(docs, category: str, access_level: int):
    for d in docs:
        meta = d.metadata or {}
        # infer category from path if auto
        if category == "auto":
            path = (meta.get("source") or meta.get("file_path") or "").lower()
            if "hr" in path:
                meta["category"] = "hr"
            elif "tech" in path or "manual" in path or "it" in path:
                meta["category"] = "technical"
            else:
                meta["category"] = "general"
        else:
            meta["category"] = category
        meta["access_level"] = access_level
        d.metadata = meta
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default="./data")
    ap.add_argument("--db", default="./vectorstore")
    ap.add_argument("--db_type", choices=["chroma", "faiss"], default=os.getenv("VECTOR_DB", "chroma"))
    ap.add_argument("--split_by", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--category", default="auto")
    ap.add_argument("--access_level", type=int, default=1)
    args = ap.parse_args()

    files = discover_files(args.source_dir)
    if not files:
        print("No files found to ingest.")
        return

    raw_docs = load_docs(files)
    raw_docs = tag_metadata(raw_docs, args.category, args.access_level)

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.split_by, chunk_overlap=args.overlap)
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    os.makedirs(args.db, exist_ok=True)

    if args.db_type == "chroma":
        vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=args.db)
        vectordb.persist()
        print(f"Ingested {len(chunks)} chunks into Chroma at {args.db}")
    else:
        vectordb = FAISS.from_documents(chunks, embedding=embeddings)
        vectordb.save_local(args.db)
        print(f"Ingested {len(chunks)} chunks into FAISS at {args.db}")

if __name__ == "__main__":
    main()
