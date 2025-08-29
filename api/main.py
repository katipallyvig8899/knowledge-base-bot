import os
import io
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

import shutil

load_dotenv()

app = FastAPI(title="Knowledge Base Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_DIR = os.getenv("PERSIST_DIR", "./vectorstore")
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

def load_roles() -> Dict[str, Any]:
    import json
    with open("config/roles.json", "r", encoding="utf-8") as f:
        return json.load(f)

ROLES = load_roles()

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if VECTOR_DB == "chroma":
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

def build_retriever(role: str):
    db = get_vectorstore()
    role_conf = ROLES.get(role, ROLES["employee"])
    allowed = role_conf.get("allowed_categories", ["general"])
    max_lvl = role_conf.get("max_access_level", 1)

    # Build metadata filter
    if "*" in allowed:
        where = {"access_level": {"$lte": max_lvl}}
    else:
        where = {"$and":[
            {"access_level": {"$lte": max_lvl}},
            {"category": {"$in": allowed}}
        ]}

    # For FAISS we don't have metadata-aware retriever; we will filter post-retrieval
    retriever = db.as_retriever(search_kwargs={"k": 5, "filter": where} if isinstance(db, Chroma) else {"k": 8})

    return retriever, db, where

class AskRequest(BaseModel):
    query: str
    role: Optional[str] = "employee"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(payload: AskRequest):
    retriever, db, where = build_retriever(payload.role)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    with open("config/prompt.txt", "r", encoding="utf-8") as f:
        instr = f.read()

    template = """{instructions}

Question: {question}
Context:
{context}

Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["instructions", "question", "context"])

    # Retrieve docs
    docs = retriever.get_relevant_documents(payload.query)

    # manual metadata filter for FAISS
    if isinstance(db, FAISS):
        def allow(doc: Document):
            meta = doc.metadata or {}
            lvl_ok = meta.get("access_level", 0) <= where["$and"][0]["access_level"]["$lte"] if "$and" in where else meta.get("access_level", 0) <= where["access_level"]["$lte"]
            if "$and" in where:
                allowed = where["$and"][1]["category"]["$in"]
                return lvl_ok and meta.get("category","") in allowed
            return lvl_ok
        docs = [d for d in docs if allow(d)]

    context = "\n\n".join([f"[{i+1}] {d.page_content[:1500]}" for i,d in enumerate(docs[:4])])

    chain_prompt = prompt.format(instructions=instr, question=payload.query, context=context)
    resp = llm.invoke(chain_prompt)

    sources = [d.metadata for d in docs[:4]]

    return {"answer": resp.content, "sources": sources}

@app.post("/upload")
async def upload(file: UploadFile = File(...), category: str = Form("general"), access_level: int = Form(1)):
    # Save uploaded file into ./data, user will re-run ingestion
    save_path = os.path.join("data", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "saved", "path": save_path, "note": "Run ingestion/ingest.py to index this file."}
