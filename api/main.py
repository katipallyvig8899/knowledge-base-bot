import os
import shutil
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

load_dotenv()

app = FastAPI(title="Knowledge Base Q&A API", version="2.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ENV Config ===
PERSIST_DIR = os.getenv("PERSIST_DIR", "./vectorstore")
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")  # lightweight; replace with mistral if GPU

# === Load Roles ===
def load_roles() -> Dict[str, Any]:
    import json
    with open("config/roles.json", "r", encoding="utf-8") as f:
        return json.load(f)

ROLES = load_roles()

# === Vectorstore Loader ===
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if VECTOR_DB == "chroma":
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

# === HuggingFace LLM Loader ===
def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
    )
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

# === Role-based retriever ===
def build_retriever(role: str):
    db = get_vectorstore()
    role_conf = ROLES.get(role, ROLES["employee"])
    allowed = role_conf.get("allowed_categories", ["general"])
    max_lvl = role_conf.get("max_access_level", 1)

    if isinstance(db, Chroma):
        if "*" in allowed:
            where = {"access_level": {"$lte": max_lvl}}
        else:
            where = {"$and":[
                {"access_level": {"$lte": max_lvl}},
                {"category": {"$in": allowed}}
            ]}
        retriever = db.as_retriever(search_kwargs={"k": 5, "filter": where})
    else:
        # FAISS fallback (no metadata filtering inside DB)
        retriever = db.as_retriever(search_kwargs={"k": 8})
    return retriever

# === Request Model ===
class AskRequest(BaseModel):
    query: str
    role: Optional[str] = "employee"

# === Routes ===
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(payload: AskRequest):
    retriever = build_retriever(payload.role)
    llm = get_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa.invoke({"query": payload.query})
    answer = result["result"]
    sources = [d.metadata for d in result["source_documents"]]

    return {"answer": answer, "sources": sources}

@app.post("/upload")
async def upload(file: UploadFile = File(...), category: str = Form("general"), access_level: int = Form(1)):
    save_path = os.path.join("data", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "saved", "path": save_path, "note": "Run ingestion/ingest.py to index this file."}
