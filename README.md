# Knowledge Base Q&A Bot (LangChain + Chroma + Streamlit + FastAPI)

An AI-powered chatbot that answers questions from your internal documents (PDFs, text, policies, manuals).

**Features**
- Retrieval-Augmented Generation using LangChain
- Vector DB: Chroma (switchable to FAISS)
- Multimodal ingestion: PDF and text now; audio-ready stub
- Web UI: Streamlit
- API: FastAPI
- Role-based guardrails via metadata filtering (e.g., HR vs Technical)
- Dockerized (local deployment ready)
- Example documents supported (put your files in `data/`)

---

## Quickstart

1) **Clone & enter**  
```bash
git clone <your-repo-url> knowledge-base-bot
cd knowledge-base-bot
```

2) **Create `.env`**  
```bash
cp .env.example .env
# edit and add your OpenAI key
```

3) **(Option A) Run locally**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Ingest documents in ./data
python ingestion/ingest.py --source_dir ./data --db ./vectorstore --db_type chroma --split_by 1000 --overlap 150 --category auto

# Start API
uvicorn api.main:app --reload --port 8000

# Start UI (new terminal)
streamlit run ui/streamlit_app.py
```

4) **(Option B) Run with Docker**
```bash
docker compose up --build
```

5) **Ask a question**
- Open Streamlit at http://localhost:8501
- Or POST to API: `POST http://localhost:8000/ask`

---

## Folder Layout
```
.
├── api/                  # FastAPI server
├── ui/                   # Streamlit frontend
├── ingestion/            # scripts to index PDFs/text to vector DB
├── data/                 # put your PDFs/text here
├── vectorstore/          # persisted Chroma/FAISS index
├── config/               # settings, prompts
├── scripts/              # helper scripts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Role-Based Guardrails
We tag documents with metadata (`category`, `access_level`). The retriever applies filters based on the role
you pass from the UI/API. Example roles: `employee`, `hr`, `engineer`, `admin`.

- HR role → can access HR docs
- Engineer role → can access Technical docs
- Employee role → default view (general)
- Admin role → full access

You can customize in `config/roles.json`.

---

## Switch Vector DB
Default: **Chroma**. To use FAISS, run ingestion with `--db_type faiss`. The API and UI auto-detect based on
what exists in `vectorstore/`.

---

## Extend to Audio (Stretch)
We include a stub endpoint and code path to ingest audio transcripts. If you add Whisper or other speech-to-text,
save transcripts as `.txt` in `data/` and re-run `ingest.py`.

---

## Security Reminder
This sample demonstrates concepts. For production, add auth (JWT/OAuth), https, rate-limits, secret management, and robust RBAC.
