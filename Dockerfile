# Build stage is optional here; simple runtime image
FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured/pypdf may be needed
RUN apt-get update && apt-get install -y build-essential poppler-utils libmagic1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PERSIST_DIR=./vectorstore
ENV VECTOR_DB=chroma
ENV EMBEDDING_MODEL=text-embedding-3-small
ENV CHAT_MODEL=gpt-4o-mini

EXPOSE 8000 8501

CMD bash -c "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
