import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Knowledge Base Q&A", page_icon="📘", layout="centered")
st.title("📘 Knowledge Base Q&A")

with st.sidebar:
    st.header("Settings")
    role = st.selectbox("Role", ["employee", "hr", "engineer", "admin"], index=0)
    st.markdown("Upload new documents (will be saved to backend `data/`) then re-run ingestion:")
    up = st.file_uploader("Upload PDF/TXT/MD", type=["pdf","txt","md"])
    category = st.selectbox("Category (tag)", ["general","policy","hr","technical","it"], index=0)
    access_level = st.slider("Access level", 1, 10, 1)
    if up is not None:
        files = {"file": (up.name, up.getbuffer())}
        data = {"category": category, "access_level": str(access_level)}
        try:
            r = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=60)
            st.success(f"Uploaded: {r.json()}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

st.write("Ask a question about your documents:")
q = st.text_input("Your question")

if st.button("Ask") and q.strip():
    payload = {"query": q, "role": role}
    try:
        r = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        if r.status_code == 200:
            data = r.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Sources")
            st.json(data["sources"])
        else:
            st.error(f"Error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
