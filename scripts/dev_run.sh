#!/usr/bin/env bash
set -e
uvicorn api.main:app --reload --port 8000 &
streamlit run ui/streamlit_app.py
