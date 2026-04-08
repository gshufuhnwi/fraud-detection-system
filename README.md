# Fraud Detection System

This project contains:

- FastAPI backend in `backend/`
- Streamlit frontend in `frontend/`
- GitHub Actions retraining workflow in `.github/workflows/train.yml`

## Run backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 10000

## Run frontend locally

```bash
cd frontend
pip install -r requirements.txt
streamlit run frontend.py