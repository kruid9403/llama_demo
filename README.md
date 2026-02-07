# llama_demo

Demo project for experimenting with a small RAG pipeline over a local pgvector store.

**Overview**
This repo provides:
1. A pgvector-backed document store (Postgres).
2. Ingest scripts for URLs and arXiv papers.
3. A LangGraph-based retrieval + generation flow.
4. A simple CLI loop and a minimal web UI.

**Setup**
1. Create a virtual environment and install deps.
2. Configure environment variables.
3. Start Postgres with pgvector.
4. Initialize the vector schema.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Configuration**
Set values in `.env`:
- `HF_TOKEN` for Hugging Face model access.
- `DB_PORT` for the Postgres host port used by Docker.
- Optional: `DB_URL` to override the default connection string.
- Optional: `ARXIV_MAX_RESULTS` to cap per-query ingestion (default 10).
- Optional: `ARXIV_DOWNLOAD_PDFS` to fetch full PDFs (`true`/`false`, default `false`).
- Optional: `USE_8BIT` to run the LLM in 8-bit quantized mode (`1`/`0`, default `1`).
- Optional: `EMBEDDING_DEVICE` for embedding model placement (`cpu`/`cuda`, default `cpu`).

Example:
```bash
HF_TOKEN=your_hf_token
DB_PORT=5435
DB_URL=postgresql://dev_user:dev_password@localhost:${DB_PORT}/embedding_db
ARXIV_MAX_RESULTS=10
ARXIV_DOWNLOAD_PDFS=false
```

**Start the Database**
```bash
docker compose up --build
```

If the port is already in use, change `DB_PORT` in `.env` and keep `docker-compose.yml` as-is.

**Initialize Schema**
```bash
python init_vector_database.py
```

**Ingest Content**
1. URL ingestion (optional):
```bash
python init_vector_database.py --url https://example.com/docs
```

2. arXiv ingestion (interactive search):
```bash
python arxiv_ingest.py
```

**Run the App**
CLI loop:
```bash
python run.py
```

Web UI (local server at `http://127.0.0.1:8000`):
```bash
python web_ui.py
```

For each question, the system automatically queries arXiv using the question text, ingests the results, and then answers using the updated vector database.
