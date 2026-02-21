# llama_demo

Demo project for experimenting with a small RAG pipeline over a local pgvector store.

For spider-only setup and operation, see `SPIDER_README.md`.

**Overview**
This repo provides:
1. A pgvector-backed document store (Postgres).
2. Ingest scripts for URLs and arXiv papers.
3. A dedicated `vector_store` package for embeddings, schema init, and retrieval.
4. A scholarly spider for crawling allowed research domains and storing citation metadata.
5. A LangGraph-based retrieval + generation flow.
6. A simple CLI loop and a minimal web UI.

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
- Optional: `ARXIV_PDF_DIR` for where downloaded PDFs are stored (supports external drives).
- Optional: `PGDATA_HOST_PATH` for where Docker stores Postgres files (supports external drives).
- Optional: `CRAWL_RAW_DIR` to store raw crawled HTML/metadata snapshots from the spider.
- Optional: `SCHOLAR_ALLOWED_DOMAINS` as a comma-separated allowlist for spider crawling.
- Optional: `USE_8BIT` to run the LLM in 8-bit quantized mode (`1`/`0`, default `1`).
- Optional: `EMBEDDING_DEVICE` for embedding model placement (`cpu`/`cuda`, default `cpu`).
- Optional: `EMBEDDING_MODEL_NAME` for embeddings (default `intfloat/e5-small-v2`, 384-dim).
- Optional: `EMBEDDING_DIM` to match your pgvector column size (default `384`).
- Optional: `DEBATE_MAX_TURNS` for two-scientist debate loop length (default `8`).
- Optional: `DEBATE_ARXIV_RESULTS` and `DEBATE_CROSSREF_RESULTS` to control ingestion per turn.

Example:
```bash
HF_TOKEN=your_hf_token
DB_PORT=5435
DB_URL=postgresql://dev_user:dev_password@localhost:${DB_PORT}/embedding_db
ARXIV_MAX_RESULTS=10
ARXIV_DOWNLOAD_PDFS=false
ARXIV_PDF_DIR=/media/jeremy/ExternalSSD/llama/arxiv_pdfs
PGDATA_HOST_PATH=/media/jeremy/ExternalSSD/llama/pgdata
CRAWL_RAW_DIR=/media/jeremy/ExternalSSD/llama/crawl_raw
SCHOLAR_ALLOWED_DOMAINS=arxiv.org,biorxiv.org,medrxiv.org
```

**Start the Database**
```bash
docker compose up --build
```

If the port is already in use, change `DB_PORT` in `.env` and keep `docker-compose.yml` as-is.
If `PGDATA_HOST_PATH` is set, Docker will store database files at that host path (for example on an external drive).

**Initialize Schema**
```bash
python -m vector_store.init_vector_database
```

**Ingest Content**
1. URL ingestion (optional):
```bash
python -m vector_store.init_vector_database --url https://example.com/docs
```

2. arXiv ingestion (interactive search):
```bash
python arxiv_ingest.py
```

3. Scholarly spider ingestion (crawl + chunk + vectorize + store references):
```bash
python scholarly_spider.py \
  --seed https://arxiv.org/list/cs.AI/recent \
  --allow-domain arxiv.org \
  --max-pages 40 \
  --max-depth 2
```

The spider stores vectors in `chunks` and citation metadata in `document_references` (`source_url`, `doi`, `venue`, `published_date`, `authors`).

**Run the App**
CLI loop:
```bash
python run.py
```

Web UI (local server at `http://127.0.0.1:8000`):
```bash
python web_ui.py
```

Spider monitor UI (local server at `http://127.0.0.1:8010`):
```bash
python -m spider_monitor
```

Legacy compatible launcher (same behavior):
```bash
python spider_ui.py
```

For each question in the Web UI, two research-scientist agents run a turn-based debate:
1. Scientist 1 searches scholarly sources (arXiv + Crossref), ingests results, queries the vector DB, and responds from retrieved chunks.
2. Scientist 2 searches again based on the latest response, ingests results, queries the vector DB, and responds from retrieved chunks.
3. The loop continues until consensus is detected, max turns are reached, or you click `Interrupt` in the UI.

The system avoids non-scholarly sources (for example, Wikipedia) in this debate flow.
