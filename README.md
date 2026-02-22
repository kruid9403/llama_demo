# llama_demo

Demo project for experimenting with a small RAG pipeline over a local pgvector store.

For spider-only setup and operation, see `SPIDER_README.md`.

**Overview**
This repo provides:
1. A pgvector-backed document store (Postgres).
2. Ingest scripts for URLs and arXiv papers.
3. A dedicated `vector_store` package for embeddings, schema init, and retrieval.
4. A scholarly spider for crawling allowed research domains and storing citation metadata.
5. A dedicated `spider_monitor` package for autonomous crawl control/monitoring.
6. A dedicated `similarity_monitor` package for reference-aware vector search.
7. A dedicated `hypothesis_monitor` package for prompt review + forward-looking research hypotheses.
8. A LangGraph-based retrieval + generation flow.
9. A simple CLI loop and a minimal web UI.

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
- Optional: `CITATION_WEIGHT_BOOST` for citation-based ranking boost in similarity search (default `0.03`).
- Optional: `CITATION_WEIGHT_MAX_BONUS` cap for citation-based ranking bonus (default `0.20`).
- Optional: `SPIDER_MONITOR_DEFAULT_SEEDS` as comma-separated fallback seed URLs when spider monitor seed input is blank.
- Optional: `SPIDER_MONITOR_DEFAULT_DOMAINS` as comma-separated fallback allowlist when domain input is blank (default `arxiv.org,biorxiv.org,medrxiv.org`).
- Optional: `SPIDER_MAX_CITED_DOIS` cap for citation DOI mentions captured per crawled document (default `200`).
- Optional: `SIMILARITY_MONITOR_PORT` for the dedicated similarity UI port (default `8020`).
- Optional: `HYPOTHESIS_MONITOR_PORT` for the dedicated hypothesis UI port (default `8030`).
- Optional: `HYPOTHESIS_MODEL_ID` and `HYPOTHESIS_TOKENIZER_ID` for hypothesis generation model selection (defaults to Llama 3 values from `MODEL_ID`/`TOKENIZER_ID`).
- Optional: `HYPOTHESIS_MAX_NEW_TOKENS`, `HYPOTHESIS_MAX_INPUT_TOKENS`, `HYPOTHESIS_CONTEXT_MAX_CHARS`.
- Optional: `HYPOTHESIS_TEMPERATURE`, `HYPOTHESIS_TOP_P`, `HYPOTHESIS_REPETITION_PENALTY`.
- Optional: `HYPOTHESIS_NOVELTY_REWRITE_ATTEMPTS` for retrying once (or more) when output looks like stitched existing theories (default `1`).
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

Similarity search UI (local server at `http://127.0.0.1:8020`):
```bash
python -m similarity_monitor
```

Hypothesis UI (local server at `http://127.0.0.1:8030`):
```bash
python -m hypothesis_monitor
```

Legacy compatible launcher (same behavior):
```bash
python spider_ui.py
```

If you leave Spider Monitor seed URLs blank, it now auto-uses this scholarly seed set:
- `https://arxiv.org/list/cs.AI/new`
- `https://arxiv.org/list/cs.LG/new`
- `https://arxiv.org/list/cs.CL/new`
- `https://arxiv.org/list/cs.CV/new`
- `https://arxiv.org/list/stat.ML/new`
- `https://www.biorxiv.org/content/early/recent`
- `https://www.medrxiv.org/content/early/recent`

For each question in the Web UI, two research-scientist agents run a turn-based debate:
1. Scientist 1 searches scholarly sources (arXiv + Crossref), ingests results, queries the vector DB, and responds from retrieved chunks.
2. Scientist 2 searches again based on the latest response, ingests results, queries the vector DB, and responds from retrieved chunks.
3. The loop continues until consensus is detected, max turns are reached, or you click `Interrupt` in the UI.

The system avoids non-scholarly sources (for example, Wikipedia) in this debate flow.

**Similarity Search For References**
Run similarity search directly against your configured `DB_URL` (including SSD-backed Postgres on `SAMSUNG_T9`):
```bash
python -m vector_store.search_database --query "retrieval augmented generation for medical QA" --top-k 8
```

Similarity ranking is citation-weighted: papers that are referenced more frequently by other ingested papers receive a ranking boost.

Optional filters:
```bash
python -m vector_store.search_database \
  --query "diffusion transformers" \
  --url https://arxiv.org/abs/2401.12345 \
  --json
```

You can run similarity search in its dedicated UI (`http://127.0.0.1:8020`) via `python -m similarity_monitor`.
You can run prompt-review + forward-looking hypothesis generation in `http://127.0.0.1:8030` via `python -m hypothesis_monitor`.
Hypothesis monitor flow is: prompt review -> similarity search over vector DB -> context prompt for Llama 3 -> generated forward-looking hypothesis with references.
Hypothesis monitor is strict-mode: if Llama generation fails, the request returns an error (no heuristic fallback).
Hypothesis outputs are novelty-validated: stitched-theory phrasing (`combine`, `integrate`, `hybrid`, etc.) is rejected and retried, then returned as an explicit error if still non-novel.
