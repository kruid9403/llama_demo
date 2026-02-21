# Spider Runbook

This guide covers the autonomous scholarly spider pipeline in this repo.

## Components

- CLI crawler and ingester: `scholarly_spider.py`
- Monitor package/UI: `spider_monitor` (run with `python -m spider_monitor`)
- Vector storage target: Postgres + pgvector

## Prerequisites

1. Activate environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Confirm `.env` has your DB URL and storage paths (example for your SSD):
```bash
DB_PORT=5435
DB_URL=postgresql://dev_user:dev_password@localhost:${DB_PORT}/embedding_db
PGDATA_HOST_PATH=/media/jeremy/SAMSUNG_T9/llama_demo/pgdata
ARXIV_PDF_DIR=/media/jeremy/SAMSUNG_T9/llama_demo/arxiv_pdfs
CRAWL_RAW_DIR=/media/jeremy/SAMSUNG_T9/llama_demo/crawl_raw
SCHOLAR_ALLOWED_DOMAINS=arxiv.org,biorxiv.org,medrxiv.org
EMBEDDING_MODEL_NAME=intfloat/e5-small-v2
EMBEDDING_DIM=384
```

3. Start Postgres:
```bash
docker compose up --build
```

## Run The Spider (CLI)

Basic crawl:
```bash
source .venv/bin/activate
python scholarly_spider.py \
  --seed https://arxiv.org/list/cs.AI/recent \
  --allow-domain arxiv.org \
  --max-pages 40 \
  --max-depth 2 \
  --delay-seconds 1.0
```

Useful flags:
- `--seed` (repeatable): starting URLs
- `--allow-domain` (repeatable): domain allowlist
- `--max-pages`: crawl cap
- `--max-depth`: traversal depth
- `--delay-seconds`: crawl politeness delay
- `--ignore-robots`: disables robots checks
- `--raw-dir`: raw HTML/metadata snapshot folder

## Run The Spider Monitor UI

Start the dedicated monitor package:
```bash
source .venv/bin/activate
python -m spider_monitor
```

Open:
- `http://127.0.0.1:8010`

The UI provides:
- Start/stop controls
- Live crawl log stream
- Counters for pages/chunks/errors/skips
- DB totals (`documents`, `chunks`, `document_references`)

## Expected Runtime Output

Typical log lines:
```text
[start] seeds=1 domains=arxiv.org max_pages=40 max_depth=2
[ingested] depth=0 chunks=6 url=https://arxiv.org/...
[robots-skip] https://...
[fetch-error] https://... -> ...
[done] pages=28 chunks=143 interrupted=False
```

## Verify Data Landed

After a run:
```bash
docker exec -it postgres-pgvector psql -U dev_user -d embedding_db -c "SELECT COUNT(*) FROM documents;"
docker exec -it postgres-pgvector psql -U dev_user -d embedding_db -c "SELECT COUNT(*) FROM chunks;"
docker exec -it postgres-pgvector psql -U dev_user -d embedding_db -c "SELECT COUNT(*) FROM document_references;"
```

## Troubleshooting

- `Permission denied` on SSD paths:
  - Ensure the target folders exist and are writable by your user.
- `tiktoken` download issues in restricted network:
  - Spider falls back to word-based chunking automatically.
- DB connection failures:
  - Verify `DB_URL`, Docker health, and exposed `DB_PORT`.
- `expected 384 dimensions, not 768`:
  - Set `EMBEDDING_MODEL_NAME=intfloat/e5-small-v2` and `EMBEDDING_DIM=384`.
  - If you intentionally want 768-dim embeddings, recreate schema with `EMBEDDING_DIM=768` and reingest.
