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
SPIDER_MONITOR_DEFAULT_SEEDS=https://arxiv.org/list/cs.AI/new,https://arxiv.org/list/cs.LG/new,https://arxiv.org/list/cs.CL/new,https://arxiv.org/list/cs.CV/new,https://arxiv.org/list/stat.ML/new,https://www.biorxiv.org/content/early/recent,https://www.medrxiv.org/content/early/recent
SPIDER_MONITOR_DEFAULT_DOMAINS=arxiv.org,biorxiv.org,medrxiv.org
EMBEDDING_MODEL_NAME=intfloat/e5-small-v2
EMBEDDING_DIM=384
SPIDER_MAX_CITED_DOIS=200
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
- Seed URL input is optional; blank input uses default scholarly seeds.

Default scholarly seeds used when seed input is blank:
- `https://arxiv.org/list/cs.AI/new`
- `https://arxiv.org/list/cs.LG/new`
- `https://arxiv.org/list/cs.CL/new`
- `https://arxiv.org/list/cs.CV/new`
- `https://arxiv.org/list/stat.ML/new`
- `https://www.biorxiv.org/content/early/recent`
- `https://www.medrxiv.org/content/early/recent`

Suggested additional scholarly sources for new-article crawling:
- `https://www.nature.com/nature/research-articles`
- `https://www.science.org/toc/science/current`
- `https://elifesciences.org/recent-articles`
- `https://journals.plos.org/plosone/browse`
- `https://academic.oup.com/bioinformatics/advance-articles`
Some publishers enforce stricter robots/paywall rules, so expect more `robots-skip` and `skip-content` events on those domains.

Citation weighting behavior:
- During spider ingestion, DOI mentions found in document body text are stored as citation references.
- Similarity search boosts papers with higher in-corpus citation counts.
- Tune weighting with:
  - `CITATION_WEIGHT_BOOST` (default `0.03`)
  - `CITATION_WEIGHT_MAX_BONUS` (default `0.20`)

## Run The Similarity Search UI (Separate Package)

Start the dedicated similarity search package:
```bash
source .venv/bin/activate
python -m similarity_monitor
```

Open:
- `http://127.0.0.1:8020`

## Run The Hypothesis UI (Separate Package)

Start the dedicated hypothesis package:
```bash
source .venv/bin/activate
python -m hypothesis_monitor
```

Open:
- `http://127.0.0.1:8030`

Hypothesis monitor generation settings (optional `.env`):
- `HYPOTHESIS_MODEL_ID` (default inherits `MODEL_ID`, typically `meta-llama/Llama-3.1-8B-Instruct`)
- `HYPOTHESIS_TOKENIZER_ID`
- `HYPOTHESIS_MAX_NEW_TOKENS` (default `420`)
- `HYPOTHESIS_MAX_INPUT_TOKENS` (default `6000`)
- `HYPOTHESIS_CONTEXT_MAX_CHARS` (default `24000`)
- `HYPOTHESIS_TEMPERATURE` (default `0.9`)
- `HYPOTHESIS_TOP_P` (default `0.92`)
- `HYPOTHESIS_REPETITION_PENALTY` (default `1.08`)
Hypothesis monitor requires successful Llama generation; it does not fall back to heuristic output.

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

Run direct similarity search (CLI):
```bash
python -m vector_store.search_database --query "topic modeling with transformers" --top-k 8
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
