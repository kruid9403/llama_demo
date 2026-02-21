# Process: Two-Scientist Scholarly Debate Pipeline

This document describes the exact runtime process used by the current branch.

## 1. Entry Point and Runtime Mode

The web app starts with:

```bash
python web_ui.py
```

`web_ui.py` serves static assets and exposes these API endpoints:

- `POST /api/stream`: starts a streamed two-scientist debate run.
- `POST /api/ask`: runs the same debate flow but returns a full final payload (non-streaming).
- `POST /api/interrupt`: signals active run cancellation.

Only one active debate run is allowed at a time. A second concurrent request receives HTTP `409`.

## 2. Frontend Request/Stream Lifecycle

Frontend (`web/app.js`) behavior:

1. User submits question from the composer.
2. Client calls `POST /api/stream` with:
   - `question`
   - `history`
   - optional `url`
3. Client reads the response stream incrementally.
4. Stream text tokens are rendered immediately.
5. Status messages are parsed from control lines with prefix:
   - `[[STATUS]] ...`
6. Auto-scroll follows output only when the user is near the bottom; manual scroll-up is respected.
7. User can click `Interrupt`, which calls `POST /api/interrupt`.

## 3. Debate Orchestration

Core loop is implemented in `research_debate.py` via `run_research_debate(...)`.

High-level turn sequence:

1. Scientist 1 (`optimist`) proposes a solution attempt with code + explanation.
2. Scientist 2 (`pessimist`) validates Scientist 1's code and reviews viability.
3. Build turn query from:
   - original user question
   - previous scientist response (if available)
4. Ingest scholarly sources for this query:
   - arXiv (`arxiv_ingest.py`)
   - Crossref (`crossref_ingest.py`)
5. Track ingestion counts for this turn.
6. Retrieve evidence chunks from pgvector using semantic vector search over the database (not URL-limited).
7. Generate scientist response from retrieved chunks only.
8. Parse response markers:
   - `Agreement: AGREE|DISAGREE`
   - `Viability: VIABLE|NOT_VIABLE`
   - `Validation: PASS|FAIL|N/A`
   - `Answer: VERIFIED|UNVERIFIED` (Scientist 2 verification verdict)
   - `Conclusion: ...`
9. Continue until one condition is met:
   - Scientist 2 marks `Viability: VIABLE` and `Validation: PASS`,
   - max cycles reached,
   - user interruption.

## 4. Scholarly Ingestion Details

### 4.1 arXiv (`arxiv_ingest.py`)

Per query:

1. Call arXiv API (`export.arxiv.org/api/query`) sorted by relevance.
2. Extract metadata + abstract (+ optional PDF text if enabled).
3. Chunk text with token-based chunking.
4. Embed chunks with `LocalEmbedder` (`intfloat/e5-base-v2`).
5. Upsert into DB tables:
   - `sources`
   - `documents`
   - `sections`
   - `chunks` (including `embedding` and `content_hash`)
6. Return ingestion count (`ingest_query`).

### 4.2 Crossref (`crossref_ingest.py`)

Per query:

1. Call Crossref API (`https://api.crossref.org/works`).
2. Restrict to scholarly publication types:
   - `journal-article`
   - `proceedings-article`
3. Extract title/DOI/abstract/year/venue.
4. Clean abstract text (including JATS/HTML stripping).
5. Chunk + embed + upsert to same DB schema.
6. Return ingestion count (`ingest_crossref_query`).

## 5. Vector Retrieval Grounding

Retrieval is in `vector_store/retrieval.py` (`retrieve_chunks`).

During debate turns:

1. Build query embedding using `LocalEmbedder.embed_query`.
2. Query pgvector using semantic vector similarity over the indexed chunks.
3. Rank by cosine distance (`c.embedding <=> query_vector`).
4. Return top chunks (`content`, `distance`, `document`, `section`).

Important grounding property:

- Scientist generation is fed chunks retrieved from the vector DB via semantic search.
- Ingestion refreshes the corpus, but retrieval is not constrained to only newly ingested URLs.

## 6. Scientist Prompting Contract

Prompt constraints in `research_debate.py` enforce:

- use scholarly/retrieved evidence only,
- no Wikipedia/unsourced claims,
- critique previous scientist evidence,
- cite chunk indices (`[n]`),
- response from the research scientist should be an opinion based on the input context from the vector search,
- response from each research scientist should be a discussion (position, evidence synthesis, critique, and next verification step),
- references should be formatted in APA 7 style,
- references should appear in a final `References (APA 7)` section at the end of each response,
- only references that appear in inline citations should be listed in the final references section,
- use up to 1000 tokens per response,
- end with strict machine-readable lines:
  - `Agreement: AGREE or DISAGREE`
  - `Viability: VIABLE or NOT_VIABLE`
  - `Validation: PASS or FAIL or N/A`
  - `Conclusion: <single sentence>`

These lines are parsed for turn-state and viability decisions.

## 7. Consensus Logic

Viability gate:

1. Scientist 2 evaluates the latest Scientist 1 proposal and code.
2. If Scientist 2 outputs `Viability: VIABLE` and `Validation: PASS`, the run ends.
3. Otherwise, Scientist 1 proposes a revised attempt.

Accepted compact reviewer ending:
- `Answer: VERIFIED` or `Answer: UNVERIFIED`
- `Verification Reason: ...`

If viability reached:

- stream includes `=== Viable Solution Reached ===`
- stream includes final conclusion summary
- run terminates.

## 8. Interrupt and Cancellation Model

Cancellation is shared through `threading.Event`:

1. `/api/interrupt` sets active stop event.
2. Debate loop checks stop flag:
   - before/after ingestion stages
   - between turns
3. Generation is stop-aware via custom Hugging Face stopping criteria:
   - `InterruptStoppingCriteria`
4. On interrupt, stream emits:
   - status update
   - `=== Debate Interrupted ===`

## 9. Streaming Format

Backend emits plain text plus control lines:

- Status control line format:
  - `[[STATUS]] <message>`
- Content tokens streamed continuously for partial rendering.

Frontend parser behavior:

- status lines update status field,
- non-status text appended to assistant bubble incrementally,
- preserves user reading position (non-forced autoscroll when scrolled up).

## 10. Environment Controls

Relevant `.env` variables:

- Model/runtime:
  - `USE_CUDA`
  - `USE_8BIT`
  - `MODEL_ID`
  - `TOKENIZER_ID`
- Embeddings:
  - `EMBEDDING_DEVICE`
- Debate controls:
  - `DEBATE_MAX_TURNS`
  - `DEBATE_ARXIV_RESULTS`
  - `DEBATE_CROSSREF_RESULTS`
  - `DEBATE_MAX_NEW_TOKENS`
- Ingestion controls:
  - `ARXIV_MAX_RESULTS`
  - `ARXIV_DOWNLOAD_PDFS`

## 11. Data Flow Summary (Single Turn)

1. User prompt arrives.
2. Scientist builds search query.
3. arXiv + Crossref searched.
4. Results ingested into pgvector tables.
5. Corpus is refreshed with newly ingested scholarly content.
6. Vector DB queried by semantic similarity for relevant chunks.
7. Retrieved chunks inserted into prompt context.
8. Scientist response generated and streamed.
9. Agreement + conclusion parsed.
10. Next scientist repeats until stop condition.

## 12. Expected Runtime Signals

During a healthy run you should see streamed milestones like:

- `Research Scientist X Search Query: ...`
- `Research Scientist X Ingested: arXiv=N, Crossref=M`
- `Research Scientist X Retrieved chunks from vector DB: K`
- `Research Scientist X Response:`
- terminal state marker (`Consensus`, `Interrupted`, or `Max Turns`).

These markers confirm the system is ingesting, retrieving, and then generating (not only listing references).
