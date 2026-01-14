import os
import time
import feedparser
import requests
import hashlib
import numpy as np
import psycopg
from pathlib import Path
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from typing import List
from pgvector.psycopg import register_vector


# =========================
# CONFIG
# =========================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://dev_user:dev_password@localhost:5432/embedding_db",
)

ARXIV_API = "http://export.arxiv.org/api/query"
PDF_DIR = Path("./arxiv_pdfs")
PDF_DIR.mkdir(exist_ok=True)

MAX_RESULTS = 10
EMBEDDING_MODEL = "intfloat/e5-base-v2"  # strong retrieval model

# =========================
# EMBEDDING
# =========================

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        ).tolist()

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
        ).tolist()

embedder = LocalEmbedder()

# =========================
# ARXIV SEARCH
# =========================

import feedparser
from urllib.parse import urlencode

ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 50

def search_arxiv(query: str):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    feed = feedparser.parse(url)

    for entry in feed.entries:
        # Robust extraction with defaults
        arxiv_id = entry.get("id", "").split("/")[-1] if entry.get("id") else None
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()

        # authors may be missing
        authors = []
        if hasattr(entry, "authors"):
            authors = [a.get("name", "").strip() for a in entry.authors if "name" in a]

        # tags / categories may be missing
        categories = []
        if hasattr(entry, "tags"):
            categories = [t.get("term", "").strip() for t in entry.tags if "term" in t]

        # published may be missing (this is what crashed)
        published = getattr(entry, "published", None)

        # pdf link may be missing or differently structured
        pdf_url = None
        if hasattr(entry, "links"):
            for link in entry.links:
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

        # skip entries that are clearly broken
        if not arxiv_id:
            continue

        yield {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": summary,
            "published": published,   # may be None
            "categories": categories,
            "pdf_url": pdf_url,
        }

# =========================
# PDF HANDLING
# =========================

def download_pdf(arxiv_id: str, pdf_url: str) -> Path:
    path = PDF_DIR / f"{arxiv_id}.pdf"
    if path.exists():
        return path
    r = requests.get(pdf_url, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)
    time.sleep(1)  # polite delay
    return path

def extract_pdf_text_safe(pdf_path: Path) -> str:
    try:
        text = extract_text(pdf_path)
    except Exception:
        text = ""
    text = text.replace("\x0c", " ")
    print(f"[extracted] {pdf_path.name} ({len(text.split())} words)")
    return " ".join(text.split())

# =========================
# CHUNKING
# =========================

import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 400
OVERLAP = 10

def chunk_text(text: str, max_tokens=MAX_TOKENS):
    tokens = TOKENIZER.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        piece = tokens[i:i + max_tokens]
        chunks.append(TOKENIZER.decode(piece))
        i += max_tokens - OVERLAP
    return chunks

# =========================
# DB INSERT
# =========================

def ingest_paper(paper: dict):
    # get full text
    full_text = ""
    if paper["pdf_url"]:
        pdf_path = download_pdf(paper["arxiv_id"], paper["pdf_url"])
        full_text = extract_pdf_text_safe(pdf_path)

    text_to_chunk = full_text if full_text.strip() else paper["abstract"]
    checksum = hashlib.sha256(text_to_chunk.encode("utf-8")).hexdigest()

    # chunk it
    chunks = chunk_text(text_to_chunk)
    embeddings = embedder.embed_passages(chunks)

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # insert source
            cur.execute("""
                INSERT INTO sources (name, base_url, doc_type)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id;
            """, ("arXiv", "https://arxiv.org", "preprint"))
            row = cur.fetchone()
            if row:
                source_id = row[0]
            else:
                cur.execute("SELECT id FROM sources WHERE base_url=%s", ("https://arxiv.org",))
                source_id = cur.fetchone()[0]

            # insert document
            cur.execute("""
                INSERT INTO documents (source_id, url, title, version, checksum)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET checksum = EXCLUDED.checksum
                RETURNING id;
            """, (
                source_id,
                f"https://arxiv.org/abs/{paper['arxiv_id']}",
                paper["title"],
                ",".join(paper["categories"]),
                checksum,
            ))
            document_id = cur.fetchone()[0]

            # insert one “full text” section
            cur.execute("""
                INSERT INTO sections (document_id, heading, level, position)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id;
            """, (document_id, "Full Text", 1, 0))
            row = cur.fetchone()
            if row:
                section_id = row[0]
            else:
                cur.execute("""
                    SELECT id FROM sections
                    WHERE document_id=%s AND heading=%s;
                """, (document_id, "Full Text"))
                section_id = cur.fetchone()[0]

            # insert chunks
            for text_chunk, emb in zip(chunks, embeddings):
                content_hash = hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()
                cur.execute("""
                    INSERT INTO chunks (
                        section_id,
                        content,
                        content_hash,
                        content_type,
                        token_count,
                        embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (section_id, content_hash) DO NOTHING;
                """, (
                    section_id,
                    text_chunk,
                    content_hash,
                    "full_text",
                    len(text_chunk.split()),
                    emb,
                ))

    print(f"[ingested] {paper['arxiv_id']}")

# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    while True:
        query = input("Enter arXiv search query: ").strip()

        if not query:
            break

        for paper in search_arxiv(query):
            try:
                ingest_paper(paper)
            except Exception as e:
                print(f"[error] {paper['arxiv_id']} → {e}")