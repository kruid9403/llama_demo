import argparse
import hashlib
import os
from typing import List
from urllib.parse import urlparse

import psycopg
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from vector_store.embeddings import EMBEDDING_DIM, LocalEmbedder

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 400
OVERLAP = 50

embedder: LocalEmbedder | None = None


def get_embedder() -> LocalEmbedder:
    global embedder
    if embedder is None:
        embedder = LocalEmbedder()
    return embedder


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def chunk_tokens(text: str) -> List[str]:
    tokens = TOKENIZER.encode(text)
    chunks: List[str] = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + MAX_TOKENS]
        chunks.append(TOKENIZER.decode(chunk))
        i += MAX_TOKENS - OVERLAP

    return chunks


def _is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _get_or_create_source(cur: psycopg.Cursor, name: str, base_url: str, doc_type: str) -> int:
    cur.execute("SELECT id FROM sources WHERE base_url = %s ORDER BY id ASC LIMIT 1;", (base_url,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO sources (name, base_url, doc_type)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (name, base_url, doc_type),
    )
    return cur.fetchone()[0]


def add_vectors_from_url(url: str) -> int:
    if not url:
        raise ValueError("URL is required.")
    if not _is_valid_http_url(url):
        raise ValueError(f"Invalid URL '{url}'. Use a full http(s) URL.")

    print(f"Scraping: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    clean_text = soup.get_text(separator=" ", strip=True)
    if not clean_text:
        print("No extractable text found.")
        return 0

    checksum = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    chunks = chunk_tokens(clean_text)
    embeddings = get_embedder().embed_passages(chunks)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            source_id = _get_or_create_source(cur, "Documentation Site", url, "docs")

            cur.execute(
                """
                INSERT INTO documents (source_id, url, checksum)
                VALUES (%s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET checksum = EXCLUDED.checksum
                RETURNING id;
                """,
                (source_id, url, checksum),
            )
            document_id = cur.fetchone()[0]

            cur.execute(
                """
                SELECT id
                FROM sections
                WHERE document_id = %s AND heading = %s AND position = %s
                ORDER BY id ASC
                LIMIT 1;
                """,
                (document_id, "Main Content", 0),
            )
            row = cur.fetchone()
            if row is not None:
                section_id = row[0]
            else:
                cur.execute(
                    """
                    INSERT INTO sections (document_id, heading, level, position)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (document_id, "Main Content", 1, 0),
                )
                section_id = cur.fetchone()[0]

            inserted = 0
            for text_chunk, emb in zip(chunks, embeddings):
                chunk_hash = hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO chunks (
                        section_id,
                        content,
                        content_hash,
                        content_type,
                        token_count,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (section_id, content_hash) DO NOTHING
                    RETURNING id;
                    """,
                    (
                        section_id,
                        text_chunk,
                        chunk_hash,
                        "prose",
                        count_tokens(text_chunk),
                        emb,
                    ),
                )
                if cur.fetchone() is not None:
                    inserted += 1

    print(f"Inserted {inserted} chunks into pgvector from {url}.")
    return inserted


def init_vector_database() -> None:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    doc_type TEXT,
                    created_at TIMESTAMP DEFAULT now()
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    version TEXT,
                    last_scraped TIMESTAMP,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT now()
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sections (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    heading TEXT,
                    level INTEGER,
                    anchor TEXT,
                    position INTEGER
                );
                """
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    content_hash TEXT,
                    content_type TEXT,
                    language TEXT,
                    token_count INTEGER,
                    embedding VECTOR({EMBEDDING_DIM}),
                    created_at TIMESTAMP DEFAULT now()
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_references (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    source_url TEXT,
                    doi TEXT,
                    venue TEXT,
                    published_date TEXT,
                    authors TEXT,
                    created_at TIMESTAMP DEFAULT now()
                );
                """
            )

            # Migrations for pre-existing databases
            cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;")
            cur.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding VECTOR({EMBEDDING_DIM});")

            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS chunks_section_content_hash_uidx
                ON chunks (section_id, content_hash);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS document_references_document_source_doi_uidx
                ON document_references (
                    document_id,
                    COALESCE(source_url, ''),
                    COALESCE(doi, '')
                );
                """
            )

    print("Vector schema initialized successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize vector DB schema and optionally ingest URLs.")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Optional URL to scrape and ingest. Can be passed multiple times.",
    )
    args = parser.parse_args()

    init_vector_database()

    if args.url:
        for url in args.url:
            try:
                add_vectors_from_url(url.strip())
            except Exception as exc:
                print(f"Failed to ingest {url}: {exc}")
    else:
        print("No URLs provided. Runtime ingestion happens automatically from searches during operations.")


if __name__ == "__main__":
    main()
